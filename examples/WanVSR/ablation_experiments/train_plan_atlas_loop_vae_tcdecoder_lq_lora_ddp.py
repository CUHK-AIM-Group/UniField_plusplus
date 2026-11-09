#!/usr/bin/env python3
"""Plan+atlas loop ablation with the parent DiT LoRA plus LQ adapters.

This is intentionally a separate entry from ``train_plan_atlas_loop_ddp.py``.
It preserves that experiment's Stage-2 plan/atlas conditioning and Stage-3
training settings, keeps the parent experiment's DiT/Mask/loop adapters
trainable, and adds a low-rank adapter to:

* ``DiT.LQ_proj_in``, the LQ-to-token injection network.

VAE and TCDecoder stay frozen; TCDecoder is used only for inference with the
requested temporal deblock strength.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
DEFAULT_VLM_DEVICE = "cuda:1"
DEFAULT_OUTPUT_DIR = "runs/decoder_components"
DEFAULT_TCDECODER_TEMPORAL_DEBLOCK_STRENGTH = 1.0
DEFAULT_LOOP_ITERATIONS = 2
DEFAULT_LOOP_BLOCK_START = 10
DEFAULT_LOOP_BLOCK_END = 19


def _has_option(argv: Sequence[str], *options: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv for option in options)


# train_plan_atlas_loop_ddp imports the base trainer, which selects CUDA devices
# before importing torch.  Make the intended three-device layout visible during
# that import even when this file is launched without CLI arguments.
_IMPORT_ARGV = sys.argv[:]
if not _has_option(_IMPORT_ARGV[1:], "--gpu"):
    sys.argv = [sys.argv[0], "--gpu", DEFAULT_GPU] + sys.argv[1:]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_GPU)
os.environ.setdefault("FLASHVSR_PHYSICAL_GPUS", DEFAULT_GPU)
try:
    import train_plan_atlas_loop_ddp as atlas_loop
finally:
    sys.argv = _IMPORT_ARGV

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils import parametrize


base_train = atlas_loop.base_train


class _WeightLoRA(nn.Module):
    """Weight parametrization for Linear, Conv2d, and Conv3d modules.

    ``BidirectionalConv3d`` has a custom forward signature and padding rule.
    A weight parametrization preserves that forward method, unlike replacing it
    with a generic adapter module.  The resulting update is the standard
    flattened LoRA form ``B @ A`` and is cast back to the frozen weight dtype
    before the original module executes.
    """

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float):
        super().__init__()
        if weight.ndim < 2:
            raise ValueError(f"LoRA requires a matrix-like weight, got {tuple(weight.shape)}")
        self.out_features = int(weight.shape[0])
        self.in_features = int(weight[0].numel())
        self.rank = min(max(1, int(rank)), self.out_features, self.in_features)
        self.scaling = float(alpha) / float(self.rank)
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, device=weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A).view_as(weight) * self.scaling
        return weight + delta.to(device=weight.device, dtype=weight.dtype)


_LORA_WEIGHT_MODULES = (nn.Linear, nn.Conv2d, nn.Conv3d)


def _add_weight_lora(component: nn.Module, rank: int, alpha: float, label: str) -> List[str]:
    """Freeze a component and attach LoRA to every linear/convolution weight."""
    component.requires_grad_(False)
    adapted = []
    for name, module in list(component.named_modules()):
        if not isinstance(module, _LORA_WEIGHT_MODULES):
            continue
        if getattr(module, "weight", None) is None:
            continue
        if parametrize.is_parametrized(module, "weight"):
            raise RuntimeError(f"{label}.{name} already has a weight parametrization")
        parametrize.register_parametrization(
            module,
            "weight",
            _WeightLoRA(module.weight, rank=rank, alpha=alpha),
        )
        module.parametrizations.weight.original.requires_grad_(False)
        adapted.append(name or "<root>")

    if not adapted:
        raise RuntimeError(f"No Linear/Conv2d/Conv3d weights found for {label} LoRA")
    return adapted


def _adapter_named_parameters(component: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    for name, param in component.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield name, param


def _adapter_state(component: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in _adapter_named_parameters(component)}


def _restore_adapter_state(
    component: nn.Module,
    state: Dict[str, torch.Tensor],
    label: str,
) -> Tuple[int, List[str]]:
    current = dict(_adapter_named_parameters(component))
    loaded = 0
    skipped = []
    for name, value in state.items():
        param = current.get(name)
        if param is None or param.shape != value.shape:
            skipped.append(name)
            continue
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))
        loaded += 1
    base_train.print0(f"  Loaded {loaded} {label} LoRA tensors")
    return loaded, skipped


class VaeTcdecoderLoraAtlasLoopPipeline(atlas_loop.AtlasLoopPipeline):

    def train_refine_process(
        self,
        LQ_video: torch.Tensor,
        HQ_video: torch.Tensor,
        stage1_image: torch.Tensor,
        conditions: List[atlas_loop.RefinementCondition],
        topk_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        local_range: int = 9,
        stage1_latents_cached: torch.Tensor = None,
        hq_latents_cached: torch.Tensor = None,
        max_plans: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """Stage-3 loss with a frozen VAE latent target."""
        del stage1_image, stage1_latents_cached
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

        if hq_latents_cached is not None:
            hq_latents = hq_latents_cached.to(device=self.device, dtype=self.torch_dtype)
        else:
            self.load_models_to_device(["vae"])
            with torch.no_grad():
                hq_latents = self.vae.encode(
                    HQ_video.to(self.device, dtype=self.torch_dtype),
                    device=self.device,
                    tiled=True,
                )
            hq_latents = hq_latents.to(device=self.device, dtype=self.torch_dtype)
        batch_size, _, latent_f, latent_h, latent_w = hq_latents.shape
        _, _, frames, height, width = LQ_video.shape
        prompt = self._condition_prompt(conditions)
        if prompt:
            self.init_cross_kv(prompt=prompt)

        t_random = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (1,),
            device=self.device,
        ).float()
        noise = torch.randn_like(hq_latents, device=hq_latents.device, dtype=hq_latents.dtype)
        latents = self.scheduler.add_noise(hq_latents, noise, t_random).to(dtype=self.torch_dtype)

        self.load_models_to_device(["dit"])
        self.timestep = t_random.to(device=self.device, dtype=torch.float32)
        time_emb_dtype = next(self.dit.time_embedding.parameters()).dtype
        time_emb_input = sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep).to(
            device=self.device,
            dtype=time_emb_dtype,
        )
        self.t = self.dit.time_embedding(time_emb_input).to(dtype=self.torch_dtype)
        self.t_mod = self.dit.time_projection(self.t).unflatten(1, (6, self.dit.dim))

        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()
        self.TCDecoder.clean_mem()
        lq_cast = LQ_video.to(device=self.device, dtype=self.torch_dtype)
        lq_latents = (
            self.dit.LQ_proj_in.stream_forward(lq_cast)
            if hasattr(self.dit, "LQ_proj_in")
            else None
        )

        mask_input, mask_weight = self._prepare_mask_inputs(
            conditions=conditions,
            batch_size=batch_size,
            image_shape=(frames, height, width),
            latent_shape=(latent_f, latent_h, latent_w),
            max_plans=max_plans,
        )
        mask_latents = None
        if hasattr(self.dit, "Mask_proj_in"):
            self.dit.Mask_proj_in.clear_cache()
            mask_dtype = next(self.dit.Mask_proj_in.parameters()).dtype
            mask_latents = self.dit.Mask_proj_in.stream_forward(mask_input.to(dtype=mask_dtype))
            if mask_latents is not None:
                mask_latents = [mask.to(dtype=self.torch_dtype) for mask in mask_latents]

        noise_pred, _, _ = self._run_looped_dit(
            x=latents,
            timestep=self.timestep,
            context=None,
            tea_cache=None,
            use_unified_sequence_parallel=False,
            LQ_latents=lq_latents,
            Mask_latents=mask_latents,
            loop_mask_latent=mask_weight,
            is_full_block=False,
            is_stream=False,
            topk_ratio=topk_ratio,
            kv_ratio=kv_ratio,
            cur_process_idx=0,
            t_mod=self.t_mod,
            t=self.t,
            local_range=local_range,
            **self._loop_kwargs(),
        )

        result = {
            "noise_pred": noise_pred,
            "target": self.scheduler.training_target(hq_latents, noise, self.timestep).detach(),
            "mask_weight": mask_weight.expand_as(noise_pred),
        }

        return result

    def _refine_single_region(
        self,
        lq_image: torch.Tensor,
        current_image: torch.Tensor,
        condition: atlas_loop.RefinementCondition = None,
        conditions: List[atlas_loop.RefinementCondition] = None,
        max_plans: int = 3,
        current_latents: torch.Tensor = None,
        return_latents: bool = False,
    ):
        """Parent inference path with the TCDecoder deblock strength applied."""
        del current_image, current_latents
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

        if conditions is None:
            conditions = [condition] if condition is not None else []
        prompt = self._condition_prompt(conditions)
        if prompt:
            self.init_cross_kv(prompt=prompt)

        batch_size, _, frames, height, width = lq_image.shape
        height, width = self.check_resize_height_width(height, width)
        latent_shape = (batch_size, 16, frames // 4, height // 8, width // 8)
        latents = self.generate_noise(latent_shape, seed=None, device=self.device, dtype=self.torch_dtype)

        self.load_models_to_device(["dit"])
        inference_timestep = 1000.0
        self.timestep = torch.tensor([inference_timestep], device=self.device, dtype=self.torch_dtype)
        time_emb_dtype = next(self.dit.time_embedding.parameters()).dtype
        time_emb_input = sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep).to(
            device=self.device,
            dtype=time_emb_dtype,
        )
        self.t = self.dit.time_embedding(time_emb_input).to(dtype=self.torch_dtype)
        self.t_mod = self.dit.time_projection(self.t).unflatten(1, (6, self.dit.dim))

        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()
        self.TCDecoder.clean_mem()
        lq_cast = lq_image.to(device=self.device, dtype=self.torch_dtype)
        lq_latents = (
            self.dit.LQ_proj_in.stream_forward(lq_cast)
            if hasattr(self.dit, "LQ_proj_in")
            else None
        )

        mask_input, loop_mask_latent = self._prepare_mask_inputs(
            conditions=conditions,
            batch_size=batch_size,
            image_shape=(frames, height, width),
            latent_shape=latent_shape[-3:],
            max_plans=max_plans,
        )
        mask_latents = None
        if hasattr(self.dit, "Mask_proj_in"):
            self.dit.Mask_proj_in.clear_cache()
            mask_dtype = next(self.dit.Mask_proj_in.parameters()).dtype
            mask_latents = self.dit.Mask_proj_in.stream_forward(mask_input.to(dtype=mask_dtype))
            if mask_latents is not None:
                mask_latents = [mask.to(dtype=self.torch_dtype) for mask in mask_latents]

        noise_pred, _, _ = self._run_looped_dit(
            x=latents,
            timestep=self.timestep,
            context=None,
            tea_cache=None,
            use_unified_sequence_parallel=False,
            LQ_latents=lq_latents,
            Mask_latents=mask_latents,
            loop_mask_latent=loop_mask_latent,
            is_full_block=False,
            is_stream=False,
            topk_ratio=2.0,
            kv_ratio=3.0,
            cur_process_idx=0,
            t_mod=self.t_mod,
            t=self.t,
            local_range=9,
            **self._loop_kwargs(),
        )

        sigma = self.scheduler.timestep_to_sigma(inference_timestep)
        sigma = torch.tensor([sigma], device=self.device, dtype=self.torch_dtype)
        pred_clean_latent = latents - sigma * noise_pred

        with torch.no_grad():
            refined = self.TCDecoder.decode_video(
                pred_clean_latent.to(dtype=self.torch_dtype).transpose(1, 2),
                parallel=False,
                show_progress_bar=False,
                cond=lq_cast[:, :, : pred_clean_latent.shape[2] * 4],
                bidirectional=self.use_bidirectional_decoder,
                temporal_deblock_strength=self.tcdecoder_temporal_deblock_strength,
            ).transpose(1, 2).mul_(2).sub_(1)

        if return_latents:
            return refined, pred_clean_latent
        return refined


class VaeTcdecoderLoraAtlasLoopTrainerDDP(atlas_loop.AtlasLoopTrainerDDP):
    """Trainer that synchronizes, optimizes, and checkpoints all new adapters."""

    def __init__(self, args):
        super().__init__(args)
        self._refresh_trainable_parameters()

    def _init_pipeline(self):
        pipe = super()._init_pipeline()
        pipe.tcdecoder_temporal_deblock_strength = float(
            self.args.tcdecoder_temporal_deblock_strength
        )
        return pipe

    def _add_lora(self):
        super()._add_lora()
        # The parent experiment deliberately keeps its Stage-1 DiT LoRA,
        # Mask_proj_in LoRA, and loop_index_embedding trainable. Do not freeze
        # them here: this run adds only the LQ injection adapter below.
        self.pipe.vae.requires_grad_(False)
        rank = int(self.args.lora_rank)
        alpha = float(self.args.lora_alpha)
        self._component_lora_layers = {
            "lq_proj_in": _add_weight_lora(
                self.pipe.denoising_model().LQ_proj_in,
                rank,
                alpha,
                "LQ_proj_in",
            ),
        }

        for _, param in _adapter_named_parameters(self.pipe.denoising_model().LQ_proj_in):
            param.data = param.data.to(dtype=torch.float32)

        if base_train.is_dist():
            for _, param in _adapter_named_parameters(self.pipe.denoising_model().LQ_proj_in):
                dist.broadcast(param.data, src=0)

        base_train.print0(
            "Added component LoRA layers: "
            + ", ".join(f"{name}={len(layers)}" for name, layers in self._component_lora_layers.items())
        )

    def _trainable_parameters(self) -> List[nn.Parameter]:
        ordered = []
        seen_ids = set()
        # LQ_proj_in is nested inside DiT, so it is included by the first pass.
        for component in (self._dit_module(),):
            for _, param in component.named_parameters():
                if param.requires_grad and id(param) not in seen_ids:
                    ordered.append(param)
                    seen_ids.add(id(param))
        if not ordered:
            raise RuntimeError("No trainable LoRA parameters were registered")
        return ordered

    def _refresh_trainable_parameters(self) -> None:
        self._lora_params = self._trainable_parameters()
        base_train.print0(f"  Trainable parameters synchronized by DDP: {len(self._lora_params)}")

    def _setup_optimizer(self, lr=None):
        if lr is None:
            lr = self.args.learning_rate
        return torch.optim.AdamW(
            self._trainable_parameters(),
            lr=lr,
            weight_decay=self.args.weight_decay,
        )

    def _extract_stage3_lora_state(self):
        # Keep the parent checkpoint contract for DiT/Mask/loop parameters.
        # The only additional adapter is saved separately below.
        state = super()._extract_stage3_lora_state()
        return {
            name: value
            for name, value in state.items()
            if "LQ_proj_in" not in name
        }

    def _component_lora_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "lq_proj_in": _adapter_state(self.pipe.denoising_model().LQ_proj_in),
        }

    def save_checkpoint(self, epoch, val_psnr, val_ssim, is_best=False):
        if self.phase == 2:
            dit_state = self._extract_stage3_lora_state()
        else:
            dit_state = self._dit_module().state_dict()
        checkpoint = {
            "epoch": epoch,
            "phase": self.phase,
            "global_step": self.global_step,
            "dit_state_dict": dit_state,
            "component_lora_state_dict": self._component_lora_state(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "best_val_loss": self.best_val_loss,
            "args": vars(self.args),
        }
        checkpoints = Path(self.output_dir) / "checkpoints"
        latest_path = checkpoints / "latest.pth"
        epoch_path = checkpoints / f"epoch_{epoch + 1}.pth"
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, epoch_path)
        base_train.print0(f"Checkpoint saved: {epoch_path}")
        if is_best:
            best_path = checkpoints / "best.pth"
            torch.save(checkpoint, best_path)
            base_train.print0(f"Best model saved: {best_path}")

    @staticmethod
    def _optimizer_state_compatible(optimizer, saved_state) -> bool:
        if not isinstance(saved_state, dict):
            return False
        saved_groups = saved_state.get("param_groups", [])
        current_groups = optimizer.state_dict().get("param_groups", [])
        return len(saved_groups) == len(current_groups) and all(
            len(saved.get("params", [])) == len(current.get("params", []))
            for saved, current in zip(saved_groups, current_groups)
        )

    def _setup_resume_lr_scheduler(self, completed_epochs: int):
        """Rebuild the cosine scheduler at the absolute resume progress."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = max(1, int(self.args.num_epochs) * steps_per_epoch)
        completed_steps = min(
            max(0, int(completed_epochs) * steps_per_epoch), total_steps
        )
        base_lr = (
            float(self.args.stage3_lr)
            if self.phase == 2
            else float(self.args.learning_rate)
        )
        min_lr = float(self.args.min_lr)

        # CosineAnnealingLR needs initial_lr when last_epoch is supplied. Its
        # constructor records completed_steps, and we then set the closed-form
        # LR explicitly for compatibility across PyTorch scheduler versions.
        for group in self.optimizer.param_groups:
            group["lr"] = base_lr
            group["initial_lr"] = base_lr
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=min_lr,
            last_epoch=completed_steps - 1,
        )

        expected_lr = min_lr + 0.5 * (base_lr - min_lr) * (
            1.0 + math.cos(math.pi * completed_steps / total_steps)
        )
        for group in self.optimizer.param_groups:
            group["lr"] = expected_lr
        scheduler._last_lr = [expected_lr for _ in self.optimizer.param_groups]
        actual_lr = scheduler.get_last_lr()[0]
        if not math.isclose(actual_lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(
                "Resume LR scheduler is misaligned: "
                f"expected {expected_lr:.8g}, got {actual_lr:.8g}"
            )
        base_train.print0(
            "Resume cosine LR at absolute progress: "
            f"epoch={completed_epochs}/{self.args.num_epochs}, "
            f"step={completed_steps}/{total_steps}, lr={actual_lr:.8g}"
        )
        return scheduler

    def _load_checkpoint(self, checkpoint_path):
        """Load new multi-component checkpoints and safely accept older ones."""
        base_train.print0(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        dit_state = checkpoint["dit_state_dict"]
        dit = self._dit_module()
        current_state = dit.state_dict()
        loaded_dit = 0
        skipped_dit = []
        for name, value in dit_state.items():
            if name not in current_state or current_state[name].shape != value.shape:
                skipped_dit.append(name)
                continue
            current_state[name] = value
            loaded_dit += 1
        dit.load_state_dict(current_state)
        base_train.print0(f"  Loaded {loaded_dit} DiT/loop/LQ tensors")
        if skipped_dit:
            base_train.print0(f"  Skipped {len(skipped_dit)} incompatible DiT tensors")

        component_state = checkpoint.get("component_lora_state_dict", {})
        for label, component, state_key in (
            ("LQ_proj_in", self.pipe.denoising_model().LQ_proj_in, "lq_proj_in"),
        ):
            saved = component_state.get(state_key)
            if saved:
                _, skipped = _restore_adapter_state(component, saved, label)
                if skipped:
                    base_train.print0(f"  Skipped {len(skipped)} incompatible {label} LoRA tensors")
            else:
                base_train.print0(f"  No {label} LoRA state in checkpoint; using fresh adapters")

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if self._optimizer_state_compatible(self.optimizer, optimizer_state):
            self.optimizer.load_state_dict(optimizer_state)
        else:
            base_train.print0("  Optimizer state not restored because adapter parameter groups differ.")

        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_psnr = checkpoint.get("val_psnr", 0.0)
        self.scheduler = self._setup_resume_lr_scheduler(self.start_epoch)
        base_train.print0(f"Resumed from epoch {self.start_epoch} (phase {checkpoint.get('phase', 1)})")

    def train(self):
        base_train.print0("  VAE encoder: frozen; VAE reconstruction loss disabled")
        base_train.print0("  Parent DiT/Mask/loop LoRA: trainable")
        base_train.print0("  LQ_proj_in LoRA: enabled with Stage-3 latent loss")
        base_train.print0(
            f"  TCDecoder temporal deblock strength: {self.args.tcdecoder_temporal_deblock_strength:g}"
        )
        return super().train()


# The inherited trainer resolves this global name at runtime in _init_pipeline.
# Rebinding it lets us reuse every plan/atlas/loop method without modifying the
# original experiment entry point.
atlas_loop.AtlasLoopPipeline = VaeTcdecoderLoraAtlasLoopPipeline


def get_args():
    """Use the parent experiment's arguments, with this run's fixed defaults."""
    original_argv = sys.argv[:]
    injected = list(original_argv)
    defaults = (
        ("--gpu", DEFAULT_GPU),
        ("--vlm_device", DEFAULT_VLM_DEVICE),
        ("--loop_iterations", str(DEFAULT_LOOP_ITERATIONS)),
        ("--loop_block_start", str(DEFAULT_LOOP_BLOCK_START)),
        ("--loop_block_end", str(DEFAULT_LOOP_BLOCK_END)),
        ("--output_dir", DEFAULT_OUTPUT_DIR),
    )
    for option, value in defaults:
        if not _has_option(injected[1:], option, option.replace("_", "-")):
            injected.extend([option, value])
    sys.argv = injected
    try:
        args = atlas_loop.get_args()
    finally:
        sys.argv = original_argv

    args.loop_apply_to_masked_tokens = True
    args.tcdecoder_temporal_deblock_strength = DEFAULT_TCDECODER_TEMPORAL_DEBLOCK_STRENGTH
    args.ablation_mode = "plan_atlas_loop_lq_all_lora_deblock"
    if not _has_option(original_argv[1:], "--output_dir", "--output-dir"):
        args.output_dir = DEFAULT_OUTPUT_DIR
    return args


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["FLASHVSR_PHYSICAL_GPUS"] = args.gpu

    layout_required_index = base_train.validate_gpu_layout(args)
    local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))
    base_train.wait_for_gpu_with_retry(
        required_index=max(local_rank_env, layout_required_index),
        max_retries=args.gpu_retry_max_retries,
        wait_seconds=args.gpu_retry_wait_seconds,
    )
    use_ddp = base_train.setup_distributed()
    if not use_ddp:
        print(f"[Single-GPU] CUDA_VISIBLE_DEVICES = {args.gpu}")

    if base_train.is_main():
        print("\n" + "=" * 64)
        print("Ablation: plan_atlas_loop_lq_all_lora_deblock")
        print(f"Training GPUs: {args.gpu}; VLM: {args.vlm_device}")
        print(
            "Loop: first pass for all tokens; one extra pass for masked tokens "
            f"on blocks {args.loop_block_start}..{args.loop_block_end}"
        )
        print(f"TCDecoder temporal deblock strength: {args.tcdecoder_temporal_deblock_strength:g}")
        print(f"Output root: {args.output_dir}")
        print("=" * 64 + "\n")

    trainer = None
    try:
        trainer = VaeTcdecoderLoraAtlasLoopTrainerDDP(args)
        trainer.train()
    finally:
        if trainer is not None and hasattr(trainer, "pipe") and hasattr(trainer.pipe, "shutdown_vllm"):
            trainer.pipe.shutdown_vllm()
        base_train.cleanup_distributed()


if __name__ == "__main__":
    main()
