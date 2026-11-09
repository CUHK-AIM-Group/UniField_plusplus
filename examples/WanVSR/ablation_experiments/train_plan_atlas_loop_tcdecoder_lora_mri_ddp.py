#!/usr/bin/env python3
"""Fine-tune only TCDecoder LoRA against target MRI volumes.

The frozen generator is restored from the requested plan/atlas/loop phase-2
checkpoint.  TCDecoder keeps the bidirectional decoder path and the
parameter-free temporal deblocking used by
``train_plan_atlas_loop_vae_tcdecoder_lq_lora_ddp.py``.  Every non-TCDecoder
parameter, including the DiT/Mask/loop/LQ adapters, is frozen.

See the repository README for single-worker and distributed launch commands.
Validation skips padding samples inserted by DistributedSampler.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
DEFAULT_VLM_DEVICE = "cuda:1"
DEFAULT_SOURCE_CHECKPOINT = None
DEFAULT_OUTPUT_DIR = "runs/decoder"
DEFAULT_TCDECODER_LORA_RANK = 128
DEFAULT_TCDECODER_LORA_ALPHA = 128.0
DEFAULT_TCDECODER_LR = 5.0e-5
DEFAULT_TCDECODER_TEMPORAL_DEBLOCK_STRENGTH = 1.0
DEFAULT_PREFLIGHT_VAL_BATCHES = 1
DEFAULT_LOOP_ITERATIONS = 2
DEFAULT_LOOP_BLOCK_START = 10
DEFAULT_LOOP_BLOCK_END = 19


def _has_option(argv: Sequence[str], *options: str) -> bool:
    return any(
        arg == option or arg.startswith(f"{option}=")
        for arg in argv
        for option in options
    )


def _parse_local_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    """Parse this entry's options and leave parent-trainer options untouched."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source_checkpoint", type=str, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument(
        "--tcdecoder_lora_rank",
        type=int,
        default=DEFAULT_TCDECODER_LORA_RANK,
    )
    parser.add_argument(
        "--tcdecoder_lora_alpha",
        type=float,
        default=DEFAULT_TCDECODER_LORA_ALPHA,
    )
    parser.add_argument("--tcdecoder_lr", type=float, default=DEFAULT_TCDECODER_LR)
    parser.add_argument(
        "--tcdecoder_temporal_deblock_strength",
        type=float,
        default=DEFAULT_TCDECODER_TEMPORAL_DEBLOCK_STRENGTH,
    )
    parser.add_argument(
        "--preflight_val_batches",
        type=int,
        default=DEFAULT_PREFLIGHT_VAL_BATCHES,
        help="Validation batches per rank to run before epoch 1; zero disables.",
    )
    return parser.parse_known_args(list(argv))


_LOCAL_ARGS, _PARENT_ARGV = _parse_local_args(sys.argv[1:])
_IMPORT_ARGV = sys.argv[:]
if not _has_option(_PARENT_ARGV, "--gpu"):
    _PARENT_ARGV.extend(["--gpu", DEFAULT_GPU])
os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_GPU)
os.environ.setdefault("FLASHVSR_PHYSICAL_GPUS", DEFAULT_GPU)
sys.argv = [sys.argv[0], *_PARENT_ARGV]
try:
    import train_plan_atlas_loop_vae_tcdecoder_lq_lora_ddp as vae_tc
finally:
    sys.argv = _IMPORT_ARGV

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


atlas_loop = vae_tc.atlas_loop
base_train = vae_tc.base_train


class _DecoderWeightLoRA(nn.Module):
    """Low-rank parametrization of a frozen convolution weight."""

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float):
        super().__init__()
        if weight.ndim < 2:
            raise ValueError(f"LoRA needs a matrix-like weight, got {tuple(weight.shape)}")
        self.out_features = int(weight.shape[0])
        self.in_features = int(weight[0].numel())
        self.rank = min(max(1, int(rank)), self.out_features, self.in_features)
        self.scaling = float(alpha) / float(self.rank)
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                self.in_features,
                device=weight.device,
                dtype=torch.float32,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                self.rank,
                device=weight.device,
                dtype=torch.float32,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A).view_as(weight) * self.scaling
        return weight + delta.to(device=weight.device, dtype=weight.dtype)


def _attach_tcdecoder_lora(
    decoder: nn.Module,
    rank: int,
    alpha: float,
) -> List[str]:
    """Freeze TCDecoder and attach LoRA to every Conv2d weight."""
    decoder.requires_grad_(False)
    adapted = []
    for name, module in list(decoder.named_modules()):
        if not isinstance(module, nn.Conv2d):
            continue
        if module.weight is None:
            continue
        if parametrize.is_parametrized(module, "weight"):
            raise RuntimeError(f"TCDecoder.{name}.weight is already parametrized")
        parametrize.register_parametrization(
            module,
            "weight",
            _DecoderWeightLoRA(module.weight, rank=rank, alpha=alpha),
        )
        module.parametrizations.weight.original.requires_grad_(False)
        adapted.append(name)
    if not adapted:
        raise RuntimeError("No Conv2d layers found in TCDecoder")
    return adapted


def _adapter_named_parameters(
    component: nn.Module,
) -> Iterable[Tuple[str, nn.Parameter]]:
    for name, parameter in component.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield name, parameter


def _adapter_state(component: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in _adapter_named_parameters(component)
    }


def _restore_adapter_state(
    component: nn.Module,
    state: Dict[str, torch.Tensor],
) -> Tuple[int, List[str]]:
    current = dict(_adapter_named_parameters(component))
    loaded = 0
    skipped = []
    for name, value in state.items():
        parameter = current.get(name)
        if parameter is None or parameter.shape != value.shape:
            skipped.append(name)
            continue
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        loaded += 1
    return loaded, skipped


class TcdecoderLoraMriPipeline(vae_tc.VaeTcdecoderLoraAtlasLoopPipeline):
    """Frozen plan/atlas/loop generator followed by trainable TCDecoder LoRA."""

    @torch.no_grad()
    def _generate_frozen_refine_latents(
        self,
        lq_video: torch.Tensor,
        conditions: List[atlas_loop.RefinementCondition],
        topk_ratio: float,
        kv_ratio: float,
        local_range: int,
        max_plans: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

        prompt = self._condition_prompt(conditions)
        if prompt:
            self.init_cross_kv(prompt=prompt)

        batch_size, _, frames, height, width = lq_video.shape
        height, width = self.check_resize_height_width(height, width)
        latent_shape = (batch_size, 16, frames // 4, height // 8, width // 8)
        latents = self.generate_noise(
            latent_shape,
            seed=None,
            device=self.device,
            dtype=self.torch_dtype,
        )

        self.load_models_to_device(["dit"])
        inference_timestep = 1000.0
        self.timestep = torch.tensor(
            [inference_timestep],
            device=self.device,
            dtype=torch.float32,
        )
        time_emb_dtype = next(self.dit.time_embedding.parameters()).dtype
        time_emb_input = sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep).to(
            device=self.device,
            dtype=time_emb_dtype,
        )
        self.t = self.dit.time_embedding(time_emb_input).to(dtype=self.torch_dtype)
        self.t_mod = self.dit.time_projection(self.t).unflatten(1, (6, self.dit.dim))

        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()
        lq_cast = lq_video.to(device=self.device, dtype=self.torch_dtype)
        lq_latents = (
            self.dit.LQ_proj_in.stream_forward(lq_cast)
            if hasattr(self.dit, "LQ_proj_in")
            else None
        )

        mask_input, mask_weight = self._prepare_mask_inputs(
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
            mask_latents = self.dit.Mask_proj_in.stream_forward(
                mask_input.to(dtype=mask_dtype)
            )
            if mask_latents is not None:
                mask_latents = [mask.to(dtype=self.torch_dtype) for mask in mask_latents]

        velocity_pred, _, _ = self._run_looped_dit(
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
        sigma = self.scheduler.timestep_to_sigma(inference_timestep)
        sigma = torch.tensor([sigma], device=self.device, dtype=self.torch_dtype)
        clean_latents = latents - sigma * velocity_pred
        return clean_latents.detach(), mask_weight.detach()

    def train_refine_process(
        self,
        LQ_video: torch.Tensor,
        HQ_video: torch.Tensor,
        stage1_image: Optional[torch.Tensor],
        conditions: List[atlas_loop.RefinementCondition],
        topk_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        local_range: int = 9,
        stage1_latents_cached: Optional[torch.Tensor] = None,
        hq_latents_cached: Optional[torch.Tensor] = None,
        max_plans: int = 3,
    ) -> Dict[str, torch.Tensor]:
        del stage1_image, stage1_latents_cached, hq_latents_cached
        clean_latents, mask_weight = self._generate_frozen_refine_latents(
            lq_video=LQ_video,
            conditions=conditions,
            topk_ratio=topk_ratio,
            kv_ratio=kv_ratio,
            local_range=local_range,
            max_plans=max_plans,
        )

        self.TCDecoder.clean_mem()
        lq_cast = LQ_video.to(device=self.device, dtype=self.torch_dtype)
        generated_mri = self.TCDecoder.decode_video(
            clean_latents.to(dtype=self.torch_dtype).transpose(1, 2),
            parallel=False,
            show_progress_bar=False,
            cond=lq_cast[:, :, : clean_latents.shape[2] * 4],
            bidirectional=self.use_bidirectional_decoder,
            temporal_deblock_strength=self.tcdecoder_temporal_deblock_strength,
        ).transpose(1, 2).mul(2).sub(1)
        # Bidirectional execution stores graph-bearing memory.  The output owns
        # the graph needed for backward, so clear the mutable decoder cache now.
        self.TCDecoder.clean_mem()

        target_mri = HQ_video.to(device=generated_mri.device, dtype=generated_mri.dtype)
        common_frames = min(generated_mri.shape[2], target_mri.shape[2])
        generated_mri = generated_mri[:, :, :common_frames]
        target_mri = target_mri[:, :, :common_frames]
        # The inherited validation loop still uses the historical
        # ``noise_pred``/``target`` names. Expose MRI-space aliases and use
        # an image-shaped zero mask under no_grad so validation computes the
        # same plain MRI L1 objective as training.
        loss_mask = (
            mask_weight
            if torch.is_grad_enabled()
            else torch.zeros_like(generated_mri[:, :1])
        )
        return {
            "generated_mri": generated_mri,
            "target_mri": target_mri,
            "noise_pred": generated_mri,
            "target": target_mri,
            "mask_weight": loss_mask,
        }


class _LimitedValidationLoader:
    """Expose the first N batches while preserving DataLoader metadata."""

    def __init__(self, loader, max_batches: int):
        self.loader = loader
        self.max_batches = min(max(1, int(max_batches)), len(loader))
        self.sampler = loader.sampler
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size

    def __len__(self):
        return self.max_batches

    def __iter__(self):
        for batch_index, batch in enumerate(self.loader):
            if batch_index >= self.max_batches:
                break
            yield batch



class TcdecoderLoraMriTrainerDDP(atlas_loop.AtlasLoopTrainerDDP):
    """Train and checkpoint only TCDecoder convolution LoRA parameters."""

    def __init__(self, args):
        self.source_checkpoint = str(args.source_checkpoint)
        super().__init__(args)
        self._load_frozen_generator_checkpoint(self.source_checkpoint)
        self._refresh_tcdecoder_trainables()
        self._assert_only_tcdecoder_lora_trainable()

    def _init_pipeline(self):
        original_pipeline_cls = atlas_loop.AtlasLoopPipeline
        atlas_loop.AtlasLoopPipeline = TcdecoderLoraMriPipeline
        try:
            pipe = super()._init_pipeline()
        finally:
            atlas_loop.AtlasLoopPipeline = original_pipeline_cls
        pipe.tcdecoder_temporal_deblock_strength = float(
            self.args.tcdecoder_temporal_deblock_strength
        )
        return pipe

    def _add_lora(self):
        # Build the exact DiT LoRA + Mask + loop architecture required by the
        # source checkpoint, then freeze it before attaching decoder LoRA.
        super()._add_lora()
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        adapted = _attach_tcdecoder_lora(
            self.pipe.TCDecoder,
            rank=int(self.args.tcdecoder_lora_rank),
            alpha=float(self.args.tcdecoder_lora_alpha),
        )
        for _, parameter in _adapter_named_parameters(self.pipe.TCDecoder):
            parameter.data = parameter.data.to(dtype=torch.float32)
        if base_train.is_dist():
            for _, parameter in _adapter_named_parameters(self.pipe.TCDecoder):
                dist.broadcast(parameter.data, src=0)
        self._tcdecoder_lora_layers = adapted
        self._refresh_tcdecoder_trainables()
        base_train.print0(
            f"Attached TCDecoder LoRA to {len(adapted)} Conv2d layers "
            f"(rank={self.args.tcdecoder_lora_rank}, "
            f"alpha={self.args.tcdecoder_lora_alpha:g})"
        )

    def _refresh_tcdecoder_trainables(self) -> None:
        self._lora_params = [
            parameter
            for _, parameter in _adapter_named_parameters(self.pipe.TCDecoder)
        ]
        if not self._lora_params:
            raise RuntimeError("TCDecoder LoRA has no trainable parameters")

    def _assert_only_tcdecoder_lora_trainable(self) -> None:
        bad = []
        trainable_names = []
        for name, parameter in self.pipe.named_parameters():
            if not parameter.requires_grad:
                continue
            trainable_names.append(name)
            if "TCDecoder" not in name or not (
                "lora_A" in name or "lora_B" in name
            ):
                bad.append(name)
        if bad:
            raise RuntimeError(
                "Non-TCDecoder-LoRA parameters are trainable: " + ", ".join(bad)
            )
        trainable_count = sum(parameter.numel() for parameter in self._lora_params)
        base_train.print0(
            f"Verified trainable scope: {len(trainable_names)} tensors, "
            f"{trainable_count:,} parameters; TCDecoder LoRA only"
        )

    def _setup_optimizer(self, lr=None):
        del lr
        parameters = [
            parameter
            for _, parameter in _adapter_named_parameters(self.pipe.TCDecoder)
        ]
        return torch.optim.AdamW(
            parameters,
            lr=float(self.args.tcdecoder_lr),
            weight_decay=float(self.args.weight_decay),
        )

    def _compute_stage3_loss(self, refine_out, stage1_latents, hq_latents):
        del stage1_latents, hq_latents
        generated = refine_out["generated_mri"].float()
        target = refine_out["target_mri"].float()
        l1_loss = F.l1_loss(generated, target)
        terms = {
            "l1": float(l1_loss.detach().item()),
            "ssim": 0.0,
            "grad": 0.0,
        }
        return l1_loss, terms, refine_out["mask_weight"]

    def _load_frozen_generator_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Source checkpoint not found: {path}")
        base_train.print0(f"Loading frozen plan/atlas/loop weights: {path}")
        checkpoint = torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if int(checkpoint.get("phase", -1)) != 2:
            raise RuntimeError(f"Expected phase-2 checkpoint: {path}")

        saved_epoch = int(checkpoint.get("epoch", -1)) + 1
        dit = self._dit_module()
        current = dit.state_dict()
        compatible = {}
        mismatched = []
        for name, value in checkpoint["dit_state_dict"].items():
            if name not in current:
                continue
            if current[name].shape != value.shape:
                mismatched.append(
                    f"{name}: current={tuple(current[name].shape)} "
                    f"saved={tuple(value.shape)}"
                )
                continue
            compatible[name] = value
        if mismatched:
            raise RuntimeError("Source checkpoint shape mismatch:\n  " + "\n  ".join(mismatched))
        if not compatible:
            raise RuntimeError(f"No compatible generator tensors in {path}")
        dit.load_state_dict(compatible, strict=False)
        dit.requires_grad_(False)
        base_train.print0(
            f"Loaded {len(compatible)} frozen generator tensors from epoch {saved_epoch}"
        )
        del checkpoint, compatible, current
        gc.collect()
        torch.cuda.empty_cache()

    def save_checkpoint(self, epoch, val_psnr, val_ssim, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "phase": self.phase,
            "global_step": self.global_step,
            "source_checkpoint": self.source_checkpoint,
            "tcdecoder_lora_state_dict": _adapter_state(self.pipe.TCDecoder),
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

    def load_tcdecoder_lora_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        loaded, skipped = _restore_adapter_state(
            self.pipe.TCDecoder,
            checkpoint["tcdecoder_lora_state_dict"],
        )
        if skipped:
            raise RuntimeError(f"Skipped {len(skipped)} TCDecoder LoRA tensors")
        base_train.print0(f"Restored {loaded} TCDecoder LoRA tensors")

    def _run_preflight_validation(self) -> None:
        max_batches = int(self.args.preflight_val_batches)
        if max_batches <= 0:
            base_train.print0("Preflight validation disabled.")
            return
        original_loader = self.val_loader
        limited_loader = _LimitedValidationLoader(original_loader, max_batches)
        global_samples = len(limited_loader) * base_train.get_world_size()
        base_train.print0(
            f"\nPreflight validation: {len(limited_loader)} batch(es) per rank, "
            f"up to {global_samples} samples globally"
        )
        self.val_loader = limited_loader
        try:
            result = self.validate(epoch=0, save_images=False)
        finally:
            self.val_loader = original_loader
        scalar_metrics = result[:4]
        if not all(math.isfinite(float(value)) for value in scalar_metrics):
            raise RuntimeError(f"Preflight validation returned non-finite metrics: {scalar_metrics}")
        base_train.print0(
            "Preflight validation PASSED: "
            f"PSNR={float(result[0]):.4f}, SSIM={float(result[1]):.4f}, "
            f"NRMSE={float(result[2]):.4f}, LPIPS={float(result[3]):.4f}"
        )
        base_train.barrier()


    def train(self):
        base_train.print0("\nTCDecoder-only fine-tuning switches:")
        base_train.print0(f"  frozen generator checkpoint: {self.source_checkpoint}")
        base_train.print0("  DiT / Mask_proj_in / loop / LQ_proj_in / VAE: frozen")
        base_train.print0("  TCDecoder base convolutions: frozen")
        base_train.print0("  TCDecoder convolution LoRA: trainable")
        base_train.print0("  objective: L1(generated MRI, target MRI)")
        base_train.print0(
            f"  bidirectional decoder: {self.pipe.use_bidirectional_decoder}"
        )
        base_train.print0(
            "  temporal deblock strength: "
            f"{self.pipe.tcdecoder_temporal_deblock_strength:g}"
        )
        self._run_preflight_validation()
        return base_train.ThreeStageTrainerDDP.train(self)


# The inherited trainer resolves this global name while building the pipeline.
atlas_loop.AtlasLoopPipeline = TcdecoderLoraMriPipeline


def get_args():
    local_args, parent_argv = _parse_local_args(sys.argv[1:])
    original_argv = sys.argv[:]
    injected = [original_argv[0], *parent_argv]
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
        args = vae_tc.get_args()
    finally:
        sys.argv = original_argv

    args.loop_apply_to_masked_tokens = True
    args.skip_stage1_train_inference = True
    args.resume_from = None
    if not local_args.source_checkpoint:
        raise ValueError("--source_checkpoint must point to a completed main-training checkpoint")
    args.source_checkpoint = local_args.source_checkpoint
    args.tcdecoder_lora_rank = int(local_args.tcdecoder_lora_rank)
    args.tcdecoder_lora_alpha = float(local_args.tcdecoder_lora_alpha)
    args.tcdecoder_lr = float(local_args.tcdecoder_lr)
    args.preflight_val_batches = int(local_args.preflight_val_batches)
    args.stage3_lr = float(local_args.tcdecoder_lr)
    args.tcdecoder_temporal_deblock_strength = float(
        local_args.tcdecoder_temporal_deblock_strength
    )
    args.stage3_ssim_weight = 0.0
    args.stage3_grad_weight = 0.0
    args.ablation_mode = "plan_atlas_loop_tcdecoder_lora_mri"
    if not _has_option(parent_argv, "--output_dir", "--output-dir"):
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
        print("\n" + "=" * 72)
        print("TCDecoder LoRA MRI fine-tuning")
        print(f"Source checkpoint: {args.source_checkpoint}")
        print(f"Training GPUs: {args.gpu}; VLM: {args.vlm_device}")
        print(
            f"TCDecoder LoRA rank={args.tcdecoder_lora_rank}, "
            f"alpha={args.tcdecoder_lora_alpha:g}, lr={args.tcdecoder_lr:.2e}"
        )
        print(f"Output root: {args.output_dir}")
        print("=" * 72 + "\n")

    trainer = None
    try:
        trainer = TcdecoderLoraMriTrainerDDP(args)
        trainer.train()
    finally:
        if (
            trainer is not None
            and hasattr(trainer, "pipe")
            and hasattr(trainer.pipe, "shutdown_vllm")
        ):
            trainer.pipe.shutdown_vllm()
        base_train.cleanup_distributed()


if __name__ == "__main__":
    main()
