#!/usr/bin/env python3
"""Standalone plan+atlas ablation with atlas-mask injection and looped DiT.

This entry keeps Stage-2 plan generation and brain-atlas spatial masks, but
adds LoopFormer-style extra depth to the middle half of the 30 DiT blocks.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

DEFAULT_GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
DEFAULT_VLM_DEVICE = "cuda:1"
DEFAULT_LOOP_MASK_THRESHOLD = 0.05
DEFAULT_MASK_LOSS_WEIGHT = 5.0
DEFAULT_NUM_EPOCHS = 100


def _early_cli_has_option(argv: Sequence[str], *options: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv for option in options)


# The visible device list is shared with the planner subprocess.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_GPU)
os.environ.setdefault("FLASHVSR_PHYSICAL_GPUS", os.environ["CUDA_VISIBLE_DEVICES"])

import numpy as np
import torch
import torch.nn.functional as F


THIS_FILE = Path(__file__).resolve()
ABLATION_DIR = THIS_FILE.parent
WANVSR_DIR = ABLATION_DIR.parent
REPO_ROOT = WANVSR_DIR.parent.parent
LOOP_DIR = WANVSR_DIR / "loop_training"
DEFAULT_MODEL_DIR = REPO_ROOT / "weights" / "FlashVSR"
DEFAULT_MODEL_PATHS = {
    "dit_path": DEFAULT_MODEL_DIR / "diffusion_pytorch_model_streaming_dmd.safetensors",
    "vae_path": DEFAULT_MODEL_DIR / "Wan2.1_VAE.pth",
    "text_encoder_path": DEFAULT_MODEL_DIR / "models_t5_umt5-xxl-enc-bf16.pth",
    "lq_proj_path": DEFAULT_MODEL_DIR / "LQ_proj_in.ckpt",
    "tcdecoder_path": DEFAULT_MODEL_DIR / "TCDecoder.ckpt",
}

for path in (str(LOOP_DIR), str(WANVSR_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

_IMPORT_ARGV = sys.argv[:]
if not _early_cli_has_option(_IMPORT_ARGV[1:], "--gpu"):
    sys.argv = [sys.argv[0], "--gpu", DEFAULT_GPU] + sys.argv[1:]
try:
    import train_loop_ddp as loop_train  # noqa: E402
finally:
    sys.argv = _IMPORT_ARGV
from diffsynth.models import ModelManager  # noqa: E402
from diffsynth.modules.plan_to_mask import (  # noqa: E402
    PlanToMaskConverter,
    RefinementCondition,
)
from diffsynth.pipelines.flashvsr_3stage import FlashVSR3StagePipeline  # noqa: E402


base_train = loop_train.base_train

DEFAULT_OUTPUT_DIR_PREFIX = str(REPO_ROOT / "runs" / "main")
DEFAULT_PLAN_VLM_LORA_PATH = (
    str(REPO_ROOT / "weights" / "planner_lora")
)
DEFAULT_PLAN_VLM_LORA_NAME = "unifield-planner"
EVALUATE_METRICS_PATH = REPO_ROOT / "unifield" / "evaluate_metrics.py"
_EVALUATE_METRICS_MODULE = None


def _cli_has_option(argv: Sequence[str], *options: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv for option in options)


def _ratio_label(mask_loss_weight: float) -> str:
    ratio = f"{float(mask_loss_weight):g}".replace(".", "p")
    return f"{ratio}to1"


def _default_output_dir(mask_loss_weight: float) -> str:
    return f"{DEFAULT_OUTPUT_DIR_PREFIX}_{_ratio_label(mask_loss_weight)}"


def _load_evaluate_metrics_module():
    """Load the requested metric implementation directly from its source file."""
    global _EVALUATE_METRICS_MODULE
    if _EVALUATE_METRICS_MODULE is not None:
        return _EVALUATE_METRICS_MODULE
    if not EVALUATE_METRICS_PATH.is_file():
        raise FileNotFoundError(f"Metric source not found: {EVALUATE_METRICS_PATH}")
    spec = importlib.util.spec_from_file_location(
        "tmi_ablation_evaluate_metrics",
        EVALUATE_METRICS_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metric source: {EVALUATE_METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EVALUATE_METRICS_MODULE = module
    return module


class AtlasConditionFactory(PlanToMaskConverter):
    """Keep atlas-derived masks while prepending the task prompt to plan text."""

    def __init__(self):
        super().__init__()
        self.current_task_prompt = ""

    def set_task_prompt(self, task_prompt: Optional[str]) -> None:
        self.current_task_prompt = str(task_prompt or "").strip()

    def _compose_text_prompt(self, plan_text: Optional[str]) -> str:
        cleaned = str(plan_text or "").strip()
        task_prompt = self.current_task_prompt
        if not task_prompt:
            return cleaned
        if not cleaned:
            return task_prompt
        if task_prompt.lower() in cleaned.lower():
            return cleaned
        return f"{task_prompt}; {cleaned}"

    def create_refinement_conditions(
        self,
        plans,
        image_size: Tuple[int, ...],
        text_encoder=None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> List[RefinementCondition]:
        conditions: List[RefinementCondition] = []
        for plan in plans:
            mask = self.convert_plan_to_mask(plan, image_size)
            text_prompt = self._compose_text_prompt(getattr(plan, "text_prompt", ""))

            text_embedding = None
            if text_encoder is not None:
                text_embedding = self._encode_text(text_prompt, text_encoder, device, dtype)

            strength = min(plan.target_problem.severity + 0.3, 1.0)
            condition = RefinementCondition(
                plan=plan,
                mask=mask,
                text_prompt=text_prompt,
                text_embedding=text_embedding,
                strength=strength,
            )
            condition.to(device, dtype)
            conditions.append(condition)
        return conditions


class AtlasLoopPipeline(loop_train.LoopFlashVSR3StagePipeline):
    """Stage-1-style DiT path with optional plan text and atlas masks.

    Stage-1 generation/finetuning starts from noise/noisy HQ latents. This
    ablation keeps that same network/input contract for Stage-3 and only adds
    the requested conditioning plus looped execution: plan text in
    cross-attention, atlas masks through ``Mask_proj_in``, and one extra pass
    through the middle half of the DiT blocks.
    """

    @staticmethod
    def from_model_manager(
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        vlm_device: Optional[str] = None,
        use_usp: bool = False,
    ) -> "AtlasLoopPipeline":
        del use_usp
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = AtlasLoopPipeline(
            device=device,
            torch_dtype=torch_dtype,
            vlm_device=vlm_device,
        )
        pipe.fetch_models(model_manager)
        pipe.use_unified_sequence_parallel = False
        pipe.vlm_base_model_name = AtlasLoopPipeline._pending_vlm_base_model_name
        pipe.vlm_lora_path = AtlasLoopPipeline._pending_vlm_lora_path
        pipe.vlm_lora_name = AtlasLoopPipeline._pending_vlm_lora_name
        return pipe

    def _condition_prompt(self, conditions: List[RefinementCondition]) -> str:
        task_prompt = str(
            getattr(getattr(self, "plan_to_mask", None), "current_task_prompt", "") or ""
        ).strip()
        task_lower = task_prompt.lower()
        plan_prompts = []
        seen = set()
        for cond in conditions:
            prompt = str(getattr(cond, "text_prompt", "") or "").strip()
            if task_prompt and prompt.lower().startswith(task_lower):
                prompt = prompt[len(task_prompt):].lstrip(" ;")
            key = prompt.lower()
            if prompt and key not in seen:
                plan_prompts.append(prompt)
                seen.add(key)
        parts = [task_prompt] if task_prompt else []
        parts.extend(plan_prompts)
        return "; ".join(parts)

    def _prepare_mask_inputs(
        self,
        conditions: List[RefinementCondition],
        batch_size: int,
        image_shape: Tuple[int, int, int],
        latent_shape: Tuple[int, int, int],
        max_plans: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build pixel-space Mask_proj_in input and latent-space loss weight."""
        frames, height, width = image_shape
        latent_f, latent_h, latent_w = latent_shape
        mask_channels = []
        combined_mask_latent = torch.zeros(
            batch_size,
            1,
            latent_f,
            latent_h,
            latent_w,
            device=self.device,
            dtype=torch.float32,
        )

        for cond in conditions[:max_plans]:
            raw_mask = cond.mask.to(self.device).float()
            strength = float(getattr(cond, "strength", 1.0))

            if raw_mask.dim() == 2:
                ch = raw_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                ch = ch.expand(batch_size, 1, frames, height, width)
                m_lat = F.interpolate(
                    raw_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    size=(1, latent_h, latent_w),
                    mode="trilinear",
                    align_corners=False,
                ).expand(batch_size, 1, latent_f, latent_h, latent_w)
            elif raw_mask.dim() == 3:
                ch = F.interpolate(
                    raw_mask.unsqueeze(0).unsqueeze(0),
                    size=(frames, height, width),
                    mode="trilinear",
                    align_corners=False,
                ).expand(batch_size, 1, frames, height, width)
                m_lat = F.interpolate(
                    raw_mask.unsqueeze(0).unsqueeze(0),
                    size=(latent_f, latent_h, latent_w),
                    mode="trilinear",
                    align_corners=False,
                ).expand(batch_size, 1, latent_f, latent_h, latent_w)
            else:
                ch = raw_mask.unsqueeze(0) if raw_mask.dim() == 4 else raw_mask
                ch = ch.to(device=self.device)
                if ch.shape[0] == 1 and batch_size > 1:
                    ch = ch.expand(batch_size, *ch.shape[1:])
                if ch.shape[-3:] != (frames, height, width):
                    ch = F.interpolate(
                        ch.float(),
                        size=(frames, height, width),
                        mode="trilinear",
                        align_corners=False,
                    )
                m_lat = ch[:, :1].float()
                if m_lat.shape[-3:] != (latent_f, latent_h, latent_w):
                    m_lat = F.interpolate(
                        m_lat,
                        size=(latent_f, latent_h, latent_w),
                        mode="trilinear",
                        align_corners=False,
                    )

            mask_channels.append(ch * strength)
            combined_mask_latent = torch.max(combined_mask_latent, m_lat * strength)

        while len(mask_channels) < max_plans:
            mask_channels.append(
                torch.zeros(
                    batch_size,
                    1,
                    frames,
                    height,
                    width,
                    device=self.device,
                    dtype=self.torch_dtype,
                )
            )

        mask_input = torch.cat(mask_channels, dim=1).to(dtype=self.torch_dtype)
        mask_weight = combined_mask_latent.to(dtype=self.torch_dtype)
        return mask_input, mask_weight

    def _run_looped_dit(self, **kwargs):
        """Run the looped DiT path with per-pass index conditioning enabled."""
        previous_state = getattr(self.dit, "_loop_index_conditioning_active", False)
        self.dit._loop_index_conditioning_active = True
        try:
            return loop_train.model_fn_wan_video_looped(self.dit, **kwargs)
        finally:
            self.dit._loop_index_conditioning_active = previous_state

    def train_refine_process(
        self,
        LQ_video: torch.Tensor,
        HQ_video: torch.Tensor,
        stage1_image: torch.Tensor,
        conditions: List[RefinementCondition],
        topk_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        local_range: int = 9,
        stage1_latents_cached: torch.Tensor = None,
        hq_latents_cached: torch.Tensor = None,
        max_plans: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """Stage-3 training with the same noisy-latent input as Stage-1."""
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

        LQ_video_cast = LQ_video.to(device=self.device, dtype=self.torch_dtype)
        LQ_latents = (
            self.dit.LQ_proj_in.stream_forward(LQ_video_cast)
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
        Mask_latents = None
        if hasattr(self.dit, "Mask_proj_in"):
            self.dit.Mask_proj_in.clear_cache()
            mask_dtype = next(self.dit.Mask_proj_in.parameters()).dtype
            Mask_latents = self.dit.Mask_proj_in.stream_forward(mask_input.to(dtype=mask_dtype))
            if Mask_latents is not None:
                Mask_latents = [m.to(dtype=self.torch_dtype) for m in Mask_latents]

        noise_pred, _, _ = self._run_looped_dit(
            x=latents,
            timestep=self.timestep,
            context=None,
            tea_cache=None,
            use_unified_sequence_parallel=False,
            LQ_latents=LQ_latents,
            Mask_latents=Mask_latents,
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

        target = self.scheduler.training_target(hq_latents, noise, self.timestep)
        return {
            "noise_pred": noise_pred,
            "target": target,
            "mask_weight": mask_weight.expand_as(noise_pred),
        }

    def _refine_single_region(
        self,
        lq_image: torch.Tensor,
        current_image: torch.Tensor,
        condition: RefinementCondition = None,
        conditions: List[RefinementCondition] = None,
        max_plans: int = 3,
        current_latents: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ):
        """Stage-3 inference path aligned with Stage-1: start from pure noise."""
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
            dtype=self.torch_dtype,
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
        self.TCDecoder.clean_mem()

        lq_cast = lq_image.to(device=self.device, dtype=self.torch_dtype)
        LQ_latents = (
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
        Mask_latents = None
        if hasattr(self.dit, "Mask_proj_in"):
            self.dit.Mask_proj_in.clear_cache()
            mask_dtype = next(self.dit.Mask_proj_in.parameters()).dtype
            Mask_latents = self.dit.Mask_proj_in.stream_forward(mask_input.to(dtype=mask_dtype))
            if Mask_latents is not None:
                Mask_latents = [m.to(dtype=self.torch_dtype) for m in Mask_latents]

        noise_pred, _, _ = self._run_looped_dit(
            x=latents,
            timestep=self.timestep,
            context=None,
            tea_cache=None,
            use_unified_sequence_parallel=False,
            LQ_latents=LQ_latents,
            Mask_latents=Mask_latents,
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
            ).transpose(1, 2).mul_(2).sub_(1)

        if return_latents:
            return refined, pred_clean_latent
        return refined


class AtlasLoopTrainerDDP(loop_train.LoopTrainerDDP):
    """Plan+atlas trainer with looped DiT execution."""

    def __init__(self, args):
        self._requested_metric_calculator = None
        super().__init__(args)

    @staticmethod
    def _make_loop_index_pre_hook(dit_ref):
        """Condition repeated passes while leaving the first pass unchanged."""
        def hook(block, inputs):
            dit = dit_ref()
            if dit is None or not getattr(dit, "_loop_index_conditioning_active", False):
                return None
            if len(inputs) < 3 or inputs[2] is None:
                return None

            loop_index = int(getattr(block, "_loop_index", 0))
            if loop_index == 0:
                return None

            loop_embeddings = dit.loop_index_embedding
            embedding_index = loop_index - 1
            if not 0 <= embedding_index < loop_embeddings.shape[0]:
                raise RuntimeError(
                    f"Loop index {loop_index} is outside the configured range "
                    f"[1, {loop_embeddings.shape[0]}]."
                )

            t_mod = inputs[2]
            loop_bias = loop_embeddings[embedding_index].to(
                device=t_mod.device,
                dtype=t_mod.dtype,
            ).unsqueeze(0)
            return inputs[:2] + (t_mod + loop_bias,) + inputs[3:]

        return hook

    def _add_lora(self):
        super()._add_lora()

        dit = self._dit_module()
        loop_iterations = max(1, int(self.args.loop_iterations))
        if hasattr(dit, "loop_index_embedding"):
            raise RuntimeError("loop_index_embedding was registered more than once.")

        reference_param = next(dit.parameters())
        dit.loop_index_embedding = torch.nn.Parameter(
            torch.zeros(
                loop_iterations - 1,
                6,
                dit.blocks[0].dim,
                device=reference_param.device,
                dtype=torch.float32,
            )
        )
        loop_start, loop_end = loop_train._loop_bounds(
            len(dit.blocks),
            self.args.loop_block_start,
            self.args.loop_block_end,
        )
        dit_ref = weakref.ref(dit)
        dit._loop_index_hook_handles = [
            block.register_forward_pre_hook(self._make_loop_index_pre_hook(dit_ref))
            for block in dit.blocks[loop_start : loop_end + 1]
        ]
        dit._loop_index_conditioning_active = False
        base_train.print0(
            "First loop pass is unmodified; added trainable loop-index "
            f"conditioning for repeated passes 1..{loop_iterations - 1} "
            f"on blocks {loop_start}..{loop_end}."
        )

    def _extract_stage3_lora_state(self):
        state = super()._extract_stage3_lora_state()
        state["loop_index_embedding"] = self._dit_module().loop_index_embedding.data.clone()
        return state

    def _load_checkpoint(self, checkpoint_path):
        """Resume old no-index checkpoints while preserving their model weights."""
        base_train.print0(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        saved_phase = checkpoint.get("phase", 1)
        dit_state = checkpoint["dit_state_dict"]
        dit = self._dit_module()
        saved_loop_embedding = dit_state.get("loop_index_embedding")
        optimizer_state_compatible = (
            saved_loop_embedding is not None
            and saved_loop_embedding.shape == dit.loop_index_embedding.shape
        )

        if saved_phase == 2:
            current_state = dit.state_dict()
            loaded_keys = []
            skipped_shape_keys = []
            for key, value in dit_state.items():
                if key not in current_state:
                    continue
                if current_state[key].shape != value.shape:
                    skipped_shape_keys.append(key)
                    continue
                current_state[key] = value
                loaded_keys.append(key)
            dit.load_state_dict(current_state)
            base_train.print0(f"  Loaded {len(loaded_keys)} Stage-3 LoRA keys")
            if skipped_shape_keys:
                base_train.print0(
                    "  Skip shape-mismatched checkpoint keys: "
                    + ", ".join(skipped_shape_keys)
                )
        else:
            dit.load_state_dict(dit_state, strict=False)

        if optimizer_state_compatible:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            # Checkpoints made before loop-index conditioning have one fewer
            # optimizer parameter. A loop-count change also changes its state
            # tensor shape, so Adam moments cannot be aligned safely.
            base_train.print0(
                "  Optimizer state not restored because loop-index conditioning "
                "is absent from or incompatible with this checkpoint."
            )

        # Keep a fresh scheduler on resume so LR follows the current run.
        base_train.print0("  Skip loading scheduler_state_dict (use fresh LR schedule)")
        resume_lr = self.args.stage3_lr if self.phase == 2 else self.args.learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = float(resume_lr)
            if "initial_lr" in group:
                group["initial_lr"] = float(resume_lr)
        base_train.print0(f"  Reset optimizer LR to {resume_lr:.2e}")

        self.scheduler = self._setup_lr_scheduler()
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_psnr = checkpoint.get("val_psnr", 0.0)
        base_train.print0(f"Resumed from epoch {self.start_epoch} (phase {saved_phase})")

    def _set_active_task_prompt(self, prompt: Optional[str]) -> None:
        task_prompt = str(prompt or "").strip()
        if hasattr(self, "pipe") and getattr(self.pipe, "plan_to_mask", None) is not None:
            setter = getattr(self.pipe.plan_to_mask, "set_task_prompt", None)
            if callable(setter):
                setter(task_prompt)

    def _extract_field_from_path(self, path: str) -> str:
        filename = os.path.basename(path).lower()
        if "1.5" in filename:
            return "1.5T"
        if "64m" in filename:
            return "64mT"
        if "7t" in filename:
            return "7T"
        if "3t" in filename:
            return "3T"
        return "low-field"

    def _extract_modality_from_path(self, path: str) -> str:
        for part in path.split(os.sep):
            part_lower = part.lower()
            if part_lower in {"t1", "t1w"}:
                return "T1-weighted"
            if part_lower in {"t2", "t2w"}:
                return "T2-weighted"
            if part_lower == "flair":
                return "FLAIR"
            if part_lower == "pd":
                return "Proton Density-weighted"
        return "MRI"

    def _generate_prompt(self, lq_path: str) -> str:
        input_field = self._extract_field_from_path(lq_path)
        modality = self._extract_modality_from_path(lq_path)
        task_type = self._infer_task_type(lq_path)
        target_field = "3T" if task_type == "64mT_to_3T" else "7T"
        prompt = (
            f"MRI {modality} sequence enhancement from "
            f"{input_field} to {target_field} magnetic field"
        )
        self._set_active_task_prompt(prompt)
        return prompt

    def _get_requested_metric_calculator(self):
        if self._requested_metric_calculator is None:
            metric_module = _load_evaluate_metrics_module()
            self._requested_metric_calculator = metric_module.MetricCalculator(
                device=self.device,
                batch_size=8,
            )
            base_train.print0(
                f"Validation metrics loaded unchanged from: {EVALUATE_METRICS_PATH}"
            )
        return self._requested_metric_calculator

    def _compute_metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> dict:
        """Match exported NIfTI volumes before calling evaluate_metrics.py APIs."""
        calculator = self._get_requested_metric_calculator()
        batch = min(pred.shape[0], gt.shape[0])
        values = {"psnr": [], "ssim": [], "nrmse": [], "lpips": []}
        for batch_index in range(batch):
            pred_volume = (
                pred[batch_index]
                .detach()
                .float()
                .mean(dim=0)
                .clamp(-1.0, 1.0)
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
            )
            gt_volume = (
                gt[batch_index]
                .detach()
                .float()
                .mean(dim=0)
                .clamp(-1.0, 1.0)
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
            )
            values["ssim"].append(float(calculator.calculate_ssim(pred_volume, gt_volume)))
            values["psnr"].append(float(calculator.calculate_psnr(pred_volume, gt_volume)))
            values["lpips"].append(
                float(calculator.calculate_lpips_volume(pred_volume, gt_volume))
            )
            values["nrmse"].append(float(calculator.calculate_nrmse(pred_volume, gt_volume)))

        if not values["psnr"]:
            return {
                "psnr": 0.0,
                "ssim": 0.0,
                "nrmse": 0.0,
                "lpips": 0.0,
                "count": 0,
                "sum_psnr": 0.0,
                "sum_ssim": 0.0,
                "sum_nrmse": 0.0,
                "sum_lpips": 0.0,
            }
        return {
            "psnr": float(np.mean(values["psnr"])),
            "ssim": float(np.mean(values["ssim"])),
            "nrmse": float(np.mean(values["nrmse"])),
            "lpips": float(np.mean(values["lpips"])),
            "count": len(values["psnr"]),
            "sum_psnr": float(np.sum(values["psnr"])),
            "sum_ssim": float(np.sum(values["ssim"])),
            "sum_nrmse": float(np.sum(values["nrmse"])),
            "sum_lpips": float(np.sum(values["lpips"])),
        }

    def _init_pipeline(self):
        original_pipeline_cls = base_train.FlashVSR3StagePipeline
        original_pending = (
            AtlasLoopPipeline._pending_vlm_base_model_name,
            AtlasLoopPipeline._pending_vlm_lora_path,
            AtlasLoopPipeline._pending_vlm_lora_name,
        )
        base_train.FlashVSR3StagePipeline = AtlasLoopPipeline
        AtlasLoopPipeline._pending_vlm_base_model_name = self.args.vlm_base_model_name
        AtlasLoopPipeline._pending_vlm_lora_path = self.args.vlm_lora_path
        AtlasLoopPipeline._pending_vlm_lora_name = self.args.vlm_lora_name
        try:
            pipe = base_train.ThreeStageTrainerDDP._init_pipeline(self)
        finally:
            base_train.FlashVSR3StagePipeline = original_pipeline_cls
            (
                AtlasLoopPipeline._pending_vlm_base_model_name,
                AtlasLoopPipeline._pending_vlm_lora_path,
                AtlasLoopPipeline._pending_vlm_lora_name,
            ) = original_pending

        pipe.plan_to_mask = AtlasConditionFactory()
        pipe.plan_to_mask.set_task_prompt("")
        pipe.stage2_lq_only = True
        loop_train._configure_loop_pipeline(pipe, self.args)
        return pipe

    def train(self):
        ratio_label = _ratio_label(self.args.mask_loss_weight).replace("to", ":")
        num_blocks = len(self.pipe.dit.blocks)
        loop_start, loop_end = loop_train._loop_bounds(
            num_blocks, self.args.loop_block_start, self.args.loop_block_end
        )
        base_train.print0("\nAblation switches:")
        base_train.print0("  conditioning: plan_text")
        base_train.print0("  call VLM / generate plan: True")
        base_train.print0("  Stage-2 input: low-field multi-slice only")
        base_train.print0(
            "  Stage-3 text prompt policy: Stage-1 task prompt + LQ-risk plan text"
        )
        base_train.print0(
            f"  VLM LoRA adapter: {self.args.vlm_lora_name}={self.args.vlm_lora_path}"
        )
        base_train.print0("  Stage-1 task prompt preserved: True")
        base_train.print0("  Stage-3 plan-text condition: True")
        base_train.print0("  atlas conversion/injection: True")
        base_train.print0("  Stage-3 input: Stage-1-style noisy latent / pure-noise inference")
        base_train.print0("  Stage-3 target: scheduler.training_target(HQ_latent, noise, timestep)")
        base_train.print0("  Stage-1 image use: baseline/metrics only, not Stage-2 or DiT input")
        base_train.print0("  Stage-1 training inference: skipped")
        base_train.print0(f"  Mask_proj_in present: {hasattr(self._dit_module(), 'Mask_proj_in')}")
        base_train.print0("  looped DiT blocks: True (model_fn_wan_video_looped)")
        base_train.print0(f"  loop_iterations: {self.args.loop_iterations}")
        base_train.print0(f"  loop block range: {loop_start}..{loop_end} of {num_blocks} blocks")
        base_train.print0(
            f"  apply extra depth to VLM/atlas masked tokens: {self.args.loop_apply_to_masked_tokens}"
        )
        base_train.print0(f"  loop mask threshold: {self.args.loop_mask_threshold:g}")
        base_train.print0(
            f"  focus/non-focus loss ratio: {ratio_label} "
            f"(mask_loss_weight={self.args.mask_loss_weight:g})"
        )
        base_train.print0("  Stage-3 structural terms: disabled (pure latent L1)")
        base_train.print0(f"  validation metrics source: {self.args.validation_metrics_source}")
        base_train.print0(f"  Stage-1 LoRA source: {self.args.stage1_ckpt}")
        base_train.print0(f"  first displayed epoch: {self.start_epoch + 1}")
        return base_train.ThreeStageTrainerDDP.train(self)


def get_args():
    original_argv = sys.argv[:]
    output_overridden = _cli_has_option(original_argv[1:], "--output_dir", "--output-dir")
    gpu_overridden = _cli_has_option(original_argv[1:], "--gpu")
    vlm_device_overridden = _cli_has_option(
        original_argv[1:], "--vlm_device", "--vlm-device"
    )
    mask_loss_weight_overridden = _cli_has_option(
        original_argv[1:], "--mask_loss_weight", "--mask-loss-weight"
    )
    num_epochs_overridden = _cli_has_option(
        original_argv[1:], "--num_epochs", "--num-epochs"
    )
    loop_iterations_overridden = _cli_has_option(
        original_argv[1:], "--loop_iterations", "--loop-iterations"
    )
    loop_block_start_overridden = _cli_has_option(
        original_argv[1:], "--loop_block_start", "--loop-block-start"
    )
    loop_block_end_overridden = _cli_has_option(
        original_argv[1:], "--loop_block_end", "--loop-block-end"
    )
    missing_loop_options = []
    if not loop_iterations_overridden:
        missing_loop_options.append("--loop_iterations")
    if not loop_block_start_overridden:
        missing_loop_options.append("--loop_block_start")
    if not loop_block_end_overridden:
        missing_loop_options.append("--loop_block_end")
    if missing_loop_options and "--help" not in original_argv:
        raise SystemExit(
            "Loop configuration must be provided explicitly for every training run. "
            f"Missing: {', '.join(missing_loop_options)}\n"
            "Example: --loop_iterations 2 --loop_block_start 7 --loop_block_end 21"
        )
    loop_mask_threshold_overridden = _cli_has_option(
        original_argv[1:], "--loop_mask_threshold", "--loop-mask-threshold"
    )
    loop_apply_overridden = _cli_has_option(
        original_argv[1:],
        "--loop_apply_to_masked_tokens",
        "--loop-apply-to-masked-tokens",
        "--no-loop_apply_to_masked_tokens",
        "--no-loop-apply-to-masked-tokens",
    )
    vlm_lora_path_overridden = _cli_has_option(
        original_argv[1:], "--vlm_lora_path", "--vlm-lora-path"
    )
    vlm_lora_name_overridden = _cli_has_option(
        original_argv[1:], "--vlm_lora_name", "--vlm-lora-name"
    )
    vlm_model_name_overridden = _cli_has_option(
        original_argv[1:], "--vlm_model_name", "--vlm-model-name"
    )
    args = loop_train.get_args()

    # Use the self-contained model assets shipped with this repository.
    for key, path in DEFAULT_MODEL_PATHS.items():
        dashed_key = key.replace("_", "-")
        if not _cli_has_option(original_argv[1:], f"--{key}", f"--{dashed_key}"):
            setattr(args, key, str(path))

    args.phase = 2
    args.continue_stage1_lora = True
    args.skip_stage1_train_inference = True
    if not gpu_overridden:
        args.gpu = DEFAULT_GPU
    if not vlm_device_overridden:
        args.vlm_device = DEFAULT_VLM_DEVICE
    if not loop_mask_threshold_overridden:
        args.loop_mask_threshold = DEFAULT_LOOP_MASK_THRESHOLD
    if not loop_apply_overridden:
        args.loop_apply_to_masked_tokens = True
    if not mask_loss_weight_overridden:
        # Atlas focus voxels contribute 5x the loss of non-focus voxels.
        args.mask_loss_weight = DEFAULT_MASK_LOSS_WEIGHT
    if not num_epochs_overridden:
        args.num_epochs = DEFAULT_NUM_EPOCHS
    args.stage3_ssim_weight = 0.0
    args.stage3_grad_weight = 0.0
    args.val_every = 5
    args.save_every = 5
    args.ablation_mode = "plan_atlas_loop"
    args.validation_metrics_source = str(EVALUATE_METRICS_PATH)
    args.enable_stage2 = True
    args.use_vlm = True

    if not output_overridden:
        args.output_dir = _default_output_dir(args.mask_loss_weight)

    if not vlm_lora_path_overridden:
        args.vlm_lora_path = DEFAULT_PLAN_VLM_LORA_PATH
    if not vlm_lora_name_overridden:
        args.vlm_lora_name = DEFAULT_PLAN_VLM_LORA_NAME
    if not vlm_model_name_overridden:
        args.vlm_model_name = str(args.vlm_lora_name)
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
        print("Ablation: plan_atlas_loop")
        print(f"Training GPUs: {args.gpu}; DDP workers: {base_train.get_world_size()}")
        print(f"VLM: {args.vlm_model_name} on {base_train.resolve_vlm_device(args)}")
        print(
            "Loop: "
            f"iterations={args.loop_iterations}, "
            f"blocks={args.loop_block_start}..{args.loop_block_end}, "
            f"masked_tokens={args.loop_apply_to_masked_tokens}"
        )
        print(f"Output root: {args.output_dir}")
        print("=" * 64 + "\n")

    trainer = None
    try:
        trainer = AtlasLoopTrainerDDP(args)
        trainer.train()
    finally:
        if trainer is not None and hasattr(trainer, "pipe") and hasattr(trainer.pipe, "shutdown_vllm"):
            trainer.pipe.shutdown_vllm()
        base_train.cleanup_distributed()


if __name__ == "__main__":
    main()
