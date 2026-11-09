#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loop-token Stage-2/Stage-3 training entry for WanVSR.

This file deliberately reuses the existing train_3stage_ddp trainer and only
swaps the pipeline forward used by Stage-3 refinement.  The base Stage-1 LoRA
checkpoint is kept as the trainable adapter by default.
"""

from __future__ import annotations

import argparse
import os
import sys

# Default loop-training layout for IDE/Run: physical GPU 0 for DiT, GPU 1 for VLM.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("FLASHVSR_PHYSICAL_GPUS", os.environ["CUDA_VISIBLE_DEVICES"])
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _activation_checkpoint
import yaml


THIS_FILE = Path(__file__).resolve()
LOOP_DIR = THIS_FILE.parent
WANVSR_DIR = LOOP_DIR.parent
REPO_ROOT = WANVSR_DIR.parent.parent
DEFAULT_CONFIG = LOOP_DIR / "loop_64mT_to_3T_ddp.yaml"
DEFAULT_VLM_FINETUNE_DIR = (
    WANVSR_DIR
    / "doctor_review_framework"
    / "finetune"
    / "llama_factory"
)

# Click Run profile: when this file is launched without CLI arguments, these
# code-side defaults are applied after YAML parsing.  Edit this block for the
# common PyCharm/IDE run path.  CLI and torchrun arguments still override when
# provided.
CLICK_RUN_USE_CODE_DEFAULTS = False
CLICK_RUN_DEFAULTS = {
    # Config file kept for readability/backward compatibility.
    "config": str(DEFAULT_CONFIG),

    # Data
    "train_list": str(WANVSR_DIR / "train.txt"),
    "val_list": str(WANVSR_DIR / "val.txt"),
    "task_type": "64mT_to_3T",
    "height": 256,
    "width": 256,
    "num_frames": 40,
    "use_monai": True,
    "monai_spatial_size": [256, 256, 160],

    # Model paths
    "dit_path": "weights/FlashVSR/diffusion_pytorch_model_streaming_dmd.safetensors",
    "vae_path": "weights/FlashVSR/Wan2.1_VAE.pth",
    "text_encoder_path": "weights/FlashVSR/models_t5_umt5-xxl-enc-bf16.pth",
    "lq_proj_path": "weights/FlashVSR/LQ_proj_in.ckpt",
    "tcdecoder_path": "weights/FlashVSR/TCDecoder.ckpt",

    # Training
    "num_epochs": 1000,
    "batch_size": 1,
    "learning_rate": 1.0e-4,
    "min_lr": 1.0e-6,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "max_grad_norm": 1.0,
    "use_amp": True,
    "val_every": 5,
    "validate_before_train": False,
    "validate_before_train_save_images": True,
    "save_every": 5,

    # DiT LoRA
    "lora_rank": 128,
    "lora_alpha": 128,
    "lora_target_modules": "q,k,v,o,ffn.0,ffn.2",

    # Stage 2/3 pipeline and VLM serving
    "enable_stage2": True,
    "enable_stage3": True,
    "max_refinement_iterations": 1,
    "stage3_timestep_max": 50.0,
    "stage3_tau_small_prob": 0.8,
    "vlm_model_name": "unifield-planner",
    "vlm_base_model_name": "Qwen/Qwen3.5-9B",
    "vlm_lora_name": "unifield-planner",
    "vlm_lora_path": (
        "weights/planner_lora"
    ),
    "vlm_finetune_dir": str(DEFAULT_VLM_FINETUNE_DIR),
    "use_vlm": True,
    "use_bidirectional": True,
    "gpu": "0,1",
    "vlm_device": "cuda:1",

    # Inference/refinement
    "topk_ratio": 2.0,
    "kv_ratio": 3.0,
    "local_range": 9,

    # Output and retry
    "output_dir": "runs/main",
    "num_workers": 4,
    "gpu_retry_wait_seconds": 300,
    "gpu_retry_max_retries": 12,

    # Phase control
    "phase": 2,
    "stage1_ckpt": "weights/stage1/",
    "stage3_lr": 5.0e-5,
    "mask_loss_weight": 5.0,
    "stage3_ssim_weight": 1.0,
    "stage3_grad_weight": 1.0,
    "stage3_ssim_kernel": 5,
    "resume_from": None,

    # Loop-token training
    "loop_iterations": 2,
    "loop_block_start": 0,
    "loop_block_end": 19,
    "loop_mask_threshold": 0.05,
    "loop_apply_to_masked_tokens": True,
    "continue_stage1_lora": True,
    "loop_debug_every": 25,
}

for path in (str(WANVSR_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import train_3stage_ddp as base_train  # noqa: E402
from diffsynth.models import ModelManager  # noqa: E402
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d  # noqa: E402
from diffsynth.pipelines.flashvsr_3stage import FlashVSR3StagePipeline  # noqa: E402
from diffsynth.modules.plan_to_mask import RefinementCondition  # noqa: E402


def _cli_has_option(argv: List[str], options: Tuple[str, ...]) -> bool:
    return any(arg == opt or arg.startswith(f"{opt}=") for arg in argv for opt in options)


def _read_yaml(path: Optional[str]) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r") as f:
        return yaml.safe_load(f) or {}


BASE_CLICK_RUN_ARG_KEYS = (
    "config",
    "train_list",
    "val_list",
    "task_type",
    "dit_path",
    "vae_path",
    "text_encoder_path",
    "lq_proj_path",
    "tcdecoder_path",
    "num_epochs",
    "batch_size",
    "learning_rate",
    "min_lr",
    "weight_decay",
    "warmup_epochs",
    "max_grad_norm",
    "lora_rank",
    "lora_alpha",
    "lora_target_modules",
    "height",
    "width",
    "num_frames",
    "use_monai",
    "monai_spatial_size",
    "enable_stage2",
    "enable_stage3",
    "max_refinement_iterations",
    "stage3_timestep_max",
    "stage3_tau_small_prob",
    "vlm_model_name",
    "use_vlm",
    "vlm_device",
    "use_bidirectional",
    "use_amp",
    "topk_ratio",
    "kv_ratio",
    "local_range",
    "output_dir",
    "save_every",
    "val_every",
    "validate_before_train",
    "validate_before_train_save_images",
    "num_workers",
    "gpu",
    "gpu_retry_wait_seconds",
    "gpu_retry_max_retries",
    "phase",
    "stage1_ckpt",
    "stage3_lr",
    "mask_loss_weight",
    "stage3_ssim_weight",
    "stage3_grad_weight",
    "stage3_ssim_kernel",
)
BASE_STORE_TRUE_KEYS = {
    "use_monai",
    "enable_stage2",
    "enable_stage3",
    "use_vlm",
    "use_bidirectional",
    "use_amp",
}
BASE_BOOLEAN_OPTIONAL_KEYS = {
    "validate_before_train",
    "validate_before_train_save_images",
}


def _click_run_base_argv() -> List[str]:
    argv: List[str] = []
    for key in BASE_CLICK_RUN_ARG_KEYS:
        value = CLICK_RUN_DEFAULTS.get(key)
        if value is None:
            continue
        option = f"--{key}"
        if isinstance(value, bool):
            if key in BASE_BOOLEAN_OPTIONAL_KEYS:
                argv.append(option if value else f"--no-{key}")
            elif value or key not in BASE_STORE_TRUE_KEYS:
                argv.append(option)
        elif isinstance(value, (list, tuple)):
            argv.append(option)
            argv.extend(str(item) for item in value)
        else:
            argv.extend([option, str(value)])
    return argv


def _parse_loop_cli(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loop_iterations", "--loop-iterations", type=int, default=None)
    parser.add_argument("--loop_block_start", "--loop-block-start", type=int, default=None)
    parser.add_argument("--loop_block_end", "--loop-block-end", type=int, default=None)
    parser.add_argument("--loop_mask_threshold", "--loop-mask-threshold", type=float, default=None)
    parser.add_argument(
        "--loop_apply_to_masked_tokens",
        "--loop-apply-to-masked-tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--loop_debug_every", "--loop-debug-every", type=int, default=None)
    parser.add_argument(
        "--continue_stage1_lora",
        "--continue-stage1-lora",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--vlm_finetune_dir", "--vlm-finetune-dir", type=str, default=None)
    parser.add_argument("--vlm_base_model_name", "--vlm-base-model-name", type=str, default=None)
    parser.add_argument("--vlm_lora_path", "--vlm-lora-path", type=str, default=None)
    parser.add_argument("--vlm_lora_name", "--vlm-lora-name", type=str, default=None)
    return parser.parse_known_args(argv)


def get_args():
    orig_argv = sys.argv[:]
    click_run_defaults_active = CLICK_RUN_USE_CODE_DEFAULTS and len(orig_argv) == 1
    loop_cli, remaining = _parse_loop_cli(orig_argv[1:])

    if click_run_defaults_active:
        remaining = _click_run_base_argv()
    elif not _cli_has_option(remaining, ("--config",)):
        default_config = CLICK_RUN_DEFAULTS.get("config", str(DEFAULT_CONFIG))
        remaining = ["--config", str(default_config)] + remaining

    sys.argv = [orig_argv[0]] + remaining
    try:
        args = base_train.get_args()
    finally:
        sys.argv = orig_argv

    yaml_cfg = _read_yaml(args.config)

    loop_default_keys = (
        "loop_iterations",
        "loop_block_start",
        "loop_block_end",
        "loop_mask_threshold",
        "loop_apply_to_masked_tokens",
        "loop_debug_every",
        "continue_stage1_lora",
        "vlm_finetune_dir",
        "vlm_base_model_name",
        "vlm_lora_path",
        "vlm_lora_name",
    )
    defaults = {key: CLICK_RUN_DEFAULTS[key] for key in loop_default_keys}
    for key, value in defaults.items():
        setattr(args, key, value)

    loop_cfg = yaml_cfg.get("loop", {}) if isinstance(yaml_cfg, dict) else {}
    for key in defaults:
        if key in loop_cfg and loop_cfg[key] is not None:
            setattr(args, key, loop_cfg[key])

    pipeline_cfg = yaml_cfg.get("pipeline", {}) if isinstance(yaml_cfg, dict) else {}
    for key in ("vlm_base_model_name", "vlm_lora_path", "vlm_lora_name"):
        if key in pipeline_cfg and pipeline_cfg[key] is not None:
            setattr(args, key, pipeline_cfg[key])
    if getattr(args, "vlm_lora_name", None) and not _cli_has_option(
        remaining, ("--vlm_model_name", "--vlm-model-name")
    ):
        args.vlm_model_name = str(args.vlm_lora_name)

    phase_cfg = yaml_cfg.get("phase", {}) if isinstance(yaml_cfg, dict) else {}
    if "phase" in phase_cfg:
        args.phase = int(phase_cfg["phase"])
    if (
        "stage1_ckpt" in phase_cfg
        and phase_cfg["stage1_ckpt"]
        and not _cli_has_option(remaining, ("--stage1_ckpt", "--stage1-ckpt"))
    ):
        args.stage1_ckpt = str(phase_cfg["stage1_ckpt"])

    resume_cfg = yaml_cfg.get("resume_from", None)
    if isinstance(yaml_cfg.get("resume", None), dict):
        resume_cfg = yaml_cfg["resume"].get("resume_from", resume_cfg)
    if not _cli_has_option(remaining, ("--resume_from", "--resume-from")):
        args.resume_from = str(resume_cfg) if resume_cfg else None

    if click_run_defaults_active:
        for key, value in CLICK_RUN_DEFAULTS.items():
            if key != "config":
                setattr(args, key, value)

    for key, value in vars(loop_cli).items():
        if value is not None:
            setattr(args, key, value)

    args.click_run_defaults_active = click_run_defaults_active

    args.loop_iterations = max(1, int(args.loop_iterations))
    args.loop_block_start = int(args.loop_block_start)
    args.loop_block_end = int(args.loop_block_end)
    args.loop_mask_threshold = float(args.loop_mask_threshold)
    args.loop_debug_every = max(0, int(args.loop_debug_every))
    args.continue_stage1_lora = bool(args.continue_stage1_lora)
    args.loop_apply_to_masked_tokens = bool(args.loop_apply_to_masked_tokens)
    args.vlm_base_model_name = str(args.vlm_base_model_name) if args.vlm_base_model_name else None
    args.vlm_lora_path = str(args.vlm_lora_path) if args.vlm_lora_path else None
    args.vlm_lora_name = str(args.vlm_lora_name) if args.vlm_lora_name else None
    return args


def _loop_bounds(num_blocks: int, start: int, end: int) -> Tuple[int, int]:
    start = 0 if start < 0 else start
    end = num_blocks - 1 if end < 0 else end
    start = min(max(start, 0), num_blocks - 1)
    end = min(max(end, 0), num_blocks - 1)
    if end < start:
        end = start
    return start, end


def _loop_context(loop_index: int, total: int) -> Dict[str, Union[int, float, str]]:
    total = max(1, int(total))
    loop_dt = 1.0 / float(total)
    loop_t = float(loop_index) * loop_dt
    return {
        "loop_index": int(loop_index),
        "loop_number": int(loop_index) + 1,
        "loop_total": int(total),
        "loop_t": loop_t,
        "loop_dt": loop_dt,
        "loop_prompt": (
            f"loop_iteration={loop_index + 1}/{total}; "
            f"loop_t={loop_t:.4f}; loop_dt={loop_dt:.4f}"
        ),
    }


def _attach_loop_context(dit: WanModel, block: torch.nn.Module, loop_index: int, total: int) -> None:
    ctx = _loop_context(loop_index, total)
    dit._loop_context = ctx
    dit._loop_index = ctx["loop_index"]
    dit._loop_t = ctx["loop_t"]
    dit._loop_dt = ctx["loop_dt"]
    dit._loop_prompt = ctx["loop_prompt"]
    block._loop_context = ctx
    block._loop_index = ctx["loop_index"]


def _prepare_token_weights(
    loop_mask_latent: Optional[torch.Tensor],
    grid_size: Tuple[int, int, int],
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> Tuple[Optional[torch.Tensor], float]:
    if loop_mask_latent is None:
        return None, 1.0

    mask = loop_mask_latent.detach().to(device=device, dtype=torch.float32)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 4:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 5:
        raise ValueError(f"loop_mask_latent must be 3D/4D/5D, got {tuple(mask.shape)}")

    mask = F.interpolate(mask, size=grid_size, mode="trilinear", align_corners=False)
    mask = mask.clamp(0.0, 1.0)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1, -1, -1)

    weights = mask.flatten(2).transpose(1, 2).contiguous()
    if threshold > 0:
        weights = torch.where(weights >= threshold, weights, torch.zeros_like(weights))
    ratio = float((weights > 0).float().mean().item())
    return weights, ratio


def model_fn_wan_video_looped(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    tea_cache=None,
    use_unified_sequence_parallel: bool = False,
    LQ_latents: Optional[List[torch.Tensor]] = None,
    Mask_latents: Optional[List[torch.Tensor]] = None,
    loop_mask_latent: Optional[torch.Tensor] = None,
    loop_iterations: int = 2,
    loop_block_start: int = 0,
    loop_block_end: int = -1,
    loop_mask_threshold: float = 0.05,
    loop_apply_to_masked_tokens: bool = True,
    is_full_block: bool = False,
    is_stream: bool = False,
    pre_cache_k: Optional[List[torch.Tensor]] = None,
    pre_cache_v: Optional[List[torch.Tensor]] = None,
    topk_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    cur_process_idx: int = 0,
    t_mod: torch.Tensor = None,
    t: torch.Tensor = None,
    local_range: int = 9,
    **kwargs,
):
    x, (f, h, w) = dit.patchify(x)
    batch_size = x.shape[0]

    win = (2, 8, 8)
    seqlen = f // win[0]
    local_num = seqlen
    window_size = win[0] * h * w // 128
    square_num = window_size * window_size
    topk = int(square_num * topk_ratio) - 1
    kv_len = int(kv_ratio)

    if cur_process_idx == 0:
        freqs = torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x.device)
    else:
        freqs = torch.cat(
            [
                dit.freqs[0][4 + cur_process_idx * 2 : 4 + cur_process_idx * 2 + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x.device)

    tea_cache_update = tea_cache.check(dit, x, t_mod) if tea_cache is not None else False

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
        )

        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[
                get_sequence_parallel_rank()
            ]

    loop_iterations = max(1, int(loop_iterations))
    num_blocks = len(dit.blocks)
    loop_start, loop_end = _loop_bounds(num_blocks, loop_block_start, loop_block_end)
    can_loop = (not is_stream) and loop_iterations > 1
    token_weights = None
    loop_mask_ratio = 1.0
    if can_loop and loop_apply_to_masked_tokens:
        token_weights, loop_mask_ratio = _prepare_token_weights(
            loop_mask_latent=loop_mask_latent,
            grid_size=(f, h, w),
            batch_size=batch_size,
            device=x.device,
            threshold=loop_mask_threshold,
        )
        can_loop = token_weights is None or bool((token_weights > 0).any().item())
    dit._last_loop_mask_ratio = loop_mask_ratio
    dit._last_loop_block_range = (loop_start, loop_end)
    dit._last_loop_iterations = loop_iterations

    def blend(base: torch.Tensor, updated: torch.Tensor) -> torch.Tensor:
        if token_weights is None:
            return updated
        weights = token_weights.to(device=updated.device, dtype=updated.dtype)
        return base + weights * (updated - base)

    def run_block(block_id: int, x_in: torch.Tensor, loop_index: int) -> torch.Tensor:
        block = dit.blocks[block_id]
        loop_index_conditioning_enabled = bool(
            getattr(dit, "_loop_index_conditioning_active", False)
        )
        if LQ_latents is not None and block_id < len(LQ_latents):
            x_in = x_in + LQ_latents[block_id]
        if Mask_latents is not None and block_id < len(Mask_latents):
            x_in = x_in + Mask_latents[block_id]

        def forward_block(block_input: torch.Tensor) -> torch.Tensor:
            # The Atlas pipeline disables this hook after the forward returns.
            # Restore its original forward state while checkpoint recomputes so
            # repeated passes receive the same trainable loop-index bias.
            previous_loop_index_conditioning = getattr(
                dit, "_loop_index_conditioning_active", False
            )
            if loop_index_conditioning_enabled:
                dit._loop_index_conditioning_active = True
            _attach_loop_context(dit, block, loop_index, loop_iterations)
            block_result = block(
                block_input,
                context,
                t_mod,
                freqs,
                f,
                h,
                w,
                local_num,
                topk,
                block_id=block_id,
                kv_len=kv_len,
                is_full_block=is_full_block,
                is_stream=is_stream,
                pre_cache_k=pre_cache_k[block_id] if pre_cache_k is not None else None,
                pre_cache_v=pre_cache_v[block_id] if pre_cache_v is not None else None,
                local_range=local_range,
            )
            if loop_index_conditioning_enabled:
                dit._loop_index_conditioning_active = previous_loop_index_conditioning
            return block_result

        use_checkpoint = (
            bool(getattr(dit, "_gradient_checkpointing", False))
            and torch.is_grad_enabled()
            and not is_stream
        )
        block_result = (
            _activation_checkpoint(forward_block, x_in, use_reentrant=False)
            if use_checkpoint else forward_block(x_in)
        )
        if is_stream:
            x_out, last_pre_cache_k, last_pre_cache_v = block_result
        else:
            x_out = block_result
            last_pre_cache_k = None
            last_pre_cache_v = None
        if pre_cache_k is not None:
            pre_cache_k[block_id] = last_pre_cache_k
        if pre_cache_v is not None:
            pre_cache_v[block_id] = last_pre_cache_v
        return x_out

    def run_stack(x_in: torch.Tensor, loop_index: int, anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_out = x_in
        for bid in range(loop_start, loop_end + 1):
            x_out = run_block(bid, x_out, loop_index)
            if anchor is not None:
                x_out = blend(anchor, x_out)
        return x_out

    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        block_id = 0
        while block_id < num_blocks:
            if can_loop and block_id == loop_start:
                x_once = run_stack(x, loop_index=0, anchor=None)
                x_loop = x_once
                for loop_index in range(1, loop_iterations):
                    x_loop = run_stack(x_loop, loop_index=loop_index, anchor=x_once)
                x = blend(x_once, x_loop)
                block_id = loop_end + 1
            else:
                x = run_block(block_id, x, loop_index=0)
                block_id += 1

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import get_sp_group

        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    return x, pre_cache_k, pre_cache_v


class LoopFlashVSR3StagePipeline(FlashVSR3StagePipeline):
    _pending_vlm_base_model_name = None
    _pending_vlm_lora_path = None
    _pending_vlm_lora_name = None

    @staticmethod
    def from_model_manager(
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        vlm_device: str = None,
        use_usp=False,
    ) -> "LoopFlashVSR3StagePipeline":
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = LoopFlashVSR3StagePipeline(
            device=device,
            torch_dtype=torch_dtype,
            vlm_device=vlm_device,
        )
        pipe.fetch_models(model_manager)
        pipe.use_unified_sequence_parallel = False
        pipe.vlm_base_model_name = LoopFlashVSR3StagePipeline._pending_vlm_base_model_name
        pipe.vlm_lora_path = LoopFlashVSR3StagePipeline._pending_vlm_lora_path
        pipe.vlm_lora_name = LoopFlashVSR3StagePipeline._pending_vlm_lora_name
        return pipe

    def _launch_vllm_server(self, model_name: str, vlm_device: str, port: int = 23794):
        import os
        import socket
        import subprocess
        import time
        import urllib.request
        from pathlib import Path

        from openai import OpenAI
        import httpx

        physical_layout = os.environ.get(
            "FLASHVSR_PHYSICAL_GPUS",
            os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        ).split(",")
        physical_layout = [g.strip() for g in physical_layout if g.strip()]
        if vlm_device.startswith("cuda:"): 
            idx = int(vlm_device.split(":")[1])
            gpu_id = physical_layout[idx] if idx < len(physical_layout) else str(idx)
        else:
            gpu_id = physical_layout[0] if physical_layout else "0"

        for p in range(port, port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", p)) != 0:
                    port = p
                    break

        base_model = getattr(self, "vlm_base_model_name", None) or model_name
        lora_path = getattr(self, "vlm_lora_path", None)
        lora_name = getattr(self, "vlm_lora_name", None) or model_name
        if lora_path:
            lora_dir = Path(lora_path)
            if not lora_dir.is_dir():
                raise FileNotFoundError(f"VLM LoRA adapter not found: {lora_path}")
            if not (lora_dir / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(
                    f"VLM LoRA adapter_model.safetensors not found under: {lora_path}"
                )

        cmd = [
            os.environ.get("UNIFIELD_VLLM_PYTHON", __import__("sys").executable),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            base_model,
            "--port",
            str(port),
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            os.environ.get("UNIFIELD_VLLM_MAX_MODEL_LEN", "32768"),
            "--dtype",
            "bfloat16",
            "--trust-remote-code",
            "--gdn-prefill-backend",
            "triton",
            "--gpu-memory-utilization",
            os.environ.get("WANVSR_VLLM_GPU_MEMORY_UTILIZATION", "0.85"),
        ]
        if os.environ.get("UNIFIELD_VLLM_EAGER", "1") == "1":
            cmd += ["--enforce-eager"]
        cmd += ["--host", "127.0.0.1"]
        if lora_path:
            cmd += [
                "--enable-lora",
                "--lora-modules",
                f"{lora_name}={lora_path}",
            ]

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}
        env["PATH"] = str(Path(cmd[0]).parent) + os.pathsep + env.get("PATH", "")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        for k in list(env.keys()):
            ku = k.upper()
            if k in (
                "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK",
                "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID",
                "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
                "TORCHELASTIC_USE_AGENT_STORE",
            ) or ku in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"): 
                del env[k]
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["no_proxy"] = "127.0.0.1,localhost"
        print(f"[LoopTraining] Starting vLLM server on GPU {gpu_id}, port {port}...")
        print(f"[LoopTraining] Base VLM model: {base_model}")
        if lora_path:
            print(f"[LoopTraining] LoRA adapter: {lora_name}={lora_path}")
        print("[LoopTraining] Command: " + " ".join(cmd))

        log_path = f"/tmp/flashvsr_loop_vllm_{port}.log"

        def _read_log_tail(max_chars: int = 4000) -> str:
            try:
                with open(log_path, "r", errors="replace") as f:
                    data = f.read()
                return data[-max_chars:]
            except OSError:
                return ""

        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True,
                start_new_session=True,
            )
        print(f"[LoopTraining] vLLM log: {log_path}", flush=True)

        base_url = f"http://127.0.0.1:{port}/v1"
        ready_url = f"{base_url}/models"
        timeout_sec = int(os.environ.get("FLASHVSR_VLLM_TIMEOUT_SEC", "600"))
        start = time.time()
        while time.time() - start < timeout_sec:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited with code {proc.returncode}.\n"
                    f"Log tail ({log_path}):\n{_read_log_tail()}"
                )
            health = subprocess.run(
                ["curl", "--noproxy", "*", "-fsS", "--max-time", "2", ready_url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            if health.returncode == 0:
                break
            time.sleep(3)
        else:
            self._terminate_vllm_process(proc, timeout=15)
            raise RuntimeError(
                f"vLLM server did not start within {timeout_sec}s.\n"
                f"Log tail ({log_path}):\n{_read_log_tail()}"
            )

        client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",
            http_client=httpx.Client(trust_env=False, timeout=120.0),
        )
        print(f"[LoopTraining] vLLM server ready at {base_url}", flush=True)
        return client, proc

    def _loop_kwargs(self) -> dict:
        return {
            "loop_iterations": int(getattr(self, "loop_iterations", 2)),
            "loop_block_start": int(getattr(self, "loop_block_start", 0)),
            "loop_block_end": int(getattr(self, "loop_block_end", -1)),
            "loop_mask_threshold": float(getattr(self, "loop_mask_threshold", 0.05)),
            "loop_apply_to_masked_tokens": bool(
                getattr(self, "loop_apply_to_masked_tokens", True)
            ),
        }

    def _mask_to_latent_grid(
        self,
        raw_mask: torch.Tensor,
        latent_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        latent_f, latent_h, latent_w = latent_shape
        raw_mask = raw_mask.to(self.device).float()
        if raw_mask.dim() == 2:
            m_lat = F.interpolate(
                raw_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                size=(1, latent_h, latent_w),
                mode="trilinear",
                align_corners=False,
            ).expand(1, 1, latent_f, latent_h, latent_w)
        elif raw_mask.dim() == 3:
            m_lat = F.interpolate(
                raw_mask.unsqueeze(0).unsqueeze(0),
                size=(latent_f, latent_h, latent_w),
                mode="trilinear",
                align_corners=False,
            )
        else:
            m_lat = raw_mask
            if m_lat.dim() == 4:
                m_lat = m_lat.unsqueeze(1)
        return m_lat

    def _build_mask_inputs(
        self,
        conditions: List[RefinementCondition],
        max_plans: int,
        image_shape: Tuple[int, int, int],
        latent_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        n_frames, height, width = image_shape
        mask_channels = []
        prompts = []
        combined_mask_latent = torch.zeros(
            1,
            1,
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            device=self.device,
            dtype=torch.float32,
        )

        for cond in conditions[:max_plans]:
            raw_mask = cond.mask.to(self.device).float()
            if raw_mask.dim() == 2:
                ch = raw_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                ch = ch.expand(1, 1, n_frames, height, width)
            elif raw_mask.dim() == 3:
                ch = F.interpolate(
                    raw_mask.unsqueeze(0).unsqueeze(0),
                    size=(n_frames, height, width),
                    mode="trilinear",
                    align_corners=False,
                )
            else:
                ch = raw_mask.unsqueeze(0) if raw_mask.dim() == 4 else raw_mask
            mask_channels.append(ch * cond.strength)

            m_lat = self._mask_to_latent_grid(raw_mask, latent_shape)
            combined_mask_latent = torch.max(combined_mask_latent, m_lat * cond.strength)

            if cond.text_prompt:
                prompts.append(cond.text_prompt)

        while len(mask_channels) < max_plans:
            mask_channels.append(
                torch.zeros(
                    1,
                    1,
                    n_frames,
                    height,
                    width,
                    device=self.device,
                    dtype=self.torch_dtype,
                )
            )
        mask_input = torch.cat(mask_channels, dim=1).to(dtype=self.torch_dtype)
        return mask_input, combined_mask_latent, prompts

    def _refine_single_region(
        self,
        lq_image: torch.Tensor,
        current_image: torch.Tensor,
        condition: "RefinementCondition" = None,
        conditions: List["RefinementCondition"] = None,
        max_plans: int = 3,
        current_latents: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if conditions is None:
            conditions = [condition] if condition is not None else []

        all_prompts = [c.text_prompt for c in conditions if c.text_prompt]
        if all_prompts:
            try:
                self.init_cross_kv(prompt="; ".join(all_prompts))
            except Exception as e:
                print(f"Warning: Failed to update cross-KV: {e}")

        _, _, n_frames, height, width = current_image.shape

        if current_latents is None:
            self.load_models_to_device(["vae"])
            with torch.no_grad():
                latents = self.vae.encode(
                    current_image.to(self.device, dtype=self.torch_dtype),
                    device=self.device,
                    tiled=True,
                )
        else:
            latents = current_latents.to(device=self.device, dtype=self.torch_dtype)

        _, _, latent_f, latent_h, latent_w = latents.shape
        mask_input, combined_mask_latent, _ = self._build_mask_inputs(
            conditions=conditions,
            max_plans=max_plans,
            image_shape=(n_frames, height, width),
            latent_shape=(latent_f, latent_h, latent_w),
        )

        self.load_models_to_device(["dit"])
        time_emb_dtype = next(self.dit.time_embedding.parameters()).dtype

        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()
            LQ_latents = self.dit.LQ_proj_in.stream_forward(lq_image)
        else:
            LQ_latents = None

        Mask_latents = None
        if hasattr(self.dit, "Mask_proj_in"):
            self.dit.Mask_proj_in.clear_cache()
            mask_dtype = next(self.dit.Mask_proj_in.parameters()).dtype
            Mask_latents = self.dit.Mask_proj_in.stream_forward(mask_input.to(dtype=mask_dtype))
            if Mask_latents is not None:
                Mask_latents = [m.to(dtype=self.torch_dtype) for m in Mask_latents]

        substeps = max(1, int(self.stage3_substeps))
        step_scale = self.stage3_step_scale / float(substeps)
        refined_latents = latents
        tau_eps = 1e-4
        for step_id in range(substeps):
            tau_step = torch.tensor(
                [max(tau_eps, min(float(step_id) * step_scale, 1.0 - tau_eps))],
                device=self.device,
                dtype=torch.float32,
            )
            timestep_step = self._bridge_tau_to_model_timestep(tau_step).to(
                device=self.device,
                dtype=self.torch_dtype,
            )
            time_emb_input = sinusoidal_embedding_1d(self.dit.freq_dim, timestep_step).to(
                device=self.device,
                dtype=time_emb_dtype,
            )
            t_embed = self.dit.time_embedding(time_emb_input).to(dtype=self.torch_dtype)
            t_mod = self.dit.time_projection(t_embed).unflatten(1, (6, self.dit.dim))

            self.timestep = timestep_step
            self.t = t_embed
            self.t_mod = t_mod

            with torch.no_grad():
                v_pred, _, _ = model_fn_wan_video_looped(
                    self.dit,
                    x=refined_latents.to(dtype=self.torch_dtype),
                    timestep=timestep_step,
                    context=None,
                    tea_cache=None,
                    use_unified_sequence_parallel=False,
                    LQ_latents=LQ_latents,
                    Mask_latents=Mask_latents,
                    loop_mask_latent=combined_mask_latent,
                    is_full_block=False,
                    is_stream=False,
                    topk_ratio=2.0,
                    kv_ratio=3.0,
                    cur_process_idx=0,
                    t_mod=t_mod,
                    t=t_embed,
                    local_range=9,
                    **self._loop_kwargs(),
                )
            refined_latents = refined_latents + step_scale * v_pred

        with torch.no_grad():
            refined = self.TCDecoder.decode_video(
                refined_latents.to(dtype=self.torch_dtype).transpose(1, 2),
                parallel=False,
                show_progress_bar=False,
                cond=lq_image[:, :, : refined_latents.shape[2] * 4].to(
                    dtype=self.torch_dtype
                ),
                bidirectional=self.use_bidirectional_decoder,
            ).transpose(1, 2).mul_(2).sub_(1)

        if return_latents:
            return refined, refined_latents
        return refined

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
        if stage1_latents_cached is not None and hq_latents_cached is not None:
            hq_latents = hq_latents_cached.to(device=self.device, dtype=self.torch_dtype)
            stage1_latents = stage1_latents_cached.to(device=self.device, dtype=self.torch_dtype)
        else:
            self.load_models_to_device(["vae"])
            with torch.no_grad():
                hq_latents = self.vae.encode(
                    HQ_video.to(self.device, dtype=self.torch_dtype),
                    device=self.device,
                    tiled=True,
                )
                stage1_latents = self.vae.encode(
                    stage1_image.to(self.device, dtype=self.torch_dtype),
                    device=self.device,
                    tiled=True,
                )

        hq_latents = hq_latents.to(device=self.device, dtype=self.torch_dtype)
        stage1_latents = stage1_latents.to(device=self.device, dtype=self.torch_dtype)

        _, _, latent_f, latent_h, latent_w = hq_latents.shape
        _, _, frames, height, width = LQ_video.shape

        mask_input, combined_mask_latent, prompts = self._build_mask_inputs(
            conditions=conditions,
            max_plans=max_plans,
            image_shape=(frames, height, width),
            latent_shape=(latent_f, latent_h, latent_w),
        )

        if prompts:
            try:
                self.init_cross_kv(prompt="; ".join(prompts))
            except Exception:
                pass

        mask_weight = combined_mask_latent.expand_as(hq_latents).to(dtype=self.torch_dtype)

        tau_eps = 1e-4
        small_prob = float(getattr(self, "stage3_tau_small_prob", 0.8))
        small_prob = max(0.0, min(1.0, small_prob))
        small_upper = float(self.stage3_step_scale)
        small_upper = max(tau_eps, min(1.0, small_upper))

        if torch.rand(1).item() < small_prob:
            if small_upper <= tau_eps:
                t_uniform = torch.full((1,), tau_eps, device=self.device, dtype=torch.float32)
            else:
                t_uniform = tau_eps + (small_upper - tau_eps) * torch.rand(
                    1, device=self.device
                )
        else:
            if small_upper >= 1.0:
                t_uniform = torch.rand(1, device=self.device)
            else:
                t_uniform = small_upper + (1.0 - small_upper) * torch.rand(
                    1, device=self.device
                )
        t_uniform = t_uniform.clamp(min=tau_eps, max=1.0)

        latents = (1 - t_uniform) * stage1_latents + t_uniform * hq_latents
        latents = latents.to(dtype=self.torch_dtype)
        t_random = self._bridge_tau_to_model_timestep(t_uniform)

        self.load_models_to_device(["dit"])
        self.timestep = t_random.to(device=self.device, dtype=self.torch_dtype)
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
        if hasattr(self.dit, "LQ_proj_in"):
            LQ_latents = self.dit.LQ_proj_in.stream_forward(LQ_video_cast)
        else:
            LQ_latents = None

        Mask_latents = None
        if hasattr(self.dit, "Mask_proj_in"):
            self.dit.Mask_proj_in.clear_cache()
            mask_dtype = next(self.dit.Mask_proj_in.parameters()).dtype
            Mask_latents = self.dit.Mask_proj_in.stream_forward(mask_input.to(dtype=mask_dtype))
            if Mask_latents is not None:
                Mask_latents = [m.to(dtype=self.torch_dtype) for m in Mask_latents]

        v_pred, _, _ = model_fn_wan_video_looped(
            self.dit,
            x=latents,
            timestep=self.timestep,
            context=None,
            tea_cache=None,
            use_unified_sequence_parallel=False,
            LQ_latents=LQ_latents,
            Mask_latents=Mask_latents,
            loop_mask_latent=combined_mask_latent,
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

        target = (hq_latents - stage1_latents).to(dtype=self.torch_dtype)

        return {
            "noise_pred": v_pred,
            "target": target,
            "mask_weight": mask_weight,
            "loop_mask_weight": combined_mask_latent.to(dtype=self.torch_dtype),
        }


def _configure_loop_pipeline(pipe: LoopFlashVSR3StagePipeline, args) -> None:
    pipe.loop_iterations = args.loop_iterations
    pipe.loop_block_start = args.loop_block_start
    pipe.loop_block_end = args.loop_block_end
    pipe.loop_mask_threshold = args.loop_mask_threshold
    pipe.loop_apply_to_masked_tokens = args.loop_apply_to_masked_tokens
    pipe.loop_debug_every = args.loop_debug_every
    pipe.vlm_finetune_dir = args.vlm_finetune_dir
    pipe.vlm_base_model_name = args.vlm_base_model_name
    pipe.vlm_lora_path = args.vlm_lora_path
    pipe.vlm_lora_name = args.vlm_lora_name


class LoopTrainerDDP(base_train.ThreeStageTrainerDDP):
    def _init_pipeline(self):
        original_pipeline_cls = base_train.FlashVSR3StagePipeline
        original_pending = (
            LoopFlashVSR3StagePipeline._pending_vlm_base_model_name,
            LoopFlashVSR3StagePipeline._pending_vlm_lora_path,
            LoopFlashVSR3StagePipeline._pending_vlm_lora_name,
        )
        base_train.FlashVSR3StagePipeline = LoopFlashVSR3StagePipeline
        LoopFlashVSR3StagePipeline._pending_vlm_base_model_name = self.args.vlm_base_model_name
        LoopFlashVSR3StagePipeline._pending_vlm_lora_path = self.args.vlm_lora_path
        LoopFlashVSR3StagePipeline._pending_vlm_lora_name = self.args.vlm_lora_name
        try:
            pipe = super()._init_pipeline()
        finally:
            base_train.FlashVSR3StagePipeline = original_pipeline_cls
            (
                LoopFlashVSR3StagePipeline._pending_vlm_base_model_name,
                LoopFlashVSR3StagePipeline._pending_vlm_lora_path,
                LoopFlashVSR3StagePipeline._pending_vlm_lora_name,
            ) = original_pending
        _configure_loop_pipeline(pipe, self.args)
        return pipe

    def _lora_named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        params = [
            (name, param)
            for name, param in self._dit_module().named_parameters()
            if "lora_" in name
        ]
        if not params:
            raise RuntimeError("No LoRA parameters found in DiT for loop training.")
        return params

    def _clone_lora_state(self) -> Dict[str, torch.Tensor]:
        return {name: param.detach().clone() for name, param in self._lora_named_parameters()}

    def _copy_lora_state_(self, state: Dict[str, torch.Tensor]) -> None:
        named_params = dict(self._lora_named_parameters())
        missing = sorted(set(state) - set(named_params))
        if missing:
            raise KeyError(f"LoRA parameters missing while restoring state: {missing[:5]}")
        for name, value in state.items():
            param = named_params[name]
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))

    def _set_stage3_lora_enabled(self, enabled: bool):
        if not getattr(self.args, "continue_stage1_lora", True):
            return super()._set_stage3_lora_enabled(enabled)

        frozen_state = getattr(self, "_frozen_stage1_lora_state", None)
        if frozen_state is None:
            return

        if enabled:
            active_state = getattr(self, "_active_lora_state_before_stage1", None)
            if active_state is not None:
                self._copy_lora_state_(active_state)
                self._active_lora_state_before_stage1 = None
            return

        if getattr(self, "_active_lora_state_before_stage1", None) is None:
            self._active_lora_state_before_stage1 = self._clone_lora_state()
        self._copy_lora_state_(frozen_state)

    def _merge_stage1_and_reinit_stage3(self):
        if not getattr(self.args, "continue_stage1_lora", True):
            return super()._merge_stage1_and_reinit_stage3()

        dit = self._dit_module()
        for param in dit.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

        if base_train.is_dist():
            import torch.distributed as dist

            for p in dit.parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=0)
            base_train.print0("Broadcast continued Stage-1 LoRA parameters from rank 0.")

        self._frozen_stage1_lora_state = self._clone_lora_state()
        self._active_lora_state_before_stage1 = None

        trainable = sum(p.numel() for p in dit.parameters() if p.requires_grad)
        base_train.print0("\nKeeping Stage-1 LoRA trainable for loop refinement.")
        base_train.print0("Frozen Stage-1 LoRA snapshot will generate stage1_image.")
        base_train.print0(f"Trainable params after Stage-1 load: {trainable:,}")

    def train(self):
        if base_train.is_main():
            num_blocks = len(self.pipe.dit.blocks)
            loop_start, loop_end = _loop_bounds(
                num_blocks, self.args.loop_block_start, self.args.loop_block_end
            )
            base_train.print0("\nLoop-token training config:")
            base_train.print0(f"  loop_iterations: {self.args.loop_iterations}")
            base_train.print0(f"  loop block range: {loop_start}..{loop_end} of {num_blocks} blocks")
            base_train.print0(
                f"  apply to VLM/atlas masked tokens: {self.args.loop_apply_to_masked_tokens}"
            )
            base_train.print0(f"  mask threshold: {self.args.loop_mask_threshold}")
            base_train.print0(f"  continue Stage-1 LoRA: {self.args.continue_stage1_lora}")
            base_train.print0(f"  VLM finetune dir note: {self.args.vlm_finetune_dir}")
            base_train.print0(f"  VLM base model: {self.args.vlm_base_model_name}")
            base_train.print0(f"  VLM LoRA adapter: {self.args.vlm_lora_name}={self.args.vlm_lora_path}")
        return super().train()


def main():
    args = get_args()

    if getattr(args, "click_run_defaults_active", False):
        print("[LoopTraining] Click Run defaults are active (no CLI arguments detected).")
        print(f"[LoopTraining] Edit CLICK_RUN_DEFAULTS in {THIS_FILE} to change this run profile.")

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
        gpu_ids = base_train.split_gpu_ids(args.gpu)
        world = base_train.get_world_size()
        local_world = base_train.get_local_world_size()
        print(f"\n{'=' * 60}")
        print("GPU Allocation Plan:")
        print(f"  Physical GPUs: {args.gpu}")
        ddp_phys = ",".join(gpu_ids[:local_world]) if len(gpu_ids) >= local_world else "N/A"
        print(f"  DDP workers: {world} (local cuda:0..{local_world - 1} -> physical {ddp_phys})")
        vlm_device = base_train.resolve_vlm_device(args, local_world)
        vlm_index = base_train.cuda_device_index(vlm_device)
        if vlm_index is not None and vlm_index < len(gpu_ids):
            print(f"  VLM server: {vlm_device} (physical GPU {gpu_ids[vlm_index]})")
        elif use_ddp:
            print("  VLM server: shared with cuda:0 (no dedicated VLM GPU configured)")
        print(f"{'=' * 60}\n")

    trainer = None
    try:
        trainer = LoopTrainerDDP(args)
        trainer.train()
    finally:
        if trainer is not None and hasattr(trainer, "pipe") and hasattr(trainer.pipe, "shutdown_vllm"):
            trainer.pipe.shutdown_vllm()
        base_train.cleanup_distributed()


if __name__ == "__main__":
    main()
