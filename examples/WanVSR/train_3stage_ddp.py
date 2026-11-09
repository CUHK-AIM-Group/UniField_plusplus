#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashVSR 3-Stage MRI Training Script — Multi-GPU DDP Version
=============================================================

Default layout (physical GPUs 0,1):
    cuda:0     -> DiT + VAE training worker
    cuda:1     -> VLM vLLM server (Phase 2 only, shared via HTTP)

Usage:
    # Launch with torchrun (recommended):
    CUDA_VISIBLE_DEVICES=0,1 torchrun \
        --nproc_per_node=1 \
        train_3stage_ddp.py \
        --config configs/3stage_64mT_to_3T_ddp.yaml \
        --gpu 0,1

    # Single-GPU fallback (same interface as original):
    python train_3stage_ddp.py \
        --config configs/3stage_64mT_to_3T.yaml --gpu 0,1
"""

import sys
import os
import warnings
import argparse
import json
import time
import yaml
from datetime import datetime, timedelta
from typing import Dict


def _early_cli_value(name, default=None):
    """Read a simple CLI option before importing torch/CUDA users."""
    prefix = f"{name}="
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


def _early_config_gpu(config_path):
    if not config_path:
        return None
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[GPU] Warning: failed to read config for early CUDA setup: {exc}", flush=True)
        return None
    pipeline = config.get("pipeline", {})
    if isinstance(pipeline, dict):
        gpu = pipeline.get("gpu")
        if gpu:
            return str(gpu)
    return None


def _set_cuda_visible_devices_early():
    default_config = os.path.join(
        os.path.dirname(__file__),
        "configs/3stage_64mT_to_3T_ddp.yaml",
    )
    config_path = _early_cli_value("--config", default_config)
    config_gpu = _early_config_gpu(config_path)
    cli_gpu = _early_cli_value("--gpu")
    gpu = cli_gpu or config_gpu or "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["FLASHVSR_PHYSICAL_GPUS"] = gpu
    print(f"[GPU] Early CUDA_VISIBLE_DEVICES = {gpu}", flush=True)


_set_cuda_visible_devices_early()

# ──────────────────────────────────────────────────────────
# CUDA_VISIBLE_DEVICES is set above before importing torch.
# torchrun + LOCAL_RANK still handle per-process device assignment.
# ──────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("NCCL_TIMEOUT", "1800000")  # 30 min (default 10 min)
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
# Disable P2P and IB to avoid NCCL barrier deadlock on some multi-GPU systems.
# If GPUs are on different PCIe roots or P2P access is blocked, NCCL's default
# P2P transport will silently hang on the first collective (e.g. barrier).
# SHM (shared memory) transport is always reliable on a single node.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "0")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast
from tqdm import tqdm
import numpy as np
from PIL import Image
import nibabel as nib
from peft import LoraConfig, inject_adapter_in_model

from diffsynth import ModelManager
from diffsynth.pipelines.flashvsr_3stage import FlashVSR3StagePipeline, ThreeStageOutput
from diffsynth.agents import AgentConfig
from diffsynth.models.wan_video_vae_bidirectional import BidirectionalWanVideoVAE

from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

try:
    import monai.transforms as mtransforms
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False
    print("Warning: MONAI not installed. Using basic data loading.")


# ═══════════════════════════════════════════════════════════
#  Distributed helpers
# ═══════════════════════════════════════════════════════════

def is_dist():
    """Return True if running inside a torchrun / DDP context."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return get_rank() == 0


def barrier():
    if is_dist():
        dist.barrier()


def print0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main():
        print(*args, **kwargs, flush=True)


def is_validation_padding_batch(loader, batch_idx):
    """Return whether a batch only exists because DistributedSampler padded it.

    Validation loaders in this trainer always use ``batch_size=1`` and a
    non-shuffled DistributedSampler.  With ``drop_last=False`` PyTorch repeats
    samples from the start of the dataset until every rank has the same number
    of batches.  Keeping the padded batch in the loader preserves the matching
    per-rank barrier count, while callers can skip its inference and metrics.
    """
    sampler = getattr(loader, "sampler", None)
    if not isinstance(sampler, DistributedSampler):
        return False
    if sampler.drop_last or sampler.shuffle:
        return False

    batch_size = loader.batch_size
    if batch_size != 1:
        raise ValueError(
            "Padding-batch detection requires validation batch_size=1; "
            f"got {batch_size!r}."
        )

    dataset_size = len(loader.dataset)
    rank = sampler.rank
    replicas = sampler.num_replicas
    unique_on_rank = max(0, (dataset_size - rank + replicas - 1) // replicas)
    return batch_idx >= unique_on_rank


def setup_distributed():
    """Initialise the NCCL process group.

    torchrun sets LOCAL_RANK, RANK, WORLD_SIZE automatically.
    If those are absent we fall back to single-GPU mode (no DDP).
    """
    if "RANK" not in os.environ:
        return False  # single-GPU fallback

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=30),
    )
    return True


def wait_for_gpu_with_retry(required_index=0, max_retries=12, wait_seconds=300):
    """Wait for CUDA/GPU to become available with retry.

    Args:
        required_index: The cuda device index that must be accessible.
        max_retries: Number of retries after the first failed attempt.
        wait_seconds: Sleep duration between retries.
    """
    rank_hint = os.environ.get("RANK", "0")
    attempt = 0
    while True:
        cuda_ok = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_ok else 0
        if cuda_ok and device_count > required_index:
            return

        if attempt >= max_retries:
            raise RuntimeError(
                f"[rank {rank_hint}] GPU not available after {attempt + 1} attempts. "
                f"cuda_available={cuda_ok}, device_count={device_count}, "
                f"required_index={required_index}."
            )

        print(
            f"[rank {rank_hint}] GPU not ready "
            f"(cuda_available={cuda_ok}, device_count={device_count}, "
            f"required_index={required_index}). "
            f"Retry in {wait_seconds}s (attempt {attempt + 1}/{max_retries + 1})...",
            flush=True,
        )
        time.sleep(wait_seconds)
        attempt += 1


def get_local_world_size():
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def split_gpu_ids(gpu_arg):
    return [g.strip() for g in str(gpu_arg).split(",") if g.strip()]


def cuda_device_index(device_str):
    if device_str is None:
        return None
    value = str(device_str).strip()
    if value == "cuda":
        return 0
    if not value.startswith("cuda:"):
        return None
    try:
        return int(value.split(":", 1)[1])
    except ValueError:
        return None


def resolve_vlm_device(args, local_world_size=None):
    gpu_ids = split_gpu_ids(args.gpu)
    local_world_size = get_local_world_size() if local_world_size is None else local_world_size
    if args.vlm_device is not None:
        return args.vlm_device
    if len(gpu_ids) > local_world_size:
        return f"cuda:{len(gpu_ids) - 1}"
    return "cuda:0"


def validate_gpu_layout(args):
    """Validate logical CUDA indices before PyTorch hits a cryptic assert."""
    gpu_ids = split_gpu_ids(args.gpu)
    if not gpu_ids:
        raise RuntimeError("--gpu must contain at least one physical GPU id.")

    local_world_size = get_local_world_size()
    if local_world_size > len(gpu_ids):
        raise RuntimeError(
            f"GPU layout invalid: torchrun requested {local_world_size} local DDP "
            f"workers, but --gpu exposes only {len(gpu_ids)} devices ({args.gpu}). "
            "For physical GPUs 0,1 use --nproc_per_node=1 so cuda:1 stays free for VLM."
        )

    worker_required_index = max(0, local_world_size - 1)
    vlm_device = resolve_vlm_device(args, local_world_size)
    vlm_index = cuda_device_index(vlm_device) if getattr(args, "use_vlm", False) else None
    if vlm_index is not None:
        if vlm_index >= len(gpu_ids):
            raise RuntimeError(
                f"GPU layout invalid: --gpu {args.gpu} exposes logical cuda:0.."
                f"{len(gpu_ids) - 1}, but vlm_device={vlm_device} requests cuda:{vlm_index}. "
                "For physical GPUs 0,1 set --gpu 0,1 and --vlm_device cuda:1."
            )
        if local_world_size > 1 and vlm_index < local_world_size:
            raise RuntimeError(
                f"GPU layout invalid: vlm_device={vlm_device} overlaps DDP worker cuda:{vlm_index}. "
                "For physical GPUs 0,1 use torchrun --nproc_per_node=1 so cuda:1 stays reserved for VLM."
            )

    cuda_ok = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_ok else 0
    if cuda_ok and device_count <= worker_required_index:
        raise RuntimeError(
            f"CUDA layout mismatch: DDP needs logical cuda:{worker_required_index}, "
            f"but PyTorch sees only {device_count} GPU(s). "
            "For physical GPUs 0,1 use torchrun --nproc_per_node=1."
        )
    if cuda_ok and vlm_index is not None and vlm_index >= device_count:
        physical = gpu_ids[vlm_index] if vlm_index < len(gpu_ids) else "unknown"
        print(
            f"[GPU] VLM {vlm_device} is external to this training process "
            f"(PyTorch sees {device_count} GPU(s)); vLLM will be launched on physical GPU {physical}.",
            flush=True,
        )

    return worker_required_index


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════
#  Config helpers (unchanged)
# ═══════════════════════════════════════════════════════════

def _flatten_yaml_config(yaml_config):
    key_map = {
        'data.train_list': 'train_list',
        'data.val_list': 'val_list',
        'data.task_type': 'task_type',
        'data.height': 'height',
        'data.width': 'width',
        'data.num_frames': 'num_frames',
        'data.use_monai': 'use_monai',
        'data.monai_spatial_size': 'monai_spatial_size',
        'model.dit_path': 'dit_path',
        'model.vae_path': 'vae_path',
        'model.text_encoder_path': 'text_encoder_path',
        'model.lq_proj_path': 'lq_proj_path',
        'model.tcdecoder_path': 'tcdecoder_path',
        'training.num_epochs': 'num_epochs',
        'training.batch_size': 'batch_size',
        'training.learning_rate': 'learning_rate',
        'training.min_lr': 'min_lr',
        'training.weight_decay': 'weight_decay',
        'training.warmup_epochs': 'warmup_epochs',
        'training.max_grad_norm': 'max_grad_norm',
        'training.use_amp': 'use_amp',
        'training.val_every': 'val_every',
        'training.validate_before_train': 'validate_before_train',
        'training.validate_before_train_save_images': 'validate_before_train_save_images',
        'training.save_every': 'save_every',
        'lora.rank': 'lora_rank',
        'lora.alpha': 'lora_alpha',
        'lora.target_modules': 'lora_target_modules',
        'pipeline.enable_stage2': 'enable_stage2',
        'pipeline.enable_stage3': 'enable_stage3',
        'pipeline.max_refinement_iterations': 'max_refinement_iterations',
        'pipeline.stage3_timestep_max': 'stage3_timestep_max',
        'pipeline.stage3_tau_small_prob': 'stage3_tau_small_prob',
        'pipeline.vlm_model_name': 'vlm_model_name',
        'pipeline.use_vlm': 'use_vlm',
        'pipeline.use_bidirectional': 'use_bidirectional',
        'pipeline.vlm_device': 'vlm_device',
        'pipeline.gpu': 'gpu',
        'inference.topk_ratio': 'topk_ratio',
        'inference.kv_ratio': 'kv_ratio',
        'inference.local_range': 'local_range',
        'output.output_dir': 'output_dir',
        'output.num_workers': 'num_workers',
        'output.gpu_retry_wait_seconds': 'gpu_retry_wait_seconds',
        'output.gpu_retry_max_retries': 'gpu_retry_max_retries',
        'phase.stage3_lr': 'stage3_lr',
        'phase.mask_loss_weight': 'mask_loss_weight',
        'phase.ssim_loss_weight': 'stage3_ssim_weight',
        'phase.grad_loss_weight': 'stage3_grad_weight',
        'phase.ssim_kernel': 'stage3_ssim_kernel',
    }
    flat = {}
    for section_name, section in yaml_config.items():
        if isinstance(section, dict):
            for k, v in section.items():
                dotkey = f"{section_name}.{k}"
                if dotkey in key_map:
                    flat[key_map[dotkey]] = v
        else:
            flat[section_name] = section
    return flat


def get_args():
    parser = argparse.ArgumentParser(description="3-Stage MRI Enhancement Training (DDP)")

    parser.add_argument("--config", type=str,
                        default='configs/train.yaml',
                        help="Path to YAML config file")

    # Data
    parser.add_argument("--train_list", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="64mT_to_3T",
                        choices=["64mT_to_3T", "3T_to_7T"])

    # Model paths
    parser.add_argument("--dit_path", type=str,
                        default="weights/FlashVSR/diffusion_pytorch_model_streaming_dmd.safetensors")
    parser.add_argument("--vae_path", type=str,
                        default="weights/FlashVSR/Wan2.1_VAE.pth")
    parser.add_argument("--text_encoder_path", type=str,
                        default="weights/FlashVSR/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--lq_proj_path", type=str,
                        default="weights/FlashVSR/LQ_proj_in.ckpt")
    parser.add_argument("--tcdecoder_path", type=str,
                        default="weights/FlashVSR/TCDecoder.ckpt")

    # Training
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q,k,v,o,ffn.0,ffn.2")

    # Data settings
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--use_monai", action="store_true", default=True)
    parser.add_argument("--monai_spatial_size", type=int, nargs=3,
                        default=[192, 192, 140])

    # 3-Stage
    parser.add_argument("--enable_stage2", action="store_true", default=True)
    parser.add_argument("--enable_stage3", action="store_true", default=True)
    parser.add_argument("--max_refinement_iterations", type=int, default=1)
    parser.add_argument("--stage3_timestep_max", type=float, default=50.0,
                        help="Stage-3 bridge timestep upper bound (tau=0 maps to this value)")
    parser.add_argument("--stage3_tau_small_prob", type=float, default=0.8,
                        help="Probability of sampling tau from (0, stage3_step_scale) in Stage-3 training")
    parser.add_argument("--vlm_model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--use_vlm", action="store_true", default=True)
    parser.add_argument("--vlm_device", type=str, default=None,
                        help="VLM device. Default: last GPU in --gpu list")

    # Model settings
    parser.add_argument("--use_bidirectional", action="store_true", default=True)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--topk_ratio", type=float, default=2.0)
    parser.add_argument("--kv_ratio", type=float, default=3.0)
    parser.add_argument("--local_range", type=int, default=9)

    # Output
    parser.add_argument("--output_dir", type=str, default="runs/main")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--validate_before_train", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Run one validation pass before the first training epoch")
    parser.add_argument("--validate_before_train_save_images", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Save validation comparison images during pre-training validation")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0,1",
                        help="Physical GPU ids in visible order, e.g. '0,1'")
    parser.add_argument("--gpu_retry_wait_seconds", type=int, default=300,
                        help="Seconds to wait before retry when GPU is not available")
    parser.add_argument("--gpu_retry_max_retries", type=int, default=12,
                        help="Max retry count when GPU is not available")

    # Phase control
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2])
    parser.add_argument("--stage1_ckpt", type=str,
                        default='weights/stage1/')
    parser.add_argument("--stage3_lr", type=float, default=5e-5)
    parser.add_argument("--mask_loss_weight", type=float, default=5.0)
    parser.add_argument("--stage3_ssim_weight", type=float, default=0.5,
                        help="Weight for Stage3 latent structural SSIM loss")
    parser.add_argument("--stage3_grad_weight", type=float, default=0.5,
                        help="Weight for Stage3 latent gradient consistency loss")
    parser.add_argument("--stage3_ssim_kernel", type=int, default=5,
                        help="Odd kernel size for Stage3 3D SSIM loss")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # YAML config
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        yaml_flat = _flatten_yaml_config(yaml_config)
        cli_set = {action.dest for action in parser._actions
                   if action.dest in sys.argv or
                   any(opt in sys.argv for opt in action.option_strings)}
        for key, value in yaml_flat.items():
            if key not in cli_set and hasattr(args, key):
                setattr(args, key, value)

    if args.train_list is None:
        parser.error("--train_list is required (via CLI or config file)")
    if args.val_list is None:
        parser.error("--val_list is required (via CLI or config file)")
    if args.phase == 2 and args.stage1_ckpt is None:
        parser.error("--stage1_ckpt is required when --phase=2")
    if not (0.0 <= args.stage3_tau_small_prob <= 1.0):
        parser.error("--stage3_tau_small_prob must be in [0, 1]")

    # Force validation cadence: run once every 5 epochs.
    args.val_every = 5

    return args


# ═══════════════════════════════════════════════════════════
#  MRI Dataset (identical to single-GPU version)
# ═══════════════════════════════════════════════════════════

from unifield.dataset import RegisteredMRIDataset as MRIDataset

# ═══════════════════════════════════════════════════════════
#  DDP Trainer
# ═══════════════════════════════════════════════════════════

class ThreeStageTrainerDDP:
    """
    Multi-GPU DDP trainer for 3-Stage MRI Enhancement.

        GPU layout (default: CUDA_VISIBLE_DEVICES=0,1):
            cuda:0 (rank 0)  -> DiT/training worker on physical GPU 0
            cuda:1           -> VLM vLLM server on physical GPU 1

    Note: The pipeline's forward pass manually iterates through dit.blocks,
    so we cannot wrap dit with nn.parallel.DistributedDataParallel.
    Instead we use manual gradient all-reduce on the LoRA parameters
    after each backward pass. This gives equivalent DDP behavior.

    Phase 1: single DiT worker for Stage 1 LoRA training.
    Phase 2: single DiT worker for Stage 3 LoRA training, VLM on cuda:1.
    """

    def __init__(self, args):
        self.args = args
        self.phase = args.phase
        self.use_ddp = is_dist()
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Each DDP worker uses its own cuda device
        if self.use_ddp:
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Determine VLM device (last local visible GPU, outside DDP workers)
        self.local_world_size = get_local_world_size()
        self.vlm_device_str = resolve_vlm_device(args, self.local_world_size)

        # Create output directory (rank 0 only, but compute path on all ranks)
        # Each rank computes the same path locally to avoid broadcast deadlock
        if args.resume_from:
            # e.g. resume_from = ".../phase2_.../checkpoints/epoch_25.pth"
            # Go up two levels to get the run directory.
            resume_dir = os.path.dirname(os.path.dirname(args.resume_from))
            if os.path.isdir(resume_dir) and os.path.exists(os.path.join(resume_dir, "args.json")):
                self.output_dir = resume_dir
                print0(f"Resuming into existing directory: {self.output_dir}")
            else:
                # Fallback: use output_dir directly (no timestamp subfolder)
                self.output_dir = args.output_dir
        else:
            phase_tag = f"phase{self.phase}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(args.output_dir, f"{phase_tag}_{timestamp}")

        # Only rank 0 creates directories
        if is_main():
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
            with open(os.path.join(self.output_dir, "args.json"), 'w') as f:
                json.dump(vars(args), f, indent=2)

        # Barrier to ensure rank 0 finishes directory setup before proceeding
        barrier()

        # Initialize pipeline
        print0("\n" + "="*60)
        print0(f"Initializing 3-Stage MRI Enhancement Pipeline  [Phase {self.phase}]")
        print0(f"DDP: {self.world_size} workers, VLM device: {self.vlm_device_str}")
        print0("="*60)

        self.pipe = self._init_pipeline()

        # Add LoRA
        self._add_lora()

        # Phase 2: load Stage 1, merge, reinit
        if self.phase == 2:
            self._load_stage1_checkpoint(args.stage1_ckpt)
            self._merge_stage1_and_reinit_stage3()
            if not self.args.enable_stage2:
                print0("Warning: Phase 2 requires Stage 2 agents. Enabling automatically.")
                self.args.enable_stage2 = True
                if is_main():
                    self.pipe.init_agents(
                        task_type=self.args.task_type,
                        vlm_model_name=self.args.vlm_model_name,
                        use_vlm=self.args.use_vlm,
                        vlm_device=self.vlm_device_str,
                    )

        # Wrap DiT with DDP — NOT possible because the pipeline's forward pass
        # manually iterates through dit.blocks (see model_fn_wan_video).
        # Instead, we manually all-reduce LoRA gradients after each backward.
        # Collect the trainable (LoRA + Mask_proj_in) parameter references
        # once for efficiency.  Mask_proj_in is a submodule of dit so its
        # params should already appear in dit.parameters(), but we add them
        # explicitly to guarantee they are included in the allreduce set.
        #
        # CRITICAL: Must use deterministic ordering!  Using set() caused each
        # DDP rank to iterate parameters in a different order (set iteration
        # depends on id()/memory address which differs across processes).
        # When _allreduce_lora_grads calls all_reduce per param, mismatched
        # shapes across ranks cause NCCL deadlock.  We use the deterministic
        # traversal order of named_parameters() instead.
        seen_ids = set()
        lora_params_ordered = []
        for _name, p in self.pipe.dit.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                lora_params_ordered.append(p)
                seen_ids.add(id(p))
        if hasattr(self.pipe.denoising_model(), 'Mask_proj_in'):
            for _name, p in self.pipe.denoising_model().Mask_proj_in.named_parameters():
                if p.requires_grad and id(p) not in seen_ids:
                    lora_params_ordered.append(p)
                    seen_ids.add(id(p))
        self._lora_params = lora_params_ordered
        print0(f"  LoRA params to sync across ranks: {len(self._lora_params)}")

        # Optimizer
        lr = self.args.stage3_lr if self.phase == 2 else self.args.learning_rate
        self.optimizer = self._setup_optimizer(lr=lr)

        # DataLoaders
        self.train_loader, self.train_sampler = self._setup_dataloader('train')
        self.val_loader, self.val_sampler = self._setup_dataloader('val')

        # Scheduler
        self.scheduler = self._setup_lr_scheduler()

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_psnr = 0.0
        self.global_step = 0
        self.lpips_metric = None
        # Note: GradScaler is intentionally NOT used. bfloat16 has the same
        # exponent range as float32 so loss scaling is unnecessary, and
        # GradScaler's per-rank found_inf flag is incompatible with manual
        # gradient allreduce (ranks can diverge when one sees inf but others don't).

        # Resume
        if args.resume_from:
            self._load_checkpoint(args.resume_from)

        print0(f"\nOutput directory: {self.output_dir}")
        print0(f"Phase: {self.phase}, LR: {lr}")
        print0("="*60 + "\n")

    # ────────────────────────────────────────
    #  Pipeline initialization
    # ────────────────────────────────────────

    def _init_pipeline(self):
        print0("Loading models...")
        mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        mm.load_models([
            self.args.dit_path,
            self.args.vae_path,
            self.args.text_encoder_path,
        ])

        gen_device = str(self.device)  # e.g. "cuda:0" for rank 0

        # Only rank 0 launches VLM server (others connect via HTTP)
        vlm_device = self.vlm_device_str

        pipe = FlashVSR3StagePipeline.from_model_manager(
            mm, device=gen_device, vlm_device=vlm_device
        )

        pipe.enable_stage2 = self.args.enable_stage2
        pipe.enable_stage3 = self.args.enable_stage3
        pipe.stage3_timestep_max = float(self.args.stage3_timestep_max)
        pipe.stage3_tau_small_prob = float(self.args.stage3_tau_small_prob)
        pipe.vlm_model_name = self.args.vlm_model_name

        # Load LQ projection
        from utils.utils import Bidirectional_LQ4x_Proj, Buffer_LQ4x_Proj, Mask_Proj_In
        if self.args.use_bidirectional:
            pipe.denoising_model().LQ_proj_in = Bidirectional_LQ4x_Proj(
                in_dim=3, out_dim=1536, layer_num=1
            ).to(gen_device, dtype=torch.bfloat16)
        else:
            pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(
                in_dim=3, out_dim=1536, layer_num=1
            ).to(gen_device, dtype=torch.bfloat16)
        pipe.denoising_model().LQ_proj_in.load_state_dict(
            torch.load(self.args.lq_proj_path, map_location='cpu'), strict=False
        )
        print0(f"Loaded LQ_proj_in from {self.args.lq_proj_path}")

        # Mask projection for Stage 3 (multi-channel, one channel per plan)
        pipe.denoising_model().Mask_proj_in = Mask_Proj_In(
            max_plans=3, out_dim=1536, layer_num=1,
        ).to(gen_device, dtype=torch.bfloat16)
        print0("Initialised Mask_proj_in (max_plans=3)")

        # Bidirectional VAE
        if self.args.use_bidirectional:
            original_vae_state = pipe.vae.state_dict()
            bidirectional_vae = BidirectionalWanVideoVAE(z_dim=16, dim=96)
            bidirectional_vae.load_state_dict(original_vae_state, strict=False)
            pipe.vae = bidirectional_vae.to(gen_device, dtype=torch.bfloat16)
            pipe.vae.eval()
            pipe.use_bidirectional_decoder = True

        # TCDecoder
        from utils.TCDecoder import build_tcdecoder
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(
            new_channels=multi_scale_channels, new_latent_channels=16+768
        )
        pipe.TCDecoder.load_state_dict(
            torch.load(self.args.tcdecoder_path, map_location='cpu'), strict=False
        )
        print0(f"Loaded TCDecoder from {self.args.tcdecoder_path}")

        pipe.to(gen_device)

        # ── Broadcast non-DIT component weights from rank 0 ──
        # torch.load without map_location can silently fail on non-rank-0
        # when checkpoint was saved with GPU tensors. Broadcast ensures
        # all ranks have identical pretrained weights.
        if is_dist():
            for p in pipe.TCDecoder.parameters():
                dist.broadcast(p.data, src=0)
            for p in pipe.denoising_model().LQ_proj_in.parameters():
                dist.broadcast(p.data, src=0)
            for p in pipe.denoising_model().Mask_proj_in.parameters():
                dist.broadcast(p.data, src=0)
            print0("Broadcast TCDecoder, LQ_proj_in & Mask_proj_in weights from rank 0.")

        # Skip VRAM management for training — each GPU has 49 GB and we only
        # use ~15 GB, so CPU↔GPU swapping is pure overhead (causes 15 s delay
        # per LQ_proj_in.stream_forward call).  Keep all models on GPU.
        # pipe.enable_vram_management(num_persistent_param_in_dit=None)

        # Init cross-attention
        default_prompt = "MRI T1 sequence enhancement"
        pipe.init_cross_kv(prompt=default_prompt)
        # Text encoder stays on GPU — init_cross_kv is called each batch.

        # Init agents (rank 0 launches VLM server, others connect via HTTP)
        # IMPORTANT: rank 0 blocks for up to 5 min waiting for vLLM health,
        # so we must use a barrier BEFORE the broadcast to prevent deadlock.
        self._vllm_port = 0  # will be set by rank 0 and broadcast
        if self.args.enable_stage2:
            if is_main():
                print0("\nInitializing Stage 2 agents (rank 0 launches VLM server)...")
                pipe.init_agents(
                    task_type=self.args.task_type,
                    vlm_model_name=self.args.vlm_model_name,
                    use_vlm=self.args.use_vlm,
                    vlm_device=vlm_device,
                )
                # Extract port from the client base_url stored by init_agents
                self._vllm_port = self._extract_vllm_port(pipe)
                print0(f"VLM server ready on port {self._vllm_port}")

            # Barrier: rank 1..N wait here while rank 0 starts vLLM.
            # This ensures all ranks reach broadcast together.
            barrier()

            # Now broadcast VLM port from rank 0 to all workers
            if self.use_ddp:
                port_tensor = torch.tensor([self._vllm_port], dtype=torch.long,
                                           device=self.device)
                dist.broadcast(port_tensor, src=0)
                self._vllm_port = port_tensor.item()

            # Non-rank-0 workers: create HTTP client to rank 0's VLM server
            if not is_main() and self.args.use_vlm and self._vllm_port > 0:
                self._connect_workers_to_vlm_server(pipe)

        return pipe

    def _extract_vllm_port(self, pipe):
        """Extract the port number from the VLM client's base_url."""
        try:
            if hasattr(pipe, 'vlm_agent') and pipe.vlm_agent is not None:
                client = pipe.vlm_agent.vlm_client
                if client is not None:
                    # base_url is like "http://127.0.0.1:23794/v1"
                    base_url = str(client.base_url)
                    import re
                    match = re.search(r':(\d+)', base_url)
                    if match:
                        return int(match.group(1))
        except Exception:
            pass
        return 23794  # default fallback

    def _connect_workers_to_vlm_server(self, pipe):
        """Non-rank-0 workers connect to the VLM server launched by rank 0."""
        import httpx
        from openai import OpenAI

        port = self._vllm_port
        client = OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="EMPTY",
            http_client=httpx.Client(trust_env=False, timeout=120.0),
        )

        # Replicate the agent setup from rank 0 but with shared HTTP client
        from diffsynth.agents import AgentConfig
        from diffsynth.agents.anomaly_detector import AnomalyDetectionAgent
        from diffsynth.agents.medical_vlm import MedicalVLMAgent

        agent_config = AgentConfig(
            task_type=self.args.task_type,
            model_name=self.args.vlm_model_name,
        )
        pipe.anomaly_agent = AnomalyDetectionAgent(
            config=agent_config, use_vlm=True, use_metrics=True, vlm_client=client,
        )
        pipe.vlm_agent = MedicalVLMAgent(
            config=agent_config, use_vlm=True, vlm_client=client,
        )
        pipe.anomaly_agent._is_initialized = True
        pipe.vlm_agent._is_initialized = True

        # Create orchestrator so run_stage2() won't trigger lazy init_agents()
        from diffsynth.agents.base_agent import AgentOrchestrator
        pipe.agent_orchestrator = AgentOrchestrator(
            anomaly_agent=pipe.anomaly_agent,
            vlm_agent=pipe.vlm_agent,
            max_iterations=3,
            quality_threshold=0.9,
        )

        print(f"[Rank {self.rank}] Connected to VLM server at port {port}")

    # ────────────────────────────────────────
    #  LoRA setup
    # ────────────────────────────────────────

    def _dit_module(self):
        """Get the underlying dit module (no DDP wrapper in this design)."""
        return self.pipe.dit

    def _allreduce_lora_grads(self):
        """Average LoRA gradients across all DDP ranks.

        Called after backward() and before optimizer.step().
        Flattens all LoRA gradients into a single buffer and does ONE
        all_reduce call instead of ~500 individual calls.  This reduces
        NCCL launch overhead from O(N_params) to O(1).
        """
        if not self.use_ddp:
            return
        # Ensure every param has a gradient tensor
        for p in self._lora_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
        # Flatten → single all_reduce → scatter back
        flat = torch.cat([p.grad.flatten() for p in self._lora_params])
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        offset = 0
        for p in self._lora_params:
            numel = p.grad.numel()
            p.grad.copy_(flat[offset:offset + numel].view_as(p.grad))
            offset += numel

    def _add_lora(self):
        print0(f"\nAdding LoRA (rank={self.args.lora_rank}, alpha={self.args.lora_alpha})...")
        self.pipe.requires_grad_(False)
        self.pipe.eval()

        dit = self.pipe.denoising_model()
        dit.train()

        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=self.args.lora_target_modules.split(","),
        )
        self.pipe.dit = inject_adapter_in_model(lora_config, dit)

        for param in self.pipe.dit.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

        self.pipe.denoising_model().LQ_proj_in.requires_grad_(False)

        # Mask_proj_in is trainable in Stage 3
        if hasattr(self.pipe.denoising_model(), 'Mask_proj_in'):
            self.pipe.denoising_model().Mask_proj_in.requires_grad_(True)
            for p in self.pipe.denoising_model().Mask_proj_in.parameters():
                p.data = p.data.to(torch.float32)

        trainable = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.pipe.dit.parameters())
        print0(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # Broadcast initial LoRA weights from rank 0 so all DDP workers
        # start with identical parameters (gaussian init is rank-local).
        if is_dist():
            for p in self.pipe.dit.parameters():
                dist.broadcast(p.data, src=0)

    def _merge_stage1_and_reinit_stage3(self):
        import math
        from peft.tuners.lora import Linear as LoraLinear

        dit = self._dit_module()
        n_merged = 0
        for _name, module in dit.named_modules():
            if not isinstance(module, LoraLinear):
                continue
            for adapter_name in list(module.lora_A.keys()):
                lora_A = module.lora_A[adapter_name]
                lora_B = module.lora_B[adapter_name]
                scaling = module.scaling[adapter_name]
                with torch.no_grad():
                    delta = (lora_B.weight @ lora_A.weight) * scaling
                    module.base_layer.weight.data += delta.to(module.base_layer.weight.dtype)
                nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(lora_B.weight)
            n_merged += 1

        for param in dit.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

        # ── Broadcast freshly initialised LoRA weights from rank 0 ──
        # Without this, each DDP worker has a different random lora_A,
        # causing the models to diverge during training.
        if is_dist():
            for p in dit.parameters():
                dist.broadcast(p.data, src=0)
            print0("Broadcast re-initialised LoRA parameters from rank 0.")

        trainable = sum(p.numel() for p in dit.parameters() if p.requires_grad)
        print0(f"\nMerged Stage-1 LoRA into base weights ({n_merged} layers).")
        print0(f"Fresh Stage-3 LoRA initialised. Trainable params: {trainable:,}")

    def _set_stage3_lora_enabled(self, enabled: bool):
        from peft.tuners.lora import Linear as LoraLinear
        dit = self._dit_module()
        for module in dit.modules():
            if isinstance(module, LoraLinear):
                module.enable_adapters(enabled)

    def _load_stage1_checkpoint(self, ckpt_path):
        print0(f"\nLoading Phase 1 checkpoint: {ckpt_path}")
        ckpt_path = os.path.normpath(ckpt_path)
        dit = self._dit_module()

        if os.path.isdir(ckpt_path):
            lora_weights_path = os.path.join(ckpt_path, "lora_weights.pth")
            if not os.path.exists(lora_weights_path):
                raise FileNotFoundError(f"lora_weights.pth not found in {ckpt_path}")
            lora_state_dict = torch.load(lora_weights_path, map_location='cpu')
            missing, unexpected = dit.load_state_dict(lora_state_dict, strict=False)
            print0(f"Loaded Phase 1 LoRA: missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state = checkpoint.get('dit_state_dict', checkpoint)
            dit.load_state_dict(state, strict=False)
            print0(f"Loaded Phase 1 LoRA from file")

    # ────────────────────────────────────────
    #  Optimizer, scheduler, data
    # ────────────────────────────────────────

    def _setup_optimizer(self, lr=None):
        if lr is None:
            lr = self.args.learning_rate
        dit = self._dit_module()
        params = filter(lambda p: p.requires_grad, dit.parameters())
        return torch.optim.AdamW(params, lr=lr, weight_decay=self.args.weight_decay)

    def _setup_lr_scheduler(self):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.num_epochs * len(self.train_loader),
            eta_min=self.args.min_lr,
        )

    def _setup_dataloader(self, mode='train'):
        data_list = self.args.train_list if mode == 'train' else self.args.val_list
        is_val = (mode != 'train')

        dataset = MRIDataset(
            data_list_file=data_list,
            spatial_size=tuple(self.args.monai_spatial_size),
            num_frames=self.args.num_frames,
            augment=(not is_val),
            use_monai=self.args.use_monai,
            full_volume=is_val,
        )

        sampler = None
        if self.use_ddp:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=(not is_val),
            )

        loader = DataLoader(
            dataset,
            batch_size=1 if is_val else self.args.batch_size,
            shuffle=(not is_val) if sampler is None else False,
            sampler=sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        return loader, sampler

    # ────────────────────────────────────────
    #  Prompt / task helpers
    # ────────────────────────────────────────

    def _infer_task_type(self, lq_path: str) -> str:
        p = lq_path.lower()
        if '64m' in p:
            return "64mT_to_3T"
        if '3t' in p:
            return "3T_to_7T"
        return self.args.task_type

    def _extract_field_from_path(self, path: str) -> str:
        filename = os.path.basename(path).lower()
        if '1.5' in filename:
            return '1.5T'
        if '64m' in filename:
            return '64mT'
        if '7t' in filename:
            return '7T'
        if '3t' in filename:
            return '3T'
        return 'low-field'

    def _extract_modality_from_path(self, path: str) -> str:
        for part in path.split(os.sep):
            part_lower = part.lower()
            if part_lower in {'t1', 't1w'}:
                return 'T1-weighted'
            if part_lower in {'t2', 't2w'}:
                return 'T2-weighted'
            if part_lower == 'flair':
                return 'FLAIR'
            if part_lower == 'pd':
                return 'Proton Density-weighted'
        return 'MRI'

    def _generate_prompt(self, lq_path):
        input_field = self._extract_field_from_path(lq_path)
        modality = self._extract_modality_from_path(lq_path)
        task_type = self._infer_task_type(lq_path)
        target_field = '3T' if task_type == "64mT_to_3T" else '7T'
        return (
            f"MRI {modality} sequence enhancement from "
            f"{input_field} to {target_field} magnetic field"
        )

    def _bind_registered_atlas(self, batch):
        from diffsynth.agents.brain_atlas import BRAIN_ATLAS
        masks = batch["region_masks"]
        if masks.shape[0] != 1:
            raise ValueError("Use batch_size=1 with one subject per DDP worker")
        atlases = [BRAIN_ATLAS, self.pipe.plan_to_mask.brain_atlas]
        for agent in (self.pipe.anomaly_agent, self.pipe.vlm_agent):
            if agent is not None and hasattr(agent, "brain_atlas"):
                atlases.append(agent.brain_atlas)
        for atlas in atlases:
            atlas.set_region_masks(masks[0])

    def _build_stage2_context(self, lq_path: str) -> Dict[str, str]:
        task_type = self._infer_task_type(lq_path)
        return {
            'modality': self._extract_modality_from_path(lq_path),
            'input_field': self._extract_field_from_path(lq_path),
            'target_field': '3T' if task_type == "64mT_to_3T" else '7T',
        }

    def _update_stage2_agent_context(self, lq_path: str) -> None:
        context = self._build_stage2_context(lq_path)
        for agent in (getattr(self.pipe, 'anomaly_agent', None), getattr(self.pipe, 'vlm_agent', None)):
            if agent is None:
                continue
            agent.config.modality = context['modality']
            agent.config.input_field = context['input_field']
            agent.config.target_field = context['target_field']

    def _get_dataset_name(self, lq_path):
        dataset_names = [
            '3_to_7-2-reg', '3_to_7-1',
            '64m_to_3-3-reg', '64m_to_3-2-reg', '64m_to_3-1',
        ]
        for name in dataset_names:
            if name in lq_path:
                return name
        return 'unknown'

    # ────────────────────────────────────────
    #  Phase 1: train_epoch
    # ────────────────────────────────────────

    def train_epoch(self, epoch):
        self.pipe.dit.train()
        epoch_loss = 0.0

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}",
                    disable=not is_main())

        for batch_idx, batch in enumerate(pbar):
            self._bind_registered_atlas(batch)
            lq = batch['lq'].to(self.device, dtype=torch.bfloat16)
            hq = batch['hq'].to(self.device, dtype=torch.bfloat16)
            B, C, T, H, W = lq.shape

            lq_path = batch['lq_path'][0]
            prompt = self._generate_prompt(lq_path)
            self.pipe.init_cross_kv(prompt=prompt)

            self.optimizer.zero_grad()

            if self.args.use_amp:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    train_out = self.pipe.train_process(
                        LQ_video=lq, HQ_video=hq,
                        num_frames=T, height=H, width=W,
                        is_full_block=False, if_buffer=True,
                        topk_ratio=self.args.topk_ratio,
                        kv_ratio=self.args.kv_ratio,
                        local_range=self.args.local_range,
                        color_fix=False,
                    )
                    loss = F.l1_loss(train_out['noise_pred'], train_out['target'])
            else:
                train_out = self.pipe.train_process(
                    LQ_video=lq, HQ_video=hq,
                    num_frames=T, height=H, width=W,
                    is_full_block=False, if_buffer=True,
                    topk_ratio=self.args.topk_ratio,
                    kv_ratio=self.args.kv_ratio,
                    local_range=self.args.local_range,
                    color_fix=False,
                )
                loss = F.l1_loss(train_out['noise_pred'], train_out['target'])

            loss.backward()
            self._allreduce_lora_grads()
            torch.nn.utils.clip_grad_norm_(
                self._dit_module().parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            self.scheduler.step()
            epoch_loss += loss.item()
            self.global_step += 1

            if is_main():
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })

            # Visualize first batch (rank 0 only)
            if batch_idx == 0 and is_main():
                self._visualize_batch(
                    lq, train_out['pred_hq'], hq, epoch, batch_idx)

        return epoch_loss / len(self.train_loader)

    @staticmethod
    def _weighted_mean(value: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
        """Compute a numerically-stable weighted mean."""
        if weight is None:
            return value.mean()
        if weight.shape != value.shape:
            weight = weight.expand_as(value)
        return (value * weight).sum() / (weight.sum() + eps)

    def _ssim3d_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
        kernel_size: int = 5,
    ) -> torch.Tensor:
        """Differentiable 3D SSIM loss over latent volumes (B, C, T, H, W)."""
        pred = pred.float()
        target = target.float()

        k = max(int(kernel_size), 3)
        if k % 2 == 0:
            k += 1
        pad = k // 2

        mu_pred = F.avg_pool3d(pred, kernel_size=k, stride=1, padding=pad)
        mu_target = F.avg_pool3d(target, kernel_size=k, stride=1, padding=pad)

        pred_sq = pred * pred
        target_sq = target * target
        pred_target = pred * target

        sigma_pred = F.avg_pool3d(pred_sq, kernel_size=k, stride=1, padding=pad) - mu_pred * mu_pred
        sigma_target = F.avg_pool3d(target_sq, kernel_size=k, stride=1, padding=pad) - mu_target * mu_target
        sigma_cross = F.avg_pool3d(pred_target, kernel_size=k, stride=1, padding=pad) - mu_pred * mu_target

        sigma_pred = torch.clamp_min(sigma_pred, 0.0)
        sigma_target = torch.clamp_min(sigma_target, 0.0)

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        numerator = (2.0 * mu_pred * mu_target + c1) * (2.0 * sigma_cross + c2)
        denominator = (mu_pred.pow(2) + mu_target.pow(2) + c1) * (sigma_pred + sigma_target + c2)
        ssim_map = numerator / (denominator + 1e-8)
        loss_map = 1.0 - torch.clamp(ssim_map, -1.0, 1.0)

        if weight is None:
            return loss_map.mean()

        return self._weighted_mean(loss_map, weight.float())

    def _gradient_consistency_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """L1 consistency of finite-difference gradients over (T, H, W)."""
        pred = pred.float()
        target = target.float()
        base_weight = weight.float() if weight is not None else None

        losses = []
        for dim in (2, 3, 4):
            if pred.shape[dim] <= 1:
                continue
            pred_diff = torch.diff(pred, dim=dim)
            target_diff = torch.diff(target, dim=dim)
            grad_err = (pred_diff - target_diff).abs()

            if base_weight is None:
                losses.append(grad_err.mean())
                continue

            if dim == 2:
                w = 0.5 * (base_weight[:, :, 1:, :, :] + base_weight[:, :, :-1, :, :])
            elif dim == 3:
                w = 0.5 * (base_weight[:, :, :, 1:, :] + base_weight[:, :, :, :-1, :])
            else:
                w = 0.5 * (base_weight[:, :, :, :, 1:] + base_weight[:, :, :, :, :-1])
            losses.append(self._weighted_mean(grad_err, w))

        if not losses:
            return pred.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _compute_stage3_loss(
        self,
        refine_out: dict,
        stage1_latents: torch.Tensor,
        hq_latents: torch.Tensor,
    ) -> tuple:
        """Compute total Stage3 loss = weighted L1 + structural terms."""
        noise_pred = refine_out['noise_pred']
        target = refine_out['target']
        mask_w = refine_out['mask_weight']

        base_weight = (1.0 + (float(self.args.mask_loss_weight) - 1.0) * mask_w).float()
        l1_loss = self._weighted_mean((noise_pred - target).abs().float(), base_weight)
        total_loss = l1_loss

        loss_terms = {
            'l1': float(l1_loss.detach().item()),
            'ssim': 0.0,
            'grad': 0.0,
        }

        use_structure = (self.args.stage3_ssim_weight > 0.0) or (self.args.stage3_grad_weight > 0.0)
        if use_structure:
            stage1_latents = stage1_latents.to(device=noise_pred.device, dtype=noise_pred.dtype)
            hq_latents = hq_latents.to(device=noise_pred.device, dtype=noise_pred.dtype)
            pred_latent = stage1_latents + noise_pred

            if self.args.stage3_ssim_weight > 0.0:
                ssim_loss = self._ssim3d_loss(
                    pred=pred_latent,
                    target=hq_latents,
                    weight=base_weight,
                    kernel_size=self.args.stage3_ssim_kernel,
                )
                total_loss = total_loss + float(self.args.stage3_ssim_weight) * ssim_loss
                loss_terms['ssim'] = float(ssim_loss.detach().item())

            if self.args.stage3_grad_weight > 0.0:
                grad_loss = self._gradient_consistency_loss(
                    pred=pred_latent,
                    target=hq_latents,
                    weight=base_weight,
                )
                total_loss = total_loss + float(self.args.stage3_grad_weight) * grad_loss
                loss_terms['grad'] = float(grad_loss.detach().item())

        return total_loss, loss_terms, mask_w

    # ────────────────────────────────────────
    #  Phase 2: train_epoch_stage3
    # ────────────────────────────────────────

    def train_epoch_stage3(self, epoch):
        from concurrent.futures import ThreadPoolExecutor

        self.pipe.dit.train()
        epoch_loss = 0.0
        num_refined = 0
        num_skipped = 0
        stage3_l1_sum = 0.0
        stage3_ssim_sum = 0.0
        stage3_grad_sum = 0.0

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        pbar = tqdm(self.train_loader,
                    desc=f"Phase2 Epoch {epoch+1}/{self.args.num_epochs}",
                    disable=not is_main())

        vlm_executor = ThreadPoolExecutor(max_workers=1)

        for batch_idx, batch in enumerate(pbar):
            self._bind_registered_atlas(batch)
            lq = batch['lq'].to(self.device, dtype=torch.bfloat16)
            hq = batch['hq'].to(self.device, dtype=torch.bfloat16)
            flip_axes = batch.get('flip_axes', [])
            if isinstance(flip_axes, torch.Tensor):
                flip_axes = flip_axes.tolist()
            if isinstance(flip_axes, (list, tuple)):
                flip_axes = [a.item() if isinstance(a, torch.Tensor) else a for a in flip_axes]

            B, C, T, H, W = lq.shape
            lq_path = batch['lq_path'][0]
            task_type = self._infer_task_type(lq_path)

            if self.pipe.anomaly_agent is not None and self.pipe.anomaly_agent.config.task_type != task_type:
                self.pipe.anomaly_agent.config.task_type = task_type
            if self.pipe.vlm_agent is not None and self.pipe.vlm_agent.config.task_type != task_type:
                self.pipe.vlm_agent.config.task_type = task_type
            self._update_stage2_agent_context(lq_path)

            prompt = self._generate_prompt(lq_path)
            skip_stage1_train = bool(
                getattr(self.args, "skip_stage1_train_inference", False)
            )

            if skip_stage1_train:
                # These ablations use LQ-only Stage 2 and Stage-1-style noisy-HQ
                # Stage 3 training, so the per-batch Stage-1 result is unused.
                stage1_image = None
                stage1_latents_cached = None
            else:
                self.pipe.init_cross_kv(prompt=prompt)

                # Stage 1 inference (no grad, LoRA disabled)
                self._set_stage3_lora_enabled(False)
                with torch.no_grad():
                    stage1_out = self.pipe.test_process(
                        LQ_video=lq, HQ_video=None,
                        num_frames=T, height=H, width=W,
                        cfg_scale=1.0, is_full_block=False, if_buffer=True,
                        topk_ratio=self.args.topk_ratio,
                        kv_ratio=self.args.kv_ratio,
                        local_range=self.args.local_range,
                        color_fix=False,
                    )
                    stage1_image = stage1_out['pred_hq'].detach()
                    stage1_latents_cached = stage1_out['pred_clean_latent'].detach()
                self._set_stage3_lora_enabled(True)

            # VLM diagnosis (async). LQ-only mode does not need Stage-1/HQ images.
            stage1_image_cpu = (
                stage1_image.detach().cpu() if stage1_image is not None else None
            )
            lq_cpu = lq.detach().cpu()
            hq_cpu = hq.detach().cpu()

            def _run_stage2(_s1_cpu=stage1_image_cpu, _lq_cpu=lq_cpu,
                            _hq_cpu=hq_cpu, _task=task_type,
                            _skip_stage1=skip_stage1_train, _lq_path=lq_path):
                self._update_stage2_agent_context(_lq_path)
                with torch.no_grad():
                    return self.pipe.run_stage2(
                        lq_image=_lq_cpu,
                        generated_image=None if _skip_stage1 else _s1_cpu,
                        gt_image=None if _skip_stage1 else _hq_cpu,
                        task_type=_task,
                        skip_global_score=True,
                        lq_only=_skip_stage1,
                    )

            vlm_future = vlm_executor.submit(_run_stage2)

            # HQ VAE encode overlap — keep DiT on GPU to avoid 22s reload
            with torch.no_grad():
                self.pipe.load_models_to_device(["dit", "vae"])
                hq_latents_cached = self.pipe.vae.encode(
                    hq.to(self.device, dtype=self.pipe.torch_dtype),
                    device=self.device, tiled=True,
                ).detach()

            stage2_output = vlm_future.result()

            # Determine if this rank needs refinement
            needs_refine = stage2_output.needs_refinement
            conditions = []

            if needs_refine:
                # Convert plans to masks
                image_size = (T, H, W) if T > 1 else (H, W)
                conditions = self.pipe.plan_to_mask.create_refinement_conditions(
                    plans=stage2_output.improvement_plan.plans,
                    image_size=image_size,
                    device=self.device, dtype=self.pipe.torch_dtype,
                )

                # Apply augmentation flips to masks
                if flip_axes:
                    _MONAI_TO_MASK_3D = {0: 1, 1: 2, 2: 0}
                    _MONAI_TO_MASK_2D = {0: 0, 1: 1}
                    for cond in conditions:
                        m = cond.mask
                        for ax in flip_axes:
                            if m.dim() == 3 and ax in _MONAI_TO_MASK_3D:
                                m = m.flip(_MONAI_TO_MASK_3D[ax])
                            elif m.dim() == 2 and ax in _MONAI_TO_MASK_2D:
                                m = m.flip(_MONAI_TO_MASK_2D[ax])
                        cond.mask = m

            local_skip = (not needs_refine) or (not conditions)

            # ── Synchronize skip decision across all DDP ranks ──
            # all_reduce is a collective op: ALL ranks must call it together.
            # If we let each rank skip independently, the non-skipping rank
            # would block on all_reduce while the skipping rank has moved on,
            # causing an NCCL timeout deadlock.
            # Strategy: only truly skip if ALL ranks want to skip.
            if self.use_ddp:
                skip_flag = torch.tensor([1.0 if local_skip else 0.0],
                                         device=self.device)
                dist.all_reduce(skip_flag, op=dist.ReduceOp.MIN)
                all_skip = skip_flag.item() > 0.5
            else:
                all_skip = local_skip

            if all_skip:
                # Every rank agrees to skip → safe to continue without allreduce
                num_skipped += 1
                self.scheduler.step()
                self.global_step += 1
                if is_main():
                    pbar.set_postfix({'loss': 'skip', 'refined': num_refined,
                                      'skipped': num_skipped})
                continue

            # Stage 3 training step (with grad)
            self.optimizer.zero_grad()

            if local_skip:
                # This rank has no conditions, but other rank(s) need to train.
                # Contribute zero gradients so allreduce doesn't deadlock.
                loss_val = 0.0
                mask_ratio = 0.0
                stage3_terms = {'l1': 0.0, 'ssim': 0.0, 'grad': 0.0}
            else:
                if self.args.use_amp:
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        refine_out = self.pipe.train_refine_process(
                            LQ_video=lq, HQ_video=hq,
                            stage1_image=stage1_image,
                            conditions=conditions,
                            topk_ratio=self.args.topk_ratio,
                            kv_ratio=self.args.kv_ratio,
                            local_range=self.args.local_range,
                            stage1_latents_cached=stage1_latents_cached,
                            hq_latents_cached=hq_latents_cached,
                        )
                        loss, stage3_terms, mask_w = self._compute_stage3_loss(
                            refine_out=refine_out,
                            stage1_latents=stage1_latents_cached,
                            hq_latents=hq_latents_cached,
                        )
                else:
                    refine_out = self.pipe.train_refine_process(
                        LQ_video=lq, HQ_video=hq,
                        stage1_image=stage1_image,
                        conditions=conditions,
                        topk_ratio=self.args.topk_ratio,
                        kv_ratio=self.args.kv_ratio,
                        local_range=self.args.local_range,
                        stage1_latents_cached=stage1_latents_cached,
                        hq_latents_cached=hq_latents_cached,
                    )
                    loss, stage3_terms, mask_w = self._compute_stage3_loss(
                        refine_out=refine_out,
                        stage1_latents=stage1_latents_cached,
                        hq_latents=hq_latents_cached,
                    )

                loss.backward()
                loss_val = loss.item()
                mask_ratio = (mask_w > 0).float().mean().item()

            # All ranks must participate in allreduce (skip ranks have zero grads)
            self._allreduce_lora_grads()
            torch.nn.utils.clip_grad_norm_(
                self._lora_params, self.args.max_grad_norm)
            self.optimizer.step()

            if not local_skip:
                epoch_loss += loss_val
                num_refined += 1
                stage3_l1_sum += stage3_terms['l1']
                stage3_ssim_sum += stage3_terms['ssim']
                stage3_grad_sum += stage3_terms['grad']
            else:
                num_skipped += 1

            self.scheduler.step()
            self.global_step += 1

            if is_main():
                postfix = {
                    'loss': f'{loss_val:.4f}' if not local_skip else 'skip(sync)',
                    'mask%': f'{mask_ratio*100:.1f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'refined': num_refined,
                    'skipped': num_skipped,
                }
                if not local_skip:
                    postfix['l1'] = f"{stage3_terms['l1']:.4f}"
                    if self.args.stage3_ssim_weight > 0.0:
                        postfix['ssim'] = f"{stage3_terms['ssim']:.4f}"
                    if self.args.stage3_grad_weight > 0.0:
                        postfix['grad'] = f"{stage3_terms['grad']:.4f}"
                pbar.set_postfix(postfix)

        vlm_executor.shutdown(wait=False)
        effective_batches = max(num_refined, 1)
        avg_loss = epoch_loss / effective_batches
        print0(f"  Refined: {num_refined}, Skipped: {num_skipped}")
        if num_refined > 0:
            print0(
                "  Stage3 terms: "
                f"L1={stage3_l1_sum/num_refined:.4f}, "
                f"SSIM={stage3_ssim_sum/num_refined:.4f} (w={self.args.stage3_ssim_weight:.3f}), "
                f"Grad={stage3_grad_sum/num_refined:.4f} (w={self.args.stage3_grad_weight:.3f})"
            )
        return avg_loss

    # ────────────────────────────────────────
    #  Validation (all ranks participate, rank 0 prints/saves)
    # ────────────────────────────────────────

    @torch.no_grad()
    def validate(self, epoch, save_images=True):
        self._dit_module().eval()

        # Set epoch on val sampler so each rank gets its shard
        if self.val_sampler is not None:
            self.val_sampler.set_epoch(epoch)

        total_psnr = 0.0
        total_ssim = 0.0
        total_nrmse = 0.0
        total_lpips = 0.0
        total_s1_psnr = 0.0
        total_s1_ssim = 0.0
        total_s1_nrmse = 0.0
        total_s1_lpips = 0.0
        total_val_loss = 0.0
        num_val_batches = 0
        num_frames = 0
        num_s1_frames = 0

        stage_stats = {
            'problems_detected': 0,
            'refinements_applied': 0,
            'quality_improvements': [],
        }

        all_datasets = [
            '3_to_7-1', '3_to_7-2-reg',
            '64m_to_3-1', '64m_to_3-2-reg', '64m_to_3-3-reg',
        ]
        dataset_metrics = {name: {'psnr': [], 'ssim': [], 's1_psnr': [], 's1_ssim': []} for name in all_datasets}
        dataset_metrics['unknown'] = {'psnr': [], 'ssim': [], 's1_psnr': [], 's1_ssim': []}

        max_samples_per_dataset = 5
        dataset_vis_samples = {name: [] for name in all_datasets}
        dataset_vis_samples['unknown'] = []

        pbar = tqdm(self.val_loader, desc="Validating", disable=not is_main())

        print(f"[Rank {get_rank()}] Validation: {len(self.val_loader)} batches on {self.device}")

        for batch_idx, batch in enumerate(pbar):
            # DistributedSampler pads 58 samples to (for example) 60 by
            # repeating dataset indices 0 and 1.  Do not run inference or add
            # metrics for those padding-only batches.  The barrier is retained
            # so every rank still executes the same number of collectives.
            if is_validation_padding_batch(self.val_loader, batch_idx):
                print(
                    f"[Rank {get_rank()}] Skipping validation padding batch "
                    f"{batch_idx + 1}/{len(self.val_loader)}"
                )
                if self.use_ddp:
                    barrier()
                continue

            self._bind_registered_atlas(batch)
            lq = batch['lq'].to(self.device, dtype=torch.bfloat16)
            hq = batch['hq'].to(self.device, dtype=torch.bfloat16)
            B, C, T, H, W = lq.shape

            lq_path = batch['lq_path'][0]
            task_type = self._infer_task_type(lq_path)
            dataset_name = self._get_dataset_name(lq_path)

            if self.pipe.anomaly_agent is not None and self.pipe.anomaly_agent.config.task_type != task_type:
                self.pipe.anomaly_agent.config.task_type = task_type
            if self.pipe.vlm_agent is not None and self.pipe.vlm_agent.config.task_type != task_type:
                self.pipe.vlm_agent.config.task_type = task_type
            self._update_stage2_agent_context(lq_path)

            prompt = self._generate_prompt(lq_path)
            self.pipe.init_cross_kv(prompt=prompt)

            if self.phase == 2:
                self._set_stage3_lora_enabled(False)

            try:
                stage1_output, stage1_latents = self.pipe.run_stage1(
                    LQ_video=lq, num_frames=T, height=H, width=W,
                    return_latent=True,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"[Rank {self.rank}] OOM on validation batch {batch_idx} "
                      f"(T={T}), skipping.")
                torch.cuda.empty_cache()
                if self.phase == 2:
                    self._set_stage3_lora_enabled(True)
                if self.use_ddp:
                    barrier()
                continue

            if self.phase == 2:
                self._set_stage3_lora_enabled(True)

            hq_latents_cached = None
            if self.phase == 2:
                with torch.no_grad():
                    self.pipe.load_models_to_device(["dit", "vae"])
                    hq_latents_cached = self.pipe.vae.encode(
                        hq.to(self.device, dtype=self.pipe.torch_dtype),
                        device=self.device, tiled=True,
                    ).detach()

            stage2_output = None
            stage3_output = None
            final_image = stage1_output

            if self.pipe.enable_stage2:
                self._update_stage2_agent_context(lq_path)
                stage2_output = self.pipe.run_stage2(
                    lq_image=lq, generated_image=stage1_output,
                    gt_image=hq, task_type=task_type)

                if (self.pipe.enable_stage3 and stage2_output is not None
                        and stage2_output.needs_refinement):
                    try:
                        stage3_output = self.pipe.run_stage3(
                            lq_image=lq, generated_image=stage1_output,
                            improvement_plan=stage2_output.improvement_plan,
                            generated_latents=stage1_latents,
                            gt_image=hq,
                            max_iterations=self.args.max_refinement_iterations,
                            task_type=task_type)
                        final_image = stage3_output.refined_image
                    except torch.cuda.OutOfMemoryError:
                        print(f"[Rank {self.rank}] OOM on Stage3 validation "
                              f"batch {batch_idx} (T={T}), using Stage1 result.")
                        torch.cuda.empty_cache()

            result = ThreeStageOutput(
                final_image=final_image, stage1_image=stage1_output,
                stage2_output=stage2_output, stage3_output=stage3_output,
                total_time=0.0)

            pred = result.final_image
            min_frames = min(pred.shape[2], hq.shape[2])
            s1_min_frames = min(result.stage1_image.shape[2], hq.shape[2])

            # Stage-1 metrics
            s1_metrics = self._compute_metrics(
                result.stage1_image[:, :, :s1_min_frames], hq[:, :, :s1_min_frames])
            total_s1_psnr += s1_metrics['sum_psnr']
            total_s1_ssim += s1_metrics['sum_ssim']
            total_s1_nrmse += s1_metrics['sum_nrmse']
            total_s1_lpips += s1_metrics['sum_lpips']
            num_s1_frames += s1_metrics['count']

            # Latent-space validation loss (consistent with training loss)
            with torch.no_grad():
                if (self.phase == 2
                        and stage2_output is not None
                        and stage2_output.needs_refinement
                        and stage2_output.improvement_plan is not None
                        and stage2_output.improvement_plan.plans):
                    # Phase 2 (Stage 3): mask-weighted L1 on train_refine_process
                    image_size = (T, H, W) if T > 1 else (H, W)
                    val_conditions = self.pipe.plan_to_mask.create_refinement_conditions(
                        plans=stage2_output.improvement_plan.plans,
                        image_size=image_size,
                        device=self.device, dtype=self.pipe.torch_dtype,
                    )
                    val_out = self.pipe.train_refine_process(
                        LQ_video=lq, HQ_video=hq,
                        stage1_image=stage1_output,
                        conditions=val_conditions,
                        topk_ratio=self.args.topk_ratio,
                        kv_ratio=self.args.kv_ratio,
                        local_range=self.args.local_range,
                        stage1_latents_cached=stage1_latents,
                        hq_latents_cached=hq_latents_cached,
                    )
                    mask_w = val_out['mask_weight']
                    mask_weight_scale = self.args.mask_loss_weight
                    weight = 1.0 + (mask_weight_scale - 1.0) * mask_w
                    val_loss = (weight * (val_out['noise_pred'] - val_out['target']).abs()).mean().item()
                else:
                    # Phase 1 (or no refinement needed): plain L1 on train_process
                    val_out = self.pipe.train_process(
                        LQ_video=lq, HQ_video=hq,
                        num_frames=T, height=H, width=W,
                        cfg_scale=1.0,
                        is_full_block=False, if_buffer=True,
                        topk_ratio=self.args.topk_ratio,
                        kv_ratio=self.args.kv_ratio,
                        local_range=self.args.local_range,
                        color_fix=False,
                    )
                    val_loss = F.l1_loss(val_out['noise_pred'], val_out['target']).item()
            total_val_loss += val_loss
            num_val_batches += 1

            # Final (stage-3 if refined, else stage-1) metrics
            final_metrics = self._compute_metrics(
                pred[:, :, :min_frames], hq[:, :, :min_frames])
            psnr = final_metrics['psnr']
            ssim = final_metrics['ssim']
            nrmse = final_metrics['nrmse']
            lpips_val = final_metrics['lpips']
            total_psnr += final_metrics['sum_psnr']
            total_ssim += final_metrics['sum_ssim']
            total_nrmse += final_metrics['sum_nrmse']
            total_lpips += final_metrics['sum_lpips']
            num_frames += final_metrics['count']

            if dataset_name in dataset_metrics:
                dataset_metrics[dataset_name]['psnr'].append(psnr)
                dataset_metrics[dataset_name]['ssim'].append(ssim)
                dataset_metrics[dataset_name]['s1_psnr'].append(s1_metrics['psnr'])
                dataset_metrics[dataset_name]['s1_ssim'].append(s1_metrics['ssim'])

            if result.stage2_output is not None:
                stage_stats['problems_detected'] += len(
                    result.stage2_output.anomaly_report.problems)
            if result.stage3_output is not None:
                stage_stats['refinements_applied'] += result.stage3_output.iterations_used
                s1_q = self.pipe._calculate_quality(result.stage1_image, hq)
                final_q = self.pipe._calculate_quality(result.final_image, hq)
                stage_stats['quality_improvements'].append(final_q - s1_q)

            pbar.set_postfix({
                'loss': f'{val_loss:.4f}',
                'psnr': f'{psnr:.2f}',
                'ssim': f'{ssim:.4f}',
                'nrmse': f'{nrmse:.4f}',
                'lpips': f'{lpips_val:.4f}',
            })

            # Only rank 0 collects vis samples
            if (save_images and is_main() and dataset_name in dataset_vis_samples
                    and len(dataset_vis_samples[dataset_name]) < max_samples_per_dataset):
                dataset_vis_samples[dataset_name].append({
                    'lq': lq.detach().cpu(), 'stage1': result.stage1_image.detach().cpu(),
                    'final': result.final_image.detach().cpu(), 'hq': hq.detach().cpu(),
                    'lq_path': lq_path, 'psnr': psnr, 'ssim': ssim,
                })

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # Per-batch barrier: each rank runs full stage2/3 on different data,
            # which can take very different amounts of time. Without a barrier the
            # drift accumulates over ALL batches and exceeds NCCL timeout at the
            # final allreduce.  With a per-batch barrier the max drift is bounded
            # to a single batch (~1-5 min), well within the 30-min timeout.
            if self.use_ddp:
                barrier()

        # ── Per-rank diagnostics (before all-reduce) ──
        if num_frames > 0:
            local_avg_psnr = total_psnr / num_frames
            local_avg_ssim = total_ssim / num_frames
            local_avg_nrmse = total_nrmse / num_frames
            local_avg_lpips = total_lpips / num_frames
            local_avg_s1_psnr = total_s1_psnr / max(num_s1_frames, 1)
            local_avg_s1_ssim = total_s1_ssim / max(num_s1_frames, 1)
            local_avg_s1_nrmse = total_s1_nrmse / max(num_s1_frames, 1)
            local_avg_s1_lpips = total_s1_lpips / max(num_s1_frames, 1)
            print(f"[Rank {get_rank()}] Local metrics (frames={num_frames}): "
                  f"S1 PSNR={local_avg_s1_psnr:.2f}, S1 SSIM={local_avg_s1_ssim:.4f}, "
                  f"S1 NRMSE={local_avg_s1_nrmse:.4f}, S1 LPIPS={local_avg_s1_lpips:.4f}, "
                  f"Final PSNR={local_avg_psnr:.2f}, Final SSIM={local_avg_ssim:.4f}, "
                  f"Final NRMSE={local_avg_nrmse:.4f}, Final LPIPS={local_avg_lpips:.4f}")
        else:
            print(f"[Rank {get_rank()}] No samples processed!")

        # ── All-reduce metrics across ranks ──
        if self.use_ddp:
            stats = torch.tensor([
                total_psnr, total_ssim, total_nrmse, total_lpips, float(num_frames),
                total_s1_psnr, total_s1_ssim, total_s1_nrmse, total_s1_lpips, float(num_s1_frames),
                total_val_loss, float(num_val_batches)
            ],
                                device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_psnr = stats[0].item()
            total_ssim = stats[1].item()
            total_nrmse = stats[2].item()
            total_lpips = stats[3].item()
            num_frames = int(stats[4].item())
            total_s1_psnr = stats[5].item()
            total_s1_ssim = stats[6].item()
            total_s1_nrmse = stats[7].item()
            total_s1_lpips = stats[8].item()
            num_s1_frames = int(stats[9].item())
            total_val_loss = stats[10].item()
            num_val_batches = int(stats[11].item())

            # Per-rank per-dataset diagnostics
            for ds_name in list(dataset_metrics.keys()):
                local_psnr = dataset_metrics[ds_name]['psnr']
                local_ssim = dataset_metrics[ds_name]['ssim']
                if local_psnr:
                    print(f"[Rank {get_rank()}] {ds_name}: "
                          f"PSNR={np.mean(local_psnr):.2f}, "
                          f"SSIM={np.mean(local_ssim):.4f} (n={len(local_psnr)})")

            # All-reduce per-dataset metrics
            for ds_name in list(dataset_metrics.keys()):
                local_psnr = dataset_metrics[ds_name]['psnr']
                local_ssim = dataset_metrics[ds_name]['ssim']
                local_s1_psnr = dataset_metrics[ds_name]['s1_psnr']
                local_s1_ssim = dataset_metrics[ds_name]['s1_ssim']
                ds_stats = torch.tensor(
                    [sum(local_psnr), sum(local_ssim), float(len(local_psnr)),
                     sum(local_s1_psnr), sum(local_s1_ssim)],
                    device=self.device)
                dist.all_reduce(ds_stats, op=dist.ReduceOp.SUM)
                n = int(ds_stats[2].item())
                if n > 0:
                    dataset_metrics[ds_name]['psnr'] = [ds_stats[0].item() / n] * n
                    dataset_metrics[ds_name]['ssim'] = [ds_stats[1].item() / n] * n
                    dataset_metrics[ds_name]['s1_psnr'] = [ds_stats[3].item() / n] * n
                    dataset_metrics[ds_name]['s1_ssim'] = [ds_stats[4].item() / n] * n
                else:
                    dataset_metrics[ds_name]['psnr'] = []
                    dataset_metrics[ds_name]['ssim'] = []
                    dataset_metrics[ds_name]['s1_psnr'] = []
                    dataset_metrics[ds_name]['s1_ssim'] = []

        if save_images and is_main():
            self._save_validation_images_by_dataset(dataset_vis_samples, epoch)

        avg_psnr = total_psnr / max(num_frames, 1)
        avg_ssim = total_ssim / max(num_frames, 1)
        avg_nrmse = total_nrmse / max(num_frames, 1)
        avg_lpips = total_lpips / max(num_frames, 1)
        avg_s1_psnr = total_s1_psnr / max(num_s1_frames, 1)
        avg_s1_ssim = total_s1_ssim / max(num_s1_frames, 1)
        avg_s1_nrmse = total_s1_nrmse / max(num_s1_frames, 1)
        avg_s1_lpips = total_s1_lpips / max(num_s1_frames, 1)

        dataset_display_names = {
            '3_to_7-1': '3T→7T (set1)',
            '3_to_7-2-reg': '3T→7T (set2-reg)',
            '64m_to_3-1': '64mT→3T (set1)',
            '64m_to_3-2-reg': '64mT→3T (set2-reg)',
            '64m_to_3-3-reg': '64mT→3T (set3-reg)',
        }

        dataset_avg_metrics = {}
        if is_main():
            print("\n" + "="*70)
            print("Validation Results by Dataset (Stage-1 → Final):")
            print("="*70)
        for ds_name, metrics in dataset_metrics.items():
            if metrics['psnr']:
                avg_ds_psnr = np.mean(metrics['psnr'])
                avg_ds_ssim = np.mean(metrics['ssim'])
                avg_ds_s1_psnr = np.mean(metrics['s1_psnr'])
                avg_ds_s1_ssim = np.mean(metrics['s1_ssim'])
                n = len(metrics['psnr'])
                dataset_avg_metrics[ds_name] = {
                    'psnr': avg_ds_psnr, 'ssim': avg_ds_ssim,
                    's1_psnr': avg_ds_s1_psnr, 's1_ssim': avg_ds_s1_ssim,
                    'count': n}
                if is_main():
                    display = dataset_display_names.get(ds_name, ds_name)
                    print(f"  {display:20s}: S1 PSNR={avg_ds_s1_psnr:.2f} → Final PSNR={avg_ds_psnr:.2f} dB, "
                          f"S1 SSIM={avg_ds_s1_ssim:.4f} → Final SSIM={avg_ds_ssim:.4f} (n={n})")
        avg_val_loss = total_val_loss / max(num_val_batches, 1)

        if is_main():
            print("-"*70)
            print("3-Stage Statistics:")
            print(f"  Total problems detected: {stage_stats['problems_detected']}")
            print(f"  Total refinements applied: {stage_stats['refinements_applied']}")
            if stage_stats['quality_improvements']:
                print(f"  Avg quality improvement: {np.mean(stage_stats['quality_improvements']):.4f}")
            val_loss_label = "mask-wtd L1 (Stage3)" if self.phase == 2 else "latent L1 (Stage1)"
            print(f"  Val Loss ({val_loss_label}): {avg_val_loss:.6f}")
            print(f"  Overall Stage-1: PSNR={avg_s1_psnr:.2f} dB, SSIM={avg_s1_ssim:.4f}, "
                  f"NRMSE={avg_s1_nrmse:.4f}, LPIPS={avg_s1_lpips:.4f}")
            print(f"  Overall Final:   PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}, "
                  f"NRMSE={avg_nrmse:.4f}, LPIPS={avg_lpips:.4f} (frames={num_frames})")
            print("="*70 + "\n")

        return avg_psnr, avg_ssim, avg_nrmse, avg_lpips, dataset_avg_metrics

    # ────────────────────────────────────────
    #  Metrics & visualization
    # ────────────────────────────────────────

    def _get_lpips_metric(self):
        if self.lpips_metric is None:
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type='alex', normalize=False
            ).to(self.device)
            self.lpips_metric.eval()
            for p in self.lpips_metric.parameters():
                p.requires_grad_(False)
        return self.lpips_metric

    def _compute_metrics(self, pred, gt):
        """Compute 3D volumetric metrics (PSNR/SSIM/NRMSE) on the full (T,H,W)
        volume per batch item.  LPIPS is inherently 2D, so it is averaged over
        slices and reported as the mean across the volume."""
        pred_np = (pred.float().cpu().numpy() + 1.0) / 2.0
        gt_np = (gt.float().cpu().numpy() + 1.0) / 2.0
        pred_np = np.clip(pred_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        psnr_vals, ssim_vals, nrmse_vals, lpips_vals = [], [], [], []
        lpips_metric = self._get_lpips_metric()
        B, C, T, H, W = pred_np.shape
        for b in range(B):
            # Average RGB channels → luminance volume (T, H, W)
            pred_vol = pred_np[b].mean(axis=0)
            gt_vol   = gt_np[b].mean(axis=0)

            # Skip near-blank volumes (e.g. all-padding batches)
            if gt_vol.max() - gt_vol.min() < 1e-6:
                continue

            # 3-D foreground mask: voxels above 5% of GT dynamic range
            fg_thresh = gt_vol.min() + 0.05 * (gt_vol.max() - gt_vol.min())
            fg_mask = gt_vol > fg_thresh
            if fg_mask.sum() < 64:
                continue

            # ── 3D PSNR ──────────────────────────────────────────────────
            psnr_vals.append(psnr_func(gt_vol, pred_vol, data_range=1.0))

            # ── 3D SSIM ──────────────────────────────────────────────────
            # win_size must be odd and fit inside every spatial dimension.
            # gaussian_weights=True ignores win_size and uses a fixed
            # truncate=3.5/sigma=1.5 kernel (~11 voxels), which may exceed T
            # for short clips, so we use an explicit odd win_size instead.
            win = min(T, H, W, 7)
            if win % 2 == 0:
                win -= 1
            win = max(win, 3)
            ssim_vals.append(ssim_func(
                gt_vol, pred_vol, data_range=1.0,
                win_size=win, use_sample_covariance=False))

            # ── 3D NRMSE (foreground voxels) ─────────────────────────────
            diff = pred_vol[fg_mask] - gt_vol[fg_mask]
            rmse = np.sqrt(np.mean(diff ** 2))
            norm = np.sqrt(np.mean(gt_vol[fg_mask] ** 2)) + 1e-8
            nrmse_vals.append(float(rmse / norm))

            # ── LPIPS (averaged over slices, inherently 2D) ──────────────
            slice_lpips = []
            for f in range(T):
                pred_luma = pred[b, :, f].float().mean(dim=0, keepdim=True)
                gt_luma   = gt[b, :, f].float().mean(dim=0, keepdim=True)
                pred_luma = pred_luma.clamp(-1.0, 1.0).repeat(3, 1, 1).unsqueeze(0)
                gt_luma   = gt_luma.clamp(-1.0, 1.0).repeat(3, 1, 1).unsqueeze(0)
                slice_lpips.append(lpips_metric(pred_luma, gt_luma).item())
            lpips_vals.append(float(np.mean(slice_lpips)))

        if not psnr_vals:
            return {
                'psnr': 0.0,
                'ssim': 0.0,
                'nrmse': 0.0,
                'lpips': 0.0,
                'count': 0,
                'sum_psnr': 0.0,
                'sum_ssim': 0.0,
                'sum_nrmse': 0.0,
                'sum_lpips': 0.0,
            }
        return {
            'psnr': float(np.mean(psnr_vals)),
            'ssim': float(np.mean(ssim_vals)),
            'nrmse': float(np.mean(nrmse_vals)),
            'lpips': float(np.mean(lpips_vals)),
            'count': len(psnr_vals),
            'sum_psnr': float(np.sum(psnr_vals)),
            'sum_ssim': float(np.sum(ssim_vals)),
            'sum_nrmse': float(np.sum(nrmse_vals)),
            'sum_lpips': float(np.sum(lpips_vals)),
        }

    def _visualize_batch(self, lq, pred, hq, epoch, batch_idx):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        mid_frame = lq.shape[2] // 2
        for ax, (img, title) in zip(axes, [
            (lq[0, :, mid_frame], 'LQ Input'),
            (pred[0, :, mid_frame], 'Predicted'),
            (hq[0, :, mid_frame], 'Ground Truth'),
        ]):
            img_np = img.detach().cpu().float()
            img_np = (img_np + 1.0) / 2.0
            img_np = img_np.clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        save_path = os.path.join(
            self.output_dir, 'visualizations',
            f'epoch_{epoch+1}_batch_{batch_idx}.png')
        plt.savefig(save_path)
        plt.close()

    def _save_validation_images_by_dataset(self, dataset_vis_samples, epoch):
        import matplotlib.pyplot as plt
        epoch_dir = os.path.join(self.output_dir, 'validation_images', f'epoch_{epoch+1:04d}')
        os.makedirs(epoch_dir, exist_ok=True)

        dataset_display_names = {
            '3_to_7-1': '3T→7T (set1)',
            '3_to_7-2-reg': '3T→7T (set2-reg)',
            '64m_to_3-1': '64mT→3T (set1)',
            '64m_to_3-2-reg': '64mT→3T (set2-reg)',
            '64m_to_3-3-reg': '64mT→3T (set3-reg)',
        }

        total_saved = 0
        for dataset_name, samples in dataset_vis_samples.items():
            if not samples:
                continue
            dataset_dir = os.path.join(epoch_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            for idx, sample in enumerate(samples):
                lq = sample['lq']
                stage1 = sample['stage1']
                final = sample['final']
                hq = sample['hq']
                lq_path = sample['lq_path']
                psnr = sample['psnr']
                ssim_val = sample['ssim']

                path_parts = lq_path.split('/')
                patient_id = 'unknown'
                modality = 'unknown'
                for part in path_parts:
                    if 'patient' in part.lower() or 'pocemr' in part.lower() or 'sub-' in part.lower():
                        patient_id = part
                    if part in ['T1', 'T1w', 'T2', 'T2w', 'FLAIR', 'PD']:
                        modality = part

                mid = min(lq.shape[2] // 2, stage1.shape[2] // 2,
                          final.shape[2] // 2, hq.shape[2] // 2)

                def _to_np(t, frame_idx):
                    f = t[0, :, frame_idx].float()
                    f = ((f + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).numpy()
                    return f

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(_to_np(lq, mid)); axes[0].set_title('LQ (Input)', fontsize=12); axes[0].axis('off')
                axes[1].imshow(_to_np(stage1, mid)); axes[1].set_title('Stage 1 Output', fontsize=12); axes[1].axis('off')
                axes[2].imshow(_to_np(final, mid)); axes[2].set_title('Final (After Refinement)', fontsize=12); axes[2].axis('off')
                axes[3].imshow(_to_np(hq, mid)); axes[3].set_title('Ground Truth HQ', fontsize=12); axes[3].axis('off')

                display_name = dataset_display_names.get(dataset_name, dataset_name)
                plt.suptitle(f'{display_name} | {patient_id} | {modality} | '
                             f'PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f}',
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                save_path = os.path.join(dataset_dir, f'{idx+1:02d}_{patient_id}_{modality}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                total_saved += 1

        print(f"Saved {total_saved} validation images to {epoch_dir}")

    # ────────────────────────────────────────
    #  Checkpoint save / load
    # ────────────────────────────────────────

    def _extract_stage3_lora_state(self):
        dit = self._dit_module()
        state = {}
        for name, param in dit.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                state[name] = param.data.clone()
            elif 'Mask_proj_in' in name:
                state[name] = param.data.clone()
        return state

    def save_checkpoint(self, epoch, val_psnr, val_ssim, is_best=False):
        if self.phase == 2:
            dit_state = self._extract_stage3_lora_state()
        else:
            dit_state = self._dit_module().state_dict()

        checkpoint = {
            'epoch': epoch,
            'phase': self.phase,
            'global_step': self.global_step,
            'dit_state_dict': dit_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args),
        }

        latest_path = os.path.join(self.output_dir, 'checkpoints', 'latest.pth')
        torch.save(checkpoint, latest_path)

        epoch_path = os.path.join(self.output_dir, 'checkpoints', f'epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_path)
        print(f"Checkpoint saved: {epoch_path}")

        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoints', 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

    def _load_checkpoint(self, checkpoint_path):
        print0(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        saved_phase = checkpoint.get('phase', 1)
        dit_state = checkpoint['dit_state_dict']
        dit = self._dit_module()

        if saved_phase == 2:
            current_state = dit.state_dict()
            loaded_keys = []
            for k, v in dit_state.items():
                if k in current_state:
                    current_state[k] = v
                    loaded_keys.append(k)
            dit.load_state_dict(current_state)
            print0(f"  Loaded {len(loaded_keys)} Stage-3 LoRA keys")
        else:
            dit.load_state_dict(dit_state)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Intentionally keep a fresh scheduler on resume so LR follows
        # the current run's schedule instead of checkpoint's schedule state.
        print0("  Skip loading scheduler_state_dict (use fresh LR schedule)")

        # Optimizer state includes per-param-group lr from checkpoint.
        # Reset it to current config so the very first resumed step uses
        # the intended LR for this run.
        resume_lr = self.args.stage3_lr if self.phase == 2 else self.args.learning_rate
        for group in self.optimizer.param_groups:
            group['lr'] = float(resume_lr)
            if 'initial_lr' in group:
                group['initial_lr'] = float(resume_lr)
        print0(f"  Reset optimizer LR to {resume_lr:.2e}")

        # Rebuild scheduler after optimizer restore+lr reset so base_lrs are clean.
        self.scheduler = self._setup_lr_scheduler()
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_psnr = checkpoint.get('val_psnr', 0.0)
        print0(f"Resumed from epoch {self.start_epoch} (phase {saved_phase})")

    # ────────────────────────────────────────
    #  Main training loop
    # ────────────────────────────────────────

    def train(self):
        print0("\n" + "="*60)
        if self.phase == 1:
            print0("Phase 1: Training Stage 1 LoRA (Base SFT)")
        else:
            print0("Phase 2: Training Stage 3 Refinement LoRA")
            print0(f"  Stage 1 checkpoint: {self.args.stage1_ckpt}")
            print0(f"  Mask loss weight: {self.args.mask_loss_weight}")
        print0(f"Task: {self.args.task_type}")
        print0(f"Epochs: {self.args.num_epochs}")
        print0(f"World size: {self.world_size}")
        print0("="*60 + "\n")

        if not hasattr(self, 'best_psnr'):
            self.best_psnr = 0.0

        if self.args.validate_before_train:
            print0("Running initial validation before training...")
            init_epoch = max(self.start_epoch - 1, 0)
            init_psnr, init_ssim, init_nrmse, init_lpips, _ = self.validate(
                epoch=init_epoch,
                save_images=self.args.validate_before_train_save_images,
            )
            print0(
                f"Initial validation: PSNR={init_psnr:.2f} dB, SSIM={init_ssim:.4f}, "
                f"NRMSE={init_nrmse:.4f}, LPIPS={init_lpips:.4f}"
            )
            barrier()

        for epoch in range(self.start_epoch, self.args.num_epochs):
            # Train
            if self.phase == 1:
                train_loss = self.train_epoch(epoch)
            else:
                train_loss = self.train_epoch_stage3(epoch)

            # Aggregate loss across ranks
            if self.use_ddp:
                loss_tensor = torch.tensor([train_loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = loss_tensor.item()

            current_lr = self.scheduler.get_last_lr()[0]
            print0(f"Epoch {epoch+1} finished. Avg loss: {train_loss:.4f}, LR: {current_lr:.2e}")

            # Validate periodically
            do_val = (epoch + 1) % self.args.val_every == 0 or epoch == self.args.num_epochs - 1
            val_psnr, val_ssim, val_nrmse, val_lpips, dataset_metrics = 0.0, 0.0, 0.0, 0.0, {}
            if do_val:
                print0("Running validation...")
                val_psnr, val_ssim, val_nrmse, val_lpips, dataset_metrics = self.validate(epoch, save_images=True)
                print0(f"Validation PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}, "
                       f"NRMSE: {val_nrmse:.4f}, LPIPS: {val_lpips:.4f}")

            # Save checkpoint & log (rank 0 only)
            do_save = (epoch + 1) % self.args.save_every == 0 or epoch == self.args.num_epochs - 1
            if is_main():
                if do_val:
                    is_best = val_psnr > self.best_psnr
                    if is_best:
                        self.best_psnr = val_psnr
                        print0(f"New best PSNR: {val_psnr:.2f} dB")
                else:
                    is_best = False

                if do_save:
                    self.save_checkpoint(epoch, val_psnr, val_ssim, is_best)

                log_file = os.path.join(self.output_dir, "training_log.txt")
                with open(log_file, 'a') as f:
                    f.write(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, "
                            f"PSNR={val_psnr:.4f} dB, SSIM={val_ssim:.6f}, "
                            f"NRMSE={val_nrmse:.6f}, LPIPS={val_lpips:.6f}, "
                            f"LR={current_lr:.2e}, best_psnr={self.best_psnr:.4f}\n")
                    if dataset_metrics:
                        f.write(f"  [Datasets]\n")
                        for ds_name, metrics in dataset_metrics.items():
                            f.write(f"    {ds_name}: PSNR={metrics['psnr']:.4f} dB, "
                                    f"SSIM={metrics['ssim']:.6f}, n={metrics['count']}\n")

            # Sync all ranks before next epoch
            barrier()

        print0("\nTraining complete!")
        print0(f"Best PSNR: {self.best_psnr:.2f} dB")


def main():
    # Parse args first (includes config file) so we get the real --gpu value
    args = get_args()

    # Keep the requested physical layout available for subprocess mapping.
    # CUDA_VISIBLE_DEVICES was already set before importing torch.
    os.environ["FLASHVSR_PHYSICAL_GPUS"] = args.gpu

    layout_required_index = validate_gpu_layout(args)

    # Wait/retry if GPU is temporarily unavailable.
    local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))
    wait_for_gpu_with_retry(
        required_index=max(local_rank_env, layout_required_index),
        max_retries=args.gpu_retry_max_retries,
        wait_seconds=args.gpu_retry_wait_seconds,
    )

    # Setup DDP (returns False if single-GPU)
    use_ddp = setup_distributed()

    if not use_ddp:
        print(f"[Single-GPU] CUDA_VISIBLE_DEVICES = {args.gpu}")

    if is_main():
        gpu_ids = split_gpu_ids(args.gpu)
        world = get_world_size()
        local_world = get_local_world_size()
        print(f"\n{'='*60}")
        print(f"GPU Allocation Plan:")
        print(f"  Physical GPUs: {args.gpu}")
        ddp_phys = ",".join(gpu_ids[:local_world]) if len(gpu_ids) >= local_world else "N/A"
        print(f"  DDP workers: {world} (local cuda:0..{local_world-1} -> physical {ddp_phys})")
        vlm_device = resolve_vlm_device(args, local_world)
        vlm_index = cuda_device_index(vlm_device)
        if vlm_index is not None and vlm_index < len(gpu_ids):
            print(f"  VLM server: {vlm_device} (physical GPU {gpu_ids[vlm_index]})")
        elif use_ddp:
            print("  VLM server: shared with cuda:0 (no dedicated VLM GPU configured)")
        print(f"{'='*60}\n")

    trainer = None
    try:
        trainer = ThreeStageTrainerDDP(args)
        trainer.train()
    finally:
        if trainer is not None and hasattr(trainer, "pipe") and hasattr(trainer.pipe, "shutdown_vllm"):
            trainer.pipe.shutdown_vllm()
        cleanup_distributed()


if __name__ == "__main__":
    main()
