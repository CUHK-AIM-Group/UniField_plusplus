"""
FlashVSR 3-Stage MRI Translation Pipeline
==========================================

Implements the 3-stage MRI enhancement framework:
- Stage 1: Base SFT (Base Generator) - Initial low-to-high field translation
- Stage 2: Diagnosis & Planning - Agent-driven problem detection and plan generation
- Stage 3: Local Refinement - Text-guided iterative local enhancement

Based on the 4-stage framework but without Stage 4 (Self-Evolving).
"""

import os
import time
from typing import Optional, Tuple, List, Dict, Any, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass

from ..models import ModelManager
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import WanVideoVAE
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters.wan_prompter import WanPrompter
from .base import BasePipeline
from .flashvsr_tiny import FlashVSRTinyPipeline, TorchColorCorrectorWavelet

# Import agent components
from ..agents import (
    AgentConfig,
    AnomalyDetectionAgent,
    AnomalyReport,
    MedicalVLMAgent,
    ImprovementPlan,
)
from ..agents.base_agent import AgentOrchestrator, Problem, Plan
from ..modules.plan_to_mask import (
    PlanToMaskConverter,
    RefinementCondition,
    IterativeRefinementController,
    LocalRefinementModule,
)


@dataclass
class Stage2Output:
    """Output from Stage 2: Diagnosis & Planning"""
    anomaly_report: AnomalyReport
    improvement_plan: ImprovementPlan
    needs_refinement: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_report": self.anomaly_report.to_dict(),
            "improvement_plan": self.improvement_plan.to_dict(),
            "needs_refinement": self.needs_refinement,
        }


@dataclass 
class Stage3Output:
    """Output from Stage 3: Local Refinement"""
    refined_image: torch.Tensor
    iterations_used: int
    final_quality: float
    refinement_summary: Dict[str, Any]


@dataclass
class ThreeStageOutput:
    """Complete output from 3-stage pipeline"""
    final_image: torch.Tensor
    stage1_image: torch.Tensor
    stage2_output: Optional[Stage2Output]
    stage3_output: Optional[Stage3Output]
    total_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage2": self.stage2_output.to_dict() if self.stage2_output else None,
            "stage3": self.stage3_output.refinement_summary if self.stage3_output else None,
            "total_time": self.total_time,
        }


class FlashVSR3StagePipeline(FlashVSRTinyPipeline):
    """
    3-Stage MRI Translation Pipeline.
    
    Extends the base FlashVSRTinyPipeline with:
    - Stage 2: Agent-driven diagnosis and planning
    - Stage 3: Local refinement with text-guided cross-attention
    
    Usage:
        pipe = FlashVSR3StagePipeline.from_model_manager(model_manager)
        pipe.init_cross_kv(prompt="MRI enhancement")
        
        # 3-stage inference
        result = pipe.run_3stage(
            LQ_video=lq_mri,
            HQ_video=hq_mri,  # Optional, for training
            task_type="64mT_to_3T",
            max_refinement_iterations=1,
        )
    """
    
    def __init__(self, device="cuda", torch_dtype=torch.float16, vlm_device: str = None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        
        # Stage 2 agents (initialized lazily)
        self.anomaly_agent = None
        self.vlm_agent = None
        self.agent_orchestrator = None
        
        # Stage 3 components
        self.plan_to_mask = PlanToMaskConverter()
        self.refinement_controller = IterativeRefinementController()
        self.local_refinement = LocalRefinementModule()
        
        # Configuration
        self.vlm_model_name = "Qwen/Qwen3.5-9B"
        self.enable_stage2 = True
        self.enable_stage3 = True
        self.stage2_lq_only = False
        self.stage3_step_scale = 0.65
        self.stage3_substeps = 3
        self.stage3_timestep_max = float(getattr(self.scheduler, "num_train_timesteps", 1000))
        self.stage3_tau_small_prob = 0.8
        
        # Multi-GPU configuration
        # vlm_device: VLM 所在的设备，默认与生成网络相同
        # 推荐双卡配置: device="cuda:0" (生成网络), vlm_device="cuda:1" (VLM)
        self.vlm_device = vlm_device if vlm_device is not None else device
        
        print(r"""
 ____  ____  _____  _    ____ _____   __  __ ____  ___
|___ \/ ___||_   _|/ \  / ___| ____|  |  \/  |  _ \|_ _|
  __) \___ \  | | / _ \| |  _|  _|    | |\/| | |_) || |
 |__ < ___) | | |/ ___ \ |_| | |___   | |  | |  _ < | |
|____/|____/  |_/_/   \_\____|_____|  |_|  |_|_| \_\___|
                  ⚡ 3-Stage MRI Enhancement Pipeline
""")

    def _bridge_tau_to_model_timestep(self, tau: torch.Tensor) -> torch.Tensor:
        """Map bridge time ``tau`` in [0,1] to Stage-1 compatible DiT timestep.

        Bridge semantics:
        - tau=0: stage1 latent (start of refinement, usually needs larger edits)
        - tau=1: GT latent (end of refinement)

        Stage-1 pretraining semantics use larger timestep for harder/noisier states,
        so Stage-3 uses reverse mapping for consistency:
            t_model = (1 - tau) * t_max

        where ``t_max`` is configurable via ``self.stage3_timestep_max``.
        """
        tau = torch.clamp(tau.float(), 0.0, 1.0)
        return (1.0 - tau) * float(self.stage3_timestep_max)

    @staticmethod
    def from_model_manager(
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        vlm_device: str = None,
        use_usp=False,
    ) -> 'FlashVSR3StagePipeline':
        """
        Create pipeline from model manager.
        
        Args:
            model_manager: Model manager instance
            torch_dtype: Data type for computation
            device: Device for generation network (DiT, VAE), e.g., "cuda:0"
            vlm_device: Device for VLM, e.g., "cuda:1". If None, uses same as device.
            use_usp: Whether to use unified sequence parallel
            
        Example (dual-GPU):
            pipe = FlashVSR3StagePipeline.from_model_manager(
                model_manager,
                device="cuda:0",      # 生成网络 (DiT + VAE)
                vlm_device="cuda:1",  # VLM (Qwen2.5-VL-7B)
            )
        """
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        
        pipe = FlashVSR3StagePipeline(device=device, torch_dtype=torch_dtype, vlm_device=vlm_device)
        pipe.fetch_models(model_manager)
        pipe.use_unified_sequence_parallel = False
        return pipe

    def init_agents(
        self,
        task_type: str = "64mT_to_3T",
        vlm_model_name: str = None,
        use_vlm: bool = True,
        vlm_device: str = None,
    ) -> None:
        """
        Initialize Stage 2 agents.
        
        Uses vLLM server for fast VLM inference. The server is launched as a
        background subprocess, and agents communicate via the OpenAI-compatible API.
        
        Args:
            task_type: Task type ("64mT_to_3T" or "3T_to_7T")
            vlm_model_name: VLM model name/path
            use_vlm: Whether to use VLM for detection/planning
            vlm_device: Override VLM device (uses self.vlm_device if None)
        """
        if vlm_model_name is not None:
            self.vlm_model_name = vlm_model_name
        
        # Determine VLM device
        vlm_dev = vlm_device if vlm_device is not None else self.vlm_device
        
        config = AgentConfig(
            model_name=self.vlm_model_name,
            device=vlm_dev,
            torch_dtype=self.torch_dtype,
            task_type=task_type,
        )
        
        print(f"[3-Stage] Initializing agents for task: {task_type}")
        print(f"[3-Stage] Generation network device: {self.device}")
        print(f"[3-Stage] VLM device: {vlm_dev}")
        
        # --- Launch vLLM server for VLM inference ---
        vlm_client = None
        self._vllm_process = None
        if use_vlm:
            try:
                vlm_client, self._vllm_process = self._launch_vllm_server(
                    config.model_name, vlm_dev,
                )
                print("[3-Stage] vLLM server ready")
            except Exception as e:
                raise RuntimeError("The requested visual planner could not start") from e
        
        # Create anomaly detection agent
        self.anomaly_agent = AnomalyDetectionAgent(
            config=config,
            use_vlm=use_vlm,
            use_metrics=True,
            vlm_client=vlm_client,
        )
        
        # Create medical VLM agent
        self.vlm_agent = MedicalVLMAgent(
            config=config,
            use_vlm=use_vlm,
            vlm_client=vlm_client,
        )
        
        # Create orchestrator
        self.agent_orchestrator = AgentOrchestrator(
            anomaly_agent=self.anomaly_agent,
            vlm_agent=self.vlm_agent,
            max_iterations=3,
            quality_threshold=0.9,
        )
        
        print("[3-Stage] Agents initialized")

    def _launch_vllm_server(self, model_name: str, vlm_device: str, port: int = 23794):
        """Launch a vLLM OpenAI-compatible server as a subprocess.
        
        Returns (openai.OpenAI client, subprocess.Popen handle).
        """
        import subprocess, time, socket, os
        from openai import OpenAI
        import httpx
        
        # Map logical VLM index from the configured physical layout.
        # With the default 0,1 layout, DiT uses cuda:0 and the vLLM
        # subprocess gets physical GPU 1 as its local cuda:0.
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
        
        # Find a free port
        for p in range(port, port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", p)) != 0:
                    port = p
                    break
        
        cmd = [
            os.environ.get("UNIFIELD_VLLM_PYTHON", __import__("sys").executable), "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(port),
            "--tensor-parallel-size", "1",
            "--max-model-len", os.environ.get("UNIFIELD_VLLM_MAX_MODEL_LEN", "32768"),
            "--dtype", "bfloat16",
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.85",
        ]
        
        env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": gpu_id}
        # Use local cache only — avoid HuggingFace network timeout
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        # Remove torchrun/DDP env vars so vLLM does not try to join
        # the parent NCCL process group, and remove proxies so localhost
        # OpenAI-compatible calls never get routed through an HTTP proxy.
        for k in list(env.keys()):
            ku = k.upper()
            if k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                      "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK",
                      "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID",
                      "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
                      "TORCHELASTIC_USE_AGENT_STORE") or ku in (
                      "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"): 
                del env[k]
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["no_proxy"] = "127.0.0.1,localhost"
        print(f"[3-Stage] Starting vLLM server on GPU {gpu_id}, port {port}...")
        print(f"[3-Stage] Command: {' '.join(cmd)}")
        
        log_path = f"/tmp/flashvsr_vllm_{port}.log"

        def _read_log_tail(max_chars: int = 4000) -> str:
            try:
                with open(log_path, "r", errors="replace") as f:
                    data = f.read()
                return data[-max_chars:]
            except OSError:
                return ""

        # Write logs to a file instead of PIPE. vLLM can emit enough startup
        # logs to fill a pipe before /health is ready, which deadlocks launch.
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=log_file, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
        print(f"[3-Stage] vLLM log: {log_path}", flush=True)

        # Wait for the OpenAI-compatible API to be ready. Some vLLM builds
        # return transient 502s from /health even after routes are mounted,
        # so probe /v1/models, which is what clients need next.
        base_url = f"http://127.0.0.1:{port}/v1"
        ready_url = f"{base_url}/models"
        timeout_sec = int(os.environ.get("FLASHVSR_VLLM_TIMEOUT_SEC", "600"))
        start = time.time()
        while time.time() - start < timeout_sec:
            # Check if process died
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
        print(f"[3-Stage] vLLM server ready at {base_url}", flush=True)
        return client, proc

    @staticmethod
    def _terminate_vllm_process(proc, timeout=10):
        """Terminate the vLLM API server and all of its worker processes."""
        import os
        import signal
        import subprocess

        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                proc.kill()
            proc.wait(timeout=timeout)

    def shutdown_vllm(self):
        """Terminate the vLLM server process group."""
        if hasattr(self, "_vllm_process") and self._vllm_process is not None:
            print("[3-Stage] Shutting down vLLM server...")
            proc = self._vllm_process
            self._vllm_process = None
            self._terminate_vllm_process(proc)

    def run_stage1(
        self,
        LQ_video: torch.Tensor,
        num_frames: int = None,
        height: int = None,
        width: int = None,
        return_latent: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Stage 1: Base SFT Generation
        
        Uses the base FlashVSR model to generate initial high-field MRI.
        
        Args:
            LQ_video: Low-field MRI input (B, C, F, H, W)
            num_frames, height, width: Optional size overrides
            return_latent: If True, also return Stage-1 clean latent for
                Stage-3 latent passthrough (avoids decode->encode round-trip)
            **kwargs: Additional arguments for base pipeline
            
        Returns:
            Generated high-field MRI tensor, or
            (generated image, clean latent) when ``return_latent=True``
        """
        B, C, F, H, W = LQ_video.shape
        
        if num_frames is None:
            num_frames = F
        if height is None:
            height = H
        if width is None:
            width = W
        
        # 使用 test_process 进行一次性推理（与旧版 train_flashvsr_tiny_lora 验证路径一致）
        # - 所有帧一次性处理，不做分块
        # - 显式设置 timestep，正确计算 σ
        # - 关闭 color_fix，if_buffer=True
        # - t=1000: 从纯噪声出发，σ=1.0，完全去噪
        result = self.test_process(
            LQ_video=LQ_video,
            HQ_video=None,
            num_frames=num_frames,
            height=height,
            width=width,
            cfg_scale=1.0,
            color_fix=False,
            if_buffer=True,
            inference_timestep=1000.0,
            **kwargs,
        )
        
        pred_hq = result['pred_hq']
        pred_clean_latent = result.get('pred_clean_latent', None)
        
        # Ensure output is (B, C, F, H, W)
        if pred_hq.dim() == 4:
            pred_hq = pred_hq.unsqueeze(0)

        if return_latent:
            if pred_clean_latent is None:
                raise RuntimeError("Stage-1 output missing 'pred_clean_latent' for latent passthrough")
            return pred_hq, pred_clean_latent

        return pred_hq

    def run_stage2(
        self,
        lq_image: torch.Tensor,
        generated_image: Optional[torch.Tensor] = None,
        gt_image: Optional[torch.Tensor] = None,
        task_type: str = "64mT_to_3T",
        unified: bool = True,
        skip_global_score: bool = False,
        lq_only: Optional[bool] = None,
    ) -> Stage2Output:
        """
        Stage 2: Agent-driven Diagnosis & Planning
        
        Args:
            lq_image: Low-field MRI input
            generated_image: Output from Stage 1. Optional when lq_only=True.
            gt_image: Ground truth (optional, for training/evaluation)
            task_type: Task type for agent configuration
            unified: If True, use a single VLM call for both detection and planning.
                     If False, use separate calls (original behavior).
            lq_only: If True, ask the VLM to pre-plan from LQ slices only,
                     without generated_image or gt_image. Defaults to
                     ``self.stage2_lq_only``.
            
        Returns:
            Stage2Output containing anomaly report and improvement plan
        """
        # Initialize agents if not done
        if self.agent_orchestrator is None:
            self.init_agents(task_type=task_type)

        use_lq_only = self.stage2_lq_only if lq_only is None else bool(lq_only)

        # Run diagnosis (unified = single VLM call, else = separate calls)
        if use_lq_only:
            problems, plans = self.agent_orchestrator.run_lq_diagnosis(
                lq_image=lq_image,
            )
        elif unified:
            if generated_image is None:
                raise ValueError("generated_image is required when lq_only=False")
            problems, plans = self.agent_orchestrator.run_unified_diagnosis(
                lq_image=lq_image,
                generated_image=generated_image,
                gt_image=gt_image,
            )
        else:
            if generated_image is None:
                raise ValueError("generated_image is required when lq_only=False")
            problems, plans = self.agent_orchestrator.run_diagnosis(
                lq_image=lq_image,
                generated_image=generated_image,
                gt_image=gt_image,
            )

        # Build anomaly report
        if problems:
            if skip_global_score:
                # 训练时跳过耗时的 GPU→CPU SSIM/PSNR 计算，VLM 已给出质量评估
                global_score = 0.5
            else:
                global_score = (
                    self.anomaly_agent._calculate_global_score(generated_image, gt_image)
                    if gt_image is not None and generated_image is not None
                    else 0.5
                )
            
            anomaly_report = AnomalyReport(
                problems=problems,
                global_quality_score=global_score,
                needs_refinement=len(problems) > 0,
                summary=self.anomaly_agent._generate_summary(problems),
            )
        else:
            anomaly_report = AnomalyReport(
                problems=[],
                global_quality_score=1.0,
                needs_refinement=False,
                summary="No problems detected.",
            )

        # Build improvement plan
        if plans:
            improvement_plan = ImprovementPlan(
                plans=plans,
                task_type=task_type,
                global_guidance=self.vlm_agent._generate_global_guidance(problems, plans),
                estimated_iterations=min(len(plans), 3),
            )
        else:
            improvement_plan = ImprovementPlan(
                plans=[],
                task_type=task_type,
                global_guidance="No refinement needed.",
                estimated_iterations=0,
            )

        return Stage2Output(
            anomaly_report=anomaly_report,
            improvement_plan=improvement_plan,
            needs_refinement=anomaly_report.needs_refinement,
        )

    def run_stage3(
        self,
        lq_image: torch.Tensor,
        generated_image: torch.Tensor,
        improvement_plan: ImprovementPlan,
        gt_image: Optional[torch.Tensor] = None,
        max_iterations: int = 1,
        task_type: str = "64mT_to_3T",
        generated_latents: Optional[torch.Tensor] = None,
    ) -> Stage3Output:
        """
        Stage 3: Local-Guided Diffusion Refinement
        
        Iteratively refines the generated image using text-guided cross-attention
        and regional masks.
        
        Args:
            lq_image: Low-field MRI input
            generated_image: Current generated image
            improvement_plan: Plans from Stage 2
            generated_latents: Optional Stage-1 clean latents for latent
                passthrough (skips first encode and iterative re-encode)
            gt_image: Ground truth (optional)
            max_iterations: Maximum refinement iterations
            task_type: Task type ("64mT_to_3T" or "3T_to_7T")
            
        Returns:
            Stage3Output containing refined image and metrics
        """
        self.refinement_controller.reset()
        current_image = generated_image.clone()
        current_latents = None
        if generated_latents is not None:
            current_latents = generated_latents.to(device=self.device, dtype=self.torch_dtype)
        
        B, C, F, H, W = current_image.shape
        image_size = (H, W) if F == 1 else (F, H, W)
        
        # Create refinement conditions
        conditions = self.plan_to_mask.create_refinement_conditions(
            plans=improvement_plan.plans,
            image_size=image_size,
            device=self.device,
            dtype=self.torch_dtype,
        )
        
        iteration = 0
        quality_scores = []
        
        while iteration < max_iterations and conditions:
            # Pass all conditions directly (multi-channel mask encoding)
            current_image, current_latents = self._refine_single_region(
                lq_image=lq_image,
                current_image=current_image,
                current_latents=current_latents,
                conditions=conditions,
                return_latents=True,
            )
            
            # Calculate quality after this iteration
            if gt_image is not None:
                quality = self._calculate_quality(current_image, gt_image)
            else:
                quality = 0.5  # Will be overridden by VLM assessment below

            if getattr(self, "stage2_lq_only", False):
                quality_scores.append(quality)
                iteration += 1
                break
            
            # Re-run Stage 2 to check for remaining problems (works with or without GT)
            stage2_check = self.run_stage2(
                lq_image=lq_image,
                generated_image=current_image,
                gt_image=gt_image,
                task_type=task_type,
            )
            
            # Use VLM-assessed quality when GT is unavailable
            if gt_image is None:
                quality = stage2_check.anomaly_report.global_quality_score
            
            quality_scores.append(quality)
            
            if not stage2_check.needs_refinement:
                break
            
            # Update conditions for next iteration
            conditions = self.plan_to_mask.create_refinement_conditions(
                plans=stage2_check.improvement_plan.plans,
                image_size=image_size,
                device=self.device,
                dtype=self.torch_dtype,
            )
            
            iteration += 1
        
        # Build summary
        summary = {
            "iterations_used": iteration,
            "quality_history": quality_scores,
            "plans_executed": len(improvement_plan.plans),
            "final_quality": quality_scores[-1] if quality_scores else 0.0,
        }
        
        return Stage3Output(
            refined_image=current_image,
            iterations_used=iteration,
            final_quality=quality_scores[-1] if quality_scores else 0.0,
            refinement_summary=summary,
        )

    def _merge_conditions(
        self,
        conditions: List[RefinementCondition],
        image_size: Tuple[int, ...],
    ) -> RefinementCondition:
        """
        Merge multiple refinement conditions into a single condition.
        
        This combines all masks (using max) and concatenates all text prompts
        to perform a single unified refinement step.
        
        Args:
            conditions: List of individual refinement conditions
            image_size: Image size for mask dimensions
            
        Returns:
            A single merged RefinementCondition
        """
        if len(conditions) == 1:
            return conditions[0]
        
        # Merge masks using max operation (union of all regions)
        masks = [cond.mask for cond in conditions]
        merged_mask = torch.stack(masks).max(dim=0)[0]
        
        # Concatenate all text prompts with separators
        text_prompts = [cond.text_prompt for cond in conditions]
        merged_prompt = "; ".join(text_prompts)
        
        # Average strength, weighted by mask area
        total_area = sum(cond.mask.sum().item() for cond in conditions)
        if total_area > 0:
            merged_strength = sum(
                cond.strength * cond.mask.sum().item() for cond in conditions
            ) / total_area
        else:
            merged_strength = sum(cond.strength for cond in conditions) / len(conditions)
        
        # Create merged condition (use first plan as reference)
        merged_condition = RefinementCondition(
            plan=conditions[0].plan,  # Reference plan
            mask=merged_mask,
            text_prompt=merged_prompt,
            text_embedding=None,  # Will be recomputed in _refine_single_region
            strength=merged_strength,
        )
        merged_condition.to(self.device, self.torch_dtype)
        
        return merged_condition

    def _refine_single_region(
        self,
        lq_image: torch.Tensor,
        current_image: torch.Tensor,
        condition: 'RefinementCondition' = None,
        conditions: List['RefinementCondition'] = None,
        max_plans: int = 3,
        current_latents: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Refine using Rectified Flow Bridge (multi-step Euler) + multi-channel mask.

        Bridge direction: tau=0 (stage1) → tau=1 (GT).
        Training target: v = x_gt - x_s1 (constant velocity, t-independent).
        Inference update uses multiple small Euler steps with a time-varying
        timestep schedule aligned to Stage-1 semantics.

        Accepts either a single merged ``condition`` (backward compat) or a list
        of ``conditions`` (new multi-channel path).  When ``conditions`` is
        provided it takes priority.

        If ``current_latents`` is provided, refinement starts directly from latent
        space and skips VAE encode for this step.
        """
        from .flashvsr_tiny import model_fn_wan_video

        # Normalise to a list
        if conditions is None:
            conditions = [condition] if condition is not None else []

        # ── Text prompt ─────────────────────────────────────────────
        all_prompts = [c.text_prompt for c in conditions if c.text_prompt]
        if all_prompts:
            try:
                self.init_cross_kv(prompt="; ".join(all_prompts))
            except Exception as e:
                print(f"Warning: Failed to update cross-KV: {e}")

        B, C, n_frames, H, W = current_image.shape

        # ── Prepare latent state (direct passthrough if provided) ───
        if current_latents is None:
            self.load_models_to_device(["vae"])
            with torch.no_grad():
                latents = self.vae.encode(
                    current_image.to(self.device, dtype=self.torch_dtype),
                    device=self.device, tiled=True,
                )
        else:
            latents = current_latents.to(device=self.device, dtype=self.torch_dtype)

        # ── Build multi-channel mask (B, max_plans, F, H, W) ───────
        mask_channels = []
        for cond in conditions[:max_plans]:
            raw_mask = cond.mask.to(self.device).float()
            if raw_mask.dim() == 2:
                ch = raw_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                ch = ch.expand(1, 1, n_frames, H, W)
            elif raw_mask.dim() == 3:
                ch = F.interpolate(
                    raw_mask.unsqueeze(0).unsqueeze(0),
                    size=(n_frames, H, W),
                    mode='trilinear', align_corners=False,
                )
            else:
                ch = raw_mask.unsqueeze(0) if raw_mask.dim() == 4 else raw_mask
            mask_channels.append(ch * cond.strength)

        while len(mask_channels) < max_plans:
            mask_channels.append(torch.zeros(1, 1, n_frames, H, W,
                                             device=self.device, dtype=self.torch_dtype))
        mask_input = torch.cat(mask_channels, dim=1).to(dtype=self.torch_dtype)

        self.load_models_to_device(["dit"])
        time_emb_dtype = next(self.dit.time_embedding.parameters()).dtype

        # LQ condition
        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()
            LQ_latents = self.dit.LQ_proj_in.stream_forward(lq_image)
        else:
            LQ_latents = None

        # Mask condition (multi-channel)
        Mask_latents = None
        if hasattr(self.dit, "Mask_proj_in"):
            self.dit.Mask_proj_in.clear_cache()
            _mask_dtype = next(self.dit.Mask_proj_in.parameters()).dtype
            Mask_latents = self.dit.Mask_proj_in.stream_forward(mask_input.to(dtype=_mask_dtype))
            # Cast back to model dtype so x + Mask_latents stays bf16 in DiT blocks
            if Mask_latents is not None:
                Mask_latents = [m.to(dtype=self.torch_dtype) for m in Mask_latents]

        # ── DiT forward with micro-steps and dynamic timestep schedule ──
        substeps = max(1, int(self.stage3_substeps))
        step_scale = self.stage3_step_scale / float(substeps)
        refined_latents = latents
        tau_eps = 1e-4
        for step_id in range(substeps):
            # Bridge progress tau starts from stage1 side (tau=0) and advances.
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
                v_pred, _, _ = model_fn_wan_video(
                    self.dit,
                    x=refined_latents.to(dtype=self.torch_dtype),
                    timestep=timestep_step,
                    context=None,
                    tea_cache=None,
                    use_unified_sequence_parallel=False,
                    LQ_latents=LQ_latents,
                    Mask_latents=Mask_latents,
                    is_full_block=False,
                    is_stream=False,
                    topk_ratio=2.0,
                    kv_ratio=3.0,
                    cur_process_idx=0,
                    t_mod=t_mod,
                    t=t_embed,
                    local_range=9,
                )
            refined_latents = refined_latents + step_scale * v_pred

        # Decode
        with torch.no_grad():
            refined = self.TCDecoder.decode_video(
                refined_latents.to(dtype=self.torch_dtype).transpose(1, 2),
                parallel=False,
                show_progress_bar=False,
                cond=lq_image[:, :, :refined_latents.shape[2]*4].to(dtype=self.torch_dtype),
                bidirectional=self.use_bidirectional_decoder,
                temporal_deblock_strength=self.tcdecoder_temporal_deblock_strength,
            ).transpose(1, 2).mul_(2).sub_(1)

        if return_latents:
            return refined, refined_latents

        return refined

    def _calculate_quality(
        self,
        generated: torch.Tensor,
        gt: torch.Tensor,
    ) -> float:
        """Calculate quality score between generated and GT"""
        try:
            from skimage.metrics import structural_similarity
            
            gen_np = generated.detach().cpu().float().numpy()
            gt_np = gt.detach().cpu().float().numpy()
            
            gen_np = (gen_np + 1.0) / 2.0
            gt_np = (gt_np + 1.0) / 2.0
            
            # Compute SSIM — uniformly sample slices to cover entire volume
            ssim_vals = []
            gen_flat = gen_np.reshape(-1, *gen_np.shape[-2:])
            gt_flat = gt_np.reshape(-1, *gt_np.shape[-2:])
            
            max_slices = 10
            if gen_flat.shape[0] <= max_slices:
                indices = range(gen_flat.shape[0])
            else:
                indices = np.linspace(0, gen_flat.shape[0] - 1, max_slices, dtype=int)
            for i in indices:
                if gt_flat[i].max() - gt_flat[i].min() < 1e-6:
                    continue
                ssim_val = structural_similarity(
                    gen_flat[i], gt_flat[i], data_range=1.0,
                    gaussian_weights=True, sigma=1.5,
                    use_sample_covariance=False,
                )
                ssim_vals.append(ssim_val)
            
            return float(np.mean(ssim_vals)) if ssim_vals else 0.5
        except:
            return 0.5

    @torch.no_grad()
    def run_3stage(
        self,
        LQ_video: torch.Tensor,
        HQ_video: Optional[torch.Tensor] = None,
        task_type: str = "64mT_to_3T",
        max_refinement_iterations: int = 1,
        skip_stage2: bool = False,
        skip_stage3: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> ThreeStageOutput:
        """
        Run the complete 3-stage MRI enhancement pipeline.
        
        Args:
            LQ_video: Low-field MRI input (B, C, F, H, W)
            HQ_video: High-field MRI ground truth (optional)
            task_type: Task type ("64mT_to_3T" or "3T_to_7T")
            max_refinement_iterations: Max iterations for Stage 3
            skip_stage2: Skip diagnosis & planning
            skip_stage3: Skip local refinement
            verbose: Print progress information
            **kwargs: Additional arguments for Stage 1
            
        Returns:
            ThreeStageOutput containing all results
        """
        start_time = time.time()
        
        # Ensure input is on correct device
        LQ_video = LQ_video.to(self.device, dtype=self.torch_dtype)
        if HQ_video is not None:
            HQ_video = HQ_video.to(self.device, dtype=self.torch_dtype)
        
        B, C, F, H, W = LQ_video.shape
        
        # ================== Stage 1: Base SFT ==================
        if verbose:
            print("\n" + "="*50)
            print("Stage 1: Base SFT Generation")
            print("="*50)

        stage1_kwargs = dict(kwargs)
        # run_3stage always enables latent passthrough internally.
        stage1_kwargs.pop("return_latent", None)
        stage1_output, stage1_latents = self.run_stage1(
            LQ_video=LQ_video,
            num_frames=F,
            height=H,
            width=W,
            return_latent=True,
            **stage1_kwargs,
        )
        
        if verbose:
            print(f"Stage 1 complete. Output shape: {stage1_output.shape}")
        
        # ================== Stage 2: Diagnosis & Planning ==================
        stage2_output = None
        
        if not skip_stage2 and self.enable_stage2:
            if verbose:
                print("\n" + "="*50)
                print("Stage 2: Agent-driven Diagnosis & Planning")
                print("="*50)
            
            stage2_output = self.run_stage2(
                lq_image=LQ_video,
                generated_image=stage1_output,
                gt_image=HQ_video,
                task_type=task_type,
            )
            
            if verbose:
                print(f"Problems detected: {len(stage2_output.anomaly_report.problems)}")
                print(f"Plans generated: {len(stage2_output.improvement_plan.plans)}")
                print(f"Needs refinement: {stage2_output.needs_refinement}")
                
                for prob in stage2_output.anomaly_report.problems[:3]:
                    print(f"  - {prob.region_name}: {prob.issue_description} (severity: {prob.severity:.2f})")
        
        # ================== Stage 3: Local Refinement ==================
        stage3_output = None
        final_image = stage1_output
        
        if not skip_stage3 and self.enable_stage3:
            if stage2_output is not None and stage2_output.needs_refinement:
                if verbose:
                    print("\n" + "="*50)
                    print("Stage 3: Local-Guided Diffusion Refinement")
                    print("="*50)
                
                stage3_output = self.run_stage3(
                    lq_image=LQ_video,
                    generated_image=stage1_output,
                    improvement_plan=stage2_output.improvement_plan,
                    generated_latents=stage1_latents,
                    gt_image=HQ_video,
                    max_iterations=max_refinement_iterations,
                    task_type=task_type,
                )
                
                final_image = stage3_output.refined_image
                
                if verbose:
                    print(f"Iterations used: {stage3_output.iterations_used}")
                    print(f"Final quality: {stage3_output.final_quality:.4f}")
            else:
                if verbose:
                    print("\nSkipping Stage 3: No refinement needed")
        
        total_time = time.time() - start_time
        
        if verbose:
            print("\n" + "="*50)
            print(f"3-Stage Pipeline Complete")
            print(f"Total time: {total_time:.2f}s")
            print("="*50)
        
        return ThreeStageOutput(
            final_image=final_image,
            stage1_image=stage1_output,
            stage2_output=stage2_output,
            stage3_output=stage3_output,
            total_time=total_time,
        )

    def train_3stage(
        self,
        LQ_video: torch.Tensor,
        HQ_video: torch.Tensor,
        task_type: str = "64mT_to_3T",
        train_stage1: bool = True,
        train_stage3: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass for 3-stage pipeline.
        
        This method is used during training to compute losses for:
        - Stage 1: Base SFT loss (flow matching)
        - Stage 3: Refinement loss (optional, guided by Stage 2 plans)
        
        Args:
            LQ_video: Low-field MRI input
            HQ_video: High-field MRI ground truth
            task_type: Task type
            train_stage1: Whether to compute Stage 1 loss
            train_stage3: Whether to compute Stage 3 loss
            
        Returns:
            Dictionary containing losses and intermediate outputs
        """
        outputs = {}
        
        # Stage 1 training (base SFT)
        if train_stage1:
            stage1_train_out = self.train_process(
                LQ_video=LQ_video,
                HQ_video=HQ_video,
                **kwargs,
            )
            outputs['stage1_noise_pred'] = stage1_train_out['noise_pred']
            outputs['stage1_target'] = stage1_train_out['target']
            outputs['stage1_pred_hq'] = stage1_train_out['pred_hq']
        
        # For Stage 3 training, we need to:
        # 1. Generate Stage 1 output (with gradients detached)
        # 2. Run Stage 2 to get plans
        # 3. Train refinement with plan conditioning
        if train_stage3:
            with torch.no_grad():
                # Generate initial output
                stage1_output = self.test_process(
                    LQ_video=LQ_video,
                    HQ_video=None,
                    **kwargs,
                )['pred_hq']
                
                # Get Stage 2 plans
                stage2_output = self.run_stage2(
                    lq_image=LQ_video,
                    generated_image=stage1_output,
                    gt_image=HQ_video,
                    task_type=task_type,
                )
            
            if stage2_output.needs_refinement:
                # Compute refinement loss
                # This could be implemented as additional loss on refined regions
                outputs['stage2_plans'] = stage2_output.improvement_plan.plans
                outputs['needs_refinement'] = True
            else:
                outputs['needs_refinement'] = False
        
        return outputs

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
        """
        Stage 3 training forward pass — Rectified Flow Bridge with mask condition.

        Changes from original:
        1. Bridge interpolation (no random noise): x_t = (1-t)*x_s1 + t*x_gt
        2. Bridge velocity target: v = x_gt - x_s1
        3. Masks encoded as multi-channel condition via Mask_proj_in
        4. Bridge interpolation is applied uniformly (no mask gating on input)
        5. Masks only used for loss weighting and as DiT condition

        Args:
            LQ_video: (B, C, F, H, W) low-field MRI
            HQ_video: (B, C, F, H, W) ground truth
            stage1_image: Stage 1 output (detached)
            conditions: Refinement conditions from Stage 2
            stage1_latents_cached: pre-computed latents (skip VAE)
            hq_latents_cached: pre-computed latents (skip VAE)
            max_plans: max number of plan channels

        Returns:
            Dict with 'v_pred', 'target', 'mask_weight'
        """
        from .flashvsr_tiny import model_fn_wan_video

        # ── Cached latents ──────────────────────────────────────────
        if stage1_latents_cached is not None and hq_latents_cached is not None:
            hq_latents = hq_latents_cached.to(device=self.device, dtype=self.torch_dtype)
            stage1_latents = stage1_latents_cached.to(device=self.device, dtype=self.torch_dtype)
        else:
            self.load_models_to_device(["vae"])
            with torch.no_grad():
                hq_latents = self.vae.encode(
                    HQ_video.to(self.device, dtype=self.torch_dtype),
                    device=self.device, tiled=True,
                )
                stage1_latents = self.vae.encode(
                    stage1_image.to(self.device, dtype=self.torch_dtype),
                    device=self.device, tiled=True,
                )

        hq_latents = hq_latents.to(device=self.device, dtype=self.torch_dtype)
        stage1_latents = stage1_latents.to(device=self.device, dtype=self.torch_dtype)

        B, C_lat, latent_f, latent_h, latent_w = hq_latents.shape
        _, _, F_img, H_img, W_img = LQ_video.shape

        # ── Build multi-channel mask (B, max_plans, F, H, W) ───────
        # Each condition occupies one channel; severity baked into value.
        mask_channels = []
        all_prompts = []
        combined_mask_latent = torch.zeros(
            1, 1, latent_f, latent_h, latent_w,
            device=self.device, dtype=torch.float32,
        )

        for i, cond in enumerate(conditions[:max_plans]):
            raw_mask = cond.mask.to(self.device).float()  # (H,W) or (D,H,W)
            # Expand to (1, 1, F, H, W) for pixel-space multi-channel stack
            if raw_mask.dim() == 2:
                ch = raw_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,H,W)
                ch = ch.expand(1, 1, F_img, H_img, W_img)
            elif raw_mask.dim() == 3:
                ch = raw_mask.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
                ch = F.interpolate(ch, size=(F_img, H_img, W_img),
                                   mode='trilinear', align_corners=False)
            else:
                ch = raw_mask.unsqueeze(0) if raw_mask.dim() == 4 else raw_mask
            mask_channels.append(ch * cond.strength)

            # Also build latent-space combined mask for loss weighting
            if raw_mask.dim() == 2:
                m_lat = F.interpolate(
                    raw_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    size=(1, latent_h, latent_w),
                    mode='trilinear', align_corners=False,
                ).expand(1, 1, latent_f, latent_h, latent_w)
            elif raw_mask.dim() == 3:
                m_lat = F.interpolate(
                    raw_mask.unsqueeze(0).unsqueeze(0),
                    size=(latent_f, latent_h, latent_w),
                    mode='trilinear', align_corners=False,
                )
            else:
                m_lat = raw_mask
            combined_mask_latent = torch.max(combined_mask_latent, m_lat * cond.strength)

            if cond.text_prompt:
                all_prompts.append(cond.text_prompt)

        # Pad to max_plans channels, then concat → (B, max_plans, F, H, W)
        while len(mask_channels) < max_plans:
            mask_channels.append(torch.zeros(1, 1, F_img, H_img, W_img,
                                             device=self.device, dtype=self.torch_dtype))
        mask_input = torch.cat(mask_channels, dim=1)  # (1, max_plans, F, H, W)
        mask_input = mask_input.to(dtype=self.torch_dtype)

        # ── Text prompt ─────────────────────────────────────────────
        if all_prompts:
            merged_prompt = "; ".join(all_prompts)
            try:
                self.init_cross_kv(prompt=merged_prompt)
            except Exception:
                pass

        # ── Loss weight mask (latent space) ─────────────────────────
        mask_weight = combined_mask_latent.expand_as(hq_latents)
        mask_weight = mask_weight.to(dtype=self.torch_dtype)

        # ============================================================
        # Rectified Flow Bridge — no random noise
        # x_t = (1-t) * x_s1 + t * x_gt   (uniform over full image)
        # target = x_gt - x_s1
        # model timestep uses reverse mapping for Stage-1 semantic alignment:
        # t_model = (1-t) * T
        # ============================================================
        # Mixed tau sampling to align training with Stage-3 inference trajectory:
        # - with probability p: tau in (0, stage3_step_scale)
        # - with probability 1-p: tau in [stage3_step_scale, 1]
        tau_eps = 1e-4
        small_prob = float(getattr(self, "stage3_tau_small_prob", 0.8))
        small_prob = max(0.0, min(1.0, small_prob))
        small_upper = float(self.stage3_step_scale)
        small_upper = max(tau_eps, min(1.0, small_upper))

        if torch.rand(1).item() < small_prob:
            if small_upper <= tau_eps:
                t_uniform = torch.full((1,), tau_eps, device=self.device, dtype=torch.float32)
            else:
                t_uniform = tau_eps + (small_upper - tau_eps) * torch.rand(1, device=self.device)
        else:
            if small_upper >= 1.0:
                t_uniform = torch.rand(1, device=self.device)
            else:
                t_uniform = small_upper + (1.0 - small_upper) * torch.rand(1, device=self.device)
        t_uniform = t_uniform.clamp(min=tau_eps, max=1.0)

        latents = (1 - t_uniform) * stage1_latents + t_uniform * hq_latents
        latents = latents.to(dtype=self.torch_dtype)

        t_random = self._bridge_tau_to_model_timestep(t_uniform)

        # ── DiT forward ─────────────────────────────────────────────
        self.load_models_to_device(["dit"])

        self.timestep = t_random.to(device=self.device, dtype=self.torch_dtype)
        time_emb_dtype = next(self.dit.time_embedding.parameters()).dtype
        time_emb_input = sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep).to(
            device=self.device,
            dtype=time_emb_dtype,
        )
        self.t = self.dit.time_embedding(time_emb_input).to(dtype=self.torch_dtype)
        self.t_mod = self.dit.time_projection(self.t).unflatten(1, (6, self.dit.dim))

        # LQ condition
        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()
        self.TCDecoder.clean_mem()

        LQ_video_cast = LQ_video.to(device=self.device, dtype=self.torch_dtype)
        if hasattr(self.dit, "LQ_proj_in"):
            LQ_latents = self.dit.LQ_proj_in.stream_forward(LQ_video_cast)
        else:
            LQ_latents = None

        # Mask condition (multi-channel)
        Mask_latents = None
        if hasattr(self.dit, "Mask_proj_in"):
            self.dit.Mask_proj_in.clear_cache()
            _mask_dtype = next(self.dit.Mask_proj_in.parameters()).dtype
            Mask_latents = self.dit.Mask_proj_in.stream_forward(mask_input.to(dtype=_mask_dtype))
            # Cast back to model dtype so x + Mask_latents stays bf16 in DiT blocks
            if Mask_latents is not None:
                Mask_latents = [m.to(dtype=self.torch_dtype) for m in Mask_latents]

        v_pred, _, _ = model_fn_wan_video(
            self.dit,
            x=latents,
            timestep=self.timestep,
            context=None,
            tea_cache=None,
            use_unified_sequence_parallel=False,
            LQ_latents=LQ_latents,
            Mask_latents=Mask_latents,
            is_full_block=False,
            is_stream=False,
            topk_ratio=topk_ratio,
            kv_ratio=kv_ratio,
            cur_process_idx=0,
            t_mod=self.t_mod,
            t=self.t,
            local_range=local_range,
        )

        # Bridge target: velocity = GT − stage1
        target = (hq_latents - stage1_latents).to(dtype=self.torch_dtype)

        return {
            'noise_pred': v_pred,       # keep key name for backward compat
            'target': target,
            'mask_weight': mask_weight,
        }
