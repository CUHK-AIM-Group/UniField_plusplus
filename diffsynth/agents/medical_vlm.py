"""
Medical VLM Agent for MRI Enhancement
======================================

Stage 2 Component: Generates improvement plans based on detected problems.

Input: Low-field MRI, Generated MRI, Problem list from Anomaly Detection Agent
Output: Structured improvement plans with text prompts for cross-attention guidance
"""

import time

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from PIL import Image

from .base_agent import BaseAgent, AgentConfig, AgentType, Problem, Plan
from .brain_atlas import BrainAtlas, BRAIN_ATLAS


@dataclass
class ImprovementPlan:
    """
    Complete improvement plan for MRI enhancement.
    
    Contains multiple Plans, each targeting a specific problem.
    Plans are ordered by priority (lower number = higher priority).
    """
    plans: List[Plan]
    task_type: str  # "64mT_to_3T" or "3T_to_7T"
    global_guidance: str  # Overall guidance text
    estimated_iterations: int  # Estimated number of refinement iterations needed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plans": [p.to_dict() for p in self.plans],
            "task_type": self.task_type,
            "global_guidance": self.global_guidance,
            "estimated_iterations": self.estimated_iterations,
        }
    
    def __len__(self) -> int:
        return len(self.plans)
    
    def __iter__(self):
        return iter(sorted(self.plans, key=lambda p: p.priority))
    
    def get_all_prompts(self) -> List[str]:
        """Get all text prompts from plans"""
        return [p.text_prompt for p in sorted(self.plans, key=lambda p: p.priority)]


class MedicalVLMAgent(BaseAgent):
    """
    Agent for generating improvement plans using medical knowledge.
    
    This agent takes detected problems and generates specific, actionable
    improvement plans that can guide the diffusion model's local refinement.
    
    Key features:
    1. Uses medical VLM (Qwen2.5-VL, InternVL, etc.) for understanding context
    2. Generates spatial descriptions that diffusion models can understand
    3. Creates text prompts for cross-attention conditioning
    4. Provides mask information for regional inpainting
    """
    
    # Pre-defined improvement templates for common issues
    IMPROVEMENT_TEMPLATES = {
        # 64mT -> 3T common issues and fixes
        "64mT_to_3T": {
            "blurry edges": {
                "action": "Enhance edge sharpness while preserving anatomical accuracy",
                "outcome": "Clear, well-defined tissue boundaries without artificial ringing",
                "prompt_template": "sharp MRI with clear {region} boundaries, high contrast edges, detailed tissue definition",
            },
            "low contrast": {
                "action": "Increase gray-white matter contrast to match 3T quality",  
                "outcome": "Natural contrast levels matching clinical 3T scans",
                "prompt_template": "high contrast MRI, clear differentiation between gray and white matter in {region}",
            },
            "structural distortion": {
                "action": "Correct structural distortions while preserving anatomy",
                "outcome": "Anatomically accurate structures without geometric errors",
                "prompt_template": "anatomically correct {region} structure, proper tissue shape and position",
            },
            "intensity mismatch": {
                "action": "Normalize intensity distribution to match 3T characteristics",
                "outcome": "Uniform intensity with proper tissue signal levels",
                "prompt_template": "proper MRI intensity in {region}, natural signal distribution",
            },
            "missing gray-white matter differentiation": {
                "action": "Enhance gray-white matter boundary visibility",
                "outcome": "Clear distinction between gray and white matter",
                "prompt_template": "clear gray matter white matter boundary in {region}, visible tissue layers",
            },
            "smoothed sulci and gyri": {
                "action": "Restore cortical folding details",
                "outcome": "Natural sulci and gyri patterns visible",
                "prompt_template": "detailed cortical folds in {region}, natural brain surface texture with sulci and gyri",
            },
            "artifacts": {
                "action": "Remove imaging artifacts while preserving signal",
                "outcome": "Clean image without streak or ring artifacts",
                "prompt_template": "artifact-free MRI of {region}, clean image quality",
            },
        },
        # 3T -> 7T common issues and fixes
        "3T_to_7T": {
            "over-sharpening artifacts": {
                "action": "Reduce over-sharpening while maintaining detail",
                "outcome": "Natural sharpness without artificial enhancement artifacts",
                "prompt_template": "natural {region} texture, smooth transitions without harsh edges",
            },
            "grid-like artifacts": {
                "action": "Remove grid artifacts while preserving tissue structure",
                "outcome": "Smooth continuous tissue without grid patterns",
                "prompt_template": "smooth continuous tissue in {region}, no grid patterns, natural texture",
            },
            "susceptibility artifacts": {
                "action": "Reduce susceptibility-induced signal loss and distortion",
                "outcome": "Recovered signal in affected areas with proper geometry",
                "prompt_template": "recovered signal in {region}, reduced distortion, proper anatomy",
            },
            "B1 inhomogeneity": {
                "action": "Correct intensity variations from B1 field inhomogeneity",
                "outcome": "Uniform intensity across the image",
                "prompt_template": "uniform intensity in {region}, consistent signal levels",
            },
            "over-enhanced noise": {
                "action": "Reduce noise amplification while keeping detail",
                "outcome": "Balance between detail and noise level",
                "prompt_template": "clean {region} with balanced detail and noise, natural appearance",
            },
        },
    }
    
    def __init__(
        self,
        config: AgentConfig,
        use_vlm: bool = True,
        vlm_client=None,
        # Legacy args (ignored when using vLLM)
        vlm_model=None,
        vlm_processor=None,
    ):
        super().__init__(config)
        self.use_vlm = use_vlm
        self.brain_atlas = BRAIN_ATLAS
        
        # OpenAI-compatible client (backed by vLLM server)
        self.vlm_client = vlm_client
        # Legacy: kept for fallback / non-vLLM mode
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        
    @property
    def agent_type(self) -> AgentType:
        return AgentType.MEDICAL_VLM
    
    def initialize(self) -> None:
        """Initialize the agent, loading VLM if needed"""
        if self.use_vlm and self.vlm_client is None and self.vlm_model is None:
            self._load_vlm()
        self._is_initialized = True
    
    def _load_vlm(self) -> None:
        """Load the Vision-Language Model"""
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            
            print(f"[MedicalVLMAgent] Loading VLM model: {self.config.model_name}")
            
            self.vlm_processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                min_pixels=256 * 28 * 28,
                max_pixels=512 * 28 * 28,
            )
            
            self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.torch_dtype,
                device_map="auto",
            )
            self.vlm_model.eval()
            
            print("[MedicalVLMAgent] VLM model loaded successfully")
            
        except Exception as e:
            print(f"[MedicalVLMAgent] Warning: Failed to load VLM model: {e}")
            print("[MedicalVLMAgent] Using template-based plan generation only")
            self.use_vlm = False
    
    def process(
        self,
        lq_image: torch.Tensor,
        generated_image: torch.Tensor,
        problems: List[Problem],
        use_templates: bool = True,
    ) -> ImprovementPlan:
        """
        Generate improvement plans for detected problems.
        
        Args:
            lq_image: Low-field MRI input
            generated_image: Generated high-field MRI from Stage 1
            problems: List of problems from Anomaly Detection Agent
            use_templates: Whether to use pre-defined templates
            
        Returns:
            ImprovementPlan containing all plans for refinement
        """
        self._ensure_initialized()
        
        if not problems:
            return ImprovementPlan(
                plans=[],
                task_type=self.config.task_type,
                global_guidance="No refinement needed.",
                estimated_iterations=0,
            )
        
        # Keep only the top-3 most severe problems
        top_problems = sorted(problems, key=lambda p: p.severity, reverse=True)[:3]
        
        plans = []
        
        for idx, problem in enumerate(top_problems):
            # Generate plan for this problem
            plan = self._generate_plan_for_problem(
                problem=problem,
                lq_image=lq_image,
                generated_image=generated_image,
                use_templates=use_templates,
                priority=idx + 1,
            )
            if plan is not None:
                plans.append(plan)
        
        # Generate global guidance
        global_guidance = self._generate_global_guidance(top_problems, plans)
        
        # Estimate iterations needed
        max_severity = max(p.severity for p in top_problems)
        estimated_iterations = 1 if max_severity < 0.4 else 2 if max_severity < 0.7 else 3
        
        return ImprovementPlan(
            plans=plans,
            task_type=self.config.task_type,
            global_guidance=global_guidance,
            estimated_iterations=estimated_iterations,
        )
    
    def _generate_plan_for_problem(
        self,
        problem: Problem,
        lq_image: torch.Tensor,
        generated_image: torch.Tensor,
        use_templates: bool = True,
        priority: int = 1,
    ) -> Optional[Plan]:
        """Generate an improvement plan for a single problem"""
        
        # First try template-based generation
        if use_templates:
            plan = self._generate_from_template(problem, priority)
            if plan is not None:
                return plan
        
        # Fall back to VLM-based generation
        if self.use_vlm and (self.vlm_client is not None or self.vlm_model is not None):
            plan = self._generate_with_vlm(
                problem, lq_image, generated_image, priority
            )
            if plan is not None:
                return plan
        
        # Last resort: basic plan generation
        return self._generate_basic_plan(problem, priority)

    def _task_description(self, prefix: Optional[str] = None) -> str:
        modality = getattr(self.config, "modality", None)
        input_field = getattr(self.config, "input_field", None)
        target_field = getattr(self.config, "target_field", None)
        modality_text = f" {modality}" if modality else ""
        if input_field and target_field:
            core = f"{input_field}{modality_text} MRI to {target_field} quality"
        elif self.config.task_type == "64mT_to_3T":
            core = f"64mT low-field{modality_text} MRI to 3T quality"
        else:
            core = f"3T{modality_text} MRI to 7T quality"
        if prefix:
            return f"{prefix} {core}"
        return core

    def _scan_context_block(self) -> str:
        lines = []
        modality = getattr(self.config, "modality", None)
        input_field = getattr(self.config, "input_field", None)
        target_field = getattr(self.config, "target_field", None)
        if modality:
            lines.append(f"Sequence modality: {modality}")
        if input_field:
            lines.append(f"Input field strength: {input_field}")
        if target_field:
            lines.append(f"Target field strength: {target_field}")
        return "\n".join(lines)

    def _scan_prompt_prefix(self, quality_descriptor: str = "high quality") -> str:
        modality = getattr(self.config, "modality", None)
        input_field = getattr(self.config, "input_field", None)
        target_field = getattr(self.config, "target_field", None)

        parts = []
        if quality_descriptor:
            if target_field:
                parts.append(f"{quality_descriptor} {target_field}")
            else:
                parts.append(quality_descriptor)
        elif target_field:
            parts.append(target_field)

        if modality:
            parts.append(modality)
        parts.append("MRI")

        prefix = " ".join(part for part in parts if part).strip()
        if input_field:
            return f"{prefix} from {input_field} input"
        return prefix

    def _strip_scan_context(self, text_prompt: str) -> str:
        cleaned = (text_prompt or "").strip()
        if not cleaned:
            return cleaned

        candidate_prefixes = []
        for quality_descriptor in ("high quality", "clinical quality", "high resolution"):
            prefix = self._scan_prompt_prefix(quality_descriptor=quality_descriptor)
            if prefix:
                candidate_prefixes.append(prefix)

        modality = getattr(self.config, "modality", None)
        target_field = getattr(self.config, "target_field", None)
        if target_field and modality:
            candidate_prefixes.extend([
                f"high quality {target_field} {modality} MRI",
                f"clinical quality {target_field} {modality} MRI",
                f"high resolution {target_field} {modality} MRI",
            ])
        if target_field:
            candidate_prefixes.extend([
                f"high quality {target_field} MRI",
                f"clinical quality {target_field} MRI",
                f"high resolution {target_field} MRI",
            ])
        if modality:
            candidate_prefixes.append(f"{modality} MRI")

        for prefix in candidate_prefixes:
            for separator in (": ", ", "):
                token = f"{prefix}{separator}"
                if cleaned.startswith(token):
                    return cleaned[len(token):].strip()
            if cleaned == prefix:
                return ""

        return cleaned

    def _inject_scan_context(self, text_prompt: str) -> str:
        cleaned = (text_prompt or "").strip()
        prefix = self._scan_prompt_prefix(quality_descriptor="high quality")
        if not prefix:
            return cleaned
        if cleaned.startswith(prefix):
            return cleaned

        stripped = self._strip_scan_context(cleaned)
        if not stripped:
            return prefix
        return f"{prefix}, {stripped}"
    
    def _generate_from_template(
        self,
        problem: Problem,
        priority: int,
    ) -> Optional[Plan]:
        """Generate plan using pre-defined templates"""
        
        task_templates = self.IMPROVEMENT_TEMPLATES.get(self.config.task_type, {})
        
        # Find matching template
        issue_lower = problem.issue_description.lower()
        matched_template = None
        
        for issue_key, template in task_templates.items():
            if issue_key in issue_lower or any(word in issue_lower for word in issue_key.split()):
                matched_template = template
                break
        
        if matched_template is None:
            return None
        
        # Get spatial description for the region
        region = self.brain_atlas.get_region(problem.region_name)
        if region:
            region_desc = region.spatial_description
        else:
            region_desc = problem.spatial_description
        
        # Generate text prompt
        text_prompt = self._inject_scan_context(
            matched_template["prompt_template"].format(region=region_desc)
        )
        
        return Plan(
            plan_id=problem.problem_id,
            target_problem=problem,
            action_description=matched_template["action"],
            expected_outcome=matched_template["outcome"],
            text_prompt=text_prompt,
            priority=priority,
        )
    
    def _generate_with_vlm(
        self,
        problem: Problem,
        lq_image: torch.Tensor,
        generated_image: torch.Tensor,
        priority: int,
    ) -> Optional[Plan]:
        """Generate plan using VLM"""
        
        try:
            # Prepare images
            gen_pil = self._tensor_to_pil(generated_image)
            lq_pil = self._tensor_to_pil(lq_image)
            
            # Construct prompt
            prompt = self._construct_plan_prompt(problem)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Low-field MRI input:"},
                        {"type": "image", "image": lq_pil},
                        {"type": "text", "text": "Generated high-field MRI:"},
                        {"type": "image", "image": gen_pil},
                        {"type": "text", "text": prompt},
                    ]
                }
            ]
            
            # Process with VLM
            text = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.vlm_processor(
                text=[text],
                images=[lq_pil, gen_pil],
                return_tensors="pt",
                padding=True,
            ).to(self.vlm_model.device)
            
            with torch.no_grad():
                output_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.config.temperature,
                    do_sample=True,
                )
            
            response = self.vlm_processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0]
            
            # Parse response into plan
            return self._parse_vlm_plan_response(response, problem, priority)
            
        except Exception as e:
            if self.config.verbose:
                print(f"[MedicalVLMAgent] VLM plan generation failed: {e}")
            return None
    
    def _construct_plan_prompt(self, problem: Problem) -> str:
        """Construct prompt for VLM plan generation"""
        
        task_desc = self._task_description(prefix="enhancing")
        context_block = self._scan_context_block()
        
        prompt = f"""You are a medical imaging expert providing guidance for MRI enhancement.

Task: {task_desc}
{context_block if context_block else ''}

A quality issue was detected:
- Region: {problem.spatial_description}
- Issue: {problem.issue_description}
- Severity: {problem.severity:.2f}

Please provide an improvement plan with:
1. Action: What specific enhancement should be applied?
2. Outcome: What should the corrected region look like?
3. Prompt: A short text description that can guide an AI image model to fix this issue.

Format your response as:
Action: [specific action to take]
Outcome: [expected result description]
Prompt: [guidance text for the image model, keep it concise and visual]

Use spatial descriptions (e.g., "deep central region" instead of "hippocampus") in the prompt."""
        
        return prompt
    
    def _parse_vlm_plan_response(
        self,
        response: str,
        problem: Problem,
        priority: int,
    ) -> Optional[Plan]:
        """Parse VLM response into a Plan"""
        
        try:
            lines = response.strip().split('\n')
            
            action = ""
            outcome = ""
            text_prompt = ""
            
            for line in lines:
                line = line.strip()
                if line.lower().startswith('action:'):
                    action = line.split(':', 1)[1].strip()
                elif line.lower().startswith('outcome:'):
                    outcome = line.split(':', 1)[1].strip()
                elif line.lower().startswith('prompt:'):
                    text_prompt = line.split(':', 1)[1].strip()
            
            if not text_prompt:
                return None
            
            return Plan(
                plan_id=problem.problem_id,
                target_problem=problem,
                action_description=action or "Enhance the affected region",
                expected_outcome=outcome or "Improved image quality",
                text_prompt=self._inject_scan_context(text_prompt),
                priority=priority,
            )
            
        except Exception as e:
            if self.config.verbose:
                print(f"[MedicalVLMAgent] Failed to parse VLM response: {e}")
            return None
    
    def _generate_basic_plan(
        self,
        problem: Problem,
        priority: int,
    ) -> Plan:
        """Generate a basic plan without templates or VLM"""
        
        region = self.brain_atlas.get_region(problem.region_name)
        if region:
            region_desc = region.spatial_description
        else:
            region_desc = problem.spatial_description
        
        # Generic text prompt based on task type
        if self.config.task_type == "64mT_to_3T":
            text_prompt = f"clear sharp {region_desc}, enhanced detail and contrast"
        else:
            text_prompt = f"natural texture in {region_desc}, balanced detail"
        text_prompt = self._inject_scan_context(text_prompt)
        
        return Plan(
            plan_id=problem.problem_id,
            target_problem=problem,
            action_description=f"Improve quality in {problem.region_name}",
            expected_outcome=f"Enhanced image quality in {region_desc}",
            text_prompt=text_prompt,
            priority=priority,
        )
    
    def _generate_global_guidance(
        self,
        problems: List[Problem],
        plans: List[Plan],
    ) -> str:
        """Generate overall guidance text"""
        
        if not plans:
            return "No specific improvements needed."
        
        task_type = self.config.task_type
        if task_type == "64mT_to_3T":
            base_guidance = "Enhance overall image quality to match clinical 3T standards."
        else:
            base_guidance = "Refine image quality to match 7T characteristics while avoiding artifacts."
        
        # Collect all regions needing attention
        regions = list(set(p.target_problem.region_name for p in plans))
        region_list = ", ".join(regions[:3])
        if len(regions) > 3:
            region_list += f" and {len(regions) - 3} more regions"
        
        return f"{base_guidance} Focus on: {region_list}."
    
    def generate_combined_prompt(
        self,
        improvement_plan: ImprovementPlan,
        max_prompts: int = 3,
    ) -> str:
        """
        Generate a combined text prompt from multiple plans.
        
        Useful for single-pass refinement with multiple conditions.
        
        Args:
            improvement_plan: The ImprovementPlan to combine
            max_prompts: Maximum number of prompts to combine
            
        Returns:
            Combined text prompt
        """
        prompts = improvement_plan.get_all_prompts()[:max_prompts]
        
        if not prompts:
            if self.config.task_type == "64mT_to_3T":
                return self._inject_scan_context(
                    "clear anatomical details, good contrast"
                )
            else:
                return self._inject_scan_context(
                    "natural texture, balanced enhancement"
                )
        
        prompt_bodies = [self._strip_scan_context(prompt) for prompt in prompts]
        prompt_bodies = [prompt for prompt in prompt_bodies if prompt]
        if not prompt_bodies:
            prompt_bodies = ["clear anatomical details, good contrast"]

        if self.config.task_type == "64mT_to_3T":
            prefix = self._scan_prompt_prefix(quality_descriptor="clinical quality")
        else:
            prefix = self._scan_prompt_prefix(quality_descriptor="high resolution")

        return f"{prefix}: {', '.join(prompt_bodies)}"

    def detect_and_plan(
        self,
        lq_image: torch.Tensor,
        generated_image: torch.Tensor,
        gt_image: Optional[torch.Tensor] = None,
        use_templates: bool = True,
    ) -> Tuple[List[Problem], List[Plan]]:
        """
        Unified detection and planning in a single VLM call.
        
        Instead of calling AnomalyDetectionAgent then MedicalVLMAgent separately,
        this method makes ONE VLM call that identifies problems and generates
        improvement plans simultaneously.
        
        Args:
            lq_image: Low-field MRI input
            generated_image: Generated high-field MRI from Stage 1
            gt_image: Ground truth (optional)
            use_templates: Whether to use templates as fallback
            
        Returns:
            Tuple of (problems, plans)
        """
        self._ensure_initialized()
        
        problems = []
        plans = []
        
        # Try unified VLM call first
        # (suppress per-call status to reduce log noise during training)
        if self.use_vlm and (self.vlm_client is not None or self.vlm_model is not None):
            problems, plans = self._unified_vlm_call(
                lq_image, generated_image, gt_image
            )
        else:
            pass  # no VLM client available; use template fallback
        
        # For any problem without a plan, fill in from templates or basic generation
        if problems and len(plans) < len(problems):
            planned_ids = {p.plan_id for p in plans}
            for idx, problem in enumerate(problems):
                if problem.problem_id not in planned_ids:
                    plan = self._generate_plan_for_problem(
                        problem=problem,
                        lq_image=lq_image,
                        generated_image=generated_image,
                        use_templates=use_templates,
                        priority=idx + 1,
                    )
                    if plan is not None:
                        plans.append(plan)
        
        return problems, plans

    def detect_and_plan_from_lq(
        self,
        lq_image: torch.Tensor,
    ) -> Tuple[List[Problem], List[Plan]]:
        """
        Predict likely hallucination/artifact risks from low-field MRI only.

        This path is used when Stage 2 should not inspect Stage-1 generated
        images or ground truth. The VLM sees multiple LQ slices, anticipates
        regions where the generator may hallucinate or over-correct, and emits
        plans whose prompts can be merged with the Stage-1 task prompt.
        """
        self._ensure_initialized()

        if not (self.use_vlm and self.vlm_client is not None):
            return [], []

        problems, plans = self._lq_only_vlm_call(lq_image)
        plans = sorted(plans, key=lambda p: p.target_problem.severity, reverse=True)[:3]
        for i, plan in enumerate(plans):
            plan.priority = i + 1
        return problems, plans

    def _unified_vlm_call(
        self,
        lq_image: torch.Tensor,
        generated_image: torch.Tensor,
        gt_image: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Problem], List[Plan]]:
        """Single VLM call that returns both problems and plans.
        
        Uses the vLLM OpenAI-compatible API for fast inference.
        Sends 5 uniformly-spaced slices + LQ center slice.
        """
        problems = []
        plans = []
        
        if self.vlm_client is None:
            return problems, plans
        
        try:
            gen_pils = self._tensor_to_5_pil_slices(generated_image)
            lq_pil = self._tensor_to_pil(lq_image)
            
            prompt = self._construct_unified_prompt()
            
            # Build OpenAI-compatible message content with base64 images
            content = [
                {"type": "text", "text": "Low-field MRI input (center slice for reference):"},
                {"type": "image_url", "image_url": {"url": self._pil_to_data_url(lq_pil)}},
                {"type": "text", "text": "Generated high-field MRI (5 axial slices spanning the volume, from superior to inferior):"},
            ]
            for pil_img in gen_pils:
                content.append({"type": "image_url", "image_url": {"url": self._pil_to_data_url(pil_img)}})
            content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "user", "content": content}]
            
            # Call vLLM via OpenAI API (non-thinking mode). Local vLLM can
            # occasionally return a transient 502 immediately after startup.
            last_error = None
            for attempt in range(3):
                try:
                    chat_response = self.vlm_client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        max_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=0.95,
                        presence_penalty=1.5,
                        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    status_code = getattr(exc, "status_code", None)
                    if status_code != 502 or attempt == 2:
                        raise
                    time.sleep(2.0 * (attempt + 1))
            else:
                raise last_error
            
            msg = chat_response.choices[0].message
            response = msg.content or ""
            
            problems, plans = self._parse_unified_response(response)
            
        except Exception as e:
            import traceback
            print(f"[MedicalVLMAgent] Unified VLM call failed: {e}")
            traceback.print_exc()
        
        return problems, plans

    def _lq_only_vlm_call(
        self,
        lq_image: torch.Tensor,
    ) -> Tuple[List[Problem], List[Plan]]:
        """Single VLM call for low-field-only hallucination-risk planning."""
        problems = []
        plans = []

        try:
            lq_pils = self._tensor_to_5_pil_slices(lq_image)
            prompt = self._construct_lq_only_prompt()

            content = [
                {
                    "type": "text",
                    "text": (
                        "Low-field MRI input (5 axial slices spanning the "
                        "sub-volume, from superior to inferior):"
                    ),
                },
            ]
            for pil_img in lq_pils:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._pil_to_data_url(pil_img)},
                    }
                )
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            last_error = None
            for attempt in range(3):
                try:
                    chat_response = self.vlm_client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        max_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=0.95,
                        presence_penalty=1.5,
                        extra_body={
                            "top_k": 20,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    status_code = getattr(exc, "status_code", None)
                    if status_code != 502 or attempt == 2:
                        raise
                    time.sleep(2.0 * (attempt + 1))
            else:
                raise last_error

            msg = chat_response.choices[0].message
            response = msg.content or ""
            problems, plans = self._parse_unified_response(response)

        except Exception as e:
            raise RuntimeError("Low-field visual planning failed") from e

        return problems, plans

    @staticmethod
    def _pil_to_data_url(pil_img: Image.Image) -> str:
        """Convert a PIL image to a base64 data URL for the OpenAI API."""
        import base64, io
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _construct_unified_prompt(self) -> str:
        """Construct a prompt that asks for both problem detection and plan generation.
        
        The prompt uses a two-phase approach:
        Phase 1 — Region identification using anatomical knowledge.
        Phase 2 — Quality defect analysis and actionable plans per region.
        """
        task_desc = self._task_description(prefix=None)
        context_block = self._scan_context_block()

        prompt = f"""You are a neuroradiology expert for MRI quality control and post-processing planning.

Task: {task_desc}
{context_block if context_block else ''}

You are given:
- low-field reference MRI (center slice)
- generated target-field MRI (5 axial slices)

Core requirement:
1) Judge field-strength mismatch between generated image and target field quality (3T or 7T).
2) Identify actionable errors.
3) Select executable refinement plans for an image-to-image diffusion model.

First answer exactly one sentence:
Where is the mismatch between your result and the target field strength?

═══════════════════════════════
PHASE 1 — REGION IDENTIFICATION
═══════════════════════════════
Identify visible regions in each slice using ONLY these 7 names:
frontal_lobe, parietal_lobe, temporal_lobe, occipital_lobe, insular_lobe, ventricles, white_matter

Format:
Slice 1: <region names>
Slice 2: ...
...
Slice 5: ...

═══════════════════════════════
PHASE 2 — FIELD-STRENGTH ERROR ANALYSIS
═══════════════════════════════
Use the issue library below. Choose the best matching issue ID(s).

ISSUE_LIBRARY:
E01 residual low-field blur / insufficient high-field sharpness
E02 over-smoothing with cortical sulci/gyri detail loss
E03 gray-white matter contrast mismatch (too low or too high)
E04 ventricular / CSF signal or boundary mismatch
E05 skull/scalp bright rim halo or ringing near outer boundary
E06 checkerboard / aliasing / grid-like artifact
E07 noise level mismatch (over-noisy or over-denoised/plastic)
E08 low-frequency intensity inhomogeneity (bias-field shading)
E09 susceptibility-like distortion around frontal-temporal base
E10 geometric/anatomical shape inconsistency or asymmetry
E11 lesion conspicuity loss or false lesion-like enhancement
E12 over-correction/hallucinated texture not clinically plausible

═══════════════════════════════
PHASE 3 — PLAN SELECTION (MAX 12, CHOOSE 1-3 PER ISSUE)
═══════════════════════════════
Choose ONLY from PLAN_LIBRARY. Plans must be concrete and directly executable by a diffusion refinement model.

PLAN_LIBRARY:
P01 Edge de-ringing: suppress boundary ringing/halo along skull and tissue interfaces.
P02 Detail-preserving denoise: reduce random noise while preserving cortical edges.
P03 GM/WM contrast calibration: adjust tissue contrast toward target-field appearance.
P04 Cortical detail recovery: restore sulci/gyri and fine cortical texture without oversharpening.
P05 Bias-field flattening: correct low-frequency intensity shading and improve uniformity.
P06 De-grid/anti-alias: remove checkerboard and interpolation artifacts.
P07 Ventricle/CSF refinement: stabilize CSF intensity and sharpen ventricular margins.
P08 Susceptibility artifact suppression: reduce frontal/temporal-base distortion and signal drop.
P09 Deep gray nuclei enhancement: improve thalamus/basal ganglia boundary visibility.
P10 Geometry stabilization: correct asymmetric deformation while preserving anatomy.
P11 Lesion-preserving enhancement: maintain lesion visibility, avoid lesion hallucination/erasure.
P12 Conservative local refinement: restrict edits to target region; minimize global drift.

Hard constraints:
- Report at most 3 issues, ordered by severity.
- Region must be one of the 7 valid names.
- Plans must be selected from P01-P12 only.
- No scanner/protocol advice (no "re-scan", no "change coil").

Output format (must keep these keys):
FieldStrengthError: <one sentence answering where your result mismatches the target field strength>

[Issue 1]
Region: <one of 7 regions>
Issue: <E## + concise description>
Severity: <0.0-1.0>
Action: <Selected Plan IDs, e.g., P03+P04; then concrete operations>
Outcome: <clinically relevant expected improvement>
Prompt: <=20 words, directly usable for cross-attention guidance>

[Issue 2]
...

If no significant issues, respond exactly: No issues detected."""
        
        return prompt

    def _construct_lq_only_prompt(self) -> str:
        """Construct a prompt for low-field-only hallucination-risk planning."""
        task_desc = self._task_description(prefix=None)
        context_block = self._scan_context_block()

        prompt = f"""You are a neuroradiology expert planning conservative MRI enhancement.

Task: {task_desc}
{context_block if context_block else ''}

You are given only the low-field MRI input slices. You have NOT seen any
generated high-field image and you have NOT seen ground truth.

Goal:
Predict where a low-to-high-field image generator is most likely to hallucinate,
over-sharpen, erase subtle anatomy, or create implausible target-field texture.
Then write a concise plan that can be merged with the base generation prompt
for a second generation pass.

Use only visible low-field evidence:
- low SNR or noisy regions
- weak gray-white contrast
- blurred sulci/gyri or ventricular margins
- susceptibility-prone frontal/temporal-base regions
- bias-field shading or intensity instability
- ambiguous lesion-like or asymmetric structures that should be preserved

Valid regions, use ONLY these 7 names:
frontal_lobe, parietal_lobe, temporal_lobe, occipital_lobe, insular_lobe, ventricles, white_matter

ISSUE_LIBRARY:
E01 likely residual low-field blur / insufficient target-field sharpness
E02 risk of over-smoothing with cortical sulci/gyri detail loss
E03 risk of gray-white matter contrast mismatch
E04 risk of ventricular / CSF signal or boundary mismatch
E05 risk of skull/scalp rim halo or ringing near outer boundary
E06 risk of checkerboard / aliasing / grid-like artifact amplification
E07 risk of noise hallucination or over-denoised plastic texture
E08 risk of low-frequency intensity inhomogeneity
E09 risk of susceptibility-like distortion around frontal-temporal base
E10 risk of geometric/anatomical shape inconsistency or asymmetry
E11 risk of lesion conspicuity loss or false lesion-like enhancement
E12 risk of over-correction/hallucinated texture not clinically plausible

PLAN_LIBRARY:
P01 Edge de-ringing: suppress boundary ringing/halo along skull and tissue interfaces.
P02 Detail-preserving denoise: reduce random noise while preserving cortical edges.
P03 GM/WM contrast calibration: adjust tissue contrast toward target-field appearance.
P04 Cortical detail recovery: restore sulci/gyri and fine cortical texture without oversharpening.
P05 Bias-field flattening: correct low-frequency intensity shading and improve uniformity.
P06 De-grid/anti-alias: remove checkerboard and interpolation artifacts.
P07 Ventricle/CSF refinement: stabilize CSF intensity and sharpen ventricular margins.
P08 Susceptibility artifact suppression: reduce frontal/temporal-base distortion and signal drop.
P09 Deep gray nuclei enhancement: improve thalamus/basal ganglia boundary visibility.
P10 Geometry stabilization: correct asymmetric deformation while preserving anatomy.
P11 Lesion-preserving enhancement: maintain lesion visibility, avoid lesion hallucination/erasure.
P12 Conservative local refinement: restrict edits to target region; minimize global drift.

Hard constraints:
- This is a pre-generation risk prediction, not a generated-image critique.
- Report at most 3 risks, ordered by severity.
- Region must be one of the 7 valid names.
- Plans must be selected from P01-P12 only.
- No scanner/protocol advice.
- Prompt must be suitable for cross-attention guidance in a second generation pass.

Output format (must keep these keys):
FieldStrengthError: <one sentence predicting the main likely hallucination or target-field mismatch>

[Issue 1]
Region: <one of 7 regions>
Issue: <E## + concise predicted risk>
Severity: <0.0-1.0>
Action: <Selected Plan IDs, e.g., P03+P04; then concrete operations>
Outcome: <clinically relevant expected improvement>
Prompt: <=20 words, directly usable for cross-attention guidance>

[Issue 2]
...

If no meaningful risk is visible, respond exactly: No issues detected."""

        return prompt

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3.5-style <think>...</think> block from a response."""
        import re
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

    def _parse_unified_response(self, response: str) -> Tuple[List[Problem], List[Plan]]:
        """Parse a unified VLM response into problems and plans.
        
        Handles multiple VLM output formats robustly by scanning line-by-line
        for Region:/Issue:/Severity:/Action:/Outcome:/Prompt: fields and
        flushing a problem+plan every time a new Region: line appears.
        """
        problems = []
        plans = []
        
        # Only treat as "no issues" if the response is very short and
        # explicitly says so — avoid false positive from Phase 1 text.
        response_stripped = response.strip()
        if len(response_stripped) < 100 and (
            "no issues detected" in response_stripped.lower()
            or "no significant issues" in response_stripped.lower()
            or "no problems detected" in response_stripped.lower()
        ):
            return problems, plans
        
        import re
        
        # Strip markdown bold/italic markers: **text** → text, *text* → text
        cleaned = re.sub(r'\*{1,2}', '', response)
        
        KNOWN_FIELDS = {'region', 'issue', 'severity', 'action', 'outcome', 'prompt'}
        problem_id = 200
        pending: Dict[str, str] = {}
        
        for line in cleaned.split('\n'):
            line = line.strip()
            # Strip leading numbering: "1." "2)" "- " etc.
            line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•]\s*', '', line)
            
            if ':' not in line:
                continue
            
            key, _, value = line.partition(':')
            key_clean = key.strip().lower()
            value_clean = value.strip()
            
            if key_clean not in KNOWN_FIELDS:
                continue
            
            # When we see a new "region", flush the previous pending block
            if key_clean == 'region' and pending.get('region') and pending.get('issue'):
                self._flush_pending(pending, problems, plans, problem_id)
                problem_id += 1
                pending = {}
            
            pending[key_clean] = value_clean
        
        # Flush the last pending block
        if pending.get('region') and pending.get('issue'):
            self._flush_pending(pending, problems, plans, problem_id)
        
        return problems, plans

    def _flush_pending(
        self,
        fields: Dict[str, str],
        problems: List[Problem],
        plans: List[Plan],
        problem_id: int,
    ) -> None:
        """Convert a dict of parsed fields into a Problem + Plan."""
        import re
        region_text = fields['region']
        issue_text = fields['issue']
        
        try:
            sev_str = re.search(r'[\d.]+', fields.get('severity', '0.5'))
            severity = float(sev_str.group()) if sev_str else 0.5
            severity = max(0.0, min(1.0, severity))
        except (ValueError, AttributeError):
            severity = 0.5
        
        region_name = self._map_description_to_region(region_text)
        if region_name is None:
            if self.config.verbose:
                print(f"Skipping non-valid region: {region_text!r}")
            return

        region = self.brain_atlas.get_region(region_name)
        spatial_desc = region.chinese_spatial_description if region else region_text
        
        problem = Problem(
            problem_id=problem_id,
            region_name=region_name,
            spatial_description=spatial_desc,
            issue_description=issue_text,
            severity=severity,
        )
        problems.append(problem)
        
        text_prompt = fields.get('prompt', '')
        action = fields.get('action', '')
        outcome = fields.get('outcome', '')
        
        if text_prompt:
            plan = Plan(
                plan_id=problem_id,
                target_problem=problem,
                action_description=action or "Enhance the affected region",
                expected_outcome=outcome or "Improved image quality",
                text_prompt=self._inject_scan_context(text_prompt),
                priority=len(plans) + 1,
            )
            plans.append(plan)

    # The 7 valid Harvard-Oxford regions for mask generation
    VALID_REGIONS = [
        "frontal_lobe", "parietal_lobe", "temporal_lobe",
        "occipital_lobe", "insular_lobe", "ventricles", "white_matter",
    ]

    def _map_description_to_region(self, description: str) -> Optional[str]:
        """Return a valid atlas region only when the VLM emitted a canonical region token."""
        import re

        desc_lower = str(description or "").strip().lower()
        normalized = re.sub(r"[^a-z0-9_]+", "_", desc_lower.replace("-", "_").replace(" ", "_")).strip("_")
        if normalized in self.VALID_REGIONS:
            return normalized

        for region in self.VALID_REGIONS:
            if (
                normalized.startswith(f"{region}_")
                or normalized.endswith(f"_{region}")
                or f"_{region}_" in normalized
            ):
                return region

        return None
    
    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image (center slice)"""
        if tensor.dim() == 5:
            B, C, F, H, W = tensor.shape
            frame_idx = F // 2
            img = tensor[0, :, frame_idx]
        elif tensor.dim() == 4:
            img = tensor[0]
        else:
            img = tensor
        
        img = (img.detach().cpu().float() + 1.0) / 2.0 * 255.0
        img = img.clamp(0, 255).byte()
        
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        
        img_np = img.permute(1, 2, 0).numpy()
        return Image.fromarray(img_np)

    @staticmethod
    def _tensor_to_5_pil_slices(tensor: torch.Tensor) -> list:
        """Convert tensor to 5 uniformly-spaced PIL slices for richer 3D context."""
        N = 5
        fracs = [i / (N + 1) for i in range(1, N + 1)]  # 1/6, 2/6, ..., 5/6
        if tensor.dim() == 5:
            F_total = tensor.shape[2]
            indices = [max(0, min(F_total - 1, round(f * F_total))) for f in fracs]
            # Batch the GPU→CPU transfer: stack all slices and do a single .cpu()
            stacked = torch.stack([tensor[0, :, idx] for idx in indices], dim=0)  # (N, C, H, W)
        elif tensor.dim() == 4:
            stacked = tensor[0].unsqueeze(0).expand(N, -1, -1, -1)
        else:
            stacked = tensor.unsqueeze(0).expand(N, -1, -1, -1)
        # Single GPU→CPU transfer + conversion
        stacked_cpu = ((stacked.detach().cpu().float() + 1.0) / 2.0 * 255.0).clamp(0, 255).byte()
        pils = []
        for i in range(N):
            s = stacked_cpu[i]
            if s.shape[0] == 1:
                s = s.repeat(3, 1, 1)
            pils.append(Image.fromarray(s.permute(1, 2, 0).numpy()))
        return pils
