"""
Anomaly Detection Agent for MRI Enhancement
============================================

Stage 2 Component: Identifies problematic regions in generated MRI images.

Input: Low-field MRI, Generated high-field MRI, (optional) Ground truth
Output: Structured list of problems with spatial descriptions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from PIL import Image

from .base_agent import BaseAgent, AgentConfig, AgentType, Problem
from .brain_atlas import BrainAtlas, BRAIN_ATLAS


@dataclass
class AnomalyReport:
    """Complete anomaly detection report"""
    problems: List[Problem]
    global_quality_score: float  # 0-1, higher is better
    needs_refinement: bool
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        top3 = sorted(self.problems, key=lambda p: p.severity, reverse=True)[:3]
        return {
            "problems": [p.to_dict() for p in top3],
            "global_quality_score": self.global_quality_score,
            "needs_refinement": self.needs_refinement,
            "summary": self.summary,
        }
    
    def __len__(self) -> int:
        return len(self.problems)
    
    def __bool__(self) -> bool:
        return len(self.problems) > 0


class AnomalyDetectionAgent(BaseAgent):
    """
    Agent for detecting anomalies and quality issues in generated MRI images.
    
    This agent can operate in two modes:
    1. VLM-based: Uses a Vision-Language Model (e.g., Qwen2.5-VL) for detection
    2. Metric-based: Uses image quality metrics (SSIM, PSNR, edge detection) for detection
    
    The agent outputs a structured list of problems, each with:
    - Region identification (using BrainAtlas spatial descriptions)
    - Issue description
    - Severity score
    - Optional mask for the affected region
    """
    
    def __init__(
        self,
        config: AgentConfig,
        use_vlm: bool = True,
        use_metrics: bool = True,
        vlm_client=None,
        # Legacy args (ignored when using vLLM)
        vlm_model=None,
        vlm_processor=None,
    ):
        super().__init__(config)
        self.use_vlm = use_vlm
        self.use_metrics = use_metrics
        self.brain_atlas = BRAIN_ATLAS
        
        # Thresholds for metric-based detection
        self.ssim_threshold = 0.85
        self.psnr_threshold = 25.0
        self.edge_threshold = 0.3
        
        # OpenAI-compatible client (backed by vLLM server)
        self.vlm_client = vlm_client
        # Legacy: kept for fallback / non-vLLM mode
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        
    @property
    def agent_type(self) -> AgentType:
        return AgentType.ANOMALY_DETECTION
    
    def initialize(self) -> None:
        """Initialize the agent, loading VLM if needed"""
        if self.use_vlm and self.vlm_client is None and self.vlm_model is None:
            self._load_vlm()
        self._is_initialized = True
    
    def _load_vlm(self) -> None:
        """Load the Vision-Language Model for anomaly detection"""
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            
            print(f"Loading VLM model: {self.config.model_name}")
            
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
            
            print("VLM model loaded successfully")
            
        except Exception as e:
            print(f"Warning: Failed to load VLM model: {e}")
            print("Falling back to metric-based detection only")
            self.use_vlm = False
    
    def process(
        self,
        lq_image: torch.Tensor,
        generated_image: torch.Tensor,
        gt_image: Optional[torch.Tensor] = None,
        return_masks: bool = False,
    ) -> AnomalyReport:
        """
        Detect anomalies in the generated MRI image.
        
        Args:
            lq_image: Low-field MRI input (B, C, F, H, W) or (B, C, H, W)
            generated_image: Generated high-field MRI from Stage 1
            gt_image: Ground truth high-field MRI (optional)
            return_masks: Whether to include region masks in the output
            
        Returns:
            AnomalyReport containing detected problems
        """
        self._ensure_initialized()
        
        problems = []
        
        # Metric-based detection (always run if gt_image is available)
        if self.use_metrics and gt_image is not None:
            metric_problems = self._detect_with_metrics(
                generated_image, gt_image, return_masks
            )
            problems.extend(metric_problems)
        
        # VLM-based detection
        if self.use_vlm and (self.vlm_client is not None or self.vlm_model is not None):
            vlm_problems = self._detect_with_vlm(
                lq_image, generated_image, gt_image
            )
            problems.extend(vlm_problems)
        
        # Remove duplicates and merge similar problems
        problems = self._merge_problems(problems)
        
        # Calculate global quality score
        global_score = self._calculate_global_score(generated_image, gt_image)
        
        # Determine if refinement is needed
        needs_refinement = len(problems) > 0 and any(p.severity > 0.3 for p in problems)
        
        # Generate summary
        summary = self._generate_summary(problems)
        
        return AnomalyReport(
            problems=problems,
            global_quality_score=global_score,
            needs_refinement=needs_refinement,
            summary=summary,
        )
    
    def _detect_with_metrics(
        self,
        generated: torch.Tensor,
        gt: torch.Tensor,
        return_masks: bool = False,
    ) -> List[Problem]:
        """
        Detect anomalies using image quality metrics.
        
        Analyzes regional SSIM, edge preservation, and contrast.
        """
        problems = []
        problem_id = 0
        
        # Ensure same shape
        if generated.shape != gt.shape:
            min_frames = min(generated.shape[2], gt.shape[2])
            generated = generated[:, :, :min_frames]
            gt = gt[:, :, :min_frames]
        
        # Convert to numpy for metric computation
        gen_np = generated.detach().cpu().float().numpy()
        gt_np = gt.detach().cpu().float().numpy()
        
        # Normalize to [0, 1]
        gen_np = (gen_np + 1.0) / 2.0
        gt_np = (gt_np + 1.0) / 2.0
        
        # Get image dimensions
        B, C, F, H, W = gen_np.shape if gen_np.ndim == 5 else (1, *gen_np.shape)
        
        # Define regions to analyze — restricted to the 7 valid regions.
        # Coordinates are (y1, y2, x1, x2) in normalised [0, 1] space.
        regions_to_check = [
            ("frontal_lobe",   (0.05, 0.45, 0.15, 0.85)),  # anterior
            ("parietal_lobe",  (0.10, 0.60, 0.20, 0.80)),  # superior-posterior
            ("temporal_lobe",  (0.30, 0.70, 0.10, 0.50)),  # lateral-inferior
            ("occipital_lobe", (0.65, 0.95, 0.25, 0.75)),  # posterior
            ("insular_lobe",   (0.35, 0.65, 0.35, 0.65)),  # deep lateral
            ("ventricles",     (0.35, 0.55, 0.40, 0.60)),  # central
            ("white_matter",   (0.25, 0.75, 0.25, 0.75)),  # inner
        ]
        
        for region_name, (y1, y2, x1, x2) in regions_to_check:
            # Extract region
            y1_px, y2_px = int(y1 * H), int(y2 * H)
            x1_px, x2_px = int(x1 * W), int(x2 * W)
            
            if gen_np.ndim == 5:
                gen_region = gen_np[:, :, :, y1_px:y2_px, x1_px:x2_px]
                gt_region = gt_np[:, :, :, y1_px:y2_px, x1_px:x2_px]
            else:
                gen_region = gen_np[:, :, y1_px:y2_px, x1_px:x2_px]
                gt_region = gt_np[:, :, y1_px:y2_px, x1_px:x2_px]
            
            # Compute regional metrics
            regional_ssim = self._compute_ssim(gen_region, gt_region)
            regional_psnr = self._compute_psnr(gen_region, gt_region)
            edge_score = self._compute_edge_similarity(gen_region, gt_region)
            
            # Check if region has issues
            issues = []
            severity = 0.0
            
            if regional_ssim < self.ssim_threshold:
                issues.append("structural distortion")
                severity = max(severity, 1.0 - regional_ssim)
            
            if regional_psnr < self.psnr_threshold:
                issues.append("intensity mismatch")
                severity = max(severity, 1.0 - regional_psnr / 40.0)
            
            if edge_score < self.edge_threshold:
                issues.append("blurry edges")
                severity = max(severity, 1.0 - edge_score)
            
            if issues:
                region = self.brain_atlas.get_region(region_name)
                spatial_desc = region.chinese_spatial_description if region else f"{region_name}区域"
                
                # Use 3D atlas mask; no bbox
                if gen_np.ndim == 5:
                    _, _, D, H_full, W_full = generated.shape
                    vol_shape = (D, H_full, W_full)
                else:
                    vol_shape = (H, W)
                atlas_mask = self.brain_atlas.generate_region_mask(
                    region_name, vol_shape, soft_boundary=True
                ) if return_masks else None
                
                problem = Problem(
                    problem_id=problem_id,
                    region_name=region_name,
                    spatial_description=spatial_desc,
                    issue_description=", ".join(issues),
                    severity=min(severity, 1.0),
                    bbox=None,
                    mask=atlas_mask,
                )
                problems.append(problem)
                problem_id += 1
        
        return problems
    
    @staticmethod
    def _pil_to_data_url(pil_img) -> str:
        """Convert a PIL image to a base64 data URL for the OpenAI API."""
        import base64, io
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _detect_with_vlm(
        self,
        lq_image: torch.Tensor,
        generated_image: torch.Tensor,
        gt_image: Optional[torch.Tensor] = None,
    ) -> List[Problem]:
        """
        Detect anomalies using Vision-Language Model.
        
        Uses the vLLM OpenAI-compatible API for fast inference.
        Sends 5 axial slices of the generated image (plus one representative LQ
        slice for context) with a structured prompt.
        After parsing, atlas-based 3D masks are generated for each detected region.
        """
        if self.vlm_client is None and self.vlm_model is None:
            return []
        
        problems = []
        
        try:
            # --- Prepare 5 slices for the generated image ---
            gen_slices = self._tensor_to_5_pil_slices(generated_image)
            lq_rep = self._tensor_to_pil(lq_image)  # single representative for context
            
            # --- Construct prompt ---
            prompt = self._construct_vlm_prompt(has_gt=gt_image is not None)
            
            # --- Build OpenAI-compatible message with base64 images ---
            content = [
                {"type": "text", "text": "Reference low-field MRI (center slice):"},
                {"type": "image_url", "image_url": {"url": self._pil_to_data_url(lq_rep)}},
                {"type": "text", "text": "Generated high-field MRI — 5 axial slices (superior → inferior):"},
            ]
            for i, sl in enumerate(gen_slices):
                content.append({"type": "text", "text": f"Slice {i+1}/5:"})
                content.append({"type": "image_url", "image_url": {"url": self._pil_to_data_url(sl)}})
            content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "user", "content": content}]
            
            # --- Call vLLM via OpenAI API (non-thinking mode) ---
            chat_response = self.vlm_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=0.95,
                presence_penalty=1.5,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            )
            
            msg = chat_response.choices[0].message
            response = msg.content or ""
            
            # Parse VLM response into problems (no bbox, no mask yet)
            problems = self._parse_vlm_response(response)
            
            # --- Assign 3D atlas masks based on the 3D volume shape ---
            if generated_image.dim() == 5:
                _, _, D, H, W = generated_image.shape
                vol_shape = (D, H, W)
            else:
                _, _, H, W = generated_image.shape
                vol_shape = (H, W)
            
            for p in problems:
                p.mask = self.brain_atlas.generate_region_mask(
                    p.region_name, vol_shape, soft_boundary=True
                )
            
        except Exception as e:
            if self.config.verbose:
                print(f"VLM detection failed: {e}")
            import traceback
            traceback.print_exc()
        
        return problems
    
    def _construct_vlm_prompt(self, has_gt: bool = False) -> str:
        """Construct the prompt for VLM anomaly detection.
        
        Uses a two-phase approach:
        Phase 1 — Identify brain regions visible in each slice using anatomical knowledge.
        Phase 2 — Report quality issues per region.
        """
        
        task_type = self.config.task_type
        modality = getattr(self.config, "modality", None) or "MRI"
        input_field = getattr(self.config, "input_field", None)
        target_field = getattr(self.config, "target_field", None)
        if input_field and target_field:
            task_desc = f"{input_field} {modality} to {target_field} MRI enhancement"
        elif task_type == "64mT_to_3T":
            task_desc = f"64mT low-field to 3T high-field {modality} enhancement"
        else:
            task_desc = f"3T to 7T {modality} enhancement"

        context_lines = []
        if modality:
            context_lines.append(f"Sequence modality: {modality}")
        if input_field:
            context_lines.append(f"Input field strength: {input_field}")
        if target_field:
            context_lines.append(f"Target field strength: {target_field}")
        context_block = "\n".join(context_lines)
        
        prompt = f"""You are a neuroradiology expert analyzing MRI quality.

Task: {task_desc}
{context_block if context_block else ''}

You are shown 5 axial slices uniformly sampled from a randomly-cropped sub-volume
of the generated high-field MRI. The absolute depth of each slice is UNKNOWN —
you must rely on visible anatomical landmarks to determine which brain regions
are present.

═══════════════════════════════
PHASE 1 — REGION IDENTIFICATION
═══════════════════════════════
Using your neuroradiology knowledge, examine the anatomical structures visible
in each slice and identify which of the 7 regions are present.
Region names (use ONLY these 7):
  frontal_lobe, parietal_lobe, temporal_lobe, occipital_lobe, insular_lobe, ventricles, white_matter

Slice 1: <visible regions>
Slice 2: ...
...
Slice 5: ...

═══════════════════════════════
PHASE 2 — QUALITY ISSUE DETECTION
═══════════════════════════════
Now report quality issues. Use your Phase 1 identification to assign the CORRECT region.

IMPORTANT RULES:
1. Region MUST be exactly one of:
   frontal_lobe, parietal_lobe, temporal_lobe, occipital_lobe,
   insular_lobe, ventricles, white_matter

2. For each problem specify spatial orientation:
   - Laterality: left / right / bilateral
   - Position (optional): anterior / posterior / superior / inferior / central

Format:
Problem 1: [region_name] | [orientation] - [issue description]
Problem 2: [region_name] | [orientation] - [issue description]
...

Example:
Problem 1: frontal_lobe | left anterior - blurry gray-white matter boundary
Problem 2: ventricles | bilateral central - irregular CSF signal intensity

If no significant problems, respond: No problems detected.

Focus on: edge sharpness, gray-white contrast, artifacts, texture, anatomical accuracy."""
        
        return prompt
    
    def _parse_vlm_response(self, response: str) -> List[Problem]:
        """Parse VLM response into structured problems"""
        problems = []
        
        if "no problem" in response.lower() or "no significant" in response.lower():
            return problems
        
        lines = response.strip().split('\n')
        problem_id = 100  # Start from 100 to differentiate from metric-based problems
        
        for line in lines:
            line = line.strip()
            if not line or not line.lower().startswith('problem'):
                continue
            
            try:
                # Parse "Problem N: [region] | [orientation] - [issue]"
                parts = line.split(':', 1)
                if len(parts) < 2:
                    continue
                
                content = parts[1].strip()
                # Split issue from region+orientation
                if ' - ' in content:
                    region_orient, issue_part = content.split(' - ', 1)
                else:
                    region_orient = content
                    issue_part = "quality issue detected"
                
                # Split region name from orientation
                if ' | ' in region_orient:
                    region_part, orientation = region_orient.split(' | ', 1)
                    orientation = orientation.strip()
                else:
                    region_part = region_orient
                    orientation = ""
                
                # Validate/map to the 7-region set
                region_name = self._map_description_to_region(region_part.strip())
                if not self.brain_atlas.is_valid_region(region_name):
                    if self.config.verbose:
                        print(f"Skipping non-valid region: {region_part!r}")
                    continue
                region = self.brain_atlas.get_region(region_name)
                base_desc = region.chinese_spatial_description if region else region_part
                
                # Embed orientation into the spatial description
                if orientation:
                    spatial_desc = f"{orientation} · {base_desc}"
                else:
                    spatial_desc = base_desc
                
                problem = Problem(
                    problem_id=problem_id,
                    region_name=region_name,
                    spatial_description=spatial_desc,
                    issue_description=issue_part.strip(),
                    severity=0.5,
                    bbox=None,  # No bbox; mask comes from atlas
                )
                problems.append(problem)
                problem_id += 1
                
            except Exception as e:
                if self.config.verbose:
                    print(f"Failed to parse line: {line}, error: {e}")
                continue
        
        return problems
    
    def _map_description_to_region(self, description: str) -> str:
        """Map a region name or description to one of the 7 valid region names.
        
        Accepts exact keys or keyword fallback.
        """
        desc_lower = description.lower().strip()
        
        # Direct match
        if self.brain_atlas.is_valid_region(desc_lower):
            return desc_lower
        
        # Keyword fallback → one of the 7 canonical names
        mappings = [
            (['frontal', '额叶', '额极', 'prefrontal', 'premotor', 'motor cortex',
              'precentral', 'orbital', 'cingulate', 'paracingulate'], 'frontal_lobe'),
            (['parietal', '顶叶', 'postcentral', 'supramarginal', 'angular',
              'precuneus', 'superior parietal', 'inferior parietal'], 'parietal_lobe'),
            (['temporal', '颞叶', 'hippocampus', '海马', 'amygdala', '杏仁',
              'heschl', 'planum', 'fusiform', 'parahippocampal'], 'temporal_lobe'),
            (['occipital', '枕叶', 'calcarine', 'cuneus', 'lingual',
              'visual cortex', '枕极'], 'occipital_lobe'),
            (['insular', '脑岛', 'insula', 'claustrum'], 'insular_lobe'),
            (['ventricle', '脑室', 'csf', 'choroid', 'lateral ventricle',
              'third ventricle', 'fourth ventricle'], 'ventricles'),
            (['white matter', '白质', 'corpus callosum', '胼胝体',
              'corona radiata', 'internal capsule'], 'white_matter'),
        ]
        
        for keywords, region in mappings:
            if any(kw in desc_lower for kw in keywords):
                return region
        
        return "unknown_region"
    
    def _merge_problems(self, problems: List[Problem]) -> List[Problem]:
        """Merge or deduplicate similar problems"""
        if len(problems) <= 1:
            return problems
        
        # Group by region
        region_groups = {}
        for p in problems:
            if p.region_name not in region_groups:
                region_groups[p.region_name] = []
            region_groups[p.region_name].append(p)
        
        # Merge problems in the same region
        merged = []
        new_id = 0
        
        for region_name, region_problems in region_groups.items():
            if len(region_problems) == 1:
                p = region_problems[0]
                p.problem_id = new_id
                merged.append(p)
            else:
                # Combine issues and take max severity
                all_issues = set()
                max_severity = 0.0
                best_bbox = None
                best_mask = None
                
                for p in region_problems:
                    all_issues.add(p.issue_description)
                    if p.severity > max_severity:
                        max_severity = p.severity
                        best_bbox = p.bbox
                        best_mask = p.mask
                
                merged_problem = Problem(
                    problem_id=new_id,
                    region_name=region_name,
                    spatial_description=region_problems[0].spatial_description,
                    issue_description="; ".join(all_issues),
                    severity=max_severity,
                    bbox=best_bbox,
                    mask=best_mask,
                )
                merged.append(merged_problem)
            new_id += 1
        
        # Sort by severity (highest first)
        merged.sort(key=lambda p: p.severity, reverse=True)
        
        return merged
    
    def _calculate_global_score(
        self,
        generated: torch.Tensor,
        gt: Optional[torch.Tensor],
    ) -> float:
        """Calculate global quality score"""
        if gt is None:
            return 0.5  # Unknown quality without GT
        
        # Ensure same shape
        if generated.shape != gt.shape:
            min_frames = min(generated.shape[2], gt.shape[2])
            generated = generated[:, :, :min_frames]
            gt = gt[:, :, :min_frames]
        
        gen_np = generated.detach().cpu().float().numpy()
        gt_np = gt.detach().cpu().float().numpy()
        
        gen_np = (gen_np + 1.0) / 2.0
        gt_np = (gt_np + 1.0) / 2.0
        
        ssim = self._compute_ssim(gen_np, gt_np)
        psnr = self._compute_psnr(gen_np, gt_np)
        
        # Combine into a single score
        ssim_score = ssim
        psnr_score = min(psnr / 40.0, 1.0)  # Normalize PSNR to [0, 1]
        
        return 0.6 * ssim_score + 0.4 * psnr_score
    
    def _generate_summary(self, problems: List[Problem]) -> str:
        """Generate a summary of detected problems"""
        if not problems:
            return "No significant quality issues detected."
        
        n_problems = len(problems)
        max_severity = max(p.severity for p in problems)
        regions = list(set(p.region_name for p in problems))
        
        severity_desc = "severe" if max_severity > 0.7 else "moderate" if max_severity > 0.4 else "minor"
        
        summary = f"Detected {n_problems} {severity_desc} quality issue(s) "
        summary += f"in regions: {', '.join(regions)}."
        
        return summary
    
    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute SSIM between two images"""
        try:
            from skimage.metrics import structural_similarity
            # Flatten batch and channel dimensions for computation
            img1_flat = img1.reshape(-1, *img1.shape[-2:])
            img2_flat = img2.reshape(-1, *img2.shape[-2:])
            
            ssim_vals = []
            for i in range(img1_flat.shape[0]):
                ssim_val = structural_similarity(
                    img1_flat[i], img2_flat[i],
                    data_range=1.0,
                )
                ssim_vals.append(ssim_val)
            return np.mean(ssim_vals)
        except:
            return 0.5
    
    @staticmethod
    def _compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute PSNR between two images"""
        try:
            mse = np.mean((img1 - img2) ** 2)
            if mse < 1e-10:
                return 100.0
            return 10 * np.log10(1.0 / mse)
        except:
            return 20.0
    
    @staticmethod
    def _compute_edge_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute edge similarity between two images"""
        try:
            from scipy import ndimage
            
            # Use Sobel filter for edge detection
            img1_flat = img1.reshape(-1, *img1.shape[-2:])
            img2_flat = img2.reshape(-1, *img2.shape[-2:])
            
            edge_scores = []
            for i in range(img1_flat.shape[0]):
                edge1 = np.hypot(
                    ndimage.sobel(img1_flat[i], axis=0),
                    ndimage.sobel(img1_flat[i], axis=1)
                )
                edge2 = np.hypot(
                    ndimage.sobel(img2_flat[i], axis=0),
                    ndimage.sobel(img2_flat[i], axis=1)
                )
                
                # Normalize
                edge1 = edge1 / (edge1.max() + 1e-8)
                edge2 = edge2 / (edge2.max() + 1e-8)
                
                # Correlation
                score = np.corrcoef(edge1.flatten(), edge2.flatten())[0, 1]
                if not np.isnan(score):
                    edge_scores.append(score)
            
            return np.mean(edge_scores) if edge_scores else 0.5
        except:
            return 0.5
    
    @staticmethod
    def _create_region_mask(
        H: int, W: int,
        y1: int, y2: int,
        x1: int, x2: int,
    ) -> torch.Tensor:
        """Create a binary mask for a region"""
        mask = torch.zeros(H, W)
        mask[y1:y2, x1:x2] = 1.0
        return mask
    
    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image for VLM input (center frame/slice)."""
        if tensor.dim() == 5:
            B, C, F, H, W = tensor.shape
            frame_idx = F // 2
            img = tensor[0, :, frame_idx]  # (C, H, W)
        elif tensor.dim() == 4:
            img = tensor[0]  # (C, H, W)
        else:
            img = tensor
        
        img = (img.detach().cpu().float() + 1.0) / 2.0 * 255.0
        img = img.clamp(0, 255).byte()
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img_np = img.permute(1, 2, 0).numpy()
        return Image.fromarray(img_np)
    
    @staticmethod
    def _tensor_to_5_pil_slices(tensor: torch.Tensor) -> List[Image.Image]:
        """
        Extract 5 uniformly-spaced axial slices from a 3-D/4-D/5-D MRI tensor.

        Slice positions follow i/(N+1) for i in 1..N (N=5), giving
        approximately [16.7 %, 33.3 %, 50 %, 66.7 %, 83.3 %] of the
        depth axis — equally spaced and symmetric, skipping skull-only edges.
        """
        N_SLICES = 5
        FRACTIONS = [i / (N_SLICES + 1) for i in range(1, N_SLICES + 1)]
        
        if tensor.dim() == 5:
            # (B, C, F, H, W) — F is the depth/frame axis
            _, _, F, _, _ = tensor.shape
            indices = [max(0, min(F - 1, round(f * F))) for f in FRACTIONS]
            slices = [tensor[0, :, idx] for idx in indices]  # list of (C, H, W)
        elif tensor.dim() == 4:
            # (B, C, H, W) — no depth axis; tile the single frame 5 times
            slices = [tensor[0]] * N_SLICES
        else:
            slices = [tensor] * N_SLICES
        
        pil_images = []
        for img in slices:
            img_f = (img.detach().cpu().float() + 1.0) / 2.0 * 255.0
            img_f = img_f.clamp(0, 255).byte()
            if img_f.shape[0] == 1:
                img_f = img_f.repeat(3, 1, 1)
            img_np = img_f.permute(1, 2, 0).numpy()
            pil_images.append(Image.fromarray(img_np))
        
        return pil_images
