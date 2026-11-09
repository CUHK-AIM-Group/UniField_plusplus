"""
Plan to Mask Conversion Module
==============================

Stage 3 Component: Converts improvement plans to spatial masks for local refinement.

This module bridges Stage 2 (planning) and Stage 3 (refinement) by:
1. Converting region descriptions to spatial masks
2. Providing text embeddings for cross-attention conditioning
3. Managing iterative refinement loops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union

from ..agents.base_agent import Plan, Problem
from ..agents.brain_atlas import BrainAtlas, BRAIN_ATLAS


@dataclass
class RefinementCondition:
    """
    Condition for local refinement in Stage 3.
    
    Contains all information needed to guide the diffusion model
    for a specific region's enhancement.
    """
    plan: Plan
    mask: torch.Tensor  # (H, W) or (D, H, W) binary/soft mask
    text_prompt: str  # Text prompt for cross-attention
    text_embedding: Optional[torch.Tensor] = None  # Pre-computed embedding
    strength: float = 1.0  # Refinement strength (0-1)
    
    def to(self, device: str, dtype: torch.dtype = None) -> 'RefinementCondition':
        """Move tensors to device"""
        self.mask = self.mask.to(device)
        if self.text_embedding is not None:
            self.text_embedding = self.text_embedding.to(device)
            if dtype is not None:
                self.text_embedding = self.text_embedding.to(dtype)
        return self


class PlanToMaskConverter:
    """
    Converts improvement plans to spatial masks for diffusion model guidance.
    
    Supports:
    - 2D masks for single slice processing
    - 3D masks for volumetric processing
    - Soft masks with Gaussian boundaries for smooth blending
    - Multiple overlapping regions
    """
    
    def __init__(
        self,
        brain_atlas: BrainAtlas = None,
        default_soft_boundary: bool = True,
        default_sigma: float = 5.0,
    ):
        self.brain_atlas = brain_atlas or BRAIN_ATLAS
        self.default_soft_boundary = default_soft_boundary
        self.default_sigma = default_sigma
    
    def convert_plan_to_mask(
        self,
        plan: Plan,
        image_size: Tuple[int, ...],  # (H, W) or (D, H, W)
        soft_boundary: bool = None,
        sigma: float = None,
    ) -> torch.Tensor:
        """
        Convert a single plan to a spatial mask.
        
        Args:
            plan: Improvement plan with region information
            image_size: Target image size (H, W) or (D, H, W)
            soft_boundary: Whether to use soft (Gaussian) boundaries
            sigma: Sigma for Gaussian smoothing
            
        Returns:
            Mask tensor of shape (H, W) or (D, H, W)
        """
        if soft_boundary is None:
            soft_boundary = self.default_soft_boundary
        if sigma is None:
            sigma = self.default_sigma
        
        problem = plan.target_problem
        
        # If problem has a pre-computed mask, use it
        if problem.mask is not None:
            mask = problem.mask
            # Resize to target size if needed
            if mask.shape != image_size:
                mask = self._resize_mask(mask, image_size)
            return mask
        
        # If problem has a bounding box, create mask from it
        if problem.bbox is not None:
            return self._create_mask_from_bbox(
                problem.bbox, image_size, soft_boundary, sigma
            )
        
        # Otherwise, use brain atlas to generate approximate mask
        return self.brain_atlas.generate_region_mask(
            problem.region_name,
            image_size,
            soft_boundary=soft_boundary,
            sigma=sigma,
        )
    
    def convert_plans_to_masks(
        self,
        plans: List[Plan],
        image_size: Tuple[int, ...],
        combine_mode: str = "max",
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Convert multiple plans to masks.
        
        Args:
            plans: List of improvement plans
            image_size: Target image size
            combine_mode: How to combine masks ("max", "sum", "separate")
            
        Returns:
            combined_mask: Combined mask for all plans
            individual_masks: List of individual masks
        """
        individual_masks = []
        
        for plan in plans:
            mask = self.convert_plan_to_mask(plan, image_size)
            individual_masks.append(mask)
        
        if not individual_masks:
            combined_mask = torch.zeros(image_size)
        elif combine_mode == "max":
            combined_mask = torch.stack(individual_masks).max(dim=0)[0]
        elif combine_mode == "sum":
            combined_mask = torch.stack(individual_masks).sum(dim=0).clamp(0, 1)
        else:  # separate
            combined_mask = individual_masks[0] if individual_masks else torch.zeros(image_size)
        
        return combined_mask, individual_masks
    
    def create_refinement_conditions(
        self,
        plans: List[Plan],
        image_size: Tuple[int, ...],
        text_encoder: nn.Module = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> List[RefinementCondition]:
        """
        Create refinement conditions for Stage 3 processing.
        
        Args:
            plans: List of improvement plans
            image_size: Target image size
            text_encoder: Optional text encoder for pre-computing embeddings
            device: Target device
            dtype: Target dtype for embeddings
            
        Returns:
            List of RefinementCondition objects
        """
        conditions = []
        
        for plan in plans:
            # Create mask
            mask = self.convert_plan_to_mask(plan, image_size)
            
            # Create text embedding if encoder is provided
            text_embedding = None
            if text_encoder is not None:
                text_embedding = self._encode_text(
                    plan.text_prompt, text_encoder, device, dtype
                )
            
            # Calculate strength based on severity
            strength = min(plan.target_problem.severity + 0.3, 1.0)
            
            condition = RefinementCondition(
                plan=plan,
                mask=mask,
                text_prompt=plan.text_prompt,
                text_embedding=text_embedding,
                strength=strength,
            )
            condition.to(device, dtype)
            conditions.append(condition)
        
        return conditions
    
    def _create_mask_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        image_size: Tuple[int, ...],
        soft_boundary: bool = True,
        sigma: float = 5.0,
    ) -> torch.Tensor:
        """Create mask from bounding box"""
        x1, y1, x2, y2 = bbox
        
        is_3d = len(image_size) == 3
        
        if is_3d:
            D, H, W = image_size
            mask = torch.zeros(D, H, W)
            # Apply to all slices for 3D
            mask[:, y1:y2, x1:x2] = 1.0
        else:
            H, W = image_size
            mask = torch.zeros(H, W)
            mask[y1:y2, x1:x2] = 1.0
        
        if soft_boundary:
            mask = self._apply_gaussian_blur(mask, sigma)
        
        return mask
    
    def _apply_gaussian_blur(
        self,
        mask: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Apply Gaussian blur for soft boundaries"""
        kernel_size = int(sigma * 3) * 2 + 1
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = F.pad(mask, (kernel_size//2,)*4, mode='replicate')
            mask = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1)
            mask = mask.squeeze(0).squeeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = F.pad(mask, (kernel_size//2,)*6, mode='replicate')
            mask = F.avg_pool3d(mask, kernel_size=kernel_size, stride=1)
            mask = mask.squeeze(0).squeeze(0)
        
        return mask
    
    def _resize_mask(
        self,
        mask: torch.Tensor,
        target_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """Resize mask to target size"""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
            mask = mask.squeeze(0).squeeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, size=target_size, mode='trilinear', align_corners=False)
            mask = mask.squeeze(0).squeeze(0)
        return mask
    
    def _encode_text(
        self,
        text: str,
        text_encoder: nn.Module,
        device: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode text to embedding"""
        try:
            with torch.no_grad():
                # This assumes the text encoder has an encode_prompt method
                # Adjust based on your specific text encoder interface
                if hasattr(text_encoder, 'encode_prompt'):
                    embedding = text_encoder.encode_prompt(text, positive=True, device=device)
                elif hasattr(text_encoder, 'encode'):
                    embedding = text_encoder.encode(text)
                else:
                    # Fallback: return None if encoder doesn't have expected methods
                    return None
                
                return embedding.to(device=device, dtype=dtype)
        except Exception as e:
            print(f"Warning: Text encoding failed: {e}")
            return None


class IterativeRefinementController:
    """
    Controls the iterative refinement loop between Stage 2 and Stage 3.
    
    Manages:
    - Iteration counting and early stopping
    - Quality threshold checking
    - Plan execution sequencing
    """
    
    def __init__(
        self,
        max_iterations: int = 3,
        quality_threshold: float = 0.9,
        min_improvement: float = 0.01,
    ):
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.min_improvement = min_improvement
        
        # State tracking
        self.current_iteration = 0
        self.quality_history = []
        self.plans_executed = []
    
    def reset(self) -> None:
        """Reset controller state for new sample"""
        self.current_iteration = 0
        self.quality_history = []
        self.plans_executed = []
    
    def should_continue(
        self,
        current_quality: float,
        remaining_problems: List[Problem],
    ) -> bool:
        """
        Determine if refinement should continue.
        
        Args:
            current_quality: Current quality score (0-1)
            remaining_problems: Problems still detected
            
        Returns:
            True if refinement should continue
        """
        # Check iteration limit
        if self.current_iteration >= self.max_iterations:
            return False
        
        # Check quality threshold
        if current_quality >= self.quality_threshold:
            return False
        
        # No problems remaining
        if not remaining_problems:
            return False
        
        # Check for improvement stagnation
        if len(self.quality_history) >= 2:
            improvement = current_quality - self.quality_history[-1]
            if improvement < self.min_improvement:
                return False
        
        return True
    
    def record_iteration(
        self,
        quality_score: float,
        executed_plans: List[Plan],
    ) -> None:
        """Record completed iteration"""
        self.quality_history.append(quality_score)
        self.plans_executed.extend(executed_plans)
        self.current_iteration += 1
    
    def get_summary(self) -> Dict:
        """Get summary of refinement process"""
        return {
            "total_iterations": self.current_iteration,
            "quality_history": self.quality_history,
            "initial_quality": self.quality_history[0] if self.quality_history else None,
            "final_quality": self.quality_history[-1] if self.quality_history else None,
            "improvement": (self.quality_history[-1] - self.quality_history[0]) 
                          if len(self.quality_history) >= 2 else 0,
            "plans_executed": len(self.plans_executed),
        }


class LocalRefinementModule(nn.Module):
    """
    Module for applying local refinements guided by plans and masks.
    
    Implements SDEdit-style or Inpainting-style refinement with:
    - Masked region selection
    - Text-conditioned cross-attention
    - Iterative denoising within masked regions
    """
    
    def __init__(
        self,
        noise_level: float = 0.5,
        blend_mode: str = "soft",
    ):
        super().__init__()
        self.noise_level = noise_level
        self.blend_mode = blend_mode
    
    def add_noise_to_region(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        noise_level: float = None,
    ) -> torch.Tensor:
        """
        Add noise to masked region (SDEdit-style).
        
        Args:
            latent: Input latent representation
            mask: Region mask (same spatial dimensions as latent)
            noise_level: Amount of noise to add (0-1)
            
        Returns:
            Noised latent with masked region partially corrupted
        """
        if noise_level is None:
            noise_level = self.noise_level
        
        # Generate noise
        noise = torch.randn_like(latent)
        
        # Expand mask to match latent dimensions
        mask = self._expand_mask_to_latent(mask, latent.shape)
        
        # Blend: noised = (1 - noise_level * mask) * latent + noise_level * mask * noise
        noised_region = (1 - noise_level) * latent + noise_level * noise
        blended = latent * (1 - mask) + noised_region * mask
        
        return blended
    
    def blend_refined_region(
        self,
        original: torch.Tensor,
        refined: torch.Tensor,
        mask: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Blend refined region back into original image.
        
        Args:
            original: Original image/latent
            refined: Refined image/latent
            mask: Region mask
            strength: Blending strength (0-1)
            
        Returns:
            Blended result
        """
        mask = self._expand_mask_to_latent(mask, original.shape)
        
        # Apply strength to mask
        effective_mask = mask * strength
        
        if self.blend_mode == "soft":
            # Smooth blending with feathered edges
            result = original * (1 - effective_mask) + refined * effective_mask
        elif self.blend_mode == "hard":
            # Hard cutoff at 0.5
            binary_mask = (effective_mask > 0.5).float()
            result = original * (1 - binary_mask) + refined * binary_mask
        else:  # "poisson" - for future implementation
            result = original * (1 - effective_mask) + refined * effective_mask
        
        return result
    
    def _expand_mask_to_latent(
        self,
        mask: torch.Tensor,
        latent_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Expand 2D/3D mask to match latent dimensions"""
        # latent_shape: (B, C, F, H, W) or (B, C, H, W)
        
        if mask.dim() == 2:
            # (H, W) -> (1, 1, 1, H, W) or (1, 1, H, W)
            while mask.dim() < len(latent_shape):
                mask = mask.unsqueeze(0)
        elif mask.dim() == 3:
            # (D, H, W) -> (1, 1, D, H, W)
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Resize spatial dimensions if needed
        if mask.shape[-2:] != latent_shape[-2:]:
            mask = F.interpolate(
                mask.float(),
                size=latent_shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        
        # Handle frame dimension for video
        if len(latent_shape) == 5 and mask.shape[2] != latent_shape[2]:
            mask = F.interpolate(
                mask.float(),
                size=latent_shape[2:],
                mode='trilinear',
                align_corners=False,
            )
        
        # Expand channel dimension
        if mask.shape[1] != latent_shape[1]:
            mask = mask.expand(-1, latent_shape[1], *mask.shape[2:])
        
        # Expand batch dimension
        if mask.shape[0] != latent_shape[0]:
            mask = mask.expand(latent_shape[0], *mask.shape[1:])
        
        return mask
    
    def forward(
        self,
        latent: torch.Tensor,
        conditions: List[RefinementCondition],
    ) -> torch.Tensor:
        """
        Apply local refinements sequentially.
        
        This is a placeholder - actual refinement requires
        integration with the diffusion model's denoising loop.
        """
        result = latent.clone()
        
        for condition in conditions:
            # Add noise to region
            noised = self.add_noise_to_region(
                result,
                condition.mask,
                noise_level=condition.strength * self.noise_level,
            )
            # Note: Actual denoising would happen here with the diffusion model
            # For now, just return the noised version as placeholder
            result = noised
        
        return result
