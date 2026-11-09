"""
Base Agent Class for MRI Enhancement System
============================================

Provides the foundation for all agents in the 4-stage MRI translation framework.
"""

import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum


class AgentType(Enum):
    """Agent types in the system"""
    ANOMALY_DETECTION = "anomaly_detection"
    MEDICAL_VLM = "medical_vlm"
    CRITIQUE = "critique"
    SKILL_EVOLUTION = "skill_evolution"


@dataclass
class AgentConfig:
    """Configuration for agents"""
    # Model settings
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    
    # Generation settings
    max_new_tokens: int = field(default_factory=lambda: int(os.environ.get("UNIFIELD_VLM_MAX_NEW_TOKENS", "16384")))
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Agent-specific settings
    max_problems_per_image: int = 5
    confidence_threshold: float = 0.5
    
    # Task settings
    task_type: str = "64mT_to_3T"  # or "3T_to_7T"
    modality: Optional[str] = None
    input_field: Optional[str] = None
    target_field: Optional[str] = None
    
    # Debug
    verbose: bool = False


@dataclass
class Problem:
    """Represents a detected problem in the MRI image"""
    problem_id: int
    region_name: str  # e.g., "hippocampus", "cerebral_cortex"
    spatial_description: str  # e.g., "大脑中心偏下方的深层区域"
    issue_description: str  # e.g., "边缘模糊，对比度不足"
    severity: float  # 0.0 - 1.0
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    mask: Optional[torch.Tensor] = None  # Binary mask for the region
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "region_name": self.region_name,
            "spatial_description": self.spatial_description,
            "issue_description": self.issue_description,
            "severity": self.severity,
        }


@dataclass 
class Plan:
    """Represents an improvement plan for a detected problem"""
    plan_id: int
    target_problem: Problem
    action_description: str  # What to do
    expected_outcome: str  # What the result should look like
    text_prompt: str  # Text prompt for cross-attention guidance
    priority: int = 1  # Lower is higher priority
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "target_problem": self.target_problem.to_dict(),
            "action_description": self.action_description,
            "expected_outcome": self.expected_outcome,
            "text_prompt": self.text_prompt,
            "priority": self.priority,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the MRI enhancement system.
    
    Agents are responsible for:
    1. Analyzing MRI images to detect problems (Stage 2)
    2. Generating improvement plans (Stage 2)
    3. Providing guidance for local refinement (Stage 3)
    4. Self-evolving through experience (Stage 4)
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = config.device
        self.torch_dtype = config.torch_dtype
        self._is_initialized = False
        
    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of this agent"""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the agent (load models, etc.)"""
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Main processing method - to be implemented by subclasses"""
        pass
    
    def _ensure_initialized(self) -> None:
        """Ensure the agent is initialized before processing"""
        if not self._is_initialized:
            self.initialize()
            self._is_initialized = True
    
    def to(self, device: str) -> 'BaseAgent':
        """Move agent to specified device"""
        self.device = device
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.agent_type.value}, device={self.device})"


class AgentMessage:
    """Message format for inter-agent communication"""
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        content: Any,
        message_type: str = "data",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "message_type": self.message_type,
            "metadata": self.metadata,
        }


class AgentOrchestrator:
    """
    Orchestrates multiple agents for the multi-stage MRI enhancement pipeline.
    
    Flow:
    1. Stage 1 generates initial high-field MRI
    2. Anomaly Detection Agent identifies problems
    3. Medical VLM Agent generates improvement plans
    4. Plans are executed in Stage 3 (local refinement)
    5. Loop back to step 2 if problems remain
    """
    
    def __init__(
        self,
        anomaly_agent: Optional['BaseAgent'] = None,
        vlm_agent: Optional['BaseAgent'] = None,
        max_iterations: int = 3,
        quality_threshold: float = 0.9,
    ):
        self.anomaly_agent = anomaly_agent
        self.vlm_agent = vlm_agent
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.iteration_history = []
        
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator"""
        if agent.agent_type == AgentType.ANOMALY_DETECTION:
            self.anomaly_agent = agent
        elif agent.agent_type == AgentType.MEDICAL_VLM:
            self.vlm_agent = agent
        else:
            raise ValueError(f"Unknown agent type: {agent.agent_type}")
    
    def run_diagnosis(
        self,
        lq_image: torch.Tensor,
        generated_image: torch.Tensor,
        gt_image: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Problem], List[Plan]]:
        """
        Run Stage 2: Diagnosis & Planning
        
        Args:
            lq_image: Low-field MRI input (B, C, F, H, W) or (B, C, H, W)
            generated_image: Generated high-field MRI from Stage 1
            gt_image: Ground truth high-field MRI (optional, for training)
            
        Returns:
            problems: List of detected problems
            plans: List of improvement plans
        """
        if self.anomaly_agent is None or self.vlm_agent is None:
            raise RuntimeError("Both anomaly_agent and vlm_agent must be registered")
        
        # Step 1: Detect anomalies
        problems = self.anomaly_agent.process(
            lq_image=lq_image,
            generated_image=generated_image,
            gt_image=gt_image,
        )
        
        # If no problems detected, return empty lists
        if not problems:
            return [], []
        
        # Step 2: Generate improvement plans (already capped to top-3 inside process())
        improvement_plan = self.vlm_agent.process(
            lq_image=lq_image,
            generated_image=generated_image,
            problems=problems,
        )
        
        return problems, improvement_plan.plans
    
    def run_unified_diagnosis(
        self,
        lq_image: 'torch.Tensor',
        generated_image: 'torch.Tensor',
        gt_image: Optional['torch.Tensor'] = None,
    ) -> Tuple[List[Problem], List[Plan]]:
        """
        Run Stage 2 with a single VLM call for both detection and planning.
        
        Instead of calling anomaly_agent then vlm_agent separately (2+ VLM calls),
        this uses vlm_agent.detect_and_plan() to do everything in one shot.
        
        Args:
            lq_image: Low-field MRI input
            generated_image: Generated high-field MRI from Stage 1
            gt_image: Ground truth (optional)
            
        Returns:
            problems: List of detected problems
            plans: List of improvement plans
        """
        if self.vlm_agent is None:
            raise RuntimeError("vlm_agent must be registered")
        
        # Single VLM call for both detection and planning
        vlm_problems, plans = self.vlm_agent.detect_and_plan(
            lq_image=lq_image,
            generated_image=generated_image,
            gt_image=gt_image,
        )
        
        # Cap to the 3 most severe problems
        plans = sorted(plans, key=lambda p: p.target_problem.severity, reverse=True)[:3]
        # Re-assign priorities to 1..N after sorting
        for i, p in enumerate(plans):
            p.priority = i + 1
        
        return vlm_problems, plans

    def run_lq_diagnosis(
        self,
        lq_image: 'torch.Tensor',
    ) -> Tuple[List[Problem], List[Plan]]:
        """Run Stage 2 planning from low-field input slices only."""
        if self.vlm_agent is None:
            raise RuntimeError("vlm_agent must be registered")

        detector = getattr(self.vlm_agent, "detect_and_plan_from_lq", None)
        if detector is None:
            raise RuntimeError("vlm_agent does not support low-field-only planning")

        problems, plans = detector(lq_image=lq_image)
        plans = sorted(plans, key=lambda p: p.target_problem.severity, reverse=True)[:3]
        for i, plan in enumerate(plans):
            plan.priority = i + 1
        return problems, plans

    def should_continue(self, problems: List[Problem]) -> bool:
        """Determine if another refinement iteration is needed"""
        if not problems:
            return False
        
        # Check if any problem exceeds severity threshold
        max_severity = max(p.severity for p in problems)
        return max_severity > (1.0 - self.quality_threshold)
    
    def reset(self) -> None:
        """Reset orchestrator state"""
        self.iteration_history = []
