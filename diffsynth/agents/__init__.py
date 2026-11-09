"""
MRI Enhancement Agent System
============================

4-Stage MRI Translation Framework:
- Stage 1: Base SFT (Base Generator)
- Stage 2: Agent-driven Diagnosis & Planning
- Stage 3: Local-Guided Diffusion Refinement
- Stage 4: Agent Self-Evolving (Skill Bank)

This module implements Stage 2 and Stage 3 components.
"""

from .base_agent import BaseAgent, AgentConfig
from .brain_atlas import BrainAtlas, BrainRegion
from .anomaly_detector import AnomalyDetectionAgent, AnomalyReport
from .medical_vlm import MedicalVLMAgent, ImprovementPlan

__all__ = [
    # Base
    'BaseAgent',
    'AgentConfig',
    # Brain Atlas
    'BrainAtlas',
    'BrainRegion',
    # Stage 2 Agents
    'AnomalyDetectionAgent',
    'AnomalyReport',
    'MedicalVLMAgent',
    'ImprovementPlan',
]
