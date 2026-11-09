"""
Brain Atlas and Anatomical Region Mapping
==========================================

Maps brain anatomical regions to spatial descriptions for Agent communication.
This enables the Agent to use natural language spatial descriptions instead of 
medical terminology that may not be understood by diffusion models.

Spatial masks are registered to the low-field volume and transformed with the image.
"""

import torch
import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class BrainRegionType(Enum):
    """Types of brain regions"""
    DEEP = "deep"  # Deep structures (hippocampus, thalamus, etc.)
    CORTICAL = "cortical"  # Cortical regions
    VENTRICULAR = "ventricular"  # Ventricles
    CEREBELLAR = "cerebellar"  # Cerebellum
    WHITE_MATTER = "white_matter"  # White matter tracts
    BOUNDARY = "boundary"  # Tissue boundaries


@dataclass
class BrainRegion:
    """
    Represents a brain anatomical region with spatial mapping.
    
    The spatial_description is designed to be understood by diffusion models
    without requiring medical knowledge.
    """
    name: str  # Medical name (e.g., "hippocampus")
    chinese_name: str  # Chinese name (e.g., "海马体")
    spatial_description: str  # Spatial description for VLM/diffusion
    chinese_spatial_description: str  # Chinese spatial description
    region_type: BrainRegionType
    
    # Approximate relative position in normalized coordinates (0-1)
    # For 3D MRI: (x_center, y_center, z_center, width, height, depth)
    relative_bbox: Optional[Tuple[float, ...]] = None
    
    # Common issues in this region for different tasks
    common_issues_64mT_to_3T: List[str] = None
    common_issues_3T_to_7T: List[str] = None
    
    def __post_init__(self):
        if self.common_issues_64mT_to_3T is None:
            self.common_issues_64mT_to_3T = []
        if self.common_issues_3T_to_7T is None:
            self.common_issues_3T_to_7T = []
    
    def get_enhancement_prompt(self, issue: str, task_type: str = "64mT_to_3T") -> str:
        """
        Generate an enhancement prompt for this region based on the issue.
        
        Args:
            issue: Description of the issue (e.g., "blurry edges")
            task_type: Either "64mT_to_3T" or "3T_to_7T"
            
        Returns:
            Text prompt for cross-attention guidance
        """
        if task_type == "64mT_to_3T":
            return f"Enhance the {self.spatial_description} with sharper edges and better contrast, fixing {issue}"
        else:  # 3T_to_7T
            return f"Refine the {self.spatial_description} with natural texture, reducing artifacts and preserving detail"


class BrainAtlas:
    """
    Brain Atlas providing anatomical region mappings and spatial descriptions.
    
    Key mappings (from PDF document):
    - 海马体 (Hippocampus) -> 大脑中心偏下方的深层区域 (Deep central-lower region of the brain)
    - 大脑皮层 (Cerebral Cortex) -> 大脑最外层的边缘轮廓区域 (Outermost boundary and surface area)
    - 脑室 (Ventricles) -> 大脑正中央的蝴蝶状空腔区域 (Central butterfly-shaped cavity area)
    - 小脑 (Cerebellum) -> 大脑底部后方的半圆形区域 (Semi-circular region at the bottom-rear)
    """
    
    # Valid region names — restricted to 5 lobes + ventricles + white matter.
    # The Agent MUST only report one of these 7 regions.
    HARVARD_OXFORD_REGIONS: List[str] = [
        "frontal_lobe",    # 额叶
        "parietal_lobe",   # 顶叶
        "temporal_lobe",   # 颞叶
        "occipital_lobe",  # 枕叶
        "insular_lobe",    # 脑岛
        "ventricles",      # 脑室
        "white_matter",    # 白质
    ]

    # Harvard-Oxford cortical atlas label indices for each of our 7 regions.
    # Labels follow nilearn's fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm').
    _CORT_LABEL_INDICES: Dict[str, List[int]] = {
        "frontal_lobe":  [1, 3, 4, 5, 6, 7, 25, 26, 27, 28, 29, 33, 41],
        "parietal_lobe": [17, 18, 19, 20, 21, 30, 31, 43],
        "temporal_lobe": [8, 9, 10, 11, 12, 13, 14, 15, 16, 34, 35, 37, 38, 39, 44, 45, 46],
        "occipital_lobe":[22, 23, 24, 32, 36, 40, 47, 48],
        "insular_lobe":  [2],
    }
    # Harvard-Oxford sub-cortical atlas label indices.
    # Labels follow nilearn's fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm').
    _SUB_LABEL_INDICES: Dict[str, List[int]] = {
        "ventricles":  [3, 14],   # Left + Right Lateral Ventricle
        "white_matter":[1, 12],   # Left + Right Cerebral White Matter
    }

    # Cache: region_name -> numpy array (D, H, W) in atlas native space (182,218,182)
    _ATLAS_MASKS: Optional[Dict[str, "np.ndarray"]] = None

    # Pre-defined brain regions with spatial mappings
    REGIONS: Dict[str, BrainRegion] = {
        "hippocampus": BrainRegion(
            name="hippocampus",
            chinese_name="海马体",
            spatial_description="deep central-lower region of the brain",
            chinese_spatial_description="大脑中心偏下方的深层区域",
            region_type=BrainRegionType.DEEP,
            relative_bbox=(0.5, 0.6, 0.4, 0.15, 0.1, 0.1),
            common_issues_64mT_to_3T=["blurry boundaries", "low contrast", "missing gray-white matter differentiation"],
            common_issues_3T_to_7T=["over-sharpening artifacts", "unnatural texture"],
        ),
        
        "cerebral_cortex": BrainRegion(
            name="cerebral_cortex",
            chinese_name="大脑皮层",
            spatial_description="outermost boundary and surface area of the brain",
            chinese_spatial_description="大脑最外层的边缘轮廓区域",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.5, 0.5, 0.9, 0.9, 0.8),
            common_issues_64mT_to_3T=["smoothed sulci and gyri", "lost cortical folding details"],
            common_issues_3T_to_7T=["grid-like artifacts", "over-enhanced noise"],
        ),
        
        "ventricles": BrainRegion(
            name="ventricles",
            chinese_name="脑室",
            spatial_description="central CSF-filled ventricular cavity of the brain",
            chinese_spatial_description="大脑中央脑脊液腔（脑室）",
            region_type=BrainRegionType.VENTRICULAR,
            relative_bbox=(0.5, 0.5, 0.5, 0.2, 0.3, 0.4),
            common_issues_64mT_to_3T=["unclear boundaries", "artifacts in CSF region"],
            common_issues_3T_to_7T=["flow artifacts", "intensity inhomogeneity"],
        ),
        
        "cerebellum": BrainRegion(
            name="cerebellum",
            chinese_name="小脑",
            spatial_description="semi-circular region at the bottom-rear of the brain",
            chinese_spatial_description="大脑底部后方的半圆形区域",
            region_type=BrainRegionType.CEREBELLAR,
            relative_bbox=(0.5, 0.8, 0.2, 0.4, 0.3, 0.25),
            common_issues_64mT_to_3T=["lost foliation details", "poor contrast"],
            common_issues_3T_to_7T=["susceptibility artifacts", "signal dropout"],
        ),
        
        "thalamus": BrainRegion(
            name="thalamus",
            chinese_name="丘脑",
            spatial_description="deep central region of the brain, above the brainstem",
            chinese_spatial_description="大脑中央深层区域，位于脑干上方",
            region_type=BrainRegionType.DEEP,
            relative_bbox=(0.5, 0.55, 0.45, 0.12, 0.12, 0.1),
            common_issues_64mT_to_3T=["poor gray matter definition", "boundary blur"],
            common_issues_3T_to_7T=["iron deposition artifacts"],
        ),
        
        "basal_ganglia": BrainRegion(
            name="basal_ganglia",
            chinese_name="基底神经节",
            spatial_description="deep central structures on both sides of the thalamus",
            chinese_spatial_description="丘脑两侧的深层中央结构",
            region_type=BrainRegionType.DEEP,
            relative_bbox=(0.5, 0.5, 0.5, 0.25, 0.15, 0.12),
            common_issues_64mT_to_3T=["indistinct nuclei boundaries", "poor contrast"],
            common_issues_3T_to_7T=["susceptibility artifacts from iron"],
        ),
        
        "corpus_callosum": BrainRegion(
            name="corpus_callosum",
            chinese_name="胼胝体",
            spatial_description="white matter bridge in the center connecting left and right brain",
            chinese_spatial_description="连接左右脑的中央白质桥",
            region_type=BrainRegionType.WHITE_MATTER,
            relative_bbox=(0.5, 0.4, 0.55, 0.3, 0.08, 0.15),
            common_issues_64mT_to_3T=["poor white matter signal", "unclear boundaries"],
            common_issues_3T_to_7T=["partial volume effects"],
        ),
        
        "brainstem": BrainRegion(
            name="brainstem",
            chinese_name="脑干",
            spatial_description="central stem-like structure at the bottom center of the brain",
            chinese_spatial_description="大脑底部中央的茎状结构",
            region_type=BrainRegionType.DEEP,
            relative_bbox=(0.5, 0.7, 0.25, 0.1, 0.15, 0.2),
            common_issues_64mT_to_3T=["low signal quality", "motion artifacts"],
            common_issues_3T_to_7T=["susceptibility artifacts"],
        ),
        
        "frontal_lobe": BrainRegion(
            name="frontal_lobe",
            chinese_name="额叶",
            spatial_description="anterior cortical region of the brain",
            chinese_spatial_description="大脑前方皮层区域（额叶）",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.3, 0.6, 0.4, 0.35, 0.35),
            common_issues_64mT_to_3T=["smoothed cortical details", "poor gray-white contrast"],
            common_issues_3T_to_7T=["edge ringing artifacts"],
        ),
        
        "temporal_lobe": BrainRegion(
            name="temporal_lobe",
            chinese_name="颞叶",
            spatial_description="lateral inferior cortical region of the brain",
            chinese_spatial_description="大脑外侧下方皮层区域（颞叶）",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.5, 0.35, 0.35, 0.25, 0.3),
            common_issues_64mT_to_3T=["susceptibility distortion", "poor medial temporal resolution"],
            common_issues_3T_to_7T=["air-tissue interface artifacts"],
        ),
        
        "occipital_lobe": BrainRegion(
            name="occipital_lobe",
            chinese_name="枕叶",
            spatial_description="posterior cortical region of the brain",
            chinese_spatial_description="大脑后方皮层区域（枕叶）",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.8, 0.55, 0.3, 0.3, 0.3),
            common_issues_64mT_to_3T=["lost visual cortex detail", "smoothed calcarine sulcus"],
            common_issues_3T_to_7T=["coil sensitivity variations"],
        ),
        
        "parietal_lobe": BrainRegion(
            name="parietal_lobe",
            chinese_name="顶叶",
            spatial_description="superior posterior cortical region of the brain",
            chinese_spatial_description="大脑上后方皮层区域（顶叶）",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.5, 0.75, 0.4, 0.35, 0.3),
            common_issues_64mT_to_3T=["smoothed cortical folding", "poor contrast"],
            common_issues_3T_to_7T=["B1 inhomogeneity"],
        ),
        
        "insular_lobe": BrainRegion(
            name="insular_lobe",
            chinese_name="脑岛",
            spatial_description="deep lateral cortical region buried within the lateral sulcus",
            chinese_spatial_description="外侧沟内的深层皮层区域（脑岛）",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.45, 0.45, 0.22, 0.22, 0.18),
            common_issues_64mT_to_3T=["blurred sulcal boundary", "low contrast"],
            common_issues_3T_to_7T=["susceptibility artifacts near air-tissue interface"],
        ),
        
        "white_matter": BrainRegion(
            name="white_matter",
            chinese_name="脑白质",
            spatial_description="inner deep white matter region of the brain",
            chinese_spatial_description="大脑内部深层白质区域",
            region_type=BrainRegionType.WHITE_MATTER,
            relative_bbox=(0.5, 0.5, 0.5, 0.6, 0.6, 0.5),
            common_issues_64mT_to_3T=["poor WM/GM contrast", "hyperintense artifacts"],
            common_issues_3T_to_7T=["WM lesion enhancement"],
        ),
        
        "lesion": BrainRegion(
            name="lesion",
            chinese_name="病灶",
            spatial_description="focal abnormal region",
            chinese_spatial_description="局灶性异常区域",
            region_type=BrainRegionType.BOUNDARY,
            relative_bbox=None,  # Variable location
            common_issues_64mT_to_3T=["missed small lesions", "poor lesion-tissue contrast"],
            common_issues_3T_to_7T=["lesion over-enhancement", "false positives"],
        ),
        # --- Additional Harvard-Oxford Regions ---
        "amygdala": BrainRegion(
            name="amygdala",
            chinese_name="杏仁核",
            spatial_description="deep medial temporal region adjacent to hippocampus",
            chinese_spatial_description="靠近海马体的颞叶内侧深部区域",
            region_type=BrainRegionType.DEEP,
            relative_bbox=(0.5, 0.62, 0.38, 0.1, 0.08, 0.08),
            common_issues_64mT_to_3T=["poor boundary definition", "low contrast with surrounding tissue"],
            common_issues_3T_to_7T=["susceptibility artifacts"],
        ),
        "caudate": BrainRegion(
            name="caudate",
            chinese_name="尾状核",
            spatial_description="C-shaped deep structure flanking the lateral ventricle",
            chinese_spatial_description="侧脑室两侧C形深部结构",
            region_type=BrainRegionType.DEEP,
            relative_bbox=(0.5, 0.45, 0.5, 0.2, 0.15, 0.15),
            common_issues_64mT_to_3T=["indistinct boundaries", "low contrast"],
            common_issues_3T_to_7T=["iron deposition artifacts"],
        ),
        "putamen": BrainRegion(
            name="putamen",
            chinese_name="壳核",
            spatial_description="lens-shaped deep structure lateral to the thalamus",
            chinese_spatial_description="丘脑外侧的透镜形深部结构",
            region_type=BrainRegionType.DEEP,
            relative_bbox=(0.5, 0.5, 0.48, 0.18, 0.13, 0.12),
            common_issues_64mT_to_3T=["indistinct boundaries", "poor contrast"],
            common_issues_3T_to_7T=["iron susceptibility artifacts"],
        ),
        "pallidum": BrainRegion(
            name="pallidum",
            chinese_name="苍白球",
            spatial_description="small deep structure medial to the putamen",
            chinese_spatial_description="壳核内侧的小型深部结构",
            region_type=BrainRegionType.DEEP,
            relative_bbox=(0.5, 0.52, 0.49, 0.1, 0.1, 0.1),
            common_issues_64mT_to_3T=["low SNR", "poor boundary visibility"],
            common_issues_3T_to_7T=["heavy iron deposition artifacts"],
        ),
        "accumbens": BrainRegion(
            name="accumbens",
            chinese_name="伏隔核",
            spatial_description="small deep structure at the junction of caudate and putamen",
            chinese_spatial_description="尾状核与壳核交汇处的小型深部结构",
            region_type=BrainRegionType.DEEP,
            relative_bbox=(0.5, 0.55, 0.47, 0.07, 0.07, 0.07),
            common_issues_64mT_to_3T=["too small to resolve", "low contrast"],
            common_issues_3T_to_7T=["partial volume effects"],
        ),
        "lateral_ventricle": BrainRegion(
            name="lateral_ventricle",
            chinese_name="侧脑室",
            spatial_description="C-shaped CSF-filled cavity flanking the basal ganglia",
            chinese_spatial_description="基底神经节两侧C形脑脊液腔",
            region_type=BrainRegionType.VENTRICULAR,
            relative_bbox=(0.5, 0.45, 0.52, 0.3, 0.2, 0.3),
            common_issues_64mT_to_3T=["blurred CSF boundaries", "signal inhomogeneity"],
            common_issues_3T_to_7T=["flow artifacts"],
        ),
        "frontal_pole": BrainRegion(
            name="frontal_pole",
            chinese_name="额极",
            spatial_description="anterior-most tip of the frontal lobe",
            chinese_spatial_description="额叶最前端",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.15, 0.65, 0.25, 0.2, 0.2),
            common_issues_64mT_to_3T=["smoothed cortical surface", "poor gray-white contrast"],
            common_issues_3T_to_7T=["B1 field inhomogeneity"],
        ),
        "insular_cortex": BrainRegion(
            name="insular_cortex",
            chinese_name="脑岛皮层",
            spatial_description="cortex buried within the lateral sulcus",
            chinese_spatial_description="外侧沟内的皮层区域",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.45, 0.45, 0.2, 0.2, 0.15),
            common_issues_64mT_to_3T=["blurred sulcal details", "low contrast"],
            common_issues_3T_to_7T=["susceptibility artifacts near air"],
        ),
        "superior_frontal_gyrus": BrainRegion(
            name="superior_frontal_gyrus",
            chinese_name="额上回",
            spatial_description="superior surface of the frontal lobe",
            chinese_spatial_description="额叶上表面",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.25, 0.7, 0.35, 0.25, 0.25),
            common_issues_64mT_to_3T=["smoothed gyral pattern", "poor WM/GM differentiation"],
            common_issues_3T_to_7T=["edge ringing"],
        ),
        "middle_frontal_gyrus": BrainRegion(
            name="middle_frontal_gyrus",
            chinese_name="额中回",
            spatial_description="lateral frontal lobe between superior and inferior frontal sulci",
            chinese_spatial_description="额上沟与额下沟之间的外侧额叶",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.3, 0.62, 0.3, 0.2, 0.2),
            common_issues_64mT_to_3T=["smoothed sulci", "low contrast"],
            common_issues_3T_to_7T=["B1 inhomogeneity"],
        ),
        "inferior_frontal_gyrus": BrainRegion(
            name="inferior_frontal_gyrus",
            chinese_name="额下回",
            spatial_description="inferior lateral frontal lobe cortex",
            chinese_spatial_description="额叶外侧下部皮层",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.35, 0.55, 0.25, 0.2, 0.2),
            common_issues_64mT_to_3T=["smoothed sulcal pattern", "poor contrast"],
            common_issues_3T_to_7T=["susceptibility from nearby sinuses"],
        ),
        "precentral_gyrus": BrainRegion(
            name="precentral_gyrus",
            chinese_name="中央前回",
            spatial_description="cortex immediately anterior to the central sulcus (motor cortex)",
            chinese_spatial_description="中央沟前方皮层（运动皮层）",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.4, 0.68, 0.3, 0.25, 0.2),
            common_issues_64mT_to_3T=["smoothed gyral boundaries", "poor cortical thickness visibility"],
            common_issues_3T_to_7T=["motion sensitivity"],
        ),
        "temporal_pole": BrainRegion(
            name="temporal_pole",
            chinese_name="颞极",
            spatial_description="anterior-most tip of the temporal lobe",
            chinese_spatial_description="颞叶最前端",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.5, 0.3, 0.2, 0.15, 0.15),
            common_issues_64mT_to_3T=["susceptibility distortion", "signal dropout"],
            common_issues_3T_to_7T=["air-tissue interface artifacts"],
        ),
        "superior_temporal_gyrus": BrainRegion(
            name="superior_temporal_gyrus",
            chinese_name="颞上回",
            spatial_description="superior surface of the temporal lobe",
            chinese_spatial_description="颞叶上表面",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.5, 0.38, 0.3, 0.2, 0.2),
            common_issues_64mT_to_3T=["smoothed Heschl's gyrus", "poor auditory cortex detail"],
            common_issues_3T_to_7T=["B1 inhomogeneity"],
        ),
        "middle_temporal_gyrus": BrainRegion(
            name="middle_temporal_gyrus",
            chinese_name="颞中回",
            spatial_description="middle lateral surface of the temporal lobe",
            chinese_spatial_description="颞叶外侧中部",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.52, 0.35, 0.3, 0.18, 0.2),
            common_issues_64mT_to_3T=["poor sulcal definition"],
            common_issues_3T_to_7T=["susceptibility artifacts"],
        ),
        "inferior_temporal_gyrus": BrainRegion(
            name="inferior_temporal_gyrus",
            chinese_name="颞下回",
            spatial_description="inferior lateral surface of the temporal lobe",
            chinese_spatial_description="颞叶外侧下部",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.55, 0.32, 0.3, 0.15, 0.18),
            common_issues_64mT_to_3T=["susceptibility distortion"],
            common_issues_3T_to_7T=["air-tissue artifacts"],
        ),
        "postcentral_gyrus": BrainRegion(
            name="postcentral_gyrus",
            chinese_name="中央后回",
            spatial_description="cortex immediately posterior to the central sulcus (somatosensory cortex)",
            chinese_spatial_description="中央沟后方皮层（躯体感觉皮层）",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.45, 0.7, 0.3, 0.22, 0.2),
            common_issues_64mT_to_3T=["smoothed gyral pattern"],
            common_issues_3T_to_7T=["motion sensitivity"],
        ),
        "superior_parietal_lobule": BrainRegion(
            name="superior_parietal_lobule",
            chinese_name="顶上小叶",
            spatial_description="superior posterior parietal cortex",
            chinese_spatial_description="顶叶后上方皮层",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.45, 0.72, 0.35, 0.25, 0.2),
            common_issues_64mT_to_3T=["smoothed lobular boundaries"],
            common_issues_3T_to_7T=["B1 inhomogeneity"],
        ),
        "supramarginal_gyrus": BrainRegion(
            name="supramarginal_gyrus",
            chinese_name="缘上回",
            spatial_description="inferior parietal lobule surrounding the end of the lateral sulcus",
            chinese_spatial_description="外侧沟末端的顶下小叶",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.48, 0.62, 0.25, 0.2, 0.2),
            common_issues_64mT_to_3T=["poor sulcal definition"],
            common_issues_3T_to_7T=["susceptibility artifacts"],
        ),
        "angular_gyrus": BrainRegion(
            name="angular_gyrus",
            chinese_name="角回",
            spatial_description="posterior inferior parietal lobule surrounding the superior temporal sulcus",
            chinese_spatial_description="颞上沟末端的顶下小叶后部",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.5, 0.65, 0.25, 0.2, 0.18),
            common_issues_64mT_to_3T=["blurred sulcal details"],
            common_issues_3T_to_7T=["B1 field variations"],
        ),
        "lateral_occipital_cortex": BrainRegion(
            name="lateral_occipital_cortex",
            chinese_name="外侧枕叶皮层",
            spatial_description="lateral surface of the occipital lobe",
            chinese_spatial_description="枕叶外侧表面",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.75, 0.6, 0.3, 0.25, 0.2),
            common_issues_64mT_to_3T=["smoothed visual area boundaries"],
            common_issues_3T_to_7T=["coil sensitivity artifacts"],
        ),
        "intracalcarine_cortex": BrainRegion(
            name="intracalcarine_cortex",
            chinese_name="距状裂皮层",
            spatial_description="medial occipital cortex along the calcarine fissure (primary visual cortex)",
            chinese_spatial_description="距状裂沿线内侧枕叶皮层（初级视皮层）",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.78, 0.52, 0.2, 0.15, 0.12),
            common_issues_64mT_to_3T=["lost calcarine detail", "poor visual cortex definition"],
            common_issues_3T_to_7T=["susceptibility artifacts"],
        ),
        "frontal_medial_cortex": BrainRegion(
            name="frontal_medial_cortex",
            chinese_name="内侧额叶皮层",
            spatial_description="medial surface of the frontal lobe",
            chinese_spatial_description="额叶内侧表面",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.2, 0.65, 0.2, 0.25, 0.18),
            common_issues_64mT_to_3T=["poor medial cortex definition"],
            common_issues_3T_to_7T=["B1 inhomogeneity at midline"],
        ),
        "paracingulate_gyrus": BrainRegion(
            name="paracingulate_gyrus",
            chinese_name="旁扣带回",
            spatial_description="medial frontal cortex parallel to the cingulate sulcus",
            chinese_spatial_description="扣带沟平行的内侧额叶皮层",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.3, 0.63, 0.15, 0.15, 0.15),
            common_issues_64mT_to_3T=["indistinct from cingulate"],
            common_issues_3T_to_7T=["partial volume effects"],
        ),
        "cingulate_gyrus": BrainRegion(
            name="cingulate_gyrus",
            chinese_name="扣带回",
            spatial_description="medial cortex arching over the corpus callosum",
            chinese_spatial_description="胼胝体上方的弧形内侧皮层",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.4, 0.6, 0.2, 0.3, 0.15),
            common_issues_64mT_to_3T=["poor cingulate sulcus definition"],
            common_issues_3T_to_7T=["B1 inhomogeneity"],
        ),
        "precuneus_cortex": BrainRegion(
            name="precuneus_cortex",
            chinese_name="楔前叶皮层",
            spatial_description="medial parietal cortex between the cingulate sulcus and parieto-occipital sulcus",
            chinese_spatial_description="扣带沟与顶枕沟之间的内侧顶叶皮层",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.55, 0.72, 0.2, 0.25, 0.18),
            common_issues_64mT_to_3T=["smoothed parietal detail"],
            common_issues_3T_to_7T=["B1 field variations"],
        ),
        "cuneal_cortex": BrainRegion(
            name="cuneal_cortex",
            chinese_name="楔叶皮层",
            spatial_description="medial occipital cortex between the parieto-occipital and calcarine sulci",
            chinese_spatial_description="顶枕沟与距状裂之间的内侧枕叶皮层",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.75, 0.68, 0.2, 0.18, 0.15),
            common_issues_64mT_to_3T=["lost visual cortex laminar detail"],
            common_issues_3T_to_7T=["coil artifacts"],
        ),
        "frontal_orbital_cortex": BrainRegion(
            name="frontal_orbital_cortex",
            chinese_name="眶额皮层",
            spatial_description="inferior surface of the frontal lobe, above the orbits",
            chinese_spatial_description="额叶下表面，眼眶上方",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.35, 0.52, 0.25, 0.15, 0.15),
            common_issues_64mT_to_3T=["susceptibility distortion near sinuses"],
            common_issues_3T_to_7T=["heavy susceptibility artifacts from air-tissue interface"],
        ),
        "parahippocampal_gyrus": BrainRegion(
            name="parahippocampal_gyrus",
            chinese_name="海马旁回",
            spatial_description="medial temporal lobe gyrus adjacent to the hippocampus",
            chinese_spatial_description="海马旁侧的内侧颞叶回",
            region_type=BrainRegionType.DEEP,
            relative_bbox=(0.5, 0.65, 0.38, 0.2, 0.12, 0.12),
            common_issues_64mT_to_3T=["poor boundary with hippocampus", "susceptibility distortion"],
            common_issues_3T_to_7T=["air-tissue artifacts"],
        ),
        "lingual_gyrus": BrainRegion(
            name="lingual_gyrus",
            chinese_name="舌回",
            spatial_description="inferior medial occipital gyrus below the calcarine fissure",
            chinese_spatial_description="距状裂下方内侧枕叶回",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.78, 0.45, 0.2, 0.15, 0.15),
            common_issues_64mT_to_3T=["poor visual cortex definition"],
            common_issues_3T_to_7T=["susceptibility artifacts"],
        ),
        "temporal_fusiform_cortex": BrainRegion(
            name="temporal_fusiform_cortex",
            chinese_name="颞梭形皮层",
            spatial_description="inferior temporal gyrus fusiform region (face/object recognition area)",
            chinese_spatial_description="颞下回梭状区域（面孔/物体识别区）",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.6, 0.35, 0.25, 0.15, 0.15),
            common_issues_64mT_to_3T=["poor inferior temporal definition"],
            common_issues_3T_to_7T=["susceptibility artifacts"],
        ),
        "occipital_fusiform_gyrus": BrainRegion(
            name="occipital_fusiform_gyrus",
            chinese_name="枕梭形回",
            spatial_description="inferior occipital fusiform region",
            chinese_spatial_description="枕叶下部梭形区域",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.78, 0.4, 0.2, 0.15, 0.12),
            common_issues_64mT_to_3T=["lost visual detail"],
            common_issues_3T_to_7T=["susceptibility artifacts"],
        ),
        "planum_temporale": BrainRegion(
            name="planum_temporale",
            chinese_name="颞平面",
            spatial_description="posterior superior temporal plane (Wernicke's area proximity)",
            chinese_spatial_description="颞叶上平面后部（邻近韦尼克区）",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.52, 0.42, 0.18, 0.12, 0.12),
            common_issues_64mT_to_3T=["poor auditory association cortex definition"],
            common_issues_3T_to_7T=["B1 inhomogeneity"],
        ),
        "planum_polare": BrainRegion(
            name="planum_polare",
            chinese_name="颞极平面",
            spatial_description="anterior superior temporal plane",
            chinese_spatial_description="颞叶上平面前部",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.48, 0.33, 0.15, 0.1, 0.1),
            common_issues_64mT_to_3T=["susceptibility distortion"],
            common_issues_3T_to_7T=["air artifacts"],
        ),
        "heschls_gyrus": BrainRegion(
            name="heschls_gyrus",
            chinese_name="颞横回",
            spatial_description="primary auditory cortex on the superior temporal plane",
            chinese_spatial_description="颞叶上平面初级听觉皮层",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.5, 0.4, 0.12, 0.08, 0.1),
            common_issues_64mT_to_3T=["too small to resolve clearly"],
            common_issues_3T_to_7T=["partial volume effects"],
        ),
        "occipital_pole": BrainRegion(
            name="occipital_pole",
            chinese_name="枕极",
            spatial_description="posterior-most tip of the occipital lobe",
            chinese_spatial_description="枕叶最后端",
            region_type=BrainRegionType.CORTICAL,
            relative_bbox=(0.5, 0.82, 0.55, 0.2, 0.2, 0.15),
            common_issues_64mT_to_3T=["lost calcarine detail", "poor visual cortex definition"],
            common_issues_3T_to_7T=["coil sensitivity variations"],
        ),
    }
    
    def __init__(self):
        self.regions = self.REGIONS.copy()
    
    def get_valid_region_names(self) -> List[str]:
        """Return the list of valid Harvard-Oxford atlas region names"""
        return list(self.HARVARD_OXFORD_REGIONS)
    
    def is_valid_region(self, name: str) -> bool:
        """Check if a region name is in the Harvard-Oxford atlas"""
        return name.lower() in self.HARVARD_OXFORD_REGIONS
    
    def get_region(self, name: str) -> Optional[BrainRegion]:
        """Get a brain region by name"""
        return self.regions.get(name.lower())
    
    def get_all_regions(self) -> List[BrainRegion]:
        """Get all brain regions"""
        return list(self.regions.values())
    
    def get_regions_by_type(self, region_type: BrainRegionType) -> List[BrainRegion]:
        """Get all regions of a specific type"""
        return [r for r in self.regions.values() if r.region_type == region_type]
    
    def get_spatial_description(self, region_name: str, language: str = "en") -> str:
        """
        Get spatial description for a region.
        
        Args:
            region_name: Name of the brain region
            language: "en" for English, "zh" for Chinese
            
        Returns:
            Spatial description string
        """
        region = self.get_region(region_name)
        if region is None:
            return f"the {region_name} region"
        
        if language == "zh":
            return region.chinese_spatial_description
        return region.spatial_description
    
    def name_to_spatial_prompt(self, region_name: str, issue: str = None) -> str:
        """
        Convert medical region name to a spatial prompt suitable for diffusion models.
        
        Args:
            region_name: Medical name of the region
            issue: Optional issue description to include
            
        Returns:
            Spatial prompt string
        """
        region = self.get_region(region_name)
        if region is None:
            base = f"the {region_name} area"
        else:
            base = f"the {region.spatial_description}"
        
        if issue:
            return f"{base}, which currently has {issue}"
        return base
    
    def set_region_masks(self, masks):
        masks = torch.as_tensor(masks).detach().cpu().float()
        if masks.ndim != 4 or masks.shape[0] != len(self.HARVARD_OXFORD_REGIONS):
            raise ValueError("Expected registered masks with shape (7, D, H, W)")
        if not torch.isfinite(masks).all():
            raise ValueError("Registered masks contain non-finite values")
        self._subject_masks = dict(zip(self.HARVARD_OXFORD_REGIONS, masks.unbind(0)))

    # Colour palette for the 7 Harvard-Oxford regions (R, G, B)
    REGION_COLORS: Dict[str, Tuple[int, int, int]] = {
        "frontal_lobe":   (255,  80,  80),   # red
        "parietal_lobe":  ( 80, 180, 255),   # blue
        "temporal_lobe":  ( 80, 255,  80),   # green
        "occipital_lobe": (255, 200,  60),   # yellow-orange
        "insular_lobe":   (200,  80, 255),   # purple
        "ventricles":     (  0, 220, 220),   # cyan
        "white_matter":   (255, 160, 200),   # pink
    }

    def _get_region_slice_2d(
        self,
        region_name: str,
        target_h: int,
        target_w: int,
        depth_fraction: float = 0.5,
    ) -> Optional[np.ndarray]:
        """Return a slice from the registered subject volume."""
        if not hasattr(self, "_subject_masks"):
            raise RuntimeError("Bind registered subject masks before generating an overlay")
        vol = self._subject_masks[region_name]
        index = max(0, min(len(vol)-1, round(depth_fraction * (len(vol)-1))))
        mask = vol[index]
        if tuple(mask.shape) != (target_h, target_w):
            raise ValueError("Overlay image and registered mask grids differ")
        return mask.numpy()

    def generate_segmentation_overlay(
        self,
        mri_pil: "Image.Image",
        depth_fraction: float = 0.5,
        alpha: float = 0.35,
        draw_labels: bool = True,
    ) -> "Image.Image":
        """Create a colour-coded brain-region overlay on an MRI slice.

        Uses the registered Harvard-Oxford labels on the subject image grid.

        Args:
            mri_pil: A PIL Image (RGB) of one axial MRI slice.
            depth_fraction: Position along the depth axis (0 = superior, 1 = inferior).
                            Used to select the correct 2-D atlas slice.
            alpha: Overlay opacity (0 = transparent, 1 = opaque).
            draw_labels: Whether to draw region-name labels on the overlay.

        Returns:
            PIL Image (RGB) of the same size with the atlas overlay composited.
        """
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        W_img, H_img = mri_pil.size  # PIL uses (W, H)

        # --- Build a combined colour overlay (H, W, 3) and weight map (H, W) ---
        overlay_np = np.zeros((H_img, W_img, 3), dtype=np.float32)
        weight_np  = np.zeros((H_img, W_img),    dtype=np.float32)

        for region_name in self.HARVARD_OXFORD_REGIONS:
            slice_2d = self._get_region_slice_2d(
                region_name, H_img, W_img, depth_fraction,
            )
            if slice_2d is None:
                continue

            mask_bin = (slice_2d > 0.3).astype(np.float32)
            if mask_bin.sum() < 10:
                continue

            colour = np.array(self.REGION_COLORS[region_name], dtype=np.float32)
            for c in range(3):
                overlay_np[:, :, c] += mask_bin * colour[c]
            weight_np += mask_bin

        # Normalise where multiple regions overlap
        overlap = weight_np > 1.0
        for c in range(3):
            overlay_np[:, :, c][overlap] /= weight_np[overlap]

        # --- Composite overlay onto the MRI ---
        mri_np = np.array(mri_pil).astype(np.float32)     # (H, W, 3)
        has_region = weight_np > 0
        for c in range(3):
            mri_np[:, :, c][has_region] = (
                (1.0 - alpha) * mri_np[:, :, c][has_region]
                + alpha * overlay_np[:, :, c][has_region]
            )
        result = Image.fromarray(mri_np.clip(0, 255).astype(np.uint8))

        # --- Draw region labels ---
        if draw_labels:
            draw = ImageDraw.Draw(result)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(10, H_img // 20))
            except Exception:
                font = ImageFont.load_default()

            for region_name in self.HARVARD_OXFORD_REGIONS:
                slice_2d = self._get_region_slice_2d(
                    region_name, H_img, W_img, depth_fraction,
                )
                if slice_2d is None:
                    continue
                mask_bin = (slice_2d > 0.3)
                ys, xs = np.where(mask_bin)
                if len(ys) < 10:
                    continue
                cy, cx = int(ys.mean()), int(xs.mean())

                colour = self.REGION_COLORS[region_name]
                label = region_name.replace("_", " ")
                draw.text((cx, cy), label, fill=colour, font=font, anchor="mm")

        return result

    def generate_region_mask(
        self,
        region_name: str,
        image_size: Tuple[int, int, int],  # (D, H, W) or (H, W)
        soft_boundary: bool = True,
        sigma: float = 5.0,
    ) -> torch.Tensor:
        """Return the current subject's registered regional mask."""
        import torch.nn.functional as F
        if not hasattr(self, "_subject_masks"):
            raise RuntimeError("Registered subject masks are required")
        if region_name not in self._subject_masks:
            raise ValueError(f"Unsupported anatomical region: {region_name}")
        mask = self._subject_masks[region_name]
        if len(image_size) == 2 and mask.shape[0] == 1:
            mask = mask[0]
        if tuple(mask.shape) != tuple(image_size):
            raise ValueError(f"Registered mask/image grid mismatch: {mask.shape} vs {image_size}")
        if soft_boundary:
            k = max(3, int(sigma) * 2 + 1)
            nd = mask.ndim
            padded = F.pad(mask[None,None], (k//2,) * (2*nd), mode="replicate")
            pool = F.avg_pool3d if nd == 3 else F.avg_pool2d
            mask = pool(padded, kernel_size=k, stride=1)[0,0]
        return mask.clamp(0, 1)

    def get_common_issues(self, region_name: str, task_type: str = "64mT_to_3T") -> List[str]:
        """
        Get common issues for a region in a specific task.
        
        Args:
            region_name: Name of the brain region
            task_type: Either "64mT_to_3T" or "3T_to_7T"
            
        Returns:
            List of common issue descriptions
        """
        region = self.get_region(region_name)
        if region is None:
            return []
        
        if task_type == "3T_to_7T":
            return region.common_issues_3T_to_7T
        return region.common_issues_64mT_to_3T


# Global instance for convenience
BRAIN_ATLAS = BrainAtlas()


def get_spatial_description(region_name: str, language: str = "en") -> str:
    """Convenience function to get spatial description"""
    return BRAIN_ATLAS.get_spatial_description(region_name, language)


def region_to_prompt(region_name: str, issue: str = None) -> str:
    """Convenience function to convert region name to prompt"""
    return BRAIN_ATLAS.name_to_spatial_prompt(region_name, issue)
