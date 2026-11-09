#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRI Enhancement Evaluation Script
对比生成图像与GT的各种质量指标
指标包括: SSIM, PSNR, LPIPS, NRMSE
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
torch.hub.set_dir(str(Path(__file__).resolve().parents[1] / 'weights' / 'metrics'))
import torch.nn.functional as F
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# MONAI imports for preprocessing
import monai.transforms as mtransforms

# LPIPS metric
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. LPIPS metric will be skipped.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MRI Enhancement Metrics")
    parser.add_argument("--gen_dir", type=str, 
                        default="runs/predictions/",
                        help="生成图像文件夹路径")
    parser.add_argument("--gt_dir", type=str, 
                        default="data/",
                        help="GT图像文件夹路径")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[256, 256, 160],
                        help="目标空间尺寸 (H, W, D)")
    parser.add_argument("--gpu", type=str, default="3",
                        help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="处理2D切片时的batch size")
    return parser.parse_args()


class MONAIPreprocessor:
    """
    使用与训练相同的MONAI预处理流程处理GT图像
    """
    def __init__(self, spatial_size=(256, 256, 160)):
        self.spatial_size = spatial_size
        
        # 定义预处理流程（与训练保持一致）
        self.preprocess = mtransforms.Compose([
            mtransforms.LoadImaged(keys=["image"], image_only=True),
            mtransforms.EnsureChannelFirstd(keys=["image"]),
            mtransforms.Spacingd(
                keys=["image"], 
                pixdim=(1.0, 1.0, 1.0), 
                mode=("bilinear",)
            ),
            mtransforms.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5, upper=99.5, 
                b_min=-1.0, b_max=1.0, 
                clip=True, relative=False
            ),
            mtransforms.Resized(
                keys=["image"],
                spatial_size=self.spatial_size,
                mode=("bilinear",)
            ),
            mtransforms.EnsureTyped(keys=["image"], dtype=torch.float32),
        ])
    
    def __call__(self, image_path):
        """
        预处理单个图像
        
        Args:
            image_path: NIfTI文件路径
            
        Returns:
            tensor: (H, W, D) 范围 [-1, 1]
        """
        data_dict = {"image": image_path}
        data_dict = self.preprocess(data_dict)
        # 输出: (1, H, W, D) -> (H, W, D)
        return data_dict["image"].squeeze(0)


def load_generated_nii(nii_path, spatial_size=(256, 256, 160)):
    """
    加载生成的NIfTI文件
    生成的文件已经是处理后的格式，range在[0, 1]
    需要转换为[-1, 1]并确保尺寸正确
    """
    nii = nib.load(nii_path)
    data = nii.get_fdata().astype(np.float32)
    
    # 转换为tensor
    data_tensor = torch.from_numpy(data)
    
    # 检查并调整尺寸
    if data_tensor.shape != tuple(spatial_size):
        # 需要resize
        # (H, W, D) -> (1, 1, H, W, D)
        data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)
        data_tensor = F.interpolate(
            data_tensor, 
            size=spatial_size, 
            mode='trilinear', 
            align_corners=False
        )
        data_tensor = data_tensor.squeeze(0).squeeze(0)
    
    # 生成图像在 [0, 1] 范围，转换为 [-1, 1]
    data_tensor = data_tensor * 2.0 - 1.0
    
    return data_tensor


def find_matching_samples(gen_dir, gt_dir):
    """
    找到生成文件夹和GT文件夹中匹配的样本
    
    生成文件夹结构: gen_dir/dataset/patient/modality/enhanced.nii
    GT文件夹结构: gt_dir/dataset/patient/modality/field.nii
    
    Returns:
        list of dict: 每个dict包含gen_path, gt_path, sample_info
    """
    matched_samples = []
    processed_folders = set()  # 避免同一文件夹被处理多次
    
    # 遍历生成文件夹
    for root, dirs, files in os.walk(gen_dir):
        # 如果这个文件夹已经处理过，跳过
        if root in processed_folders:
            continue
            
        # 优先找 enhanced.nii，其次 enhanced.nii.gz
        gen_file = None
        if 'enhanced.nii' in files:
            gen_file = 'enhanced.nii'
        elif 'enhanced.nii.gz' in files:
            gen_file = 'enhanced.nii.gz'
        
        if gen_file:
            processed_folders.add(root)
            gen_path = os.path.join(root, gen_file)
            
            # 获取相对路径
            rel_path = os.path.relpath(root, gen_dir)
            
            # 解析路径结构: dataset/patient/modality
            parts = rel_path.split(os.sep)
            if len(parts) < 3:
                continue
            
            dataset = parts[0]  # e.g., "64m_to_3-1", "3_to_7-1"
            patient = parts[1]
            modality = parts[2] if len(parts) > 2 else None
            
            # 确定GT文件名
            # 根据任务类型确定GT场强
            if '64m_to_3' in dataset or '15_to_3' in dataset:
                gt_field = '3T'
            elif '3_to_7' in dataset:
                gt_field = '7T'
            else:
                # 尝试从路径推断
                gt_field = None
            
            # 构建可能的GT路径
            gt_folder = os.path.join(gt_dir, rel_path)
            
            # 尝试多种可能的GT文件名
            possible_gt_names = []
            if gt_field:
                possible_gt_names.extend([
                    f'{gt_field}.nii',
                    f'{gt_field}.nii.gz',
                    f'{gt_field.lower()}.nii',
                    f'{gt_field.lower()}.nii.gz',
                ])
            # 添加其他可能的名称
            possible_gt_names.extend([
                '3T.nii', '3T.nii.gz', '3t.nii', '3t.nii.gz',
                '7T.nii', '7T.nii.gz', '7t.nii', '7t.nii.gz',
                '1.5T.nii', '1.5T.nii.gz',
            ])
            
            gt_path = None
            for gt_name in possible_gt_names:
                candidate = os.path.join(gt_folder, gt_name)
                if os.path.exists(candidate):
                    gt_path = candidate
                    break
            
            if gt_path is None:
                print(f"Warning: GT not found for {gen_path}")
                print(f"  Searched in: {gt_folder}")
                continue
            
            matched_samples.append({
                'gen_path': gen_path,
                'gt_path': gt_path,
                'dataset': dataset,
                'patient': patient,
                'modality': modality,
                'rel_path': rel_path
            })
    
    return matched_samples


class MetricCalculator:
    """计算各种图像质量指标: SSIM, PSNR, LPIPS, NRMSE"""
    
    def __init__(self, device='cuda', batch_size=8):
        self.device = device
        self.batch_size = batch_size
        
        # 初始化LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None
    
    def tensor_to_0_1(self, tensor):
        """将[-1, 1]范围的tensor转换为[0, 1]"""
        return (tensor + 1.0) / 2.0
    
    def calculate_ssim(self, pred, gt):
        """
        计算SSIM (整个3D volume)
        
        Args:
            pred: (H, W, D) tensor, range [-1, 1]
            gt: (H, W, D) tensor, range [-1, 1]
        """
        pred_np = self.tensor_to_0_1(pred).cpu().numpy()
        gt_np = self.tensor_to_0_1(gt).cpu().numpy()
        
        # 计算3D SSIM
        return ssim(pred_np, gt_np, data_range=1.0)
    
    def calculate_psnr(self, pred, gt):
        """
        计算PSNR (整个3D volume)
        """
        pred_np = self.tensor_to_0_1(pred).cpu().numpy()
        gt_np = self.tensor_to_0_1(gt).cpu().numpy()
        
        return psnr(gt_np, pred_np, data_range=1.0)
    
    def calculate_lpips_volume(self, pred, gt):
        """
        计算LPIPS (对所有2D切片求平均)
        
        Args:
            pred: (H, W, D) tensor
            gt: (H, W, D) tensor
        """
        if self.lpips_fn is None:
            return np.nan
        
        pred = self.tensor_to_0_1(pred)  # [0, 1]
        gt = self.tensor_to_0_1(gt)
        
        # 转换为 [0, 1] -> [-1, 1] for LPIPS
        pred = pred * 2 - 1
        gt = gt * 2 - 1
        
        # 遍历所有切片
        D = pred.shape[2]
        lpips_values = []
        
        for i in range(0, D, self.batch_size):
            batch_end = min(i + self.batch_size, D)
            
            # 获取切片并扩展为3通道
            pred_slices = pred[:, :, i:batch_end].permute(2, 0, 1).unsqueeze(1)  # (B, 1, H, W)
            gt_slices = gt[:, :, i:batch_end].permute(2, 0, 1).unsqueeze(1)
            
            # 复制为3通道
            pred_slices = pred_slices.expand(-1, 3, -1, -1).to(self.device)
            gt_slices = gt_slices.expand(-1, 3, -1, -1).to(self.device)
            
            with torch.no_grad():
                lpips_val = self.lpips_fn(pred_slices, gt_slices)
                lpips_values.extend(lpips_val.cpu().numpy().flatten().tolist())
        
        return np.mean(lpips_values)
    
    def calculate_nrmse(self, pred, gt):
        """
        计算NRMSE (Normalized Root Mean Square Error)
        归一化方式: 使用GT的数据范围 (max - min)
        
        Args:
            pred: (H, W, D) tensor, range [-1, 1]
            gt: (H, W, D) tensor, range [-1, 1]
        """
        pred_np = self.tensor_to_0_1(pred).cpu().numpy()
        gt_np = self.tensor_to_0_1(gt).cpu().numpy()
        
        # 计算MSE
        mse = np.mean((pred_np - gt_np) ** 2)
        rmse = np.sqrt(mse)
        
        # 使用GT的范围进行归一化
        gt_range = gt_np.max() - gt_np.min()
        if gt_range == 0:
            return np.nan
        
        nrmse = rmse / gt_range
        return nrmse


def main():
    args = parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化预处理器
    spatial_size = tuple(args.spatial_size)
    print(f"Target spatial size: {spatial_size}")
    preprocessor = MONAIPreprocessor(spatial_size=spatial_size)
    
    # 初始化指标计算器
    metric_calc = MetricCalculator(device=device, batch_size=args.batch_size)
    
    # 找到匹配的样本
    print(f"\nSearching for matching samples...")
    print(f"Generated folder: {args.gen_dir}")
    print(f"GT folder: {args.gt_dir}")
    
    matched_samples = find_matching_samples(args.gen_dir, args.gt_dir)
    print(f"Found {len(matched_samples)} matching samples")
    
    if len(matched_samples) == 0:
        print("No matching samples found. Please check the folder structure.")
        return
    
    # 存储结果
    results = []
    
    def get_task_group(dataset):
        if '64m_to_3' in dataset or '15_to_3' in dataset:
            return '64mT->3T'
        elif '3_to_7' in dataset:
            return '3T->7T'
        else:
            return 'Other'
    
    # 通样本计算指标
    print("\nCalculating metrics...")
    for sample in tqdm(matched_samples, desc="Evaluating"):
        try:
            # 加载生成图像
            gen = load_generated_nii(sample['gen_path'], spatial_size)
            
            # 加载并预处理GT图像
            gt = preprocessor(sample['gt_path'])
            
            # 计算指标
            sample_result = {
                'dataset': sample['dataset'],
                'patient': sample['patient'],
                'modality': sample['modality'],
                'rel_path': sample['rel_path'],
                'gen_path': sample['gen_path'],
                'gt_path': sample['gt_path'],
            }
            
            # SSIM
            sample_result['ssim'] = metric_calc.calculate_ssim(gen, gt)
            
            # PSNR
            sample_result['psnr'] = metric_calc.calculate_psnr(gen, gt)
            
            # LPIPS
            sample_result['lpips'] = metric_calc.calculate_lpips_volume(gen, gt)
            
            # NRMSE
            sample_result['nrmse'] = metric_calc.calculate_nrmse(gen, gt)
            
            results.append(sample_result)
            
            # 打印单样本结果
            print(f"\n{sample['rel_path']}:")
            print(f"  SSIM: {sample_result['ssim']:.4f}, PSNR: {sample_result['psnr']:.2f}, "
                  f"LPIPS: {sample_result['lpips']:.4f}, NRMSE: {sample_result['nrmse']:.4f}")
            
        except Exception as e:
            print(f"\nError processing {sample['gen_path']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(results) == 0:
        print("No results computed. Please check the data.")
        return
    
    # 创建DataFrame用于统计
    df = pd.DataFrame(results)
    
    # 计算汇总统计
    metric_columns = ['ssim', 'psnr', 'lpips', 'nrmse']
    
    df['task_group'] = df['dataset'].apply(get_task_group)
    
    # 获取所有模态并标准化（提前定义，供后续使用）
    def normalize_modality(mod):
        """标准化模态名称"""
        if mod is None:
            return 'Unknown'
        mod_upper = mod.upper()
        if 'T1' in mod_upper:
            return 'T1'
        elif 'T2' in mod_upper:
            return 'T2'
        elif 'FLAIR' in mod_upper:
            return 'FLAIR'
        else:
            return mod
    
    df['modality_normalized'] = df['modality'].apply(normalize_modality)
    
    # 打印汇总结果
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    # 总体统计
    print(f"\n【Overall】 (n={len(results)})")
    for metric in metric_columns:
        values = df[metric].dropna().values
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric.upper():8s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 64mT->3T 统计
    subset_64m = df[df['task_group'] == '64mT->3T']
    if len(subset_64m) > 0:
        print(f"\n【64mT->3T】 (n={len(subset_64m)})")
        for metric in metric_columns:
            values = subset_64m[metric].dropna().values
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric.upper():8s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 3T->7T 统计
    subset_3t = df[df['task_group'] == '3T->7T']
    if len(subset_3t) > 0:
        print(f"\n【3T->7T】 (n={len(subset_3t)})")
        for metric in metric_columns:
            values = subset_3t[metric].dropna().values
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric.upper():8s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # ========== 按模态统计 ==========
    print("\n" + "-" * 60)
    print("STATISTICS BY MODALITY")
    print("-" * 60)
    
    modalities = sorted(df['modality_normalized'].unique())
    
    # 1. 总体按模态统计
    print("\n【Overall by Modality】")
    for mod in modalities:
        subset_mod = df[df['modality_normalized'] == mod]
        if len(subset_mod) > 0:
            print(f"\n  ◆ {mod} (n={len(subset_mod)})")
            for metric in metric_columns:
                values = subset_mod[metric].dropna().values
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"    {metric.upper():8s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 2. 64mT->3T 按模态统计
    if len(subset_64m) > 0:
        print(f"\n【64mT->3T by Modality】")
        for mod in modalities:
            subset = subset_64m[subset_64m['modality_normalized'] == mod]
            if len(subset) > 0:
                print(f"\n  ◆ {mod} (n={len(subset)})")
                for metric in metric_columns:
                    values = subset[metric].dropna().values
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        print(f"    {metric.upper():8s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 3. 3T->7T 按模态统计
    if len(subset_3t) > 0:
        print(f"\n【3T->7T by Modality】")
        for mod in modalities:
            subset = subset_3t[subset_3t['modality_normalized'] == mod]
            if len(subset) > 0:
                print(f"\n  ◆ {mod} (n={len(subset)})")
                for metric in metric_columns:
                    values = subset[metric].dropna().values
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        print(f"    {metric.upper():8s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
