"""
量化敏感度标定脚本

对每个压缩组 × 每个量化级别，测量精度下降 S_g^(p)。
对应论文公式：S_g^(p) = A^{full} - A^{(g,p)}

使用方法：
    python scripts/calibrate_sensitivity.py --model mtan --dataset nyuv2 --output results/accuracy_modeling/sensitivity_mtan.pkl
    python scripts/calibrate_sensitivity.py --model split --split_type standard --dataset nyuv2 --output results/accuracy_modeling/sensitivity_split.pkl
    python scripts/calibrate_sensitivity.py --model dense --dataset nyuv2 --output results/accuracy_modeling/sensitivity_dense.pkl
    python scripts/calibrate_sensitivity.py --model cross --dataset nyuv2 --output results/accuracy_modeling/sensitivity_cross.pkl
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import pickle
from typing import Optional, Dict, List
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_mtl import BaseMTLModel
from src.models.mtan import MTAN
from src.models.segnet_split import SplitSegNet
from src.models.segnet_dense import DenseSegNet
from src.models.segnet_cross import CrossStitchSegNet
from src.accuracy_modeling.quantization import uniform_quantize
from src.accuracy_modeling.aggregate_accuracy import compute_aggregate_accuracy_relative
from src.data.nyuv2 import create_nyuv2_dataloader


def _atomic_pickle_dump(obj, path: Path) -> None:
    """原子写入 pickle，避免中途崩溃导致输出文件损坏。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)

ACCURACY_AGGREGATION = "relative_degradation_v1"
AGG_WEIGHTS = (1.0, 1.0, 1.0)  # (seg, depth, normal)


# ============================================================
# 评估工具函数（复用自 mtan-reference/im2im_pred/utils.py）
# ============================================================

class ConfMatrix(object):
    """混淆矩阵（用于语义分割 mIoU 和 Accuracy 计算）"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item(), acc.item()


def depth_error(x_pred, x_output):
    """深度估计误差（abs_err 和 rel_err）"""
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output):
    """表面法向量误差（mean/median angle error）"""
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


# ============================================================


def evaluate_mtan(model: nn.Module, dataloader: DataLoader, device: str = 'cuda') -> dict:
    """
    评估 MTAN 模型在 NYUv2 验证集上的精度。

    仅返回论文建模所需的三项核心指标，避免无关输出造成困惑。

    Returns:
        metrics: {
            'seg_miou': float,       # Semantic Segmentation mIoU
            'depth_abs_err': float,  # Depth Abs Error
            'normal_mean': float,    # Normal Mean Angle Error
        }
    """
    model.eval()
    model.to(device)
    
    # 初始化混淆矩阵（语义分割，13 类）
    conf_mat = ConfMatrix(num_classes=13)
    
    # 累加器
    total_depth_abs = 0.0
    total_normal_mean = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['image'].to(device)        # [B, 3, H, W]
            seg_gt = batch['semantic'].to(device)     # [B, H, W]
            depth_gt = batch['depth'].to(device)      # [B, 1, H, W]
            normal_gt = batch['normal'].to(device)    # [B, 3, H, W]
            
            # 前向推理（直接调用 core_net，保持和 mtan-reference 一致）
            test_pred, _ = model.core_net(images)
            # test_pred = [seg_pred, depth_pred, normal_pred]
            
            seg_pred = test_pred[0]     # [B, 13, H, W] log_softmax 输出
            depth_pred = test_pred[1]   # [B, 1, H, W]
            normal_pred = test_pred[2]  # [B, 3, H, W] 归一化的法向量
            
            # 1. Semantic Segmentation: 更新混淆矩阵
            conf_mat.update(seg_pred.argmax(1).flatten(), seg_gt.flatten())
            
            # 2. Depth Estimation: abs_err
            abs_err, _ = depth_error(depth_pred, depth_gt)
            total_depth_abs += abs_err
            
            # 3. Surface Normal: mean angle error
            mean_err, _, _, _, _ = normal_error(normal_pred, normal_gt)
            total_normal_mean += mean_err
            
            num_batches += 1
    
    # 计算最终指标
    seg_miou, _ = conf_mat.get_metrics()
    depth_abs_err = total_depth_abs / num_batches
    normal_mean_err = total_normal_mean / num_batches

    return {
        'seg_miou': seg_miou,
        'depth_abs_err': depth_abs_err,
        'normal_mean': normal_mean_err,
    }


def calibrate_activation_ranges(
    model: MTAN,
    dataloader: DataLoader,
    split_points: List[str],
    device: str = "cuda",
    per_channel: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    离线校准固定量化范围（min/max）。

    Returns:
        calib_stats: {split_name: {'f_min': tensor, 'f_max': tensor}}
    """
    model.eval()
    model.to(device)

    calib_stats: Dict[str, Dict[str, torch.Tensor]] = {
        sp: {"f_min": None, "f_max": None} for sp in split_points
    }

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Calibrating ranges", leave=False)):
            if max_batches is not None and i >= max_batches:
                break

            images = batch["image"].to(device)
            features = model.extract_features(images, split_points=split_points)

            for name, feat in features.items():
                if per_channel:
                    cur_min = feat.amin(dim=(0, 2, 3), keepdim=True)
                    cur_max = feat.amax(dim=(0, 2, 3), keepdim=True)
                else:
                    cur_min = feat.min()
                    cur_max = feat.max()

                cur_min = cur_min.detach().cpu()
                cur_max = cur_max.detach().cpu()

                if calib_stats[name]["f_min"] is None:
                    calib_stats[name]["f_min"] = cur_min
                    calib_stats[name]["f_max"] = cur_max
                else:
                    calib_stats[name]["f_min"] = torch.minimum(calib_stats[name]["f_min"], cur_min)
                    calib_stats[name]["f_max"] = torch.maximum(calib_stats[name]["f_max"], cur_max)

            # 释放临时特征，避免显存碎片
            del features
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    missing = [sp for sp, v in calib_stats.items() if v["f_min"] is None or v["f_max"] is None]
    if missing:
        raise ValueError(f"Calibration failed for split points: {missing}")

    return calib_stats


def calibrate_sensitivity(
    model: BaseMTLModel,
    dataloader: DataLoader,
    bit_widths: list = [32, 16, 8, 4],
    device: str = 'cuda',
    save_path: Optional[Path] = None,
    resume: bool = False,
    per_channel: bool = False,
    quant_mode: str = "dynamic",
    calib_batches: Optional[int] = 50,
) -> dict:
    """
    标定所有压缩组的量化敏感度
    
    Args:
        model: MTAN 模型
        dataloader: NYUv2 验证集 DataLoader
        bit_widths: 量化比特宽度列表
        device: 计算设备
        save_path: 输出文件路径（用于边跑边保存/断点续跑）
        resume: 若 save_path 已存在，则加载并跳过已完成的 (group, bit-width)
        per_channel: 是否按通道量化（True=per-channel，误差更小；False=per-tensor，更贴近通信量化）
        quant_mode: 量化范围模式（dynamic=每次自适应；fixed=离线校准固定范围）
        calib_batches: 固定量化的校准批次数（-1 或 None 表示全量）
    
    Returns:
        sensitivity_matrix: {
            'full_precision_acc': float,
            'sensitivity': {
                'enc_stage_0': {32: 0.0, 16: 0.01, 8: 0.03, 4: 0.08},
                'enc_stage_1': {32: 0.0, 16: 0.01, 8: 0.04, 4: 0.10},
                ...
            },
            'metadata': {
                'compression_groups': [...],
                'bit_widths': [32, 16, 8, 4],
            }
        }
    """
    model.to(device)
    compression_groups = model.get_compression_groups()
    if quant_mode not in {"dynamic", "fixed"}:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}")

    # ------------------------------------------------------------
    # Resume：加载已有结果（避免系统崩溃/重启后从头再跑）
    # ------------------------------------------------------------
    sensitivity = {}
    baseline_metrics = None
    A_full = None
    reference_metrics = None
    calib_stats = None
    if resume and save_path is not None and save_path.exists():
        try:
            with open(save_path, "rb") as f:
                existing = pickle.load(f)
            meta = existing.get("metadata", {}) or {}
            agg = meta.get("accuracy_aggregation", None)
            if agg != ACCURACY_AGGREGATION:
                # Back up the old file to avoid mixing different A definitions.
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                backup = save_path.with_suffix(save_path.suffix + f".bak-{ts}")
                try:
                    save_path.replace(backup)
                    print(f"⚠ Existing file uses different aggregation ({agg}); backed up to: {backup}")
                except Exception:
                    print(f"⚠ Existing file uses different aggregation ({agg}); cannot backup automatically. Start from scratch.")
                sensitivity = {}
                baseline_metrics = None
                A_full = None
                reference_metrics = None
            else:
                sensitivity = existing.get("sensitivity", {}) or {}
                baseline_metrics = existing.get("baseline_metrics", None)
                A_full = existing.get("full_precision_acc", None)
                reference_metrics = meta.get("reference_metrics", None)
                # 复用已有的固定量化统计
                if quant_mode == "fixed" and meta.get("quant_mode") == "fixed":
                    calib_stats = meta.get("calibration_stats", None)
                if reference_metrics is None and baseline_metrics is not None:
                    # Backward-compatible: derive reference metrics from stored baseline.
                    try:
                        reference_metrics = {
                            "seg_miou": float(baseline_metrics["seg_miou"]),
                            "depth_abs_err": float(baseline_metrics["depth_abs_err"]),
                            "normal_mean": float(baseline_metrics["normal_mean"]),
                        }
                    except Exception:
                        reference_metrics = None
                print(f"✓ Resume from existing file: {save_path}")
        except Exception as e:
            print(f"⚠ Failed to resume from {save_path}: {e}. Start from scratch.")
            sensitivity = {}
            baseline_metrics = None
            A_full = None
            reference_metrics = None
            calib_stats = None

    # 预创建所有组键，便于后续跳过判断
    for g in compression_groups.keys():
        sensitivity.setdefault(g, {})

    print("="*70)
    print("量化敏感度标定（Quantization Sensitivity Profiling）")
    print("="*70)
    print(f"压缩组数量: {len(compression_groups)}")
    print(f"量化级别: {bit_widths}")
    print(f"量化方式: {'per-channel' if per_channel else 'per-tensor'}")
    print(f"范围模式: {quant_mode}")
    effective_bws = [int(bw) for bw in bit_widths if int(bw) < 32]
    print(f"总实验次数(名义): {len(compression_groups) * len(bit_widths) + 1} (含基线)")
    print(f"总实验次数(实际): {len(compression_groups) * len(effective_bws) + 1} (跳过 32-bit 重复评估)")
    print("="*70)
    
    # Step 0: 固定量化范围校准（可选）
    if quant_mode == "fixed" and calib_stats is None:
        max_batches = None if calib_batches is None or calib_batches < 0 else int(calib_batches)
        print("\n[Step 0/2] 校准固定量化范围（min/max）...")
        split_points = model.get_split_points()
        calib_stats = calibrate_activation_ranges(
            model=model,
            dataloader=dataloader,
            split_points=split_points,
            device=device,
            per_channel=per_channel,
            max_batches=max_batches,
        )
        print(f"  ✓ Calibration done (batches={max_batches if max_batches is not None else 'ALL'})")

    # Step 1: 测量基线精度（全精度）
    if A_full is None or baseline_metrics is None or reference_metrics is None:
        print("\n[Step 1/2] 测量基线精度（Full Precision）...")
        baseline_metrics = evaluate_mtan(model, dataloader, device)
        reference_metrics = {
            "seg_miou": float(baseline_metrics["seg_miou"]),
            "depth_abs_err": float(baseline_metrics["depth_abs_err"]),
            "normal_mean": float(baseline_metrics["normal_mean"]),
        }
        agg_full = compute_aggregate_accuracy_relative(
            metrics=baseline_metrics,
            reference_metrics=reference_metrics,
            weights=AGG_WEIGHTS,
        )
        A_full = float(agg_full["A"])  # by definition, should be 1
        baseline_metrics.update(
            {
                "avg_metric_normalized": A_full,
                "A_seg": agg_full["A_seg"],
                "A_depth": agg_full["A_depth"],
                "A_normal": agg_full["A_normal"],
                "delta_seg": agg_full["delta_seg"],
                "delta_depth": agg_full["delta_depth"],
                "delta_normal": agg_full["delta_normal"],
            }
        )

        print(f"  ✓ Baseline Aggregate Accuracy (Normalized): {A_full:.6f} (w.r.t. full precision)")
        print(f"  📌 Full-precision metrics:")
        print(f"    - Seg mIoU: {baseline_metrics['seg_miou']:.4f}")
        print(f"    - Depth Abs Err: {baseline_metrics['depth_abs_err']:.4f} m")
        print(f"    - Normal Mean: {baseline_metrics['normal_mean']:.2f}°")
    else:
        print("\n[Step 1/2] 基线精度已存在（Resume），跳过评估。")
        print(f"  ✓ Baseline Aggregate Accuracy (Normalized): {float(A_full):.6f} (w.r.t. full precision)")

    # 先保存一次 baseline（避免后续崩溃连 baseline 都丢）
    result = {
        'full_precision_acc': A_full,
        'baseline_metrics': baseline_metrics,
        'sensitivity': sensitivity,
        'metadata': {
            'compression_groups': list(compression_groups.keys()),
            'bit_widths': bit_widths,
            'num_groups': len(compression_groups),
            'per_channel': per_channel,
            'quant_mode': quant_mode,
            'calibration_batches': calib_batches if quant_mode == "fixed" else None,
            'calibration_stats': calib_stats if quant_mode == "fixed" else None,
            'accuracy_aggregation': ACCURACY_AGGREGATION,
            'accuracy_weights': list(AGG_WEIGHTS),
            'reference_metrics': reference_metrics,
        }
    }
    if save_path is not None:
        _atomic_pickle_dump(result, save_path)
    
    print(f"\n[Step 2/2] 标定每个压缩组的敏感度...")
    
    for group_idx, (group_name, split_points) in enumerate(compression_groups.items(), 1):
        print(f"\n  [{group_idx}/{len(compression_groups)}] {group_name} ({len(split_points)} split points)")
        sensitivity.setdefault(group_name, {})
        
        for bw in bit_widths:
            if bw in sensitivity[group_name]:
                print(f"    - {bw}-bit: 已存在（Resume），跳过。")
                continue

            # 32-bit 视为全精度：不需要重复评估，直接置零敏感度
            if int(bw) >= 32:
                S_g_p = 0.0
                sensitivity[group_name][bw] = S_g_p
                print(f"    - {bw}-bit: A = {A_full:.6f}, S_g^(p) = {S_g_p:.6f} (full precision)")
                if save_path is not None:
                    result['sensitivity'] = sensitivity
                    _atomic_pickle_dump(result, save_path)
                continue

            # 注册量化 Hook
            handles = model.apply_quantization_to_group(
                group_name,
                bit_width=bw,
                per_channel=per_channel,
                quant_mode=quant_mode,
                calib_stats=calib_stats,
            )
            
            # 测量量化后的精度
            quant_metrics = evaluate_mtan(model, dataloader, device)
            agg_quant = compute_aggregate_accuracy_relative(
                metrics=quant_metrics,
                reference_metrics=reference_metrics,
                weights=AGG_WEIGHTS,
            )
            A_quant = float(agg_quant["A"])
            
            # 计算敏感度 S_g^(p) = A_full - A^(g,p)
            S_raw = A_full - A_quant
            # 数值噪声或偶发“量化改善”会导致 S_g^(p) < 0；为匹配论文建模假设，做非负截断
            if S_raw < 0:
                if S_raw < -1e-6:
                    print(f"    ⚠ Warning: A_quant > A_full by {-S_raw:.6f}; clamp S_g^(p) to 0.")
                S_g_p = 0.0
            else:
                S_g_p = S_raw
            sensitivity[group_name][bw] = S_g_p
            
            print(f"    - {bw}-bit: A = {A_quant:.6f}, S_g^(p) = {S_g_p:.6f}")
            
            # 移除 Hook
            for h in handles:
                h.remove()

            # 释放缓存（不影响数值，只降低显存碎片/峰值风险）
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 边跑边保存（防止系统崩溃/蓝屏导致进度丢失）
            if save_path is not None:
                result['sensitivity'] = sensitivity
                _atomic_pickle_dump(result, save_path)
    
    # 汇总结果（最终保存）
    result['sensitivity'] = sensitivity
    # metadata 里的 bit_widths 可能因为 resume/扩展而变化，做一次归一化
    all_bws = sorted({bw for g in sensitivity.values() for bw in g.keys()})
    result['metadata']['bit_widths'] = all_bws
    result['metadata']['per_channel'] = per_channel
    result['metadata']['quant_mode'] = quant_mode
    if quant_mode == "fixed":
        result['metadata']['calibration_batches'] = calib_batches
        result['metadata']['calibration_stats'] = calib_stats
    if save_path is not None:
        _atomic_pickle_dump(result, save_path)
    
    return result


def build_model(args) -> BaseMTLModel:
    if args.model == "mtan":
        config = {"architecture": {"task_names": ["semantic", "depth", "normal"]}}
        model = MTAN(config)
    elif args.model == "split":
        config = {
            "architecture": {
                "task_names": ["semantic", "depth", "normal"],
                "split_type": args.split_type,
                "num_classes": 13,
                "input_resolution": [3, 288, 288],
            }
        }
        model = SplitSegNet(config)
    elif args.model == "dense":
        config = {
            "architecture": {
                "task_names": ["semantic", "depth", "normal"],
                "num_classes": 13,
                "input_resolution": [3, 288, 288],
            }
        }
        model = DenseSegNet(config)
    elif args.model == "cross":
        config = {
            "architecture": {
                "task_names": ["semantic", "depth", "normal"],
                "num_classes": 13,
                "input_resolution": [3, 288, 288],
            }
        }
        model = CrossStitchSegNet(config)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Default checkpoint by model type
    if args.checkpoint is None:
        if args.model == "mtan":
            args.checkpoint = "../mtan-reference/im2im_pred/checkpoints/mtan/best_model_equal_standard.pth"
        elif args.model == "split":
            args.checkpoint = "../mtan-reference/im2im_pred/checkpoints/split_network/best_model_equal_standard.pth"
        elif args.model == "dense":
            args.checkpoint = "../mtan-reference/im2im_pred/checkpoints/dense/best_model_equal_standard.pth"
        elif args.model == "cross":
            args.checkpoint = "../mtan-reference/im2im_pred/checkpoints/crossstitch/best_model_equal_standard.pth"

    model.load_pretrained(args.checkpoint)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="量化敏感度标定")
    parser.add_argument('--model', type=str, default='mtan', choices=['mtan', 'split', 'dense', 'cross'],
                        help='模型类型')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='预训练权重路径（默认随模型类型切换）')
    parser.add_argument('--split_type', type=str, default='standard',
                        choices=['standard', 'wide', 'deep'],
                        help='Split SegNet 类型（仅 model=split 时生效）')
    parser.add_argument('--dataset', type=str, default='data/nyuv2', 
                        help='数据集路径（NYUv2）')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size')
    parser.add_argument('--num_workers', type=int,
                        default=0 if os.name == 'nt' else 4,
                        help='DataLoader worker 数（Windows 建议 0-2；崩溃可先设 0）')
    parser.add_argument('--pin_memory', type=int, default=1, choices=[0, 1],
                        help='是否使用 pinned memory（1=开启，0=关闭；系统不稳定可设 0）')
    parser.add_argument('--bit_widths', type=int, nargs='+', default=[32, 16, 8, 4],
                        help='量化比特宽度列表（对应论文 R={1, 1/2, 1/4, 1/8}）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径（.pkl）；默认随模型类型切换')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--resume', action='store_true',
                        help='若输出文件已存在，则断点续跑（跳过已完成的组/比特宽度）')
    parser.add_argument('--per_channel', type=int, default=0, choices=[0, 1],
                        help='量化方式：1=per-channel(误差更小)，0=per-tensor(默认)')
    parser.add_argument('--quant_mode', type=str, default='dynamic', choices=['dynamic', 'fixed'],
                        help='量化范围模式：dynamic=每次自适应 min/max；fixed=离线校准固定范围')
    parser.add_argument('--calib_batches', type=int, default=50,
                        help='固定量化校准批次数（-1=全量；仅 quant_mode=fixed 时生效）')
    args = parser.parse_args()
    
    # 创建输出目录（每个模型一个子文件夹）
    # 默认使用 fixed 模式，文件名包含 _fixed 后缀以明确标识
    if args.output is None:
        mode_suffix = "_fixed" if args.quant_mode == "fixed" else f"_{args.quant_mode}"
        args.output = f"results/accuracy_modeling/{args.model}/sensitivity_{args.model}{mode_suffix}.pkl"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    model = build_model(args)
    
    # 加载 NYUv2 数据集
    print("\n加载 NYUv2 数据集...")
    dataloader = create_nyuv2_dataloader(
        root=args.dataset,
        split='val',  # 使用验证集进行敏感度标定
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        augmentation=False,  # 标定时不使用数据增强
        pin_memory=bool(args.pin_memory),
    )
    
    # 执行敏感度标定
    result = calibrate_sensitivity(
        model=model,
        dataloader=dataloader,
        bit_widths=args.bit_widths,
        device=args.device,
        save_path=output_path,
        resume=args.resume,
        per_channel=bool(args.per_channel),
        quant_mode=args.quant_mode,
        calib_batches=args.calib_batches,
    )
    
    # 保存结果
    _atomic_pickle_dump(result, output_path)
    
    print("\n" + "="*70)
    print("✓ 标定完成！")
    print(f"  结果已保存至: {args.output}")
    print("="*70)
    
    # 打印摘要
    print("\n敏感度摘要（前 5 组）:")
    for i, (group_name, sens) in enumerate(list(result['sensitivity'].items())[:5], 1):
        print(f"  {i}. {group_name}:")
        for bw, s in sens.items():
            print(f"       {bw}-bit: S_g^(p) = {s:.6f}")


if __name__ == '__main__':
    main()
