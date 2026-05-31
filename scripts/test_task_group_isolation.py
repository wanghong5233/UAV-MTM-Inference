"""
验证：task{k}_att/task{k}_branch 量化只影响对应任务分支的输出。

目的：
1) 检查 apply_quantization_to_group() 中 task-specific 组的 Hook gating 是否正确
2) 避免出现"量化 task0 但 depth/normal 也被同时量化"的隐藏不一致

使用方法：
    python scripts/test_task_group_isolation.py --model mtan
    python scripts/test_task_group_isolation.py --model split
    python scripts/test_task_group_isolation.py --model dense
    python scripts/test_task_group_isolation.py --model cross
"""

import sys
from pathlib import Path
import torch
import argparse

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_mtl import BaseMTLModel
from src.models.mtan import MTAN
from src.models.segnet_split import SplitSegNet
from src.models.segnet_dense import DenseSegNet
from src.models.segnet_cross import CrossStitchSegNet
from src.data.nyuv2 import create_nyuv2_dataloader


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item()


def build_model(model_name: str, split_type: str = "standard", checkpoint: str = None) -> BaseMTLModel:
    """构建模型（复用calibrate_sensitivity.py的逻辑）"""
    if model_name == "mtan":
        config = {"architecture": {"task_names": ["semantic", "depth", "normal"]}}
        model = MTAN(config)
    elif model_name == "split":
        config = {
            "architecture": {
                "task_names": ["semantic", "depth", "normal"],
                "split_type": split_type,
                "num_classes": 13,
                "input_resolution": [3, 288, 288],
            }
        }
        model = SplitSegNet(config)
    elif model_name == "dense":
        config = {
            "architecture": {
                "task_names": ["semantic", "depth", "normal"],
                "num_classes": 13,
                "input_resolution": [3, 288, 288],
            }
        }
        model = DenseSegNet(config)
    elif model_name == "cross":
        config = {
            "architecture": {
                "task_names": ["semantic", "depth", "normal"],
                "num_classes": 13,
                "input_resolution": [3, 288, 288],
            }
        }
        model = CrossStitchSegNet(config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Default checkpoint by model type
    if checkpoint is None:
        if model_name == "mtan":
            checkpoint = "../mtan-reference/im2im_pred/checkpoints/mtan/best_model_equal_standard.pth"
        elif model_name == "split":
            checkpoint = "../mtan-reference/im2im_pred/checkpoints/split_network/best_model_equal_standard.pth"
        elif model_name == "dense":
            checkpoint = "../mtan-reference/im2im_pred/checkpoints/dense/best_model_equal_standard.pth"
        elif model_name == "cross":
            checkpoint = "../mtan-reference/im2im_pred/checkpoints/crossstitch/best_model_equal_standard.pth"

    model.load_pretrained(checkpoint)
    model.eval()
    return model


def get_task_specific_groups(model_name: str):
    """获取模型的任务特定组名称"""
    if model_name == "mtan":
        # MTAN: task{k}_att_enc, task{k}_att_dec (新的优化后分组)
        return [
            ("task0_att_enc", 0, "semantic(seg)-enc"),
            ("task0_att_dec", 0, "semantic(seg)-dec"),
            ("task1_att_enc", 1, "depth-enc"),
            ("task1_att_dec", 1, "depth-dec"),
            ("task2_att_enc", 2, "normal-enc"),
            ("task2_att_dec", 2, "normal-dec"),
        ]
    elif model_name == "split":
        # Split: 没有任务特定组（全部是shared），跳过隔离测试
        return []
    elif model_name == "dense":
        # Dense: task{k}_enc, task{k}_dec
        return [
            ("task0_enc", 0, "semantic(seg)-enc"),
            ("task0_dec", 0, "semantic(seg)-dec"),
            ("task1_enc", 1, "depth-enc"),
            ("task1_dec", 1, "depth-dec"),
            ("task2_enc", 2, "normal-enc"),
            ("task2_dec", 2, "normal-dec"),
        ]
    elif model_name == "cross":
        # Cross-Stitch: 每个任务有独立的encoder/decoder，但是通过cross-stitch共享
        # 量化某个任务的特定阶段理论上只影响该任务（由于cross-stitch的作用可能有轻微串扰）
        # 这里测试 enc_stage_0（所有任务共享）vs enc_stage_0对某个任务的影响
        # Cross-Stitch的组是按stage分的，每个stage包含3个任务的对应层
        # 实际上Cross-Stitch没有"task-specific"的组，所有组都包含3个任务
        return []
    else:
        return []


def main():
    parser = argparse.ArgumentParser(description="任务组隔离性验证")
    parser.add_argument('--model', type=str, default='mtan', choices=['mtan', 'split', 'dense', 'cross'],
                        help='模型类型')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='预训练权重路径（默认随模型类型切换）')
    parser.add_argument('--split_type', type=str, default='standard',
                        choices=['standard', 'wide', 'deep'],
                        help='Split SegNet 类型（仅 model=split 时生效）')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备（默认自动选择）')
    args = parser.parse_args()
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    bit_width = 4

    model = build_model(args.model, args.split_type, args.checkpoint)
    model.to(device)

    dataloader = create_nyuv2_dataloader(
        root="data/nyuv2",
        split="val",
        batch_size=1,
        shuffle=False,
        num_workers=0,
        augmentation=False,
    )
    batch = next(iter(dataloader))
    images = batch["image"].to(device)

    with torch.no_grad():
        preds_full, _ = model.core_net(images)

    print("\n" + "=" * 70)
    print(f"Task-group isolation test (model: {args.model})")
    print("=" * 70)
    print(f"device = {device}")
    print(f"bit_width = {bit_width}")

    # 获取任务特定组
    task_groups = get_task_specific_groups(args.model)
    
    if not task_groups:
        print(f"\n  ⚠ 注意：模型 {args.model} 没有任务特定的压缩组（全部shared或不适用隔离测试）")
        print(f"  跳过任务隔离测试。")
    else:
        for group_name, target_idx, target_name in task_groups:
            try:
                handles = model.apply_quantization_to_group(group_name, bit_width=bit_width)
                with torch.no_grad():
                    preds_q, _ = model.core_net(images)
                for h in handles:
                    h.remove()

                diffs = [mean_abs_diff(preds_q[i], preds_full[i]) for i in range(3)]
                target_diff = diffs[target_idx]
                other_diffs = [d for i, d in enumerate(diffs) if i != target_idx]

                print(f"\nGroup: {group_name} (target: {target_name})")
                print(f"  seg diff   : {diffs[0]:.6e}")
                print(f"  depth diff : {diffs[1]:.6e}")
                print(f"  normal diff: {diffs[2]:.6e}")

                # 经验性判据：目标任务的 diff 应明显大于非目标任务（避免数值噪声误判）
                if target_diff > (max(other_diffs) * 10.0 + 1e-8):
                    print("  ✓ Looks isolated (target diff dominates).")
                else:
                    print("  ⚠ Isolation may be weak; check hook gating / numerical noise.")
            except KeyError:
                print(f"\n  ⚠ Group '{group_name}' not found in model (may use old naming)")

    # 共享组对照：量化 shared encoder 理论上会影响所有任务输出
    shared_group = "enc_stage_0"
    handles = model.apply_quantization_to_group(shared_group, bit_width=bit_width)
    with torch.no_grad():
        preds_q, _ = model.core_net(images)
    for h in handles:
        h.remove()
    diffs = [mean_abs_diff(preds_q[i], preds_full[i]) for i in range(3)]

    print(f"\nGroup: {shared_group} (shared, expected to affect all tasks)")
    print(f"  seg diff   : {diffs[0]:.6e}")
    print(f"  depth diff : {diffs[1]:.6e}")
    print(f"  normal diff: {diffs[2]:.6e}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
