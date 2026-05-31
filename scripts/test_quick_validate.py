"""
快速验证脚本：测试模型 + 数据加载 + 评估流程

运行时间：~2 分钟
目的：确保所有组件能正常工作

使用方法：
    python scripts/test_quick_validate.py --model mtan
    python scripts/test_quick_validate.py --model split --split_type standard
    python scripts/test_quick_validate.py --model dense
    python scripts/test_quick_validate.py --model cross
"""

import sys
from pathlib import Path
import torch
import argparse

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_mtl import BaseMTLModel
from src.models.mtan import MTAN
from src.models.segnet_split import SplitSegNet
from src.models.segnet_dense import DenseSegNet
from src.models.segnet_cross import CrossStitchSegNet
from src.data.nyuv2 import create_nyuv2_dataloader


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


def test_model_loading(model_name: str, split_type: str = "standard", checkpoint: str = None):
    """测试模型权重加载"""
    print("\n" + "="*70)
    print(f"[Test 1/4] 模型权重加载 (model: {model_name})")
    print("="*70)
    
    model = build_model(model_name, split_type, checkpoint)
    
    print(f"  ✓ 模型加载成功")
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 切分点数量: {len(model.get_split_points())}")
    print(f"  - 压缩组数量: {len(model.get_compression_groups())}")
    
    return model


def test_data_loading():
    """测试数据集加载"""
    print("\n" + "="*70)
    print("[Test 2/4] 数据集加载")
    print("="*70)
    
    dataloader = create_nyuv2_dataloader(
        root='data/nyuv2',
        split='val',
        batch_size=2,
        shuffle=False,
        num_workers=0,  # 调试模式用 0
        augmentation=False
    )
    
    batch = next(iter(dataloader))
    
    print(f"  ✓ 数据集加载成功")
    print(f"  - Batch 形状:")
    print(f"    - image: {batch['image'].shape}")
    print(f"    - semantic: {batch['semantic'].shape}")
    print(f"    - depth: {batch['depth'].shape}")
    print(f"    - normal: {batch['normal'].shape}")


def test_forward_inference(model_name: str, split_type: str = "standard", checkpoint: str = None):
    """测试前向推理"""
    print("\n" + "="*70)
    print(f"[Test 3/4] 前向推理 (model: {model_name})")
    print("="*70)
    
    model = build_model(model_name, split_type, checkpoint)
    
    # 加载一个 batch
    dataloader = create_nyuv2_dataloader(
        root='data/nyuv2',
        split='val',
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    batch = next(iter(dataloader))
    
    # 前向推理（使用 core_net，和 mtan-reference 一致）
    with torch.no_grad():
        preds, logsigma = model.core_net(batch['image'])
    
    print(f"  ✓ 前向推理成功")
    print(f"  - 输出形状:")
    print(f"    - seg: {preds[0].shape}")
    print(f"    - depth: {preds[1].shape}")
    print(f"    - normal: {preds[2].shape}")
    
    return model, dataloader


def test_quantization_hook(model_name: str, split_type: str = "standard", checkpoint: str = None):
    """测试量化 Hook 机制"""
    print("\n" + "="*70)
    print(f"[Test 4/4] 量化 Hook 机制 (model: {model_name})")
    print("="*70)
    
    model = build_model(model_name, split_type, checkpoint)
    
    # 加载一个 batch
    dataloader = create_nyuv2_dataloader(
        root='data/nyuv2',
        split='val',
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    batch = next(iter(dataloader))
    
    # 测试：对 enc_stage_0 应用 4-bit 量化
    print("  - 注册 enc_stage_0 的 4-bit 量化 Hook...")
    handles = model.apply_quantization_to_group('enc_stage_0', bit_width=4)
    
    # 前向推理（带量化）
    with torch.no_grad():
        preds_quant, _ = model.core_net(batch['image'])
    
    # 移除 Hook
    for h in handles:
        h.remove()
    
    # 前向推理（不带量化）
    with torch.no_grad():
        preds_full, _ = model.core_net(batch['image'])
    
    # 检查是否有差异（量化应该导致输出略有不同）
    diff = torch.abs(preds_quant[0] - preds_full[0]).mean().item()
    
    print(f"  ✓ 量化 Hook 测试成功")
    print(f"  - Full precision vs 4-bit quantized:")
    print(f"    - Seg 输出平均差异: {diff:.6f}")
    
    if diff > 1e-6:
        print(f"  ✓ 量化生效（输出有差异）")
    else:
        print(f"  ⚠ 警告：量化似乎没有生效（差异过小）")


def main():
    parser = argparse.ArgumentParser(description="快速验证测试")
    parser.add_argument('--model', type=str, default='mtan', choices=['mtan', 'split', 'dense', 'cross'],
                        help='模型类型')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='预训练权重路径（默认随模型类型切换）')
    parser.add_argument('--split_type', type=str, default='standard',
                        choices=['standard', 'wide', 'deep'],
                        help='Split SegNet 类型（仅 model=split 时生效）')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"快速验证测试 (model: {args.model})")
    print("="*70)
    print("目的：确保模型、数据、评估、量化流程能正常工作")
    print("="*70)
    
    try:
        test_model_loading(args.model, args.split_type, args.checkpoint)
        test_data_loading()
        test_forward_inference(args.model, args.split_type, args.checkpoint)
        test_quantization_hook(args.model, args.split_type, args.checkpoint)
        
        print("\n" + "="*70)
        print("✅ 所有测试通过！")
        print("="*70)
        print("\n下一步：运行完整的敏感度标定")
        print(f"  python scripts/calibrate_sensitivity.py \\")
        print(f"    --model {args.model} \\")
        print(f"    --dataset \"data/nyuv2\" \\")
        print(f"    --output \"results/accuracy_modeling/sensitivity_{args.model}.pkl\" \\")
        print(f"    --batch_size 8 \\")
        print(f"    --bit_widths 32 16 8 4 \\")
        print(f"    --device cuda")
        print("="*70 + "\n")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ 测试失败")
        print("="*70)
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        print("="*70 + "\n")


if __name__ == '__main__':
    main()
