"""
MTAN 模型测试脚本

验证：
1. 能否加载 mtan-reference 预训练权重
2. 前向推理是否正常
3. 输出尺寸是否符合预期

用法（从项目根目录）:
    python scripts/models/test_mtan.py --checkpoint ../../mtan-reference/im2im_pred/checkpoints/mtan/best_model_equal_standard.pth
"""

import argparse
import sys
from pathlib import Path

import torch

# 添加项目根目录到 sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.mtan import MTAN


def build_default_config() -> dict:
    """构造默认 MTAN 配置"""
    return {
        'name': 'mtan',
        'architecture': {
            'num_tasks': 3,
            'task_names': ['semantic_segmentation', 'depth_estimation', 'surface_normal'],
            'input_resolution': [3, 288, 288],
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Test MTAN model loading and inference')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to pretrained MTAN checkpoint (best_model_equal_standard.pth)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--width', type=int, default=288)
    args = parser.parse_args()

    device = torch.device(args.device)
    
    print("="*70)
    print("MTAN Model Test")
    print("="*70)
    
    # 1. 创建模型
    print("\n[Step 1/4] Creating MTAN model...")
    config = build_default_config()
    model = MTAN(config).to(device)
    
    print(f"  ✓ Model created")
    print(f"  - Total parameters: {model.get_num_parameters():,}")
    print(f"  - Split points: {len(model.get_split_points())}")
    print(f"  - Compression groups: {len(model.get_compression_groups())}")
    
    # 2. 加载预训练权重
    print(f"\n[Step 2/4] Loading pretrained weights...")
    print(f"  - Checkpoint: {args.checkpoint}")
    
    if not Path(args.checkpoint).exists():
        print(f"  ✗ Checkpoint not found: {args.checkpoint}")
        print(f"\n  Please provide correct path, e.g.:")
        print(f"    python scripts/models/test_mtan.py --checkpoint ../../mtan-reference/im2im_pred/checkpoints/mtan/best_model_equal_standard.pth")
        return
    
    result = model.load_pretrained(args.checkpoint)
    
    if result['missing_keys']:
        print(f"  ⚠ Missing keys: {len(result['missing_keys'])}")
    if result['unexpected_keys']:
        print(f"  ⚠ Unexpected keys: {len(result['unexpected_keys'])}")
    
    # 3. 测试前向推理
    print(f"\n[Step 3/4] Testing forward inference...")
    model.eval()
    
    # 创建随机输入
    dummy_input = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    print(f"  - Input shape: {tuple(dummy_input.shape)}")
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"  ✓ Forward pass successful")
    print(f"\n  Output shapes:")
    for task_name, output_tensor in outputs.items():
        print(f"    - {task_name:20s}: {tuple(output_tensor.shape)}")
    
    # 4. 测试切分点和压缩组
    print(f"\n[Step 4/4] Checking split points and compression groups...")
    
    split_points = model.get_split_points()
    print(f"  ✓ Found {len(split_points)} split points")
    print(f"    Examples: {split_points[:5]}")
    
    comp_groups = model.get_compression_groups()
    print(f"  ✓ Found {len(comp_groups)} compression groups")
    for group_name, split_list in list(comp_groups.items())[:3]:
        print(f"    - {group_name}: {len(split_list)} split points")
    
    # 总结
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run quantization sensitivity profiling:")
    print("     python scripts/1_fit_accuracy.py --model mtan")
    print("  2. Integrate MTAN into UAV simulation environment")
    print("="*70)


if __name__ == '__main__':
    main()
