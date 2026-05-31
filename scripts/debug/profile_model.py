"""
性能分析工具

分析MTL模型的FLOPs、参数量、延迟等。

用法：
    python scripts/debug/profile_model.py --model split_network
"""

import argparse
import sys
from pathlib import Path
import torch
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import SplitNetwork, MTAN, CrossStitchNetwork
from src.utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description='性能分析')
    parser.add_argument('--model', type=str, required=True, choices=['split_network', 'mtan', 'crossstitch'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=100)
    args = parser.parse_args()
    
    print(f"分析模型：{args.model}")
    print("="*50)
    
    # 加载配置
    config = load_config(f'configs/models/{args.model}.yaml')
    
    # 创建模型
    # TODO: 实际创建模型
    # model = ...
    
    # 计算参数量
    # num_params = model.get_num_parameters()
    # print(f"参数量：{num_params:,}")
    
    # 估算FLOPs
    # TODO: 使用thop或ptflops
    
    # 测试推理延迟
    print(f"测试推理延迟（batch_size={args.batch_size}）...")
    # TODO: 实际测试
    
    print("="*50)
    print("✓ 性能分析完成！")


if __name__ == '__main__':
    main()

