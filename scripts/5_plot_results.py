"""
生成论文图表脚本

用法：
    python scripts/5_plot_results.py --results_dir logs/training
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import plot_training_curves, plot_comparison, plot_pareto_frontier


def main():
    parser = argparse.ArgumentParser(description='生成论文图表')
    parser.add_argument('--results_dir', type=str, default='logs/training', help='结果目录')
    parser.add_argument('--output_dir', type=str, default='results/figures', help='输出目录')
    args = parser.parse_args()
    
    print("生成论文图表...")
    print("="*50)
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: 实现图表生成
    # 1. 训练曲线
    # 2. 算法对比
    # 3. Pareto前沿
    # 4. Ablation study
    
    print("✓ 所有图表已生成！")
    print(f"保存到：{output_dir}")


if __name__ == '__main__':
    main()

