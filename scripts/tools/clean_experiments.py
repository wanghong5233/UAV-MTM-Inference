"""
清理实验文件

用法：
    python scripts/tools/clean_experiments.py --keep_best
"""

import argparse
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description='清理实验文件')
    parser.add_argument('--keep_best', action='store_true', help='保留最佳模型')
    parser.add_argument('--experiments_dir', type=str, default='experiments', help='实验目录')
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    
    print("清理实验文件...")
    print("="*50)
    
    if args.keep_best:
        print("保留最佳模型，删除其他checkpoint...")
        # TODO: 实现清理逻辑
    else:
        confirm = input(f"确定要删除 {experiments_dir} 中的所有实验吗？(yes/no): ")
        if confirm.lower() == 'yes':
            if experiments_dir.exists():
                shutil.rmtree(experiments_dir)
                experiments_dir.mkdir()
                print("✓ 已清理")
        else:
            print("取消清理")
    
    print("="*50)


if __name__ == '__main__':
    main()

