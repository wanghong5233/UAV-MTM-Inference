#!/usr/bin/env python3
"""
批量收敛性分析图表生成脚本
遍历 analysis 下所有 seeds_* 目录，生成收敛性分析图表并保存在 convergence/figures 下。

使用现有 scripts/plot_convergence_figures.py 的能力，按每个 seeds_* 目录作为数据根目录，
专门生成收敛性相关的所有图表。
"""

import argparse
from pathlib import Path
import subprocess
import sys


def find_seed_groups(analysis_dir: Path):
    """查找所有种子组合目录"""
    if not analysis_dir.exists():
        return []
    groups = []
    # 若传入的目录本身就是 seeds_*，直接返回该目录
    if analysis_dir.is_dir() and analysis_dir.name.startswith("seeds_"):
        return [analysis_dir]
    for p in sorted(analysis_dir.iterdir()):
        if p.is_dir() and p.name.startswith("seeds_"):
            groups.append(p)
    return groups


def run_convergence_plot_for_group(group_dir: Path, figures: list, extra_args: list):
    """为指定种子组合目录生成收敛性图表"""
    cmd = [
        sys.executable,
        "scripts/plot_convergence_figures.py",
        "--data-dir", str(group_dir).replace("\\", "/"),
        "--figures",
    ] + figures

    # 输出到每个 seeds_* 目录下的 figures
    out_dir = group_dir / "figures"
    cmd += ["--output-dir", str(out_dir).replace("\\", "/")]

    # 追加额外参数（如 --smooth 等）
    cmd += extra_args

    print(f"🖼️  生成收敛性图表: {group_dir.name} → {out_dir}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ 完成: {group_dir.name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 生成失败: {group_dir.name} -> {e}")


def has_convergence_data(group_dir: Path):
    """检查是否有收敛性数据"""
    # 首先检查直接在group_dir下是否有数据文件
    required_files = ["raw_data.csv", "summary_stats.csv"]
    
    # 检查新的扁平结构
    if all((group_dir / file).exists() for file in required_files):
        return True
    
    # 兼容旧的目录结构
    convergence_dir = group_dir / "convergence"
    if convergence_dir.exists() and all((convergence_dir / file).exists() for file in required_files):
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description="批量生成收敛性分析图表")
    parser.add_argument(
        "--analysis-dir", default="analysis/analysis_convergence",
        help="聚合结果根目录，默认 analysis/analysis_convergence"
    )
    parser.add_argument(
        "--figures", nargs='+', 
        choices=['curves', 'comparison', 'metrics', 'metrics_curves', 'table', 'all', 'curves_4_metrics'],
        default=['all'],
        help="要生成的收敛性图表类型，默认生成所有图表"
    )
    
    # 透传常用绘图参数，设置适合收敛性分析的默认值
    parser.add_argument("--smooth", type=int, default=10, 
                       help="收敛曲线滑动平均窗口，默认10")
    parser.add_argument("--ci", action="store_true", default=True,
                       help="使用置信区间，默认开启")
    parser.add_argument("--tail-n", dest="tail_n", type=int, default=50,
                       help="最终对比使用的尾段更新数，默认50")
    parser.add_argument("--latency-threshold", type=float, default=None,
                       help="时延阈值，若不指定则从配置读取")
    parser.add_argument("--max-points", type=int, default=300,
                       help="收敛曲线最大可视化点数，默认300")
    parser.add_argument("--paper", action="store_true", default=True,
                       help="启用论文风格，默认开启")
    parser.add_argument("--legend-cols", type=int, default=1,
                       help="图例列数，默认1")
    parser.add_argument("--dpi-save", type=int, default=400,
                       help="保存图片DPI，默认400")
    parser.add_argument("--colorblind", action="store_true", 
                       help="使用色盲友好调色板")
    parser.add_argument("--show", action="store_true",
                       help="显示图表窗口")

    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)

    # 支持传入 seeds_*/convergence，自动推断回到种子目录
    if analysis_dir.name == "convergence" and analysis_dir.parent.name.startswith("seeds_"):
        # 形如 analysis/seeds_xxx/convergence
        analysis_dir = analysis_dir.parent

    groups = find_seed_groups(analysis_dir)

    if not groups:
        print(f"❌ 未找到任何种子组合目录 seeds_* 于: {analysis_dir}")
        return

    # 构建透传参数到 plot_convergence_figures.py
    pass_args = []
    if args.smooth is not None:
        pass_args += ["--smooth", str(args.smooth)]
    if args.ci:
        pass_args += ["--ci"]
    if args.tail_n is not None:
        pass_args += ["--tail-n", str(args.tail_n)]
    if args.latency_threshold is not None:
        pass_args += ["--latency-threshold", str(args.latency_threshold)]
    if args.max_points is not None and args.max_points > 0:
        pass_args += ["--max-points", str(args.max_points)]
    if args.paper:
        pass_args += ["--paper"]
    if args.legend_cols is not None:
        pass_args += ["--legend-cols", str(args.legend_cols)]
    if args.dpi_save is not None:
        pass_args += ["--dpi-save", str(args.dpi_save)]
    if args.colorblind:
        pass_args += ["--colorblind"]
    if args.show:
        pass_args += ["--show"]

    processed_count = 0
    for group_dir in groups:
        if not has_convergence_data(group_dir):
            print(f"⚠️ 跳过无收敛性数据的目录: {group_dir.name}")
            continue

        run_convergence_plot_for_group(group_dir, args.figures, pass_args)
        processed_count += 1

    if processed_count == 0:
        print("❌ 未找到任何包含收敛性数据的种子组合目录")
    else:
        print(f"🎉 批量收敛性图表生成完成！处理了 {processed_count} 个种子组合目录")


if __name__ == "__main__":
    main()
