#!/usr/bin/env python3
"""
批量画图脚本
遍历 analysis 下所有 seeds_* 目录，按类型生成图表并保存在最底层的 convergence/figures 下。

使用现有 scripts/plot_figures.py 的能力，按每个 seeds_* 目录作为数据根目录，
输出到该 seeds_* 目录的 figures 子目录。
"""

import argparse
from pathlib import Path
import subprocess
import sys


def find_seed_groups(analysis_dir: Path):
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


def run_plot_for_group(group_dir: Path, exp_type: str, figures: list, extra_args: list):
    # data-dir 指向该 seeds_* 目录
    cmd = [
        sys.executable,
        "scripts/plot_figures.py",
        "--data-dir", str(group_dir).replace("\\", "/"),
        "--figures",
    ] + figures

    # 输出到每个 seeds_* 目录下的 <exp_type>/figures
    out_dir = group_dir / exp_type / "figures"
    cmd += ["--output-dir", str(out_dir).replace("\\", "/")]

    # 追加额外参数（如 --smooth 等）
    cmd += extra_args

    print(f"🖼️  生成图表: {group_dir.name}/{exp_type} → {out_dir}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 生成失败: {group_dir}/{exp_type} -> {e}")


def find_present_types(group_dir: Path):
    present = []
    for t in ["convergence", "scalability", "sensitivity"]:
        t_dir = group_dir / t
        if not t_dir.exists():
            continue
        # 判断是否有数据
        has_summary = (t_dir / "summary_stats.csv").exists()
        has_raw = (t_dir / "raw_data.csv").exists()
        if has_summary or has_raw:
            present.append(t)
    return present


def main():
    parser = argparse.ArgumentParser(description="批量生成图表")
    parser.add_argument(
        "--analysis-dir", default="analysis",
        help="聚合结果根目录，默认 analysis"
    )
    parser.add_argument(
        "--figures", nargs='+', default=None,
        help="手动指定要生成的图表；留空将按类型自动选择"
    )
    # 透传常用绘图参数
    # 将常用的样式参数设置为默认值，无需每次传参
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--ci", action="store_true", default=True)
    parser.add_argument("--tail-n", dest="tail_n", type=int, default=50)
    parser.add_argument("--latency-threshold", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=300)
    parser.add_argument("--paper", action="store_true", default=True)
    parser.add_argument("--legend-cols", type=int, default=1)
    parser.add_argument("--dpi-save", type=int, default=400)

    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)

    # 支持传入 seeds_*/<type>，自动推断类型
    forced_types = None
    known_types = {"convergence", "scalability", "sensitivity"}
    if analysis_dir.name in known_types and analysis_dir.parent.name.startswith("seeds_"):
        # 形如 analysis/seeds_xxx/convergence
        forced_types = [analysis_dir.name]
        analysis_dir = analysis_dir.parent

    groups = find_seed_groups(analysis_dir)

    if not groups:
        print(f"❌ 未找到任何种子组合目录 seeds_* 于: {analysis_dir}")
        return

    # 构建透传参数到 plot_figures.py（统一使用默认样式参数）
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

    for group_dir in groups:
        present_types = find_present_types(group_dir)
        if forced_types is not None:
            # 仅处理被强制指定的类型（若目录中无该类型数据则跳过）
            present_types = [t for t in present_types if t in forced_types]
        if not present_types:
            print(f"⚠️ 跳过无数据目录: {group_dir}")
            continue

        for t in present_types:
            # 自动按类型选择要生成的图表
            if args.figures is None:
                if t == "convergence":
                    figs = ['convergence', 'comparison', 'metrics', 'metrics_curves', 'table']
                elif t == "scalability":
                    figs = ['scalability']
                elif t == "sensitivity":
                    print(f"ℹ️  检测到 {group_dir.name}/sensitivity，但当前绘图脚本未实现该类型，先跳过。")
                    continue
                else:
                    figs = ['convergence']
            else:
                figs = args.figures

            run_plot_for_group(group_dir, t, figs, pass_args)

    print("🎉 批量图表生成完成！")


if __name__ == "__main__":
    main()


