#!/usr/bin/env python3
"""
批量种子组合聚合与作图脚本

功能：
- 接收多组种子组合（命令行或文件），或根据给定种子数量k自动对可用种子做全组合；逐组执行：聚合→作图
- 每组结果独立保存到 analysis/sweeps/seeds_*/ 下（analysis 与 figures 分离）
- 便于人工浏览不同组合的图，最终挑选最佳组合再精修

示例：
python scripts/seed_sweep.py --seed-sets "5,2,4; 5,1,3; 2,0" \
  --figures convergence comparison metrics_curves \
  --smooth 10 --ci --tail-n 50 --max-points 300 --paper --legend-cols 1 --dpi-save 400
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from itertools import combinations


def parse_seed_sets(seed_sets_str: str):
    sets = []
    if not seed_sets_str:
        return sets
    for part in seed_sets_str.split(';'):
        s = part.strip().replace(' ', ',')
        s = ','.join([p for p in s.split(',') if p.strip()])
        if s:
            sets.append(s)
    return sets


def read_seed_sets_file(path: str):
    p = Path(path)
    if not p.exists():
        return []
    rows = p.read_text(encoding='utf-8').splitlines()
    return parse_seed_sets(';'.join(rows))


def run(cmd, cwd):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}")


def discover_seeds(base_dir: str):
    seeds = []
    p = Path(base_dir)
    if not p.exists():
        return seeds
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith('seed_'):
            try:
                seeds.append(int(d.name.split('_')[1]))
            except Exception:
                pass
    return sorted(list(set(seeds)))


def main():
    parser = argparse.ArgumentParser(description='批量种子组聚合与作图')
    parser.add_argument('--seed-sets', type=str, default='', help='多组种子，分号分隔，如 "5,2,4; 5,1,3; 2,0"')
    parser.add_argument('--seed-sets-file', type=str, default='', help='种子组合文件，每行一组（逗号或空格分隔）')
    parser.add_argument('--k', type=int, default=0, help='自动组合的组大小k，k>0时对种子集合做全组合')
    parser.add_argument('--seeds-universe', type=str, default='', help='全体候选种子（逗号或空格分隔）；不提供则自动扫描 experiment_result/seed_*')
    parser.add_argument('--max-combinations', type=int, default=0, help='最多取前N个组合，0表示不限')
    parser.add_argument('--skip-existing', action='store_true', help='若目标组合目录已存在图，则跳过')
    parser.add_argument('--figures', nargs='+', default=['convergence', 'comparison', 'metrics_curves'],
                        choices=['convergence', 'comparison', 'scalability', 'metrics', 'metrics_curves', 'table', 'all'],
                        help='要生成的图表类型，默认 convergence comparison')
    # 作图参数透传
    parser.add_argument('--smooth', type=int, default=10)
    parser.add_argument('--ci', action='store_true')
    parser.add_argument('--tail-n', dest='tail_n', type=int, default=50)
    parser.add_argument('--max-points', type=int, default=300)
    parser.add_argument('--paper', action='store_true')
    parser.add_argument('--legend-cols', type=int, default=1)
    parser.add_argument('--dpi-save', type=int, default=400)
    parser.add_argument('--colorblind', action='store_true')
    parser.add_argument('--latency-threshold', type=float, default=None)

    args = parser.parse_args()

    # 解析/生成种子组
    seed_sets = []
    seed_sets += parse_seed_sets(args.seed_sets)
    if args.seed_sets_file:
        seed_sets += read_seed_sets_file(args.seed_sets_file)

    # 自动组合模式
    if args.k and args.k > 0:
        # 获取候选种子集合
        if args.seeds_universe:
            universe = [int(s) for s in parse_seed_sets(args.seeds_universe)[0].split(',')]
        else:
            universe = discover_seeds('experiment_result')
        if len(universe) < args.k:
            print(f"❌ 候选种子数量不足：len={len(universe)} < k={args.k}")
            sys.exit(1)
        # 生成全组合
        combs = list(combinations(universe, args.k))
        if args.max_combinations and args.max_combinations > 0:
            combs = combs[: args.max_combinations]
        seed_sets += [','.join(str(x) for x in comb) for comb in combs]

    seed_sets = [s for s in seed_sets if s]
    if not seed_sets:
        print('❌ 未提供任何种子组合（--seed-sets/--seed-sets-file 或 --k）')
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent
    sweeps_root = repo_root / 'analysis' / 'sweeps'
    sweeps_root.mkdir(parents=True, exist_ok=True)

    for seeds in seed_sets:
        slug = f"seeds_{'-'.join([x.strip() for x in seeds.split(',')])}"
        out_root = sweeps_root / slug
        out_analysis = out_root / 'analysis'
        out_fig = out_root / 'figures'
        out_analysis.mkdir(parents=True, exist_ok=True)
        out_fig.mkdir(parents=True, exist_ok=True)

        print(f"\n=== 处理种子组合: {seeds} → {slug} ===")

        # 1) 可跳过已存在
        if args.skip_existing and (out_fig / 'convergence_curves.png').exists():
            print(f"⏩ 跳过已存在: {slug}")
            continue

        # 2) 聚合（仅收敛性）
        agg_cmd = [
            sys.executable, 'scripts/aggregate_results.py',
            '--experiment-type', 'convergence',
            '--seeds', seeds,
            '--output-dir', str(out_analysis)
        ]
        run(agg_cmd, cwd=str(repo_root))

        # 3) 作图
        plot_cmd = [
            sys.executable, 'scripts/plot_figures.py',
            '--data-dir', str(out_analysis),
            '--output-dir', str(out_fig),
            '--figures', *args.figures,
            '--smooth', str(args.smooth),
            '--tail-n', str(args.tail_n),
            '--max-points', str(args.max_points),
            '--legend-cols', str(args.legend_cols),
            '--dpi-save', str(args.dpi_save)
        ]
        if args.ci:
            plot_cmd.append('--ci')
        if args.paper:
            plot_cmd.append('--paper')
        if args.colorblind:
            plot_cmd.append('--colorblind')
        if args.latency_threshold is not None:
            plot_cmd += ['--latency-threshold', str(args.latency_threshold)]

        run(plot_cmd, cwd=str(repo_root))

        print(f"✅ 完成: {slug} → {out_fig}")

    print(f"\n🎉 全部完成！汇总路径: {sweeps_root}")


if __name__ == '__main__':
    main()


