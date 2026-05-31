#!/usr/bin/env python3
"""
收敛性实验结果聚合脚本
专用于从收敛性实验结果目录收集CSV数据并进行统计分析
"""

import os
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
import argparse
from collections import defaultdict

def find_convergence_experiment_results(base_dir="experiment_result_convergence", allowed_seeds=None):
    """扫描收敛性实验结果目录
    allowed_seeds: 可选的整数种子集合；若提供，仅聚合这些种子
    """
    results = []
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ 结果目录不存在: {base_dir}")
        return results
    
    # 遍历所有种子目录
    for seed_dir in base_path.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        
        seed = seed_dir.name.split("_")[1]
        # 仅保留允许的种子
        try:
            seed_int = int(seed)
        except Exception:
            continue
        if allowed_seeds is not None and seed_int not in allowed_seeds:
            continue
        
        # 遍历种子目录下的实验
        for exp_dir in seed_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            # 跳过figures目录（这是输出目录，不是实验目录）
            if exp_dir.name == "figures":
                continue

            # 解析实验名称
            exp_name = exp_dir.name

            # 查找metrics.csv文件
            metrics_file = exp_dir / "metrics.csv"
            config_file = exp_dir / "config_snapshot.json"

            if not metrics_file.exists():
                print(f"⚠️  跳过无效目录（缺少metrics.csv）: {exp_dir}")
                continue
            
            # 提取实验信息
            experiment_info = parse_convergence_experiment_name(exp_name)
            experiment_info['seed'] = seed_int
            experiment_info['exp_dir'] = str(exp_dir)
            experiment_info['metrics_file'] = str(metrics_file)
            experiment_info['config_file'] = str(config_file) if config_file.exists() else None
            
            results.append(experiment_info)
    
    return results

def parse_convergence_experiment_name(exp_name):
    """解析收敛性实验名称，提取算法等信息"""
    info = {'raw_name': exp_name}
    
    # 尝试匹配不同的命名模式
    patterns = [
        # 带实验名称的: convergence__h_mappo_l__50users_10servers__timestamp
        r'(?P<experiment_type>\w+)__(?P<algorithm>\w+)__(?P<users>\d+)users_(?P<servers>\d+)servers__(?P<timestamp>\d+)',
        # 不带实验名称的: h_mappo_l__50users_10servers__timestamp
        r'(?P<algorithm>\w+)__(?P<users>\d+)users_(?P<servers>\d+)servers__(?P<timestamp>\d+)',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, exp_name)
        if match:
            info.update(match.groupdict())
            break
    
    # 设置默认值
    info.setdefault('experiment_type', 'convergence')
    info.setdefault('algorithm', 'unknown')
    
    return info

def load_metrics_data(metrics_file):
    """加载metrics.csv文件"""
    try:
        df = pd.read_csv(metrics_file)
        # 确保有必要的列
        required_cols = ['update']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  文件 {metrics_file} 缺少必要列: {missing_cols}")
            return None
        return df
    except Exception as e:
        print(f"❌ 加载 {metrics_file} 失败: {e}")
        return None

def load_config_data(config_file):
    """加载配置文件"""
    if not config_file or not Path(config_file).exists():
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  加载配置文件 {config_file} 失败: {e}")
        return {}

def aggregate_convergence_results(experiments):
    """聚合收敛性实验结果"""
    all_data = []
    
    for exp_info in experiments:
        metrics_df = load_metrics_data(exp_info['metrics_file'])
        if metrics_df is None:
            continue
        
        # 添加实验元信息
        metrics_df['algorithm'] = exp_info['algorithm']
        metrics_df['seed'] = exp_info['seed']
        metrics_df['experiment_type'] = exp_info['experiment_type']
        
        all_data.append(metrics_df)
    
    if not all_data:
        return None
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 计算统计信息（补充奖励列）
    summary_stats = combined_df.groupby(['algorithm', 'update']).agg({
        'success_rate': ['mean', 'std', 'count'],
        'avg_delay_s': ['mean', 'std'],
        'avg_constraint_violation': ['mean', 'std'],
        'avg_energy_j': ['mean', 'std'],
        'avg_privacy_cost': ['mean', 'std'],
        'avg_user_reward': ['mean', 'std'],
        'avg_alloc_reward': ['mean', 'std'],
        'avg_deploy_reward': ['mean', 'std'],
        'lagrangian_multiplier': ['mean', 'std'],
        'deployment_coverage': ['mean', 'std']
    }).reset_index()
    
    # 展平多级列名
    summary_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary_stats.columns]
    
    return {
        'raw_data': combined_df,
        'summary_stats': summary_stats
    }

def save_convergence_results(convergence_data, output_dir="analysis/convergence", sub_dir=None):
    """保存收敛性聚合结果"""
    output_path = Path(output_dir)

    # 如果指定了子目录，则在基础目录下创建子目录
    if sub_dir:
        output_path = output_path / sub_dir

    # 直接使用output_path作为保存目录，不再嵌套convergence子目录
    convergence_dir = output_path
    convergence_dir.mkdir(parents=True, exist_ok=True)

    if convergence_data is None:
        print("⚠️ 没有收敛性数据可保存")
        return

    # 保存原始数据
    if 'raw_data' in convergence_data:
        convergence_data['raw_data'].to_csv(convergence_dir / "raw_data.csv", index=False)
        print(f"💾 保存原始数据: {convergence_dir / 'raw_data.csv'}")

    # 保存统计数据
    if 'summary_stats' in convergence_data:
        convergence_data['summary_stats'].to_csv(convergence_dir / "summary_stats.csv", index=False)
        print(f"💾 保存统计数据: {convergence_dir / 'summary_stats.csv'}")

    print(f"✅ 收敛性结果保存完成: {convergence_dir}")

def parse_seed_list(seed_str: str):
    """解析逗号或空格分隔的种子列表，如 "5,2,0" 或 "5 2 0"""
    if not seed_str:
        return None
    parts = [p for p in seed_str.replace(',', ' ').split(' ') if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(int(p))
        except Exception:
            pass
    return set(vals) if vals else None


def main():
    parser = argparse.ArgumentParser(description="聚合收敛性实验结果")
    parser.add_argument(
        '--base-dir',
        default='experiment_result_convergence',
        help='收敛性实验结果基础目录 (默认: experiment_result_convergence)'
    )
    parser.add_argument(
        '--output-dir',
        default='analysis/analysis_convergence',
        help='输出目录 (默认: analysis/analysis_convergence)'
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help='仅聚合指定种子，逗号或空格分隔，如: "5,2,0" 或 "5 2 0"'
    )
    parser.add_argument(
        '--sub-dir',
        type=str,
        default=None,
        help='子目录名称，用于区分不同种子组合，如: "seeds_0_1_2"'
    )
    
    args = parser.parse_args()
    allowed_seeds = parse_seed_list(args.seeds)

    # 如果指定了种子但没有指定子目录，自动生成子目录名称
    if allowed_seeds and not args.sub_dir:
        seed_list = sorted(list(allowed_seeds))
        args.sub_dir = f"seeds_{'_'.join(map(str, seed_list))}"

    print(f"🔍 扫描收敛性实验结果: {args.base_dir}")
    if allowed_seeds:
        print(f"   仅聚合种子: {sorted(list(allowed_seeds))}")
    if args.sub_dir:
        print(f"   输出子目录: {args.sub_dir}")
    
    # 找到所有收敛性实验结果
    experiments = find_convergence_experiment_results(args.base_dir, allowed_seeds)

    if not experiments:
        print("❌ 没有找到收敛性实验结果")
        return

    print(f"📊 找到 {len(experiments)} 个收敛性实验")
    
    # 按算法统计
    algorithm_counts = defaultdict(int)
    seed_counts = defaultdict(int)
    for exp in experiments:
        algorithm_counts[exp['algorithm']] += 1
        seed_counts[exp['seed']] += 1
    
    print("   算法分布:")
    for alg, count in sorted(algorithm_counts.items()):
        print(f"     {alg}: {count} 个实验")
    
    print("   种子分布:")
    for seed, count in sorted(seed_counts.items()):
        print(f"     种子 {seed}: {count} 个实验")

    # 如果没有指定种子且没有指定子目录，使用简短的"seeds_all"名称
    if allowed_seeds is None and not args.sub_dir:
        all_seeds = {exp['seed'] for exp in experiments}
        if all_seeds:
            args.sub_dir = "seeds_all"
            print(f"   自动生成子目录: {args.sub_dir} (包含所有种子: {sorted(list(all_seeds))})")

    # 聚合收敛性结果
    print(f"\n📈 聚合收敛性实验结果...")
    
    convergence_result = aggregate_convergence_results(experiments)
    
    if convergence_result is not None:
        print(f"✅ 成功聚合收敛性结果")
        
        # 显示数据概览
        if 'raw_data' in convergence_result:
            raw_df = convergence_result['raw_data']
            print(f"   原始数据: {len(raw_df)} 行记录")
        
        if 'summary_stats' in convergence_result:
            summary_df = convergence_result['summary_stats']
            print(f"   统计数据: {len(summary_df)} 行记录")
            unique_algorithms = summary_df['algorithm'].nunique()
            unique_updates = summary_df['update'].nunique()
            print(f"   算法数量: {unique_algorithms}")
            print(f"   训练步数范围: {summary_df['update'].min()}-{summary_df['update'].max()}")
    else:
        print(f"❌ 聚合收敛性结果失败")
        return
    
    # 保存结果
    save_convergence_results(convergence_result, args.output_dir, args.sub_dir)
    
    # 计算最终输出路径
    output_path = Path(args.output_dir)
    if args.sub_dir:
        output_path = output_path / args.sub_dir
    
    print(f"\n🎉 收敛性聚合完成! 结果保存在 {output_path}")

if __name__ == "__main__":
    main()
