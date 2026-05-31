#!/usr/bin/env python3
"""
论文定量分析数据导出脚本

一次性汇总所有实验的收敛数据，生成用于论文定量分析的 CSV 文件。
包括：
1. 收敛性分析：每个算法在四个核心指标上的最终收敛值
2. 敏感性分析：每个参数下每个算法的最终值
3. 权重分析：不同能量/隐私权重比下的能耗和隐私成本

使用方法：
    python scripts/export_quantitative_data.py
    python scripts/export_quantitative_data.py --convergence-dir analysis/analysis_convergence
    python scripts/export_quantitative_data.py --sensitivity-dir experiment_result_sensitivity
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import glob
import yaml


# 算法配置（用于标签映射）
ALGORITHM_CONFIG = {
    'h_mappo_l': {'label': 'HC-MAPPO-L'},
    'mappo_no_constraint': {'label': 'H-MAPPO'},
    'ippo': {'label': 'H-IPPO'},
    'hc_ippo_l': {'label': 'HC-IPPO-L'},
    'greedy_policy': {'label': 'Greedy Policy'},
    'local_only': {'label': 'Local-Only'},
    'edge_only': {'label': 'Edge-Only'},
    'random_policy': {'label': 'Random Policy'},
    'lru_avg': {'label': 'Heuristic-MAPPO-L'},
}

ALGORITHM_ORDER = ['h_mappo_l', 'mappo_no_constraint', 'ippo', 'hc_ippo_l',
                   'greedy_policy', 'local_only', 'edge_only', 'random_policy', 'lru_avg']

NON_LEARNING_ALGORITHMS = ['local_only', 'edge_only', 'greedy_policy']

# 参数显示配置
PARAMETER_CONFIG = {
    'num_users': {'name_en': 'Number of Users', 'unit': ''},
    'num_edges': {'name_en': 'Number of Edge Servers', 'unit': ''},
    'services_per_model': {'name_en': 'Services per Model', 'unit': ''},
    'input_size_range': {'name_en': 'Input Data Size', 'unit': 'MB'},
    'latency_constraint': {'name_en': 'Delay Constraint', 'unit': 's'},
    'server_storage_range': {'name_en': 'Server Storage Capacity', 'unit': 'GB'},
    'server_compute_range': {'name_en': 'Server Compute Capacity', 'unit': 'GFLOPS'},
    'server_bandwidth_range': {'name_en': 'Server Bandwidth', 'unit': 'MHz'},
    'user_compute_range': {'name_en': 'User Compute Capacity', 'unit': 'GFLOPS'},
    'energy_privacy_weights': {'name_en': 'Energy-Privacy Weights', 'unit': ''},
}


def export_convergence_data(convergence_dir: Path, output_dir: Path, tail_n: int = 50):
    """
    导出收敛性分析的最终收敛值
    
    Args:
        convergence_dir: 收敛性分析结果目录（包含 summary_stats.csv）
        output_dir: 输出目录
        tail_n: 取最后 N 个 update 的均值作为收敛值
    
    Returns:
        DataFrame: 收敛值数据
    """
    print("\n" + "="*60)
    print("📊 导出收敛性分析数据")
    print("="*60)
    
    # 智能搜索 summary_stats.csv（支持子目录）
    summary_file = convergence_dir / "summary_stats.csv"
    if not summary_file.exists():
        # 尝试在子目录中查找
        possible_locations = list(convergence_dir.glob("*/summary_stats.csv"))
        if possible_locations:
            summary_file = possible_locations[0]
            print(f"📁 在子目录中找到数据: {summary_file.relative_to(convergence_dir)}")
        else:
            print(f"⚠️  未找到收敛性汇总数据: {summary_file}")
            print(f"   已搜索: {convergence_dir} 及其子目录")
            return None
    
    print(f"📁 读取数据: {summary_file}")
    summary_data = pd.read_csv(summary_file)
    
    # 计算每个算法的最终收敛值（取最后 tail_n 个 update 的均值）
    final_values_data = []
    
    for alg in ALGORITHM_ORDER:
        if alg not in summary_data['algorithm'].values:
            continue
            
        df_alg = summary_data[summary_data['algorithm'] == alg]
        if df_alg.empty:
            continue
        
        # 取最后 tail_n 个 update
        max_update = df_alg['update'].max()
        df_tail = df_alg[df_alg['update'] >= max_update - (tail_n - 1)]
        if df_tail.empty:
            df_tail = df_alg.tail(tail_n)
        
        # 计算均值和标准差
        mean_vals = df_tail.mean(numeric_only=True)
        std_vals = df_tail.std(numeric_only=True)
        
        # 提取四个核心指标
        final_values_data.append({
            'algorithm': alg,
            'algorithm_label': ALGORITHM_CONFIG.get(alg, {}).get('label', alg),
            'user_cost_mean': abs(mean_vals.get('avg_user_reward_mean', 0)),  # 转为正的代价
            'user_cost_std': abs(std_vals.get('avg_user_reward_mean', 0)),
            'delay_mean': mean_vals.get('avg_delay_s_mean', 0),
            'delay_std': std_vals.get('avg_delay_s_mean', 0),
            'energy_mean': mean_vals.get('avg_energy_j_mean', 0),
            'energy_std': std_vals.get('avg_energy_j_mean', 0),
            'privacy_mean': mean_vals.get('avg_privacy_cost_mean', 0),
            'privacy_std': std_vals.get('avg_privacy_cost_mean', 0),
            'success_rate_mean': mean_vals.get('success_rate_mean', 0),
            'success_rate_std': std_vals.get('success_rate_mean', 0),
            'num_updates_averaged': len(df_tail),
            'final_update': max_update,
        })
    
    if not final_values_data:
        print("⚠️  没有提取到任何收敛数据")
        return None
    
    df_final = pd.DataFrame(final_values_data)
    
    # 数值列保留两位小数
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    df_final[numeric_cols] = df_final[numeric_cols].round(2)
    
    # 保存
    output_file = output_dir / "convergence_final_values.csv"
    df_final.to_csv(output_file, index=False)
    
    print(f"\n✅ 导出收敛性分析数据: {output_file}")
    print(f"   包含 {len(df_final)} 个算法")
    print(f"   指标: User Cost, Delay, Energy, Privacy, Success Rate")
    
    # 打印摘要
    print("\n📈 收敛值摘要:")
    for _, row in df_final.iterrows():
        print(f"  {row['algorithm_label']:20s}: "
              f"Cost={row['user_cost_mean']:.2f}±{row['user_cost_std']:.2f}, "
              f"Delay={row['delay_mean']:.2f}±{row['delay_std']:.2f}s, "
              f"Energy={row['energy_mean']:.2f}±{row['energy_std']:.2f}J, "
              f"Privacy={row['privacy_mean']:.2f}±{row['privacy_std']:.2f}")
    
    return df_final


def export_sensitivity_data(sensitivity_dir: Path, output_dir: Path, tail_n: int = 50):
    """
    导出敏感性分析的最终值
    
    Args:
        sensitivity_dir: 敏感性分析结果目录
        output_dir: 输出目录
        tail_n: 对于学习型算法，取最后 N 个 update 的均值
    
    Returns:
        DataFrame: 敏感性分析数据
    """
    print("\n" + "="*60)
    print("📊 导出敏感性分析数据")
    print("="*60)
    
    if not sensitivity_dir.exists():
        print(f"⚠️  敏感性分析目录不存在: {sensitivity_dir}")
        return None
    
    all_data = []
    
    # 遍历每个参数
    for param_dir in sensitivity_dir.iterdir():
        if not param_dir.is_dir():
            continue
        
        param_name = param_dir.name
        print(f"\n📁 处理参数: {param_name}")
        
        # 遍历每个参数值
        for value_dir in param_dir.iterdir():
            if not value_dir.is_dir():
                continue
            
            param_value = value_dir.name
            
            # 遍历每个算法
            for algo_dir in value_dir.iterdir():
                if not algo_dir.is_dir():
                    continue
                
                algorithm = algo_dir.name
                if algorithm not in ALGORITHM_CONFIG:
                    continue
                
                # 收集所有种子的数据
                seed_dfs = []
                for seed_dir in algo_dir.glob('seed_*'):
                    metrics_file = seed_dir / 'metrics.csv'
                    if metrics_file.exists():
                        try:
                            df = pd.read_csv(metrics_file)
                            seed_dfs.append(df)
                        except Exception as e:
                            print(f"    ⚠️  加载失败: {metrics_file} - {e}")
                
                if not seed_dfs:
                    continue
                
                # 对每个种子取均值
                seed_means = []
                for df in seed_dfs:
                    if algorithm in NON_LEARNING_ALGORITHMS:
                        # 非学习型：所有步骤的平均值
                        processed_data = df
                    else:
                        # 学习型：最后 tail_n 个更新的均值
                        if len(df) > tail_n:
                            processed_data = df.tail(tail_n)
                        else:
                            processed_data = df
                    seed_means.append(processed_data.mean(numeric_only=True))
                
                # 计算跨种子的统计量
                if len(seed_means) == 1:
                    mean_vals = seed_means[0].to_dict()
                    std_vals = {k: 0.0 for k in mean_vals.keys()}
                else:
                    seed_df = pd.DataFrame(seed_means)
                    mean_vals = seed_df.mean().to_dict()
                    std_vals = seed_df.std().to_dict()
                
                # 记录四个核心指标
                all_data.append({
                    'parameter': param_name,
                    'parameter_name': PARAMETER_CONFIG.get(param_name, {}).get('name_en', param_name),
                    'parameter_value': param_value,
                    'algorithm': algorithm,
                    'algorithm_label': ALGORITHM_CONFIG[algorithm]['label'],
                    'num_seeds': len(seed_dfs),
                    'user_cost_mean': abs(mean_vals.get('avg_user_reward', 0)),
                    'user_cost_std': abs(std_vals.get('avg_user_reward', 0)),
                    'delay_mean': mean_vals.get('avg_delay_s', 0),
                    'delay_std': std_vals.get('avg_delay_s', 0),
                    'energy_mean': mean_vals.get('avg_energy_j', 0),
                    'energy_std': std_vals.get('avg_energy_j', 0),
                    'privacy_mean': mean_vals.get('avg_privacy_cost', 0),
                    'privacy_std': std_vals.get('avg_privacy_cost', 0),
                    'success_rate_mean': mean_vals.get('success_rate', 0),
                    'success_rate_std': std_vals.get('success_rate', 0),
                })
    
    if not all_data:
        print("⚠️  没有提取到任何敏感性分析数据")
        return None
    
    df_sensitivity = pd.DataFrame(all_data)
    
    # 数值列保留两位小数
    numeric_cols = df_sensitivity.select_dtypes(include=[np.number]).columns
    df_sensitivity[numeric_cols] = df_sensitivity[numeric_cols].round(2)
    
    # 保存
    output_file = output_dir / "sensitivity_final_values.csv"
    df_sensitivity.to_csv(output_file, index=False)
    
    print(f"\n✅ 导出敏感性分析数据: {output_file}")
    print(f"   包含 {len(df_sensitivity)} 条记录")
    print(f"   参数: {df_sensitivity['parameter'].nunique()} 个")
    print(f"   算法: {df_sensitivity['algorithm'].nunique()} 个")
    
    # 按参数汇总
    print("\n📈 各参数数据量:")
    for param in df_sensitivity['parameter'].unique():
        count = len(df_sensitivity[df_sensitivity['parameter'] == param])
        print(f"  {param:30s}: {count:3d} 条记录")
    
    return df_sensitivity


def export_weight_analysis_data(weight_dir: Path, output_dir: Path):
    """
    导出能量/隐私权重分析的数据（仅针对 h_mappo_l 算法）
    
    Args:
        weight_dir: 权重分析结果目录
        output_dir: 输出目录
    
    Returns:
        DataFrame: 权重分析数据
    """
    print("\n" + "="*60)
    print("📊 导出权重分析数据（h_mappo_l）")
    print("="*60)
    
    if not weight_dir.exists():
        print(f"⚠️  权重分析目录不存在: {weight_dir}")
        return None
    
    all_data = []
    
    DEFAULT_PRIVACY_WEIGHT = 5.0
    
    # 遍历每个权重值
    for weight_value_dir in weight_dir.iterdir():
        if not weight_value_dir.is_dir():
            continue
        
        try:
            # 验证目录名是否为浮点数
            weight_value = float(weight_value_dir.name)
        except ValueError:
            print(f"  跳过非权重目录: {weight_value_dir.name}")
            continue
        
        # 只处理 h_mappo_l 算法
        algo_dir = weight_value_dir / 'h_mappo_l'
        if not algo_dir.exists() or not algo_dir.is_dir():
            continue
        
        # 收集所有种子的数据
        seed_dfs = []
        for seed_dir in algo_dir.glob('seed_*'):
            metrics_file = seed_dir / 'metrics.csv'
            if metrics_file.exists():
                try:
                    df = pd.read_csv(metrics_file)
                    seed_dfs.append(df)
                except Exception as e:
                    print(f"    ⚠️  加载失败: {metrics_file} - {e}")
        
        if not seed_dfs:
            continue
        
        print(f"  权重 {weight_value}: {len(seed_dfs)} 个种子")
        
        # 对每个种子取最后 50 个 update 的均值
        seed_means = []
        for df in seed_dfs:
            if len(df) > 50:
                processed_data = df.tail(50)
            else:
                processed_data = df
            seed_means.append(processed_data.mean(numeric_only=True))
        
        # 计算跨种子的统计量
        if len(seed_means) == 1:
            mean_vals = seed_means[0].to_dict()
            std_vals = {k: 0.0 for k in mean_vals.keys()}
        else:
            seed_df = pd.DataFrame(seed_means)
            mean_vals = seed_df.mean().to_dict()
            std_vals = seed_df.std().to_dict()
        
        # 计算权重比
        ratio = weight_value / DEFAULT_PRIVACY_WEIGHT
        
        all_data.append({
            'energy_weight': weight_value,
            'privacy_weight': DEFAULT_PRIVACY_WEIGHT,
            'weight_ratio': ratio,
            'num_seeds': len(seed_dfs),
            'user_cost_mean': abs(mean_vals.get('avg_user_reward', 0)),
            'user_cost_std': abs(std_vals.get('avg_user_reward', 0)),
            'delay_mean': mean_vals.get('avg_delay_s', 0),
            'delay_std': std_vals.get('avg_delay_s', 0),
            'energy_mean': mean_vals.get('avg_energy_j', 0),
            'energy_std': std_vals.get('avg_energy_j', 0),
            'privacy_mean': mean_vals.get('avg_privacy_cost', 0),
            'privacy_std': std_vals.get('avg_privacy_cost', 0),
            'success_rate_mean': mean_vals.get('success_rate', 0),
            'success_rate_std': std_vals.get('success_rate', 0),
        })
    
    if not all_data:
        print("⚠️  没有提取到任何权重分析数据")
        return None
    
    df_weights = pd.DataFrame(all_data)
    # 按权重比排序
    df_weights = df_weights.sort_values('weight_ratio')
    
    # 数值列保留两位小数
    numeric_cols = df_weights.select_dtypes(include=[np.number]).columns
    df_weights[numeric_cols] = df_weights[numeric_cols].round(2)
    
    # 保存
    output_file = output_dir / "weight_analysis_final_values.csv"
    df_weights.to_csv(output_file, index=False)
    
    print(f"\n✅ 导出权重分析数据: {output_file}")
    print(f"   包含 {len(df_weights)} 个权重配置")
    
    # 打印摘要
    print("\n📈 权重分析摘要:")
    for _, row in df_weights.iterrows():
        print(f"  Ratio={row['weight_ratio']:.2f}: "
              f"Energy={row['energy_mean']:.2f}±{row['energy_std']:.2f}J, "
              f"Privacy={row['privacy_mean']:.2f}±{row['privacy_std']:.2f}")
    
    return df_weights


def export_heatmap_data(convergence_dir: Path, output_dir: Path, heatmap_dir: Path = None):
    """
    导出热力图的统计数据（客户端级别的排名和指标分布）
    
    Args:
        convergence_dir: 收敛性分析结果目录（包含 raw_data.csv）
        output_dir: 输出目录
        heatmap_dir: 热力图原始数据目录（如 experiment_result_convergence/heatmap）
    
    Returns:
        tuple: (排名统计DataFrame, 客户端指标DataFrame)
    """
    print("\n" + "="*60)
    print("📊 导出热力图统计数据")
    print("="*60)
    
    raw_data = None
    
    # 优先使用 heatmap_dir（如果提供）
    if heatmap_dir and heatmap_dir.exists():
        print(f"📁 从热力图目录读取数据: {heatmap_dir}")
        
        # 查找所有算法的 per_client_metrics.csv
        all_data = []
        
        # 遍历 seed_* 目录
        for seed_dir in heatmap_dir.glob("seed_*"):
            if not seed_dir.is_dir():
                continue
            
            print(f"  处理种子: {seed_dir.name}")
            
            # 遍历每个算法目录
            for algo_dir in seed_dir.iterdir():
                if not algo_dir.is_dir():
                    continue
                
                # 从目录名提取算法名（格式：convergence__算法名__其他）
                dir_name = algo_dir.name
                if not dir_name.startswith('convergence__'):
                    continue
                
                parts = dir_name.split('__')
                if len(parts) < 2:
                    continue
                
                algorithm = parts[1]
                
                # 读取 per_client_metrics.csv
                metrics_file = algo_dir / 'per_client_metrics.csv'
                if not metrics_file.exists():
                    print(f"    ⚠️  未找到: {metrics_file}")
                    continue
                
                try:
                    df = pd.read_csv(metrics_file)
                    df['algorithm'] = algorithm
                    all_data.append(df)
                    print(f"    ✓ {algorithm}: {len(df)} 条记录")
                except Exception as e:
                    print(f"    ⚠️  读取失败 {metrics_file}: {e}")
        
        if not all_data:
            print("⚠️  未找到任何热力图数据")
            return None, None
        
        raw_data = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ 合并数据: {len(raw_data)} 条记录，{raw_data['algorithm'].nunique()} 个算法")
    
    else:
        # 回退到原有逻辑：从 raw_data.csv 读取
        raw_data_file = convergence_dir / "raw_data.csv"
        if not raw_data_file.exists():
            # 尝试在子目录中查找
            possible_locations = list(convergence_dir.glob("*/raw_data.csv"))
            if possible_locations:
                raw_data_file = possible_locations[0]
                print(f"📁 在子目录中找到数据: {raw_data_file.relative_to(convergence_dir)}")
            else:
                print(f"⚠️  未找到原始数据: {raw_data_file}")
                print(f"   已搜索: {convergence_dir} 及其子目录")
                return None, None
        
        print(f"📁 读取数据: {raw_data_file}")
        raw_data = pd.read_csv(raw_data_file)
    
    # 只保留成功的任务
    successful_data = raw_data[raw_data['service_hit'] == 1].copy()
    
    # 1. 计算每个算法-客户端的统一代价 (unified cost)
    print("\n📈 计算统一代价...")
    
    # 计算每个算法-客户端的平均指标
    agg_dict = {
        'delay_s': 'mean',
        'energy_j': 'mean',
        'privacy_cost': 'mean',
    }
    
    # 如果有 user_reward 字段，也包含进来
    if 'user_reward' in successful_data.columns:
        agg_dict['user_reward'] = 'mean'
    
    client_metrics = successful_data.groupby(['algorithm', 'client_id']).agg(agg_dict).reset_index()
    
    # 计算成功率
    success_rate = raw_data.groupby(['algorithm', 'client_id'])['service_hit'].mean().reset_index()
    success_rate.columns = ['algorithm', 'client_id', 'success_rate']
    
    # 合并
    client_metrics = pd.merge(client_metrics, success_rate, on=['algorithm', 'client_id'], how='outer')
    
    # 计算统一代价
    if 'user_reward' in client_metrics.columns:
        # 有 user_reward：base_cost / success_rate
        client_metrics['base_cost'] = -client_metrics['user_reward']
        client_metrics['unified_cost'] = client_metrics['base_cost'] / client_metrics['success_rate']
    else:
        # 没有 user_reward：使用归一化的加权代价
        # unified_cost = (delay + energy + privacy) / success_rate
        client_metrics['base_cost'] = (
            client_metrics['delay_s'] + 
            client_metrics['energy_j'] + 
            client_metrics['privacy_cost']
        )
        client_metrics['unified_cost'] = client_metrics['base_cost'] / client_metrics['success_rate']
    
    client_metrics['unified_cost'] = client_metrics['unified_cost'].fillna(np.inf)
    
    # 2. 计算排名（每个客户端对所有算法排名）
    print("📊 计算排名统计...")
    
    rank_data = []
    for client_id in client_metrics['client_id'].unique():
        client_df = client_metrics[client_metrics['client_id'] == client_id].copy()
        
        # 按统一代价排名（越小越好，rank=1最好）
        client_df = client_df.sort_values('unified_cost')
        client_df['rank'] = range(1, len(client_df) + 1)
        
        rank_data.append(client_df[['algorithm', 'client_id', 'rank', 'unified_cost']])
    
    rank_df = pd.concat(rank_data, ignore_index=True)
    
    # 3. 生成排名统计摘要
    print("📋 生成排名分布统计...")
    
    rank_summary = []
    for alg in ALGORITHM_ORDER:
        if alg not in rank_df['algorithm'].values:
            continue
        
        alg_ranks = rank_df[rank_df['algorithm'] == alg]['rank']
        
        # 统计每个排名的数量
        rank_counts = alg_ranks.value_counts().sort_index().to_dict()
        
        summary_row = {
            'algorithm': alg,
            'algorithm_label': ALGORITHM_CONFIG.get(alg, {}).get('label', alg),
            'mean_rank': alg_ranks.mean(),
            'median_rank': alg_ranks.median(),
            'std_rank': alg_ranks.std(),
            'num_clients': len(alg_ranks),
        }
        
        # 添加每个排名的客户端数量
        max_rank = int(alg_ranks.max())
        for r in range(1, max_rank + 1):
            summary_row[f'rank_{r}_count'] = rank_counts.get(r, 0)
            summary_row[f'rank_{r}_percent'] = (rank_counts.get(r, 0) / len(alg_ranks) * 100) if len(alg_ranks) > 0 else 0
        
        rank_summary.append(summary_row)
    
    df_rank_summary = pd.DataFrame(rank_summary)
    
    # 数值列保留两位小数
    numeric_cols = df_rank_summary.select_dtypes(include=[np.number]).columns
    df_rank_summary[numeric_cols] = df_rank_summary[numeric_cols].round(2)
    
    # 保存排名统计
    rank_summary_file = output_dir / "heatmap_rank_statistics.csv"
    df_rank_summary.to_csv(rank_summary_file, index=False)
    
    print(f"\n✅ 导出排名统计: {rank_summary_file}")
    print(f"   包含 {len(df_rank_summary)} 个算法的排名分布")
    
    # 4. 保存详细的客户端指标数据
    print("\n📊 导出客户端级别指标...")
    
    # 透视表：每个客户端在各算法下的指标
    heatmap_detail = client_metrics.copy()
    heatmap_detail = heatmap_detail.round(2)
    
    heatmap_detail_file = output_dir / "heatmap_client_metrics.csv"
    heatmap_detail.to_csv(heatmap_detail_file, index=False)
    
    print(f"✅ 导出客户端指标: {heatmap_detail_file}")
    print(f"   包含 {len(heatmap_detail)} 条记录（{heatmap_detail['client_id'].nunique()} 个客户端）")
    
    # 打印排名摘要
    print("\n📈 排名统计摘要:")
    for _, row in df_rank_summary.iterrows():
        rank_1_pct = row.get('rank_1_percent', 0)
        print(f"  {row['algorithm_label']:20s}: "
              f"平均排名={row['mean_rank']:.2f}, "
              f"第1名占比={rank_1_pct:.1f}%")
    
    return df_rank_summary, heatmap_detail


def main():
    parser = argparse.ArgumentParser(
        description="导出所有实验的定量分析数据（用于论文写作）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--convergence-dir',
        type=Path,
        default=Path('analysis/analysis_convergence'),
        help='收敛性分析结果目录'
    )
    parser.add_argument(
        '--sensitivity-dir',
        type=Path,
        default=Path('experiment_result_sensitivity'),
        help='敏感性分析结果目录'
    )
    parser.add_argument(
        '--weight-dir',
        type=Path,
        default=Path('experiment_result_sensitivity/energy_privacy_weights'),
        help='权重分析结果目录'
    )
    parser.add_argument(
        '--heatmap-dir',
        type=Path,
        default=None,
        help='热力图原始数据目录（如 experiment_result_convergence/heatmap），如果不提供则从 convergence-dir 中查找'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('analysis/quantitative_data'),
        help='数据输出目录'
    )
    parser.add_argument(
        '--tail-n',
        type=int,
        default=50,
        help='取最后 N 个 update 的均值作为收敛值'
    )
    parser.add_argument(
        '--skip-convergence',
        action='store_true',
        help='跳过收敛性分析数据导出'
    )
    parser.add_argument(
        '--skip-sensitivity',
        action='store_true',
        help='跳过敏感性分析数据导出'
    )
    parser.add_argument(
        '--skip-weight',
        action='store_true',
        help='跳过权重分析数据导出'
    )
    parser.add_argument(
        '--skip-heatmap',
        action='store_true',
        help='跳过热力图统计数据导出'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("🎯 论文定量分析数据导出工具")
    print("="*60)
    print(f"输出目录: {args.output_dir}")
    
    # 1. 导出收敛性分析数据
    if not args.skip_convergence:
        df_convergence = export_convergence_data(
            args.convergence_dir,
            args.output_dir,
            args.tail_n
        )
    
    # 2. 导出敏感性分析数据
    if not args.skip_sensitivity:
        df_sensitivity = export_sensitivity_data(
            args.sensitivity_dir,
            args.output_dir,
            args.tail_n
        )
    
    # 3. 导出权重分析数据
    if not args.skip_weight:
        df_weights = export_weight_analysis_data(
            args.weight_dir,
            args.output_dir
        )
    
    # 4. 导出热力图统计数据
    if not args.skip_heatmap:
        df_rank_stats, df_client_metrics = export_heatmap_data(
            args.convergence_dir,
            args.output_dir,
            args.heatmap_dir
        )
    
    print("\n" + "="*60)
    print("🎉 数据导出完成！")
    print("="*60)
    print(f"\n所有数据已保存到: {args.output_dir}")
    print("\n生成的文件:")
    for f in sorted(args.output_dir.glob("*.csv")):
        print(f"  - {f.name}")
    print("\n💡 提示: 这些 CSV 文件可以直接用于论文中的定量分析和表格制作")


if __name__ == "__main__":
    main()

