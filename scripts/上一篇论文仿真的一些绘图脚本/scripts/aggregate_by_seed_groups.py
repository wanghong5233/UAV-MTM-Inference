#!/usr/bin/env python3
"""
按算法分组并为每组指定不同种子进行聚合的脚本
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 导入现有脚本的核心功能
from aggregate_convergence_results import (
    find_convergence_experiment_results,
    aggregate_convergence_results,
    save_convergence_results
)

def main():
    # --- START: 默认分组配置 ---
    # 在此修改您的分组配置。
    # "group_name": {
    #   "algorithms": ["list_of_algorithm_names"],
    #   "seeds": [list_of_seed_numbers]
    # }
    DEFAULT_GROUPS_CONFIG = {
      "group_ippo": {
        "algorithms": ["ippo", "hc_ippo_l", "mappo_no_constraint", "lru_avg"],
        "seeds": [4, 5, 6, 8,11, 12, 13]
      },
      "group_others": {
        "algorithms": ["h_mappo_l"],
        "seeds": [1, 2, 3, 9]
      }
    }
    # --- END: 默认分组配置 ---

    parser = argparse.ArgumentParser(
        description="按算法分组并指定不同种子进行聚合（分组配置在脚本内修改）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--base-dir',
        default='experiment_result_convergence',
        help='收敛性实验结果基础目录'
    )
    parser.add_argument(
        '--output-dir',
        default='analysis_convergence',
        help='聚合结果的输出目录'
    )
    parser.add_argument(
        '--sub-dir',
        type=str,
        required=True,
        help='必须提供一个子目录名称，用于保存本次特定分组聚合的结果'
    )

    args = parser.parse_args()

    groups_config = DEFAULT_GROUPS_CONFIG
    try:
        # 基本验证
        if not isinstance(groups_config, dict):
            raise ValueError("配置应为一个对象/字典")
        for name, config in groups_config.items():
            if not all(k in config for k in ['algorithms', 'seeds']):
                raise ValueError(f"组 '{name}' 缺少 'algorithms' 或 'seeds' 键")
            if not isinstance(config['algorithms'], list) or not isinstance(config['seeds'], list):
                 raise ValueError(f"组 '{name}' 的 'algorithms' 和 'seeds' 必须是列表")
    except ValueError as e:
        print(f"❌ 脚本内默认配置无效: {e}")
        return

    print("📖 分组聚合配置 (来自脚本内部):")
    all_seeds = set()
    for name, config in groups_config.items():
        print(f"  - 组 '{name}':")
        print(f"    算法: {config['algorithms']}")
        print(f"    种子: {config['seeds']}")
        all_seeds.update(config['seeds'])
    
    print(f"\n🔍 扫描所有相关种子 {sorted(list(all_seeds))} 的实验...")
    all_experiments = find_convergence_experiment_results(args.base_dir, allowed_seeds=all_seeds)

    if not all_experiments:
        print("❌ 未找到任何匹配的实验结果。")
        return
    
    print(f"📊 找到 {len(all_experiments)} 个相关实验，开始按组筛选和聚合...")

    all_raw_data_dfs = []
    all_summary_stats_dfs = []

    for name, config in groups_config.items():
        print(f"\n🔄 处理组 '{name}'...")
        
        algos_in_group = set(config['algorithms'])
        seeds_in_group = set(config['seeds'])

        # 从已扫描的全部实验中筛选出当前组的实验
        experiments_for_group = [
            exp for exp in all_experiments
            if exp['algorithm'] in algos_in_group and exp['seed'] in seeds_in_group
        ]
        
        if not experiments_for_group:
            print(f"  ⚠️ 在组 '{name}' 中未找到任何实验数据，跳过。")
            continue

        print(f"  - 找到 {len(experiments_for_group)} 个实验进行聚合。")

        # 对筛选出的实验进行聚合
        group_result = aggregate_convergence_results(experiments_for_group)

        if group_result and group_result.get('raw_data') is not None and group_result.get('summary_stats') is not None:
            # 过滤掉由于聚合`groupby`产生的、不属于本组算法的数据
            raw_df = group_result['raw_data']
            summary_df = group_result['summary_stats']
            
            filtered_raw = raw_df[raw_df['algorithm'].isin(algos_in_group)]
            filtered_summary = summary_df[summary_df['algorithm'].isin(algos_in_group)]
            
            all_raw_data_dfs.append(filtered_raw)
            all_summary_stats_dfs.append(filtered_summary)
            print(f"  - 成功聚合，得到 {len(filtered_summary)} 条统计记录。")
        else:
            print(f"  - 组 '{name}' 聚合失败或无数据。")

    if not all_summary_stats_dfs:
        print("\n❌ 所有组均未能成功聚合数据，操作终止。")
        return

    # 合并所有组的结果
    print("\n🔗 合并所有组的聚合结果...")
    final_raw_df = pd.concat(all_raw_data_dfs, ignore_index=True)
    final_summary_df = pd.concat(all_summary_stats_dfs, ignore_index=True)

    final_data = {
        'raw_data': final_raw_df,
        'summary_stats': final_summary_df
    }
    
    print(f"✅ 合并完成。总计 {len(final_summary_df)} 条统计记录。")
    
    # 保存最终结果
    save_convergence_results(final_data, args.output_dir, args.sub_dir)

    output_path = Path(args.output_dir) / args.sub_dir
    print(f"\n🎉 分组聚合完成！结果保存在 {output_path}")

if __name__ == "__main__":
    main()
