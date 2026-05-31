#!/usr/bin/env python3
"""
敏感性分析图表生成脚本
专用于生成不同系统参数下算法性能对比的折线图和柱状图
符合IEEE期刊高质量标准
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, PercentFormatter
import seaborn as sns
from pathlib import Path
import argparse
import json
import yaml
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from matplotlib import patheffects as pe

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置高质量绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.figsize'] = (12, 8)
# 论文级配色与背景（浅灰坐标轴背景，深色坐标轴边框，柔和网格）
plt.rcParams['axes.facecolor'] = '#f2f3f5'
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#c9c9c9'
plt.rcParams['grid.alpha'] = 0.35

# 字体尺寸统一（与 plot_convergence_by_weight 保持一致）
BASE_FONT_SIZE = 20
plt.rcParams.update({
    'font.size': BASE_FONT_SIZE,            # 默认字体大小
    'axes.titlesize': BASE_FONT_SIZE + 2,   # 子图标题大小
    'axes.labelsize': BASE_FONT_SIZE,       # x、y轴标签大小
    'xtick.labelsize': BASE_FONT_SIZE - 2,  # x轴刻度标签大小
    'ytick.labelsize': BASE_FONT_SIZE - 2,  # y轴刻度标签大小
    'legend.fontsize': BASE_FONT_SIZE - 4,  # 图例字体大小（略小）
    'figure.titlesize': BASE_FONT_SIZE + 4, # Figure总标题大小 (suptitle)
})

# 算法颜色和标记配置（沿用收敛性分析的配置以保持一致性）
ALGORITHM_CONFIG = {
    'h_mappo_l': {
        'label': 'HC-MAPPO-L',
        'color': '#D55E00',  # 主方法 - 朱红 (将被调色板覆盖)
        'marker': 'o',
        'linestyle': '-',
    },
    'mappo_no_constraint': {
        'label': 'H-MAPPO',
        'color': '#286a9e',
        'marker': 's',
        'linestyle': '--',
    },
    'ippo': {
        'label': 'H-IPPO',
        'color': '#87ae41',
        'marker': '^',
        'linestyle': '-.',
    },
    'hc_ippo_l': {
        'label': 'HC-IPPO-L',
        'color': '#694898',
        'marker': '>',
        'linestyle': ':',
    },
    'greedy_policy': {
        'label': 'Greedy Policy',
        'color': '#ea7f2d',
        'marker': 'v',
        'linestyle': (0, (5, 2)),
    },
    'local_only': {
        'label': 'Local-Only',
        'color': '#2e3d28',
        'marker': 'D',
        'linestyle': (0, (3, 1, 1, 1)),
    },
    'edge_only': {
        'label': 'Edge-Only',
        'color': '#9c89bb',
        'marker': 'X',
        'linestyle': (0, (1, 1)),
    },
    'random_policy': {
        'label': 'Random Policy',
        'color': '#7d3326',
        'marker': 'P',
        'linestyle': (0, (5, 1, 1, 1)),
    },
    'lru_avg': {
        'label': 'Heuristic-MAPPO-L',
        'color': '#cec73b',
        'marker': 'h',
        'linestyle': (0, (3, 5, 1, 5, 1, 5)),
    },
}

# 学术常用且色盲友好的调色板
PALETTES = {
    # Okabe–Ito (colorblind safe). 我们将主方法优先使用 Vermillion
    'okabe': ['#D55E00', '#0072B2', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000', '#999999'],
    # Tableau 10（学界常用，区分度高）
    'tableau': ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#e377c2', '#17becf', '#7f7f7f', '#bcbd22'],
    # Candy（高饱和、易区分，近 ColorBrewer Set1 扩展）
    'candy': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'],
    # Pastel（柔和、论文常用，基于 ColorBrewer Set3）
    'pastel': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9'],
    # Original（脚本最初的原始配色）
    'original': ['#FF0000', '#0066CC', '#00AA00', '#FF6600', '#9900FF', '#FF9900', '#00CCCC', '#666666', '#CC0066']
}

# 新增：高对比度、学术友好、易于区分的调色板
PALETTES['vibrant'] = [
    '#D55E00',  # 朱红 (Vermillion, HC-MAPPO-L - 我们的方法)
    '#1f77b4',  # 亮蓝 (H-MAPPO)
    '#2ca02c',  # 鲜绿 (H-IPPO)
    '#9467bd',  # 紫色 (HC-IPPO-L)
    '#ff7f0e',  # 橙色 (Greedy Policy)
    '#8c564b',  # 棕色 (Local-Only - 替换易混淆的粉色)
    '#17becf',  # 青色 (Edge-Only)
    '#7f7f7f',  # 中灰 (Random Policy)
    '#bcbd22'   # 黄绿 (Heuristic-MAPPO-L - 替换过浅的黄色)
]

ALGORITHM_ORDER = ['h_mappo_l', 'mappo_no_constraint', 'ippo', 'hc_ippo_l',
                   'greedy_policy', 'local_only', 'edge_only', 'random_policy', 'lru_avg']

NON_LEARNING_ALGORITHMS = ['local_only', 'edge_only', 'greedy_policy']


"""
图例位置预设说明（适用于折线图和柱状图两套预设字典）：
  - axes: 选择放置图例的子图，取值 'ax1'/'ax2'/'ax3'/'ax4'
  - loc: 经典的 legend 锚点位置（如 'upper right'、'upper left' 等）
  - bbox_to_anchor: 用于对 legend 进行细粒度微调的坐标，默认坐标系为目标子图的 Axes 分数坐标系：
      bbox_to_anchor=(x, y) 表示 loc 指定的那个角将被放置到 (x, y) 位置。
      其中 (0, 0) 是子图左下角，(1, 1) 是子图右上角。
      例如 loc='upper right', bbox_to_anchor=(0.98, 0.88) 表示将图例右上角放到子图宽度的 98%、高度的 88% 处，
      相比默认右上角(1.0, 1.0)略微向内、向下移动，避免遮挡曲线。
"""

# 每个参数的图例位置预设（折线图）
LEGEND_PRESETS_LINES = {
    # 延迟约束：放在 ax1 右上角并略微下移，避免遮挡主曲线
    'latency_constraint': {'axes': 'ax1', 'loc': 'upper right', 'bbox_to_anchor': (0.995, 0.93)},
    # 输入数据大小：放在 ax1 左上角，一般空白较多；提供锚点便于微调
    'input_size_range': {'axes': 'ax1', 'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},
    # 用户数量：右下角通常较空；可根据实际曲线密度调整 y 值
    'num_users': {'axes': 'ax1', 'loc': 'lower right', 'bbox_to_anchor': (0.98, 0.02)},
    # 边缘服务器数量：右下角初值
    'num_edges': {'axes': 'ax1', 'loc': 'lower right', 'bbox_to_anchor': (0.98, 0.02)},
    # 每模型服务实例数：右下角初值
    'services_per_model': {'axes': 'ax1', 'loc': 'lower right', 'bbox_to_anchor': (0.98, 0.02)},
    # 服务器存储：右上角初值
    'server_storage_range': {'axes': 'ax1', 'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
    # 服务器计算：右上角初值
    'server_compute_range': {'axes': 'ax1', 'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
    # 服务器带宽：右上角初值
    'server_bandwidth_range': {'axes': 'ax1', 'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
    # 用户计算：右上角初值
    'user_compute_range': {'axes': 'ax1', 'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
    # 能量隐私权重：左中部区域通常更空；可按需要上下微调 y
    'energy_privacy_weights': {'axes': 'ax1', 'loc': 'center left', 'bbox_to_anchor': (0.02, 0.5)},
    # 能量隐私权重 0：与上同
    'energy_privacy_weights_0': {'axes': 'ax1', 'loc': 'center left', 'bbox_to_anchor': (0.02, 0.5)},
}

# 每个参数的图例位置预设（柱状图）
LEGEND_PRESETS_BARS = {
    # 延迟约束：放在 ax1 右上角并略微下移，避免遮挡柱顶
    'latency_constraint': {'axes': 'ax4', 'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.90)},
    # 输入数据大小：左上角初值
    'input_size_range': {'axes': 'ax1', 'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},
    # 用户数量：右下角初值
    'num_users': {'axes': 'ax1', 'loc': 'lower right', 'bbox_to_anchor': (0.98, 0.02)},
    # 边缘服务器数量：右下角初值
    'num_edges': {'axes': 'ax1', 'loc': 'lower right', 'bbox_to_anchor': (0.98, 0.02)},
    # 每模型服务实例数：右下角初值
    'services_per_model': {'axes': 'ax1', 'loc': 'lower right', 'bbox_to_anchor': (0.98, 0.02)},
    # 服务器存储：右上角初值
    'server_storage_range': {'axes': 'ax1', 'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
    # 服务器计算：右上角初值
    'server_compute_range': {'axes': 'ax1', 'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
    # 服务器带宽：右上角初值
    'server_bandwidth_range': {'axes': 'ax1', 'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
    # 用户计算：右上角初值
    'user_compute_range': {'axes': 'ax1', 'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
    # 能量隐私权重：左中部初值
    'energy_privacy_weights': {'axes': 'ax1', 'loc': 'center left', 'bbox_to_anchor': (0.02, 0.5)},
    # 能量隐私权重 0：左中部初值
    'energy_privacy_weights_0': {'axes': 'ax1', 'loc': 'center left', 'bbox_to_anchor': (0.02, 0.5)},
}


def apply_palette(palette_name: str):
    colors = PALETTES.get(palette_name, PALETTES['okabe'])
    for i, alg in enumerate(ALGORITHM_ORDER):
        if alg in ALGORITHM_CONFIG:
            ALGORITHM_CONFIG[alg]['color'] = colors[i % len(colors)]

# 参数显示配置
PARAMETER_CONFIG = {
    'num_users': {
        'name': '用户数量',
        'name_en': 'Number of Users',
        'unit': '',
        'sort_key': lambda x: int(x)
    },
    'num_edges': {
        'name': '边缘服务器数量',
        'name_en': 'Number of Edge Servers',
        'unit': '',
        'sort_key': lambda x: int(x)
    },
    'services_per_model': {
        'name': '每模型服务实例数',
        'name_en': 'Services per Model',
        'unit': '',
        'sort_key': lambda x: int(x)
    },
    'input_size_range': {
        'name': '输入数据大小',
        'name_en': 'Input Data Size',
        'unit': 'MB',
        'sort_key': lambda x: sum(map(float, x.replace('_', '-').split('-'))) / len(x.replace('_', '-').split('-'))
    },
    'latency_constraint': {
        'name': '延迟约束',
        'name_en': 'Delay Constraint',
        'unit': 's',
        'sort_key': lambda x: float(x)
    },
    'server_storage_range': {
        'name': '服务器存储容量',
        'name_en': 'Server Storage Capacity',
        'unit': 'GB',
        'sort_key': lambda x: sum(map(float, x.replace('_', '-').split('-'))) / len(x.replace('_', '-').split('-'))
    },
    'server_compute_range': {
        'name': '服务器计算能力',
        'name_en': 'Server Compute Capacity',
        'unit': 'GFLOPS',
        'sort_key': lambda x: sum(map(float, x.replace('_', '-').split('-'))) / len(x.replace('_', '-').split('-'))
    },
    'server_bandwidth_range': {
        'name': '服务器带宽',
        'name_en': 'Server Bandwidth',
        'unit': 'MHz',
        'sort_key': lambda x: sum(map(float, x.replace('_', '-').split('-'))) / len(x.replace('_', '-').split('-'))
    },
    'user_compute_range': {
        'name': '用户计算能力',
        'name_en': 'User Compute Capacity',
        'unit': 'GFLOPS',
        'sort_key': lambda x: sum(map(float, x.replace('_', '-').split('-'))) / len(x.replace('_', '-').split('-'))
    },
    'energy_privacy_weights': {
        'name': '能量隐私权重',
        'name_en': 'Energy-Privacy Weights',
        'unit': '',
        'sort_key': lambda x: float(x)
    },
    'energy_privacy_weights_0': {
        'name': '能量隐私权重 0',
        'name_en': 'Energy-Privacy Weights 0',
        'unit': '',
        'sort_key': lambda x: float(x)
    },
}

def _load_latency_threshold_from_sensitivity_config(default_tau=3.0):
    """从敏感性实验配置文件中读取时延阈值"""
    cfg_path = Path('configs/experiments/sensitivity.yaml')
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                # 敏感性分析配置的结构可能在 fixed_params -> framework -> latency_constraint_s
                tau = data.get('fixed_params', {}).get('framework', {}).get('latency_constraint_s', default_tau)
                return float(tau)
        except (IOError, yaml.YAMLError, ValueError):
            # 如果文件读取或解析失败，则返回默认值
            return default_tau
    return default_tau

def load_sensitivity_data(data_dir, parameters_to_load):
    """加载敏感性分析数据"""
    data_dir = Path(data_dir)
    sensitivity_data = {}
    
    print("🔍 正在扫描敏感性分析数据...")
    
    # 只加载指定的参数
    for param_name in parameters_to_load:
        param_dir = data_dir / param_name
        if not param_dir.is_dir():
            continue
            
        print(f"  📁 处理参数: {param_name}")
        
        param_data = {}
        
        for value_dir in param_dir.iterdir():
            if not value_dir.is_dir():
                continue
                
            param_value = value_dir.name
            value_data = {}
            
            for algo_dir in value_dir.iterdir():
                if not algo_dir.is_dir():
                    continue
                    
                algorithm = algo_dir.name
                seed_data = []
                
                # 收集所有种子的数据
                for seed_dir in algo_dir.iterdir():
                    if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
                        continue
                        
                    metrics_file = seed_dir / 'metrics.csv'
                    if metrics_file.exists():
                        try:
                            df = pd.read_csv(metrics_file)
                            seed_data.append(df)
                        except Exception as e:
                            print(f"    ⚠️  加载失败: {metrics_file} - {e}")
                
                if seed_data:
                    value_data[algorithm] = seed_data
                    print(f"    ✅ {algorithm}: {len(seed_data)} 个种子")
            
            if value_data:
                param_data[param_value] = value_data
        
        if param_data:
            sensitivity_data[param_name] = param_data
            print(f"  ✅ {param_name}: {len(param_data)} 个参数值")
    
    print(f"📊 数据加载完成: {len(sensitivity_data)} 个参数")
    return sensitivity_data

def process_algorithm_data(seed_data_list, algorithm, tail_n=50):
    """处理单个算法的多种子数据，返回均值和标准差"""
    if not seed_data_list:
        return None
    
    # 对每个种子取均值
    seed_means = []
    for df in seed_data_list:
        if algorithm in NON_LEARNING_ALGORITHMS:
            # 对于非学习型算法，对所有记录的步骤取平均值
            processed_data = df
        else:
            # 对于学习型算法，取最后tail_n个更新的均值以获得收敛性能
            if len(df) > tail_n:
                processed_data = df.tail(tail_n)
            else:
                processed_data = df
        seed_means.append(processed_data.mean(numeric_only=True))
    
    # 计算跨种子的统计量
    if len(seed_means) == 1:
        # 只有一个种子，无法计算标准差
        result = seed_means[0].to_dict()
        result_std = {k: 0.0 for k in result.keys()}
    else:
        # 多个种子，计算均值和标准差
        seed_df = pd.DataFrame(seed_means)
        result = seed_df.mean().to_dict()
        result_std = seed_df.std().to_dict()
    
    return result, result_std

def plot_parameter_sensitivity_lines(data, param_name, output_dir, args):
    """绘制参数敏感性折线图"""
    if param_name not in data:
        print(f"⚠️  没有参数 {param_name} 的数据")
        return
    
    DEFAULT_PRIVACY_WEIGHT = 5.0

    param_data = data[param_name]
    param_config = PARAMETER_CONFIG.get(param_name, {})
    
    # 排序参数值
    param_values = list(param_data.keys())
    if 'sort_key' in param_config:
        param_values.sort(key=param_config['sort_key'])
    else:
        param_values.sort()
    
    # 提前处理x轴坐标，确保所有算法对齐
    x_coords_map = {}
    is_numeric_x = False
    numeric_params = ['energy_privacy_weights', 'latency_constraint', 'num_users', 'services_per_model',
                      'server_storage_range', 'server_compute_range', 'user_compute_range', 'energy_privacy_weights_0',
                      'input_size_range', 'server_bandwidth_range']
    
    if param_name in numeric_params:
        is_numeric_x = True
        for pv in param_values:
            try:
                if param_name == 'energy_privacy_weights':
                    x_coords_map[pv] = float(pv) / DEFAULT_PRIVACY_WEIGHT
                elif 'range' in param_name:
                    parts = list(map(float, pv.replace('_', '-').split('-')))
                    x_coords_map[pv] = sum(parts) / len(parts)
                else:
                    x_coords_map[pv] = float(pv)
            except ValueError:
                is_numeric_x = False # 转换失败，回退到分类处理
                break
    
    if not is_numeric_x:
        x_coords_map = {pv: i for i, pv in enumerate(param_values)}

    # 移除全局x范围计算，让 margins 自动处理
    # if is_numeric_x:
    #     x_min_global = min(x_coords_map.values()) if x_coords_map else 0
    #     x_max_global = max(x_coords_map.values()) if x_coords_map else 1
    # else:
    #     x_min_global = 0
    #     x_max_global = max(x_coords_map.values()) if x_coords_map else 0

    # 收集所有算法
    all_algorithms = set()
    for value_data in param_data.values():
        all_algorithms.update(value_data.keys())
    
    # 过滤出配置中的算法并应用用户选择
    selected_algorithms = getattr(args, 'selected_algorithms', list(ALGORITHM_CONFIG.keys()))
    algorithms = [alg for alg in selected_algorithms if alg in all_algorithms]
    
    # 创建子图 - 2x2布局展示核心指标
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 添加 (a), (b), (c), (d) 标签到每个子图底部居中
    label_fontsize = plt.rcParams['axes.labelsize']
    y_pos = -0.2 # 调整此值以控制标签与x轴的垂直距离
    ax1.text(0.5, y_pos, '(a)', transform=ax1.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax2.text(0.5, y_pos, '(b)', transform=ax2.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax3.text(0.5, y_pos, '(c)', transform=ax3.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax4.text(0.5, y_pos, '(d)', transform=ax4.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    
    # 处理每个算法的数据
    for algorithm in algorithms:
        config = ALGORITHM_CONFIG[algorithm]
        
        x_vals = []
        user_rewards = []
        delays = []
        energies = []
        privacies = []
        
        user_rewards_err = []
        delays_err = []
        energies_err = []
        privacies_err = []
        
        # 处理范围参数的显示
        # x_labels = [] # 由 numeric_x_coords 和 param_values 替代
        
        for param_value in param_values:
            if algorithm not in param_data[param_value]:
                continue
                
            result = process_algorithm_data(param_data[param_value][algorithm], algorithm, args.tail_n)
            if result is None:
                continue
                
            means, stds = result
            
            x_vals.append(x_coords_map[param_value])
            
            # 将负的用户奖励转换为正的成本值（专业做法）
            raw_reward = means.get('avg_user_reward', 0)
            user_cost = abs(raw_reward) if raw_reward < 0 else raw_reward
            user_rewards.append(user_cost)
            delays.append(means.get('avg_delay_s', 0))
            energies.append(means.get('avg_energy_j', 0))
            privacies.append(means.get('avg_privacy_cost', 0))
            
            # 误差也需要取绝对值
            raw_reward_err = stds.get('avg_user_reward', 0)
            user_rewards_err.append(abs(raw_reward_err))
            delays_err.append(stds.get('avg_delay_s', 0))
            energies_err.append(stds.get('avg_energy_j', 0))
            privacies_err.append(stds.get('avg_privacy_cost', 0))
        
        if not x_vals:
            continue
        
        # 绘制折线
        markevery = 1
        is_ours = (algorithm == 'h_mappo_l')
        lw = 2.5 if is_ours else 2.0
        alpha = 1.0 if is_ours else 0.8
        zorder = 10 if is_ours else 5
        
        # 线条描边增强可读性
        line_effects = [pe.Stroke(linewidth=lw + 1.4, foreground='white'), pe.Normal()]
        
        # 绘制逻辑
        if not args.no_error and args.error_shade and (is_ours or not args.shade_only_ours):
            # 方案一：带误差阴影的折线图
            # 用户奖励
            ln1, = ax1.plot(x_vals, user_rewards, label=config['label'], color=config['color'], marker=config['marker'], linestyle=config['linestyle'], markevery=markevery, markersize=6, linewidth=lw, alpha=alpha, zorder=zorder, markerfacecolor='white', markeredgewidth=1.2)
            ln1.set_path_effects(line_effects)
            fill_alpha = max(0.02, min(0.2, args.shade_alpha)) * (1.0 if is_ours else 0.75)
            ax1.fill_between(x_vals, np.array(user_rewards) - np.array(user_rewards_err), np.array(user_rewards) + np.array(user_rewards_err), color=config['color'], alpha=fill_alpha, zorder=zorder-1)
            # 平均延迟
            ln2, = ax2.plot(x_vals, delays, label=config['label'], color=config['color'], marker=config['marker'], linestyle=config['linestyle'], markevery=markevery, markersize=6, linewidth=lw, alpha=alpha, zorder=zorder, markerfacecolor='white', markeredgewidth=1.2)
            ln2.set_path_effects(line_effects)
            ax2.fill_between(x_vals, np.array(delays) - np.array(delays_err), np.array(delays) + np.array(delays_err), color=config['color'], alpha=fill_alpha, zorder=zorder-1)
            # 能耗
            ln3, = ax3.plot(x_vals, energies, label=config['label'], color=config['color'], marker=config['marker'], linestyle=config['linestyle'], markevery=markevery, markersize=6, linewidth=lw, alpha=alpha, zorder=zorder, markerfacecolor='white', markeredgewidth=1.2)
            ln3.set_path_effects(line_effects)
            ax3.fill_between(x_vals, np.array(energies) - np.array(energies_err), np.array(energies) + np.array(energies_err), color=config['color'], alpha=fill_alpha, zorder=zorder-1)
            # 隐私
            ln4, = ax4.plot(x_vals, privacies, label=config['label'], color=config['color'], marker=config['marker'], linestyle=config['linestyle'], markevery=markevery, markersize=6, linewidth=lw, alpha=alpha, zorder=zorder, markerfacecolor='white', markeredgewidth=1.2)
            ln4.set_path_effects(line_effects)
            ax4.fill_between(x_vals, np.array(privacies) - np.array(privacies_err), np.array(privacies) + np.array(privacies_err), color=config['color'], alpha=fill_alpha, zorder=zorder-1)
        else:
            # 方案二：带误差棒或不带误差的折线图
            yerr_user_rewards = user_rewards_err if not args.no_error else None
            yerr_delays = delays_err if not args.no_error else None
            yerr_energies = energies_err if not args.no_error else None
            yerr_privacies = privacies_err if not args.no_error else None
            capsize_val = 3 if not args.no_error else 0
            # 用户奖励
            cont1 = ax1.errorbar(x_vals, user_rewards, yerr=yerr_user_rewards, 
                        label=config['label'], color=config['color'],
                        marker=config['marker'], linestyle=config['linestyle'],
                        markevery=markevery, markersize=6, linewidth=lw,
                        alpha=alpha, zorder=zorder, capsize=capsize_val,
                        markerfacecolor='white', markeredgewidth=1.2)
            try:
                cont1[0].set_path_effects(line_effects)
            except Exception:
                pass
            # 平均延迟
            cont2 = ax2.errorbar(x_vals, delays, yerr=yerr_delays,
                        label=config['label'], color=config['color'],
                        marker=config['marker'], linestyle=config['linestyle'],
                        markevery=markevery, markersize=6, linewidth=lw,
                        alpha=alpha, zorder=zorder, capsize=capsize_val,
                        markerfacecolor='white', markeredgewidth=1.2)
            try:
                cont2[0].set_path_effects(line_effects)
            except Exception:
                pass
            # 能耗
            cont3 = ax3.errorbar(x_vals, energies, yerr=yerr_energies,
                        label=config['label'], color=config['color'],
                        marker=config['marker'], linestyle=config['linestyle'],
                        markevery=markevery, markersize=6, linewidth=lw,
                        alpha=alpha, zorder=zorder, capsize=capsize_val,
                        markerfacecolor='white', markeredgewidth=1.2)
            try:
                cont3[0].set_path_effects(line_effects)
            except Exception:
                pass
            # 隐私
            cont4 = ax4.errorbar(x_vals, privacies, yerr=yerr_privacies,
                        label=config['label'], color=config['color'],
                        marker=config['marker'], linestyle=config['linestyle'],
                        markevery=markevery, markersize=6, linewidth=lw,
                        alpha=alpha, zorder=zorder, capsize=capsize_val,
                        markerfacecolor='white', markeredgewidth=1.2)
            try:
                cont4[0].set_path_effects(line_effects)
            except Exception:
                pass
    
    # 设置坐标轴
    param_name_display = param_config.get('name_en', param_name)
    unit = param_config.get('unit', '')

    if param_name == 'energy_privacy_weights':
        xlabel = 'Energy/Privacy Weight Ratio'
    elif 'range' in param_name:
        xlabel = f"Average {param_name_display} ({unit})" if unit else f"Average {param_name_display}"
    else:
        xlabel = f"{param_name_display} ({unit})" if unit else param_name_display
    
    # IEEE风格优化：若是数值型但仅有少量离散数据点（≤8），则强制在"数据点"处打刻度，避免误读；
    # 否则交给 matplotlib 自动取刻度，保证可读性
    force_ticks_at_data = is_numeric_x and len(param_values) <= 8

    for ax in [ax1, ax2, ax3, ax4]:
        if not is_numeric_x:
            # 分类变量：刻度与数据点一一对应
            ax.set_xticks(list(x_coords_map.values()))
            ax.set_xticklabels(list(x_coords_map.keys()), rotation=0, ha='center')
        elif force_ticks_at_data:
            # 数值型且点数不多：刻度严格对齐到数据点
            tick_positions = [x_coords_map[pv] for pv in param_values]
            # 标签使用绘制时的 x 值（平均/比值等），并做整/一位小数显示
            def _fmt(v):
                return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.1f}"
            tick_labels = [_fmt(x_coords_map[pv]) for pv in param_values]
            ax.set_xticks(tick_positions)
            # 数值短标签：IEEE 更常用水平居中，避免不必要倾斜
            ax.set_xticklabels(tick_labels, rotation=0, ha='center')
        # 对于其他数值类型，让matplotlib自动选择刻度

        ax.grid(True, alpha=0.35)
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_axisbelow(True)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which='minor', axis='y', alpha=0.18)
        # 设置 y 方向 5% 边距, x 方向 2% 边距
        ax.margins(y=0.05, x=0.02)
    
    ax1.set_ylabel('User Cost')
    ax2.set_ylabel('Delay (s)')
    ax3.set_ylabel('Energy (J)')
    ax4.set_ylabel('Privacy Cost')
    
    # 无论哪个参数变化，都为时延图添加阈值线
    if param_name == 'latency_constraint':
        # 当x轴是延迟约束时，绘制 y=x 对角线以表示约束满足的边界
        all_coords = list(x_coords_map.values())
        min_val = min(all_coords) if all_coords else 0
        max_val = max(all_coords) if all_coords else 1
        
        # 关键修复：仅在数据的x轴范围内绘制y=x线，不让它错误地延展x轴
        ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2.0, label='Delay Threshold (y=x)', zorder=3)
    else:
        # 当x轴是其他参数时，绘制水平的延迟阈值线
        tau = _load_latency_threshold_from_sensitivity_config()
        if tau is not None:
            tau_str = f"{int(tau)}" if tau == int(tau) else f"{tau:.2f}"
            ax2.axhline(y=tau, color='red', linestyle='--', linewidth=2.0, label=f'Delay Threshold τ̄ ({tau_str}s)', zorder=3)
            # 确保阈值在Y轴上有刻度并标红，同时移除与阈值过于接近的刻度
            original_y_ticks = list(ax2.get_yticks())
            
            # 定义一个最小间距，避免刻度重叠。例如Y轴范围的4%
            min_tick_dist = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.04
            
            # 最终的刻度列表，强制包含阈值 tau
            final_y_ticks = [tau]
            
            # 遍历自动生成的刻度，如果和 tau 不冲突，则加入
            for tick in original_y_ticks:
                if abs(tick - tau) >= min_tick_dist:
                    final_y_ticks.append(tick)
            
            # 设置新的、经过筛选的刻度
            ax2.set_yticks(sorted(list(set(final_y_ticks))))
            
            # 将阈值刻度标为红色
            for label in ax2.get_yticklabels():
                try:
                    if np.isclose(float(label.get_text().replace('−', '-')), tau):
                        label.set_color('red')
                except (ValueError, IndexError, TypeError):
                    pass
    
    # 图例（合并子图图例项，使用底部居中多列，全局一致，避免遮挡）
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1
    all_labels = labels1
    for h, l in zip(handles2, labels2):
        if l not in all_labels:
            all_handles.append(h)
            all_labels.append(l)

    # 图例参数可手动微调：
    #  - n_cols        : 每行列数（建议 4~6，取决于系列数量与版面）
    #  - bbox_to_anchor: (x, y) 为 Figure 分数坐标，(0.5, -0.02) 表示居中且略低于图形
    #                    y < 0 会把图例放到图形下方，绝对值越大越靠下
    #  - loc           : 'upper center' 搭配 bbox_to_anchor 居中放置
    #  - tight_layout  : rect=[left, bottom, right, top] 这里 bottom=0.12 预留下边距
    #                    如图例被截断或过近，可把 0.12 调到 0.10~0.18
    n_cols = min(len(all_labels), 5)
    leg = fig.legend(
        all_handles,
        all_labels,
        loc='upper center',
        bbox_to_anchor=(0.51, 0.00),  # 底部居中，向下微移
        ncol=n_cols,
        fontsize=plt.rcParams['legend.fontsize'],
        frameon=True,
    )
    leg.get_frame().set_facecolor('#ffffff')
    leg.get_frame().set_alpha(1)
    leg.get_frame().set_edgecolor('#cccccc')
    # 为底部图例预留空间
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    # 保存图表到参数专属目录
    param_output_dir = Path(output_dir) / param_name
    param_output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(param_output_dir / "lines.pdf", bbox_inches='tight')
    
    if args.show:
        plt.show()
    else:
        plt.close()
    
    print(f"📊 保存敏感性折线图: {param_output_dir / 'lines.pdf'}")


def plot_parameter_success_rate_lines(data, param_name, output_dir, args):
    """绘制参数敏感性成功率折线图（单独图表）"""
    if param_name not in data:
        return
    
    param_data = data[param_name]
    param_config = PARAMETER_CONFIG.get(param_name, {})
    
    # 收集所有算法
    all_algorithms = set()
    for value_data in param_data.values():
        all_algorithms.update(value_data.keys())
    
    # 过滤出配置中的算法并应用用户选择
    selected_algorithms = getattr(args, 'selected_algorithms', list(ALGORITHM_CONFIG.keys()))
    algorithms = [alg for alg in selected_algorithms if alg in all_algorithms]
    
    # 创建子图 - 单独成功率图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 处理每个算法的数据
    for algorithm in algorithms:
        config = ALGORITHM_CONFIG[algorithm]
        
        x_vals = []
        success_rates = []
        success_rates_err = []
        
        # 处理范围参数的显示
        x_labels = []
        
        param_values = list(param_data.keys())
        if 'sort_key' in param_config:
            param_values.sort(key=param_config['sort_key'])
        
        for param_value in param_values:
            if param_value not in param_data or algorithm not in param_data[param_value]:
                continue
            
            seed_data_list = param_data[param_value][algorithm]
            result = process_algorithm_data(seed_data_list, algorithm, args.tail_n)
            if result is None:
                continue
            
            means, stds = result
            
            x_vals.append(len(success_rates))
            success_rates.append(means.get('success_rate', 0))
            success_rates_err.append(stds.get('success_rate', 0))
            
            # 生成标签
            if 'range' in param_name:
                # 范围参数，学术论文中通常使用范围的代表值（如中点）
                try:
                    parts = list(map(float, param_value.replace('_', '-').split('-')))
                    mid_point = sum(parts) / len(parts)
                    if mid_point == int(mid_point):
                        x_label = str(int(mid_point))
                    else:
                        x_label = f"{mid_point:.1f}"
                except (ValueError, IndexError):
                    x_label = param_value
            else:
                x_label = param_value
            x_labels.append(x_label)
        
        if not x_vals:
            continue
        
        # 图表样式设置
        alpha = 0.9 if algorithm in ['h_mappo_l', 'mappo_no_constraint', 'ippo'] else 0.7
        lw = 2.5 if algorithm in ['h_mappo_l'] else 2.0
        zorder = 10 if algorithm == 'h_mappo_l' else 5
        markevery = 1
        
        line_effects = [pe.Stroke(linewidth=lw + 1.4, foreground='white'), pe.Normal()]
        
        # 绘制逻辑
        if not args.no_error and args.error_shade and (algorithm == 'h_mappo_l' or not args.shade_only_ours):
            # 方案一：带误差阴影
            ln, = ax.plot(x_vals, success_rates, label=config['label'], color=config['color'], marker=config['marker'], linestyle=config['linestyle'], markevery=markevery, markersize=6, linewidth=lw, alpha=alpha, zorder=zorder, markerfacecolor='white', markeredgewidth=1.2)
            ln.set_path_effects(line_effects)
            fill_alpha = max(0.02, min(0.2, args.shade_alpha)) * (1.0 if algorithm == 'h_mappo_l' else 0.75)
            ax.fill_between(x_vals, np.array(success_rates) - np.array(success_rates_err), np.array(success_rates) + np.array(success_rates_err), color=config['color'], alpha=fill_alpha, zorder=zorder-1)
        else:
            # 方案二：带误差棒或不带误差
            yerr_success_rates = success_rates_err if not args.no_error else None
            capsize_val = 3 if not args.no_error else 0
            cont = ax.errorbar(x_vals, success_rates, yerr=yerr_success_rates,
                       label=config['label'], color=config['color'],
                       marker=config['marker'], linestyle=config['linestyle'],
                       markevery=markevery, markersize=6, linewidth=lw,
                       alpha=alpha, zorder=zorder, capsize=capsize_val,
                       markerfacecolor='white', markeredgewidth=1.2)
            try:
                cont[0].set_path_effects(line_effects)
            except Exception:
                pass

    # 设置坐标轴
    # 对于 services_per_model，使用更明确的标签
    if param_name == 'services_per_model':
        xlabel = 'Number of Services per Model'
    else:
        param_name_display = param_config.get('name_en', param_name)
        unit = param_config.get('unit', '')
        if 'range' in param_name:
            xlabel = f"Average {param_name_display} ({unit})" if unit else f"Average {param_name_display}"
        else:
            xlabel = f"{param_name_display} ({unit})" if unit else param_name_display
    
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0, ha='center')
    ax.grid(True, alpha=0.35)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel('Success Rate')
    ax.set_axisbelow(True)
    
    # 设置 y 轴为百分比格式 (0-1 显示为 0%-100%，保留一位小数)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='minor', axis='y', alpha=0.18)
    # 设置 y 方向 5% 边距, x 方向 2% 边距
    ax.margins(y=0.05, x=0.02)
    
    # 图例（放置在图内左下角空白处，避免遮挡数据）
    # 将成功率图的图例字号相对全局略小一点，避免喧宾夺主
    leg = ax.legend(bbox_to_anchor=(0.02, 0.02), loc='lower left', 
              frameon=True, fancybox=False, shadow=False, fontsize=plt.rcParams['legend.fontsize'] - 2,
              borderpad=0.3, labelspacing=0.3, handletextpad=0.5, columnspacing=0.8)
    leg.get_frame().set_facecolor('#ffffff')
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_edgecolor('#cccccc')
    
    # 图例在图内，不需要为右侧预留空间
    plt.tight_layout()
    
    # 保存图表到参数专属目录
    param_output_dir = Path(output_dir) / param_name
    param_output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(param_output_dir / "success_rate.pdf", bbox_inches='tight')
    
    if args.show:
        plt.show()
    else:
        plt.close()
    
    print(f"📊 保存成功率敏感性折线图: {param_output_dir / 'success_rate.pdf'}")


def plot_parameter_sensitivity_bars(data, param_name, output_dir, args):
    """绘制参数敏感性分组柱状图（IEEE期刊风格）"""
    if param_name not in data:
        return
    
    param_data = data[param_name]
    param_config = PARAMETER_CONFIG.get(param_name, {})
    
    # 排序参数值
    param_values = list(param_data.keys())
    if 'sort_key' in param_config:
        param_values.sort(key=param_config['sort_key'])
    else:
        param_values.sort()
    
    # 收集所有算法
    all_algorithms = set()
    for value_data in param_data.values():
        all_algorithms.update(value_data.keys())
    
    # 过滤出配置中的算法并应用用户选择
    selected_algorithms = getattr(args, 'selected_algorithms', list(ALGORITHM_CONFIG.keys()))
    algorithms = [alg for alg in selected_algorithms if alg in all_algorithms]
    
    # 处理所有数据
    results_matrix = {}  # {algorithm: {param_value: (means, stds)}}
    for algorithm in algorithms:
        results_matrix[algorithm] = {}
        for param_value in param_values:
            if algorithm in param_data[param_value]:
                result = process_algorithm_data(param_data[param_value][algorithm], algorithm, args.tail_n)
                if result is not None:
                    results_matrix[algorithm][param_value] = result
    
    # 过滤掉没有足够数据的算法
    algorithms = [alg for alg in algorithms if len(results_matrix[alg]) >= len(param_values) // 2]
    
    if not algorithms:
        return
    
    # 创建分组柱状图 - 2x2布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 添加 (a), (b), (c), (d) 标签到每个子图底部居中
    label_fontsize = plt.rcParams['axes.labelsize']
    y_pos = -0.2 # 调整此值以控制标签与x轴的垂直距离
    ax1.text(0.5, y_pos, '(a)', transform=ax1.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax2.text(0.5, y_pos, '(b)', transform=ax2.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax3.text(0.5, y_pos, '(c)', transform=ax3.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax4.text(0.5, y_pos, '(d)', transform=ax4.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    
    # 分组柱状图参数
    n_groups = len(param_values)
    n_algorithms = len(algorithms)
    bar_width = 0.8 / n_algorithms
    r = np.arange(n_groups)
    
    # 处理x轴标签
    x_labels = []
    for param_value in param_values:
        if 'range' in param_name:
            # 范围参数，学术论文中通常使用范围的代表值（如中点）
            try:
                parts = list(map(float, param_value.replace('_', '-').split('-')))
                mid_point = sum(parts) / len(parts)
                if mid_point == int(mid_point):
                    x_label = str(int(mid_point))
                else:
                    x_label = f"{mid_point:.1f}"
            except (ValueError, IndexError):
                x_label = param_value
        else:
            x_label = param_value
        x_labels.append(x_label)
    
    # IEEE期刊风格的填充图案
    hatches = ['', '///', '...', '+++', 'xxx', '|||', '---', '\\\\\\']
    
    # 绘制每个算法的柱子
    for i, algorithm in enumerate(algorithms):
        config = ALGORITHM_CONFIG[algorithm]
        
        # 收集该算法在所有参数值下的数据
        user_rewards = []
        delays = []
        energies = []
        privacies = []
        
        user_rewards_err = []
        delays_err = []
        energies_err = []
        privacies_err = []
        
        for param_value in param_values:
            if param_value in results_matrix[algorithm]:
                means, stds = results_matrix[algorithm][param_value]
                # 将负的用户奖励转换为正的成本值（专业做法）
                raw_reward = means.get('avg_user_reward', 0)
                user_cost = abs(raw_reward) if raw_reward < 0 else raw_reward
                user_rewards.append(user_cost)
                delays.append(means.get('avg_delay_s', 0))
                energies.append(means.get('avg_energy_j', 0))
                privacies.append(means.get('avg_privacy_cost', 0))
                
                # 误差也需要取绝对值
                raw_reward_err = stds.get('avg_user_reward', 0)
                user_rewards_err.append(abs(raw_reward_err))
                delays_err.append(stds.get('avg_delay_s', 0))
                energies_err.append(stds.get('avg_energy_j', 0))
                privacies_err.append(stds.get('avg_privacy_cost', 0))
            else:
                # 填充缺失数据
                user_rewards.append(0)
                delays.append(0)
                energies.append(0)
                privacies.append(0)
                user_rewards_err.append(0)
                delays_err.append(0)
                energies_err.append(0)
                privacies_err.append(0)
        
        # 计算每组柱子的位置
        positions = r + i * bar_width
        
        # 突出我们的算法
        is_ours = (algorithm == 'h_mappo_l')
        alpha = 0.9 if is_ours else 0.7
        edge_color = 'black' if is_ours else config['color']
        edge_width = 2 if is_ours else 1
        hatch = hatches[i % len(hatches)]
        
        # 根据参数控制是否绘制误差棒
        yerr_user_rewards = user_rewards_err if not args.no_error else None
        yerr_delays = delays_err if not args.no_error else None
        yerr_energies = energies_err if not args.no_error else None
        yerr_privacies = privacies_err if not args.no_error else None
        capsize_val = 3 if not args.no_error else 0

        # 绘制两个子图
        ax1.bar(positions, user_rewards, bar_width, 
                label=config['label'], color=config['color'], alpha=alpha,
                yerr=yerr_user_rewards, capsize=capsize_val, ecolor='gray',
                edgecolor=edge_color, linewidth=edge_width, hatch=hatch)
        
        ax2.bar(positions, delays, bar_width,
                label=config['label'], color=config['color'], alpha=alpha,
                yerr=yerr_delays, capsize=capsize_val, ecolor='gray',
                edgecolor=edge_color, linewidth=edge_width, hatch=hatch)

        ax3.bar(positions, energies, bar_width,
                label=config['label'], color=config['color'], alpha=alpha,
                yerr=yerr_energies, capsize=capsize_val, ecolor='gray',
                edgecolor=edge_color, linewidth=edge_width, hatch=hatch)
                
        ax4.bar(positions, privacies, bar_width,
                label=config['label'], color=config['color'], alpha=alpha,
                yerr=yerr_privacies, capsize=capsize_val, ecolor='gray',
                edgecolor=edge_color, linewidth=edge_width, hatch=hatch)
    
    # 设置坐标轴
    param_name_display = param_config.get('name_en', param_name)
    unit = param_config.get('unit', '')
    if 'range' in param_name:
        xlabel = f"Average {param_name_display} ({unit})" if unit else f"Average {param_name_display}"
    else:
        xlabel = f"{param_name_display} ({unit})" if unit else param_name_display
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(r + bar_width * (n_algorithms - 1) / 2)
        ax.set_xticklabels(x_labels, rotation=0, ha='center')
        ax.grid(True, axis='y', alpha=0.35)
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_axisbelow(True)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which='minor', axis='y', alpha=0.18)
        # 设置 y 方向 5% 边距, x 方向 2% 边距
        ax.margins(y=0.05, x=0.02)
    
    ax1.set_ylabel('User Cost')
    ax2.set_ylabel('Delay (s)')
    ax3.set_ylabel('Energy (J)')
    ax4.set_ylabel('Privacy Cost')
    
    # 为延迟约束的条形图添加阈值线
    if param_name == 'latency_constraint':
        try:
            # 获取每个分组中心对应的约束数值
            constraint_values = [float(pv) for pv in param_values]
            group_centers = r + bar_width * (n_algorithms - 1) / 2
            
            # 绘制阈值线
            ax2.plot(group_centers, constraint_values, 
                     color='red', linestyle='--', linewidth=2.0, 
                    #  marker='x', markersize=8,
                     label='Delay Threshold', zorder=10)
        except (ValueError, IndexError) as e:
            print(f"⚠️  无法在条形图上绘制延迟阈值线: {e}")
    else:
        # 为其他参数的条形图添加水平阈值线
        tau = _load_latency_threshold_from_sensitivity_config()
        if tau is not None:
            tau_str = f"{int(tau)}" if tau == int(tau) else f"{tau:.2f}"
            ax2.axhline(y=tau, color='red', linestyle='--', linewidth=2.0, label=f'Delay Threshold τ̄ ({tau_str}s)', zorder=10)
            
            # 确保阈值在Y轴上有刻度并标红，同时移除与阈值过于接近的刻度
            original_y_ticks = list(ax2.get_yticks())
            
            # 定义一个最小间距，避免刻度重叠。例如Y轴范围的4%
            min_tick_dist = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.04
            
            # 最终的刻度列表，强制包含阈值 tau
            final_y_ticks = [tau]
            
            # 遍历自动生成的刻度，如果和 tau 不冲突，则加入
            for tick in original_y_ticks:
                if abs(tick - tau) >= min_tick_dist:
                    final_y_ticks.append(tick)
            
            ax2.set_yticks(sorted(list(set(final_y_ticks))))
            
            # 将阈值刻度标为红色
            for label in ax2.get_yticklabels():
                try:
                    if np.isclose(float(label.get_text().replace('−', '-')), tau):
                        label.set_color('red')
                except (ValueError, IndexError, TypeError):
                    pass

    # 图例（合并所有子图的图例项）
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    for h, l in zip(handles2, labels2):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    
    # 所有柱状图统一使用：底部居中、多列的全局图例（避免遮挡，论文友好）
    # 图例参数可手动微调：
    #  - n_cols        : 每行列数（建议 4~6）
    #  - bbox_to_anchor: (x, y) 为 Figure 分数坐标，(0.5, -0.02) 表示居中且略低于图形
    #  - loc           : 'upper center' 居中；若想更靠下，调小 y；想更靠上，调大 y
    #  - tight_layout  : bottom=0.12 预留下边距，图例过挤时可适当增大
    n_cols = min(len(labels), 5)
    leg = fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.51, 0.00),  # 底部居中，向下微移
        ncol=n_cols,
        fontsize=plt.rcParams['legend.fontsize'],
        frameon=True,
    )
    leg.get_frame().set_facecolor('#ffffff')
    leg.get_frame().set_alpha(1)
    leg.get_frame().set_edgecolor('#cccccc')
    # 为底部图例预留空间（可按需要将 0.12 调整到 0.10~0.18）
    fig.subplots_adjust(hspace=0.6, wspace=0.25)
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    # 保存图表到参数专属目录
    param_output_dir = Path(output_dir) / param_name
    param_output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(param_output_dir / "bars.pdf", bbox_inches='tight')
    
    if args.show:
        plt.show()
    else:
        plt.close()
    
    print(f"📊 保存敏感性分组柱状图: {param_output_dir / 'bars.pdf'}")

def generate_sensitivity_summary(data, output_dir, args):
    """生成敏感性分析汇总报告"""
    all_summary_data = []
    
    for param_name, param_data in data.items():
        param_config = PARAMETER_CONFIG.get(param_name, {})
        param_values = list(param_data.keys())
        
        if 'sort_key' in param_config:
            param_values.sort(key=param_config['sort_key'])
        
        param_summary_data = []
        
        for param_value in param_values:
            value_data = param_data[param_value]
            
            for algorithm, seed_data_list in value_data.items():
                if algorithm not in ALGORITHM_CONFIG:
                    continue
                
                # 应用算法筛选
                selected_algorithms = getattr(args, 'selected_algorithms', list(ALGORITHM_CONFIG.keys()))
                if algorithm not in selected_algorithms:
                    continue
                
                result = process_algorithm_data(seed_data_list, algorithm, args.tail_n)
                if result is None:
                    continue
                
                means, stds = result
                
                row_data = {
                    'parameter': param_name,
                    'parameter_value': param_value,
                    'algorithm': algorithm,
                    'algorithm_label': ALGORITHM_CONFIG[algorithm]['label'],
                    'n_seeds': len(seed_data_list),
                    'user_reward_mean': means.get('avg_user_reward', 0),
                    'user_reward_std': stds.get('avg_user_reward', 0),
                    'delay_mean': means.get('avg_delay_s', 0),
                    'delay_std': stds.get('avg_delay_s', 0),
                    'success_rate_mean': means.get('success_rate', 0),
                    'success_rate_std': stds.get('success_rate', 0),
                    'energy_mean': means.get('avg_energy_j', 0),
                    'energy_std': stds.get('avg_energy_j', 0),
                    'privacy_mean': means.get('avg_privacy_cost', 0),
                    'privacy_std': stds.get('avg_privacy_cost', 0),
                }
                
                all_summary_data.append(row_data)
                param_summary_data.append(row_data)
        
        # 保存每个参数的汇总数据
        if param_summary_data:
            param_df = pd.DataFrame(param_summary_data)
            param_output_dir = Path(output_dir) / param_name
            param_output_dir.mkdir(parents=True, exist_ok=True)
            
            param_df.to_csv(param_output_dir / "summary.csv", index=False)
            print(f"📋 保存参数 {param_name} 汇总: {param_output_dir / 'summary.csv'}")
    
    # 保存全局汇总数据
    if all_summary_data:
        all_summary_df = pd.DataFrame(all_summary_data)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_summary_df.to_csv(output_path / "all_parameters_summary.csv", index=False)
        print(f"📋 保存全局敏感性分析汇总: {output_path / 'all_parameters_summary.csv'}")
        
        return all_summary_df
    
    return None

def main():
    parser = argparse.ArgumentParser(description="生成敏感性分析图表")
    parser.add_argument(
        '--data-dir',
        default='experiment_result_sensitivity',
        help='敏感性分析数据目录 (默认: experiment_result_sensitivity)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='图表输出目录 (默认: 从数据目录自动生成, e.g., analysis/analysis_sensitivity_dw_1)'
    )
    parser.add_argument(
        '--parameters',
        nargs='+',
        default=['all'],
        help='要分析的参数列表，或者"all"表示全部'
    )
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=['all'],
        help='要绘制的算法列表，可选: h_mappo_l, mappo_no_constraint, ippo, hc_ippo_l, '
             'greedy_policy, local_only, edge_only, random_policy, lru_avg，或者"all"表示全部'
    )
    parser.add_argument(
        '--figures',
        nargs='+',
        choices=['lines', 'bars', 'success_rate', 'summary', 'all'],
        default=['all'],
        help='要生成的图表类型'
    )
    parser.add_argument(
        '--lines-legend', choices=['auto', 'inside', 'bottom'], default='auto',
        help='折线图图例位置：auto 按参数放在子图内部、bottom 使用底部全局图例、inside 强制使用内部预设'
    )
    parser.add_argument(
        '--tail-n', dest='tail_n', type=int, default=50,
        help='最终对比使用的尾段更新数，默认50'
    )
    parser.add_argument(
        '--no-error', action='store_true', default=True,
        help='不显示误差棒或误差阴影（默认不显示）'
    )
    parser.add_argument(
        '--error-shade', action='store_true',
        help='使用误差阴影替代误差棒'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='显示绘图窗口（默认不显示，仅保存）'
    )
    parser.add_argument(
        '--no-gray-bg',
        action='store_true',
        help='关闭浅灰坐标轴背景，使用白底背景'
    )
    parser.add_argument(
        '--palette', choices=['okabe', 'tableau', 'candy', 'pastel', 'original', 'vibrant'], default='vibrant',
        help='选择配色方案（okabe: Okabe–Ito；tableau: Tableau-10；candy: 糖果；pastel: 柔和Set3；original: 初始配色；vibrant: 高对比度）'
    )
    parser.add_argument(
        '--shade-alpha', type=float, default=0.08,
        help='误差阴影透明度（默认0.08，范围0.02-0.2）'
    )
    parser.add_argument(
        '--shade-only-ours', action='store_true',
        help='仅为我们的算法绘制误差阴影，其他算法不绘制阴影以减少遮挡'
    )
    
    args = parser.parse_args()
    
    # 根据参数切换背景
    if args.no_gray_bg:
        plt.rcParams['axes.facecolor'] = '#ffffff'
        plt.rcParams['grid.alpha'] = 0.25

    # 应用学术调色板
    apply_palette(args.palette)
    
    # 处理算法筛选参数
    if 'all' in args.algorithms:
        selected_algorithms = list(ALGORITHM_CONFIG.keys())
    else:
        # 验证算法名称
        valid_algorithms = []
        for alg in args.algorithms:
            if alg in ALGORITHM_CONFIG:
                valid_algorithms.append(alg)
            else:
                print(f"⚠️  未知算法: {alg}")
        
        if not valid_algorithms:
            print("❌ 没有有效的算法名称")
            print(f"   可用算法: {list(ALGORITHM_CONFIG.keys())}")
            return
        
        selected_algorithms = valid_algorithms
    
    print(f"🎯 选择的算法: {[ALGORITHM_CONFIG[alg]['label'] for alg in selected_algorithms]}")
    
    # 将算法列表传递给args对象
    args.selected_algorithms = selected_algorithms
    
    # 生成基于算法组合的输出目录
    if 'all' in args.algorithms:
        algo_suffix = "all_algorithms"
    else:
        # 使用算法名称创建目录后缀
        algo_suffix = "_".join(selected_algorithms)
    
    # 动态确定输出目录
    if args.output_dir is None:
        # 检查是否使用了默认的 data-dir
        data_dir_default = parser.get_default('data_dir')
        if args.data_dir == data_dir_default:
            # 如果是默认数据目录，则使用原始的默认输出目录，以保持完全一致的行为
            base_output_dir = Path('analysis/analysis_sensitivity')
        else:
            # 否则，根据传入的数据目录动态生成输出目录
            data_dir_path = Path(args.data_dir)
            analysis_dir_name = data_dir_path.name.replace('experiment_result_', 'analysis_')
            base_output_dir = Path('analysis') / analysis_dir_name
    else:
        # 使用用户指定的输出目录
        base_output_dir = Path(args.output_dir)

    # 更新输出目录
    args.output_dir = str(base_output_dir / algo_suffix)
    print(f"📁 算法组合输出目录: {args.output_dir}")
    
    # 修正BUG：在加载数据前，先根据 --parameters 参数确定要处理的参数列表
    data_dir_path = Path(args.data_dir)
    if not data_dir_path.is_dir():
        print(f"❌ 错误: 数据目录不存在: {args.data_dir}")
        return
        
    available_params = [p.name for p in data_dir_path.iterdir() if p.is_dir()]
    
    parameters_to_plot = []
    if 'all' in args.parameters:
        parameters_to_plot = available_params
    else:
        parameters_to_plot = [p for p in args.parameters if p in available_params]
        # 检查用户是否提供了无效的参数名
        for p_arg in args.parameters:
            if p_arg not in available_params:
                print(f"⚠️  警告: 参数 '{p_arg}' 在目录 '{args.data_dir}' 中未找到，将被忽略。")

    if not parameters_to_plot:
        print(f"❌ 错误: 经过筛选，没有找到任何有效的参数进行处理。可用参数: {available_params}")
        return

    print(f"🎯 将处理以下参数: {parameters_to_plot}")
    
    # 加载敏感性分析数据（只加载指定参数）
    print("📊 开始加载敏感性分析数据...")
    data = load_sensitivity_data(args.data_dir, parameters_to_plot)
    
    if not data:
        print("❌ 没有可用的敏感性分析数据")
        return
    
    # 生成图表
    for param_name in parameters_to_plot:
        print(f"\n📈 正在处理参数: {param_name}")
        
        if 'all' in args.figures or 'lines' in args.figures:
            plot_parameter_sensitivity_lines(data, param_name, args.output_dir, args)
        
        if 'all' in args.figures or 'success_rate' in args.figures:
            plot_parameter_success_rate_lines(data, param_name, args.output_dir, args)
        
        if 'all' in args.figures or 'bars' in args.figures:
            plot_parameter_sensitivity_bars(data, param_name, args.output_dir, args)
    
    # 生成汇总报告
    if 'all' in args.figures or 'summary' in args.figures:
        print("\n📋 生成敏感性分析汇总报告...")
        generate_sensitivity_summary(data, args.output_dir, args)
    
    print(f"\n🎉 敏感性分析图表生成完成! 保存在 {args.output_dir}")

if __name__ == "__main__":
    main()
