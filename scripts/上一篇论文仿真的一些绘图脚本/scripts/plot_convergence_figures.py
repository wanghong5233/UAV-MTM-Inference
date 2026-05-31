#!/usr/bin/env python3
"""
收敛性分析图表生成脚本
专用于生成收敛性实验的各种图表和分析结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from pathlib import Path
import argparse
import json
import yaml
import glob
import re
from matplotlib.colors import PowerNorm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- Matplotlib 全局样式设置 (与 plot_convergence_by_weight.py 统一) ---
# 设置高质量绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600

# 论文级配色与背景
plt.rcParams['axes.facecolor'] = '#f2f3f5'
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#c9c9c9'
plt.rcParams['grid.alpha'] = 0.35

# 字体尺寸统一
BASE_FONT_SIZE = 16  # 为2x2图稍微调整基础字号
plt.rcParams.update({
    'font.size': BASE_FONT_SIZE,
    'axes.titlesize': BASE_FONT_SIZE + 2,
    'axes.labelsize': BASE_FONT_SIZE,
    'xtick.labelsize': BASE_FONT_SIZE - 2,
    'ytick.labelsize': BASE_FONT_SIZE - 2,
    'legend.fontsize': BASE_FONT_SIZE - 2,
    'figure.titlesize': BASE_FONT_SIZE + 4,
})

# 算法颜色和标记配置
ALGORITHM_CONFIG = {
    'h_mappo_l': {
        'label': 'HC-MAPPO-L',
        'color': '#D55E00',  # Vermillion (Okabe–Ito)
        'marker': 'o',
        'linestyle': '-',
    },
    'mappo_no_constraint': {
        'label': 'H-MAPPO',
        'color': '#0072B2',  # Blue (Okabe–Ito)
        'marker': 's',
        'linestyle': '--',
    },
    'ippo': {
        'label': 'H-IPPO',
        'color': '#009E73',  # Bluish green (Okabe–Ito)
        'marker': '^',
        'linestyle': '-.',
    },
    'hc_ippo_l': {
        'label': 'HC-IPPO-L',
        'color': '#CC79A7',  # Reddish purple (Okabe–Ito)
        'marker': '>',
        'linestyle': ':',
    },
    'greedy_policy': {
        'label': 'Greedy Policy',
        'color': '#1f77b4',
        'marker': 'v',
        'linestyle': ':',
    },
    'local_only': {
        'label': 'Local-Only',
        'color': '#9467bd',
        'marker': 'D',
        'linestyle': '-',
    },
    'edge_only': {
        'label': 'Edge-Only',
        'color': '#8c564b',
        'marker': 'X',
        'linestyle': '-',
    },
    'lru_avg': {
        'label': 'Heuristic-MAPPO-L',
        'color': '#56B4E9',  # Sky blue (Okabe–Ito)
        'marker': 'h',
        'linestyle': (0, (5, 2)),  # Loosely dashed, distinct from '--' and '-.'
    },
}

# 仅学习型算法用于"收敛曲线"类图
LEARNING_ALGOS = {'h_mappo_l', 'mappo_no_constraint', 'ippo', 'hc_ippo_l', 'lru_avg'}

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
    return sorted(vals) if vals else None

def get_seed_suffix(seed_list):
    """根据种子列表生成文件名后缀"""
    if not seed_list:
        return "_allseeds"
    return f"_seeds_{'_'.join(map(str, seed_list))}"

def load_convergence_data(data_dir, sub_dir=None):
    """加载收敛性数据"""
    data_path = Path(data_dir)

    # 如果指定了子目录，则在子目录下加载
    if sub_dir:
        data_path = data_path / sub_dir

    data = {}
    
    # 尝试直接从指定目录加载数据文件
    raw_file = data_path / "raw_data.csv"
    summary_file = data_path / "summary_stats.csv"
    
    if raw_file.exists() and summary_file.exists():
        try:
            data['convergence'] = {
                'raw': pd.read_csv(raw_file),
                'summary': pd.read_csv(summary_file)
            }
            print(f"✅ 加载收敛性数据: {len(data['convergence']['raw'])} 条记录")
        except Exception as e:
            print(f"❌ 加载收敛性数据失败: {e}")
    else:
        # 兼容旧的目录结构（如果存在convergence子目录）
        convergence_dir = data_path / "convergence"
        if convergence_dir.exists():
            try:
                data['convergence'] = {
                    'raw': pd.read_csv(convergence_dir / "raw_data.csv"),
                    'summary': pd.read_csv(convergence_dir / "summary_stats.csv")
                }
                print(f"✅ 加载收敛性数据: {len(data['convergence']['raw'])} 条记录")
            except Exception as e:
                print(f"❌ 加载收敛性数据失败: {e}")
    
    return data

def aggregate_per_client_data(base_exp_dir, algorithms=None):
    """
    从基础实验目录递归扫描、聚合所有per_client_metrics.csv文件。
    智能地从路径中解析算法和种子信息。
    如果提供了 algorithms 列表，则只加载指定算法的数据。
    """
    print(f"🔍 开始从 {base_exp_dir} 聚合客户端数据...")
    
    search_pattern = str(Path(base_exp_dir) / '**' / 'per_client_metrics.csv')
    file_paths = glob.glob(search_pattern, recursive=True)
    
    if not file_paths:
        print("   -> ⚠️ 未找到任何 per_client_metrics.csv 文件。")
        return None

    all_dfs = []
    
    # 正则表达式，用于从路径中提取算法和种子
    # 匹配 .../seed_123/... 或 ...__h_mappo_l__...
    seed_regex = re.compile(r'seed_(\d+)')
    
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            
            # 从路径中解析元数据
            path_str = str(Path(path).parent)
            
            # 1. 解析种子
            seed_match = seed_regex.search(path_str)
            seed = int(seed_match.group(1)) if seed_match else -1
            
            # 2. 解析算法
            alg_name = "unknown"
            # 优先从最深的目录名片段匹配，避免匹配到上层目录名
            dir_parts = Path(path).parent.name.split('__')
            found_alg = False
            for part in dir_parts:
                if part in ALGORITHM_CONFIG:
                    alg_name = part
                    found_alg = True
                    break
            
            if not found_alg:
                 print(f"   -> ⚠️ 无法从路径 {path_str} 中解析算法名称，跳过此文件。")
                 continue

            # --- 新增：如果指定了算法列表，则只加载匹配的算法 ---
            if algorithms and alg_name not in algorithms:
                continue

            df['algorithm'] = alg_name
            df['seed'] = seed
            all_dfs.append(df)
            print(f"   -> ✅ 已加载: {path} (算法: {alg_name}, 种子: {seed})")
            
        except Exception as e:
            print(f"   -> ❌ 加载或解析 {path} 时出错: {e}")

    if not all_dfs:
        print("   -> ⚠️ 聚合数据失败，没有成功加载任何文件。")
        return None
        
    aggregated_df = pd.concat(all_dfs, ignore_index=True)
    print(f"聚合完成! 共 {len(aggregated_df)} 条记录, 来自 {len(file_paths)} 个文件。")
    return aggregated_df

def _load_latency_threshold(default_tau=3.0):
    """从收敛性实验配置文件中读取时延阈值"""
    cfg_path = Path('configs/experiments/convergence.yaml')
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                tau = data.get('fixed_params', {}).get('framework', {}).get('latency_constraint_s', default_tau)
                return float(tau)
        except Exception:
            return default_tau
    return default_tau


def _moving_average(x: pd.Series, window: int) -> pd.Series:
    if window is None or window <= 1:
        return x
    return x.rolling(window=window, min_periods=1, center=False).mean()


def _ordered_algorithms(df_algorithms):
    present = list(df_algorithms.unique())
    ordered = [alg for alg in ALGORITHM_CONFIG.keys() if alg in present]
    return ordered


def plot_convergence_curves(data, output_dir, args, seed_list=None):
    """绘制收敛性曲线图"""
    if 'convergence' not in data:
        print("⚠️  没有收敛性数据，跳过绘制")
        return
    
    summary_data = data['convergence']['summary']
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 参数
    smooth_w = int(args.smooth) if hasattr(args, 'smooth') and args.smooth is not None else 0
    use_ci = bool(args.ci) if hasattr(args, 'ci') else False
    tau = args.latency_threshold

    # 计算全局 x 轴范围
    max_update_global = int(summary_data['update'].max()) if 'update' in summary_data.columns else None

    # 移除旧的、局部的 paper 模式，使用全局样式
    # if getattr(args, 'paper', False):
    #     plt.rcParams.update({ ... })

    # 左图：用户奖励收敛曲线（User Reward R_user）
    for algorithm in _ordered_algorithms(summary_data['algorithm']):
        # 仅绘制学习型算法
        if algorithm not in LEARNING_ALGOS:
            continue
        if algorithm not in ALGORITHM_CONFIG:
            continue
        
        alg_data = summary_data[summary_data['algorithm'] == algorithm].sort_values('update')
        config = ALGORITHM_CONFIG[algorithm]
        
        # 仅用户奖励 R_user
        user_reward = alg_data['avg_user_reward_mean']
        # 误差：std 或 95%CI（近似用 success_rate_count 作为种子数）
        base_std = alg_data['avg_user_reward_std']
        n = alg_data['success_rate_count'].replace(0, np.nan)
        err = base_std if not use_ci else (1.96 * base_std / np.sqrt(n))

        # 平滑
        user_reward = _moving_average(user_reward, smooth_w)
        err = _moving_average(err, smooth_w)
        
        # 下采样（仅用于可视化，不影响统计），最多 max_points 个点
        max_points = int(getattr(args, 'max_points', 0) or 0)
        upd = alg_data['update']
        if max_points > 0 and len(upd) > max_points:
            idx = np.linspace(0, len(upd) - 1, max_points, dtype=int)
            upd = upd.iloc[idx].reset_index(drop=True)
            user_reward = user_reward.iloc[idx].reset_index(drop=True)
            err = err.iloc[idx].reset_index(drop=True)

        # 动态标记密度
        markevery = max(1, len(upd) // 20)

        is_ours = (algorithm == 'h_mappo_l')
        lw = 2.5 if is_ours else 2.0
        z = 10 if is_ours else 5
        ax1.plot(upd, user_reward, 
                label=config['label'], color=config['color'], 
                marker=config['marker'], linestyle=config['linestyle'],
                markevery=markevery, markersize=5, linewidth=lw, zorder=z)
        ax1.fill_between(upd, 
                        (user_reward - err).bfill().ffill(),
                        (user_reward + err).bfill().ffill(),
                        alpha=0.18 if is_ours else 0.15, color=config['color'], zorder=z-1)
    
    ax1.set_xlabel('Training Iteration', labelpad=10)
    ax1.set_ylabel('User Reward')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=getattr(args, 'legend_cols', 1), frameon=False)
    ax1.grid(True, alpha=0.3)
    ax1.margins(y=0.05)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    # 固定 x 轴范围并移除 x 方向留白
    if max_update_global is not None:
        ax1.set_xlim(0, max_update_global)
    ax1.margins(x=0)
    
    # 右图：时延收敛曲线
    for algorithm in _ordered_algorithms(summary_data['algorithm']):
        # 仅绘制学习型算法
        if algorithm not in LEARNING_ALGOS:
            continue
        if algorithm not in ALGORITHM_CONFIG:
            continue
        
        alg_data = summary_data[summary_data['algorithm'] == algorithm].sort_values('update')
        config = ALGORITHM_CONFIG[algorithm]
        
        mean_delay = alg_data['avg_delay_s_mean']
        base_std = alg_data['avg_delay_s_std']
        n = alg_data['success_rate_count'].replace(0, np.nan)
        err = base_std if not use_ci else (1.96 * base_std / np.sqrt(n))
        mean_delay = _moving_average(mean_delay, smooth_w)
        err = _moving_average(err, smooth_w)
        
        # 下采样（仅用于可视化）
        max_points = int(getattr(args, 'max_points', 0) or 0)
        upd = alg_data['update']
        if max_points > 0 and len(upd) > max_points:
            idx = np.linspace(0, len(upd) - 1, max_points, dtype=int)
            upd = upd.iloc[idx].reset_index(drop=True)
            mean_delay = mean_delay.iloc[idx].reset_index(drop=True)
            err = err.iloc[idx].reset_index(drop=True)

        markevery = max(1, len(upd) // 20)
        is_ours = (algorithm == 'h_mappo_l')
        lw = 2.5 if is_ours else 2.0
        z = 10 if is_ours else 5
        ax2.plot(upd, mean_delay, 
                label=config['label'], color=config['color'],
                marker=config['marker'], linestyle=config['linestyle'],
                markevery=markevery, markersize=5, linewidth=lw, zorder=z)
        ax2.fill_between(upd, 
                        (mean_delay - err).bfill().ffill(),
                        (mean_delay + err).bfill().ffill(),
                        alpha=0.18 if is_ours else 0.15, color=config['color'], zorder=z-1)
    
    # 添加时延约束线（从配置读取）
    ax2.axhline(y=tau, color='red', linestyle='--', linewidth=2, label='Latency Threshold τ̄', zorder=1)
    
    ax2.set_xlabel('Training Iteration', labelpad=10)
    ax2.set_ylabel('Delay (s)')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=getattr(args, 'legend_cols', 1), frameon=False)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.margins(y=0.05)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # 固定 x 轴范围并移除 x 方向留白
    if max_update_global is not None:
        ax2.set_xlim(0, max_update_global)
    ax2.margins(x=0)
    
    # 确保时延阈值在Y轴上有刻度
    y_ticks = list(ax2.get_yticks())
    if tau not in y_ticks:
        y_ticks.append(tau)
    ax2.set_yticks(sorted(y_ticks))
    
    # 将阈值刻度标为红色
    for label in ax2.get_yticklabels():
        try:
            if np.isclose(float(label.get_text().replace('−', '-')), tau):
                label.set_color('red')
        except ValueError:
            pass
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dpi_save = int(getattr(args, 'dpi_save', 300) or 300)
    seed_suffix = get_seed_suffix(seed_list)
    plt.savefig(output_path / f"convergence_curves{seed_suffix}.png", dpi=dpi_save, bbox_inches='tight')
    plt.savefig(output_path / f"convergence_curves{seed_suffix}.pdf", bbox_inches='tight')
    if getattr(args, 'show', False):
        plt.show()
    else:
        plt.close()
    
    print(f"📊 保存收敛性曲线图: {output_path / f'convergence_curves{seed_suffix}.png'}")

def plot_convergence_curves_four_metrics(data, output_dir, args, seed_list=None):
    """绘制四种核心指标的收敛性曲线图（用户奖励、时延、能耗、隐私）"""
    if 'convergence' not in data:
        print("⚠️  没有收敛性数据，跳过绘制")
        return
    
    summary_data = data['convergence']['summary']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    (ax1, ax2), (ax3, ax4) = axes
    
    # 添加 (a), (b), (c), (d) 标签到每个子图底部居中
    label_fontsize = plt.rcParams['axes.labelsize']
    y_pos = -0.18 # 调整此值以控制标签与x轴的垂直距离
    ax1.text(0.5, y_pos, '(a)', transform=ax1.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax2.text(0.5, y_pos, '(b)', transform=ax2.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax3.text(0.5, y_pos, '(c)', transform=ax3.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax4.text(0.5, y_pos, '(d)', transform=ax4.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    
    smooth_w = int(args.smooth) if hasattr(args, 'smooth') and args.smooth is not None else 0
    use_ci = bool(args.ci) if hasattr(args, 'ci') else False
    tau = args.latency_threshold

    # 计算全局 x 轴范围
    max_update_global = int(summary_data['update'].max()) if 'update' in summary_data.columns else None

    # 移除旧的、局部的 paper 模式，使用全局样式
    # if getattr(args, 'paper', False):
    #     plt.rcParams.update({ ... })

    metrics_to_plot = [
        (ax1, 'avg_user_reward_mean', 'avg_user_reward_std', 'User Reward', None),
        (ax2, 'avg_delay_s_mean', 'avg_delay_s_std', 'Delay (s)', tau),
        (ax3, 'avg_energy_j_mean', 'avg_energy_j_std', 'Energy (J)', None),
        (ax4, 'avg_privacy_cost_mean', 'avg_privacy_cost_std', 'Privacy Cost', None),
    ]

    for ax, mean_col, std_col, ylabel, hline in metrics_to_plot:
        for algorithm in _ordered_algorithms(summary_data['algorithm']):
            if algorithm not in LEARNING_ALGOS or algorithm not in ALGORITHM_CONFIG:
                continue
            
            alg_data = summary_data[summary_data['algorithm'] == algorithm].sort_values('update')
            config = ALGORITHM_CONFIG[algorithm]
            
            mean_vals = alg_data[mean_col]
            base_std = alg_data[std_col]
            n = alg_data['success_rate_count'].replace(0, np.nan)
            err = base_std if not use_ci else (1.96 * base_std / np.sqrt(n))

            mean_vals = _moving_average(mean_vals, smooth_w)
            err = _moving_average(err, smooth_w)
            
            max_points = int(getattr(args, 'max_points', 0) or 0)
            upd = alg_data['update']
            if max_points > 0 and len(upd) > max_points:
                idx = np.linspace(0, len(upd) - 1, max_points, dtype=int)
                upd = upd.iloc[idx].reset_index(drop=True)
                mean_vals = mean_vals.iloc[idx].reset_index(drop=True)
                err = err.iloc[idx].reset_index(drop=True)

            markevery = max(1, len(upd) // 20)
            is_ours = (algorithm == 'h_mappo_l')
            lw = 2.5 if is_ours else 2.0
            z = 10 if is_ours else 5

            ax.plot(upd, mean_vals, label=config['label'], color=config['color'],
                    marker=config['marker'], linestyle=config['linestyle'],
                    markevery=markevery, markersize=5, linewidth=lw, zorder=z)
            ax.fill_between(upd, (mean_vals - err).bfill().ffill(), (mean_vals + err).bfill().ffill(),
                            alpha=0.18 if is_ours else 0.15, color=config['color'], zorder=z-1)

        ax.set_xlabel('Training Iteration', labelpad=10)
        ax.set_ylabel(ylabel)
        if hline is not None:
            ax.axhline(y=hline, color='red', linestyle='--', linewidth=2, label='Latency Threshold τ̄', zorder=1)
            # 确保阈值在Y轴上有刻度
            y_ticks = list(ax.get_yticks())
            if not any(np.isclose(hline, t) for t in y_ticks):
                y_ticks.append(hline)
            ax.set_yticks(sorted(y_ticks))
            # 将阈值刻度标为红色
            for label in ax.get_yticklabels():
                try:
                    if np.isclose(float(label.get_text().replace('−', '-')), hline):
                        label.set_color('red')
                except ValueError:
                    pass
        if "Delay" in ylabel:
            ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.margins(y=0.05)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # 固定 x 轴范围并移除 x 方向留白
        if max_update_global is not None:
            ax.set_xlim(0, max_update_global)
        ax.margins(x=0)

    all_handles, all_labels = [], []
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in all_labels:
                all_labels.append(l)
                all_handles.append(h)
    # 恢复图例：统一放置在第一个图（用户奖励）的右下角，并设置白色背景
    ax1.legend(all_handles, all_labels, loc='lower right', 
               ncol=getattr(args, 'legend_cols', 1), 
               frameon=True, 
               fontsize=plt.rcParams['legend.fontsize'],
               facecolor='white', 
               framealpha=1)

    # 调整子图间距，为底部的 a,b,c,d 标签留出空间
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    plt.tight_layout()

    # 保存图表
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dpi_save = int(getattr(args, 'dpi_save', 300) or 300)
    seed_suffix = get_seed_suffix(seed_list)
    plt.savefig(output_path / f"convergence_curves_4_metrics{seed_suffix}.png", dpi=dpi_save, bbox_inches='tight')
    plt.savefig(output_path / f"convergence_curves_4_metrics{seed_suffix}.pdf", bbox_inches='tight')
    if getattr(args, 'show', False):
        plt.show()
    else:
        plt.close()
    
    save_path_png = output_path / f"convergence_curves_4_metrics{seed_suffix}.png"
    print(f"📊 保存四指标收敛性曲线图: {save_path_png}")

def plot_final_comparison(data, output_dir, args, seed_list=None):
    """绘制收敛后指标对比图"""
    if 'convergence' not in data:
        print("⚠️  没有收敛性数据，跳过绘制")
        return
    
    summary_data = data['convergence']['summary']
    
    # 取最后的收敛值：每算法末尾N个update的均值（N可配置）
    tail_n = int(getattr(args, 'tail_n', 50))
    rows = []
    rows_std = []
    for alg, df_alg in summary_data.groupby('algorithm'):
        if df_alg.empty:
            continue
        max_update = df_alg['update'].max()
        df_tail = df_alg[df_alg['update'] >= max_update - (tail_n - 1)]
        if df_tail.empty:
            df_tail = df_alg.tail(tail_n)
        mean_row = df_tail.mean(numeric_only=True)
        std_row = df_tail.mean(numeric_only=True)  # 占位，下面单独取std列的均值
        mean_row['algorithm'] = alg
        rows.append(mean_row)
        rows_std.append({
            'algorithm': alg,
            'avg_user_reward_std': df_tail['avg_user_reward_std'].mean() if 'avg_user_reward_std' in df_tail else 0.0,
            'avg_alloc_reward_std': df_tail['avg_alloc_reward_std'].mean() if 'avg_alloc_reward_std' in df_tail else 0.0,
            'avg_delay_s_std': df_tail['avg_delay_s_std'].mean() if 'avg_delay_s_std' in df_tail else 0.0,
            'avg_energy_j_std': df_tail['avg_energy_j_std'].mean() if 'avg_energy_j_std' in df_tail else 0.0,
            'avg_privacy_cost_std': df_tail['avg_privacy_cost_std'].mean() if 'avg_privacy_cost_std' in df_tail else 0.0,
        })
    if not rows:
        print("⚠️ 最终对比数据为空，跳过绘制")
        return
    final_df = pd.DataFrame(rows)
    final_std_df = pd.DataFrame(rows_std)
    final_merged = pd.merge(final_df, final_std_df, on='algorithm', how='left')
    
    # 另存一份用于附录（同时导出 Reward 与等价 Cost，便于论文/附录两用）
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tmp = final_merged.copy()
    tmp['user_reward'] = tmp.get('avg_user_reward_mean', 0.0)
    tmp.to_csv(output_path / "final_comparison_data.csv", index=False)

    # 准备数据
    algorithms = []          # labels
    alg_keys = []            # raw keys for colors
    user_rewards = []
    delays = []
    energies = []
    privacy_costs = []
    
    cost_errs = []
    delay_errs = []
    energy_errs = []
    privacy_errs = []

    for _, row in final_merged.iterrows():
        algorithm = row['algorithm']
        if algorithm not in ALGORITHM_CONFIG:
            continue
        alg_keys.append(algorithm)
        algorithms.append(ALGORITHM_CONFIG[algorithm]['label'])
        
        # 计算指标
        user_reward = row.get('avg_user_reward_mean', 0.0)
        user_rewards.append(user_reward)
        delays.append(row.get('avg_delay_s_mean', 0.0))
        energies.append(row.get('avg_energy_j_mean', 0.0))
        privacy_costs.append(row.get('avg_privacy_cost_mean', 0.0))
        # 误差条（仅用户奖励）
        cu = row.get('avg_user_reward_std', 0.0)
        cost_errs.append(cu)
        delay_errs.append(row.get('avg_delay_s_std', 0.0))
        energy_errs.append(row.get('avg_energy_j_std', 0.0))
        privacy_errs.append(row.get('avg_privacy_cost_std', 0.0))
    
    # 创建分组柱状图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 添加 (a), (b), (c), (d) 标签到每个子图底部居中
    label_fontsize = plt.rcParams['axes.labelsize']
    y_pos = -0.18 # 调整此值以控制标签与x轴的垂直距离
    ax1.text(0.5, y_pos, '(a)', transform=ax1.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax2.text(0.5, y_pos, '(b)', transform=ax2.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax3.text(0.5, y_pos, '(c)', transform=ax3.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    ax4.text(0.5, y_pos, '(d)', transform=ax4.transAxes, fontsize=label_fontsize + 4, va='top', ha='center')
    
    x = np.arange(len(algorithms))
    width = 0.6
    
    # 获取颜色
    colors = [ALGORITHM_CONFIG[alg]['color'] for alg in alg_keys]
    
    # 用户奖励
    bars1 = ax1.bar(x, user_rewards, width, color=colors, alpha=0.85, yerr=cost_errs, capsize=3.5, ecolor='gray')
    ax1.set_ylabel('User Reward')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars1, user_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 平均时延
    bars2 = ax2.bar(x, delays, width, color=colors, alpha=0.85, yerr=delay_errs, capsize=3.5, ecolor='gray')
    tau = args.latency_threshold
    ax2.axhline(y=tau, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Delay (s)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 能耗
    bars3 = ax3.bar(x, energies, width, color=colors, alpha=0.85, yerr=energy_errs, capsize=3.5, ecolor='gray')
    ax3.set_ylabel('Energy (J)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 隐私成本
    bars4 = ax4.bar(x, privacy_costs, width, color=colors, alpha=0.85, yerr=privacy_errs, capsize=3.5, ecolor='gray')
    ax4.set_ylabel('Privacy Cost')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 调整子图间距，为底部的 a,b,c,d 标签留出空间
    fig.subplots_adjust(hspace=0.6, wspace=0.25)
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dpi_save = int(getattr(args, 'dpi_save', 300) or 300)
    seed_suffix = get_seed_suffix(seed_list)
    plt.savefig(output_path / f"final_comparison{seed_suffix}.png", dpi=dpi_save, bbox_inches='tight')
    plt.savefig(output_path / f"final_comparison{seed_suffix}.pdf", bbox_inches='tight')
    if getattr(args, 'show', False):
        plt.show()
    else:
        plt.close()
    
    print(f"📊 保存最终对比图: {output_path / f'final_comparison{seed_suffix}.png'}")

def plot_metrics_comparison(data, output_dir, args, seed_list=None):
    """绘制成功率/能耗/隐私的最终对比（使用尾段均值）。"""
    if 'convergence' not in data:
        print("⚠️  没有收敛性数据，跳过绘制")
        return
    summary_data = data['convergence']['summary']

    tail_n = int(getattr(args, 'tail_n', 50))
    rows = []
    for alg, df_alg in summary_data.groupby('algorithm'):
        if df_alg.empty:
            continue
        max_update = df_alg['update'].max()
        df_tail = df_alg[df_alg['update'] >= max_update - (tail_n - 1)]
        if df_tail.empty:
            df_tail = df_alg.tail(tail_n)
        rows.append({
            'algorithm': alg,
            'success_rate_mean': df_tail['success_rate_mean'].mean(),
            'success_rate_std': df_tail['success_rate_std'].mean(),
            'avg_energy_j_mean': df_tail['avg_energy_j_mean'].mean(),
            'avg_energy_j_std': df_tail['avg_energy_j_std'].mean(),
            'avg_privacy_cost_mean': df_tail['avg_privacy_cost_mean'].mean(),
            'avg_privacy_cost_std': df_tail['avg_privacy_cost_std'].mean(),
        })

    if not rows:
        print("⚠️ 指标对比数据为空，跳过绘制")
        return

    df = pd.DataFrame(rows)
    alg_keys = [alg for alg in _ordered_algorithms(df['algorithm'])]
    df = df.set_index('algorithm').loc[alg_keys].reset_index()

    labels = [ALGORITHM_CONFIG[alg]['label'] for alg in alg_keys]
    colors = [ALGORITHM_CONFIG[alg]['color'] for alg in alg_keys]

    x = np.arange(len(labels))
    width = 0.65

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # 成功率
    ax1.bar(x, df['success_rate_mean'], width, color=colors, alpha=0.85, yerr=df['success_rate_std'], capsize=3.5, ecolor='gray')
    
    ax1.set_ylabel('Success Rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(True, axis='y', alpha=0.3)

    # 能耗
    ax2.bar(x, df['avg_energy_j_mean'], width, color=colors, alpha=0.85, yerr=df['avg_energy_j_std'], capsize=3.5, ecolor='gray')
    
    ax2.set_ylabel('Energy (J)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)

    # 隐私
    ax3.bar(x, df['avg_privacy_cost_mean'], width, color=colors, alpha=0.85, yerr=df['avg_privacy_cost_std'], capsize=3.5, ecolor='gray')
    
    ax3.set_ylabel('Privacy Cost')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dpi_save = int(getattr(args, 'dpi_save', 300) or 300)
    seed_suffix = get_seed_suffix(seed_list)
    plt.savefig(output_path / f'metrics_comparison{seed_suffix}.png', dpi=dpi_save, bbox_inches='tight')
    plt.savefig(output_path / f'metrics_comparison{seed_suffix}.pdf', bbox_inches='tight')
    if getattr(args, 'show', False):
        plt.show()
    else:
        plt.close()
    print(f"📊 保存指标对比图: {output_path / f'metrics_comparison{seed_suffix}.png'}")

def plot_metrics_convergence_curves(data, output_dir, args, seed_list=None):
    """绘制成功率/能耗/隐私的收敛曲线（均值±误差带）。"""
    if 'convergence' not in data:
        print("⚠️  没有收敛性数据，跳过绘制")
        return
    summary_data = data['convergence']['summary']

    smooth_w = int(args.smooth) if hasattr(args, 'smooth') and args.smooth is not None else 0
    use_ci = bool(args.ci) if hasattr(args, 'ci') else False

    metrics = [
        ('success_rate_mean', 'success_rate_std', 'Success Rate', '(a) Success Rate Convergence'),
        ('avg_energy_j_mean', 'avg_energy_j_std', 'Energy (J)', '(b) Energy Convergence'),
        ('avg_privacy_cost_mean', 'avg_privacy_cost_std', 'Privacy Cost', '(c) Privacy Convergence'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (m_mean, m_std, ylabel, title) in zip(axes, metrics):
        for algorithm in _ordered_algorithms(summary_data['algorithm']):
            # 仅绘制学习型算法
            if algorithm not in LEARNING_ALGOS:
                continue
            if algorithm not in ALGORITHM_CONFIG:
                continue
            alg_data = summary_data[summary_data['algorithm'] == algorithm].sort_values('update')
            config = ALGORITHM_CONFIG[algorithm]

            mean_series = alg_data[m_mean]
            base_std = alg_data[m_std]
            n = alg_data['success_rate_count'].replace(0, np.nan)
            err = base_std if not use_ci else (1.96 * base_std / np.sqrt(n))

            mean_series = _moving_average(mean_series, smooth_w)
            err = _moving_average(err, smooth_w)

            max_points = int(getattr(args, 'max_points', 0) or 0)
            upd = alg_data['update']
            if max_points > 0 and len(upd) > max_points:
                idx = np.linspace(0, len(upd) - 1, max_points, dtype=int)
                upd = upd.iloc[idx].reset_index(drop=True)
                mean_series = mean_series.iloc[idx].reset_index(drop=True)
                err = err.iloc[idx].reset_index(drop=True)

            markevery = max(1, len(upd) // 20)
            is_ours = (algorithm == 'h_mappo_l')
            lw = 2.5 if is_ours else 2.0
            z = 10 if is_ours else 5

            ax.plot(upd, mean_series, label=config['label'], color=config['color'],
                    marker=config['marker'], linestyle=config['linestyle'],
                    markevery=markevery, markersize=5, linewidth=lw, zorder=z)
            ax.fill_between(upd, (mean_series - err).bfill().ffill(), (mean_series + err).bfill().ffill(),
                            alpha=0.18 if is_ours else 0.15, color=config['color'], zorder=z-1)

        
        ax.set_xlabel('Training Iteration', labelpad=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=getattr(args, 'legend_cols', 1), frameon=False)
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dpi_save = int(getattr(args, 'dpi_save', 300) or 300)
    seed_suffix = get_seed_suffix(seed_list)
    plt.savefig(output_path / f'metrics_convergence_curves{seed_suffix}.png', dpi=dpi_save, bbox_inches='tight')
    plt.savefig(output_path / f'metrics_convergence_curves{seed_suffix}.pdf', bbox_inches='tight')
    if getattr(args, 'show', False):
        plt.show()
    else:
        plt.close()
    print(f"📊 保存指标收敛曲线图: {output_path / f'metrics_convergence_curves{seed_suffix}.png'}")

def generate_summary_table(data, output_dir, args, seed_list=None):
    """生成结果汇总表格"""
    if 'convergence' not in data:
        return
    
    summary_data = data['convergence']['summary']
    # 复用最终对比的提取逻辑
    tail_n = int(getattr(args, 'tail_n', 50))
    rows = []
    for alg, df_alg in summary_data.groupby('algorithm'):
        if df_alg.empty:
            continue
        max_update = df_alg['update'].max()
        df_tail = df_alg[df_alg['update'] >= max_update - (tail_n - 1)]
        if df_tail.empty:
            df_tail = df_alg.tail(tail_n)
        mean_row = df_tail.mean(numeric_only=True)
        mean_row['algorithm'] = alg
        rows.append(mean_row)
    if not rows:
        return
    final_data = pd.DataFrame(rows)
    
    # 创建汇总表格
    table_data = []
    for _, row in final_data.iterrows():
        algorithm = row['algorithm']
        if algorithm not in ALGORITHM_CONFIG:
            continue
        
        total_cost = -(row['avg_user_reward_mean'] + row['avg_alloc_reward_mean'])
        tau = args.latency_threshold
        constraint_violation = max(0, row['avg_delay_s_mean'] - tau)
        
        table_data.append({
            'Algorithm': ALGORITHM_CONFIG[algorithm]['label'],
            'Success Rate': f"{row['success_rate_mean']:.3f} ± {row['success_rate_std']:.3f}",
            'Avg Delay (s)': f"{row['avg_delay_s_mean']:.3f} ± {row['avg_delay_s_std']:.3f}",
            'Constraint Violation (s)': f"{constraint_violation:.3f}",
            'Energy (J)': f"{row['avg_energy_j_mean']:.2f} ± {row['avg_energy_j_std']:.2f}",
            'Privacy Cost': f"{row['avg_privacy_cost_mean']:.4f} ± {row['avg_privacy_cost_std']:.4f}",
            'Total Cost': f"{total_cost:.2f}"
        })
    
    df_table = pd.DataFrame(table_data)
    
    # 保存表格
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    seed_suffix = get_seed_suffix(seed_list)
    df_table.to_csv(output_path / f"summary_table{seed_suffix}.csv", index=False)
    df_table.to_latex(output_path / f"summary_table{seed_suffix}.tex", index=False, float_format="%.3f")
    
    print(f"📋 保存汇总表格: {output_path / f'summary_table{seed_suffix}.csv'}")
    print(df_table.to_string(index=False))

def plot_distribution_heatmaps(data, output_dir, args, seed_list=None):
    """为能耗、时延、隐私等指标绘制客户端分布热力图"""
    if 'convergence' not in data or 'raw' not in data['convergence']:
        print("⚠️  没有原始数据 (raw_data.csv)，跳过热力图绘制")
        return

    raw_data = data['convergence']['raw']

    # 筛选数据：如果数据包含update列，则只用最后的数据
    final_data = raw_data
    if 'update' in raw_data.columns:
        # 修复：对每个算法，只取其各自的最大更新次数的数据
        # 这可以处理不同算法运行不同更新次数的情况
        final_data = raw_data.groupby('algorithm', group_keys=False).apply(
            lambda df: df[df['update'] == df['update'].max()]
        )
    
    # 选取算法进行绘图（包含学习 + 非学习）
    present_algos = final_data['algorithm'].unique()
    # 使用所有在配置中且出现在数据里的算法（学习 + 非学习）
    all_available_algos = _ordered_algorithms(pd.Series(present_algos))

    if args.algorithms:
        # 如果用户指定了算法列表，则使用用户指定的
        algos_to_plot = [alg for alg in all_available_algos if alg in args.algorithms]
        print(f"✅ 根据用户指定，将绘制以下算法: {[ALGORITHM_CONFIG.get(a, {}).get('label', a) for a in algos_to_plot]}")
    else:
        # 否则，使用所有可用的算法
        algos_to_plot = all_available_algos

    if not algos_to_plot:
        print("⚠️  没有可用于绘制热力图的算法数据")
        return
        
    metrics_to_plot = [
        # (聚合列名, 原始列名, 颜色条标签, Seaborn颜色映射)
        ('avg_energy_j', 'energy_j', 'Energy Consumption (J)', 'YlGnBu'),
        ('avg_delay_s', 'delay_s', 'Average Delay (s)', 'YlOrRd'),
        ('avg_privacy_cost', 'privacy_cost', 'Privacy Cost', 'PuBuGn'),
        ('avg_service_hit', 'service_hit', 'Success Rate', 'viridis')
    ]
    
    for agg_col, raw_col, cbar_label, cmap in metrics_to_plot:
        # 智能选择列名
        metric_col = None
        if agg_col in final_data.columns:
            metric_col = agg_col
        elif raw_col in final_data.columns:
            metric_col = raw_col
        else:
            print(f"⚠️  数据中缺少 '{agg_col}' 或 '{raw_col}' 列，跳过绘制 {cbar_label} 热力图")
            continue
            
        # --- 核心修正：数据处理逻辑 ---
        # 1. 获取当前算法下所有客户端的完整列表
        all_clients_df = final_data[['algorithm', 'client_id']].drop_duplicates()

        if 'Success Rate' in cbar_label:
            # 2a. 对于成功率，在所有任务上计算平均值
            client_metrics = final_data.groupby(['algorithm', 'client_id'])[metric_col].mean().reset_index()
        else:
            # 2b. 对于其他指标，仅在成功任务上计算平均值
            successful_hits = final_data[final_data['service_hit'] == 1]
            metrics_on_success = successful_hits.groupby(['algorithm', 'client_id'])[metric_col].mean().reset_index()

            # 3. 使用左连接，确保所有客户端都被包含。没有成功任务的客户端，其指标为 NaN
            client_metrics = pd.merge(all_clients_df, metrics_on_success, on=['algorithm', 'client_id'], how='left')

        # --- Print Statistics ---
        print(f"\n--- {cbar_label} Statistics per Algorithm ---")
        for algo in algos_to_plot:
            # 从修正后的 client_metrics 中取数据
            algo_costs = client_metrics[client_metrics['algorithm'] == algo][metric_col]
            if not algo_costs.dropna().empty:
                print(f"  - {algo:<20}: Min={algo_costs.min():.2f}, Mean={algo_costs.mean():.2f}, Max={algo_costs.max():.2f}")
            else:
                print(f"  - {algo:<20}: No successful tasks.")
        print("---------------------------------------------------\n")

        # 确定绘图网格布局（自适应至最多3x3；超过则近似方阵）
        num_algos = len(algos_to_plot)
        if num_algos <= 3:
            fig, axes = plt.subplots(1, num_algos, figsize=(8 * num_algos, 7), squeeze=False)
            axes = axes.flatten()
        elif num_algos <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
        elif num_algos <= 6:
            fig, axes = plt.subplots(2, 3, figsize=(22, 12))
            axes = axes.flatten()
        elif num_algos <= 8:
            fig, axes = plt.subplots(2, 4, figsize=(28, 12))
            axes = axes.flatten()
        elif num_algos <= 9:
            fig, axes = plt.subplots(3, 3, figsize=(22, 16))
            axes = axes.flatten()
        else:
            rows = int(np.ceil(np.sqrt(num_algos)))
            cols = int(np.ceil(num_algos / rows))
            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)
            axes = axes.flatten()

        # --- Robust Color Scaling ---
        # 使用分位数来确定颜色范围，避免极端值影响整体的可视化效果
        all_values = client_metrics[client_metrics['algorithm'].isin(algos_to_plot)][metric_col]

        # 对于非负指标，最小值可以设为0
        if "delay" in cbar_label or "Energy" in cbar_label or "Cost" in cbar_label:
            vmin = 0
        elif "Success Rate" in cbar_label:
            vmin = 0
            vmax = 1
        else:
            # 对于可能为负的指标（如奖励），使用5%分位数
            vmin = all_values.quantile(0.05)

        # 使用95%分位数作为上限，使得大部分数据的颜色区分更明显
        if "Success Rate" not in cbar_label:
            vmax = all_values.quantile(0.95)

        # 确保 vmin < vmax，并在数据完全相同时提供回退
        if vmin >= vmax:
            vmin = all_values.min()
            vmax = all_values.max()
            if vmin >= vmax:
                vmax = vmin + 1.0
        
        if "Success Rate" in cbar_label:
            print(f"   - Fixed color range for '{cbar_label}': [{vmin:.2f}, {vmax:.2f}]")
        else:
            print(f"   - Robust color range for '{cbar_label}': [{vmin:.2f}, {vmax:.2f}] (vmin=0, vmax=95th percentile)")

        for i, alg in enumerate(algos_to_plot):
            ax = axes[i]
            alg_data = client_metrics[client_metrics['algorithm'] == alg]
            
            # --- 动态网格大小：使用 final_data 确保网格能容纳所有客户端 ---
            num_clients = final_data['client_id'].max() + 1
            grid_size = int(np.ceil(np.sqrt(num_clients)))
            
            heatmap_data = np.full((grid_size, grid_size), np.nan)
            annot_grid = np.full((grid_size, grid_size), '', dtype=object)
            
            for _, row in alg_data.iterrows():
                client_id = int(row['client_id'])
                if client_id >= 0:
                    row_idx, col_idx = divmod(client_id, grid_size)
                    # 核心：无论指标是否为NaN，都填入ID
                    annot_grid[row_idx, col_idx] = str(client_id)
                    # 指标值（可能为NaN）用于颜色渲染
                    heatmap_data[row_idx, col_idx] = row[metric_col]

            # --- 核心修改：手动绘制注释，以根据背景色动态调整文本颜色 ---
            sns.heatmap(heatmap_data, ax=ax, annot=False, cmap=cmap, 
                        cbar_kws={'shrink': 0.8, 'pad': 0.035}, vmin=vmin, vmax=vmax,
                        linewidths=.5, linecolor='gray', square=True)

            # 依赖 seaborn 默认范围，避免手动设置导致边缘裁切

            # 设置颜色条刻度字号
            if ax.collections and hasattr(ax.collections[0], 'colorbar') and ax.collections[0].colorbar:
                ax.collections[0].colorbar.ax.tick_params(labelsize=10)

            # 获取归一化函数和颜色映射
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap_obj = plt.get_cmap(cmap)

            # 遍历单元格以添加带颜色的文本
            for i in range(heatmap_data.shape[0]):
                for j in range(heatmap_data.shape[1]):
                    # 只为有ID的格子添加文本
                    if annot_grid[i, j] != '':
                        val = heatmap_data[i, j]
                        
                        # 如果值为NaN（例如，无成功任务的客户端），背景是默认色，用黑色文本
                        if pd.isna(val):
                            text_color = 'black'
                        else:
                            # 根据背景亮度决定文本颜色
                            rgba = cmap_obj(norm(val))
                            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                            text_color = 'white' if luminance < 0.5 else 'black'
                        
                        ax.text(j + 0.5, i + 0.59, annot_grid[i, j],
                                ha='center', va='center', color=text_color, size=16)

            ax.set_title(ALGORITHM_CONFIG.get(alg, {}).get('label', alg), fontsize=18)
            ax.set_xticks([])
            ax.set_yticks([])

        # 隐藏多余的子图
        for i in range(num_algos, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # 增加子图间距（对当前图生效）
        fig.subplots_adjust(hspace=0.11, wspace=-0.34)
        
        # 保存图表
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dpi_save = int(getattr(args, 'dpi_save', 300) or 300)
        seed_suffix = get_seed_suffix(seed_list)
        filename_base = f"heatmap_distribution_{metric_col.replace('_', '').replace('j', '').replace('s', '')}"
        
        plt.savefig(output_path / f"{filename_base}{seed_suffix}.png", dpi=dpi_save, bbox_inches='tight')
        plt.savefig(output_path / f"{filename_base}{seed_suffix}.pdf", bbox_inches='tight')
        
        if getattr(args, 'show', False):
            plt.show()
        else:
            plt.close(fig)
        
        print(f"📊 保存 {cbar_label} 分布热力图: {output_path / f'{filename_base}{seed_suffix}.png'}")


def plot_unified_cost_heatmap(data, output_dir, args, seed_list=None):
    """为统一的用户代价指标绘制客户端分布热力图"""
    if 'convergence' not in data or 'raw' not in data['convergence']:
        print("⚠️  没有原始数据，跳过统一代价热力图绘制")
        return

    raw_data = data['convergence']['raw']
    tau = args.latency_threshold

    # 筛选数据：如果数据包含update列，则只用最后的数据
    final_data = raw_data
    if 'update' in raw_data.columns:
        # 修复：对每个算法，只取其各自的最大更新次数的数据
        final_data = raw_data.groupby('algorithm', group_keys=False).apply(
            lambda df: df[df['update'] == df['update'].max()]
        )

    # 智能选择列名
    def get_col_name(df, agg_name, raw_name):
        if agg_name in df.columns: return agg_name
        if raw_name in df.columns: return raw_name
        return None

    energy_col = get_col_name(final_data, 'avg_energy_j', 'energy_j')
    delay_col = get_col_name(final_data, 'avg_delay_s', 'delay_s')
    privacy_col = get_col_name(final_data, 'avg_privacy_cost', 'privacy_cost')

    if not all([energy_col, delay_col, privacy_col, 'service_hit' in final_data.columns]):
        print("⚠️  数据中缺少能耗、时延、隐私成本或命中率列，无法计算统一代价，跳过绘制。")
        return

    # --- 核心修改: 分别计算成本和成功率 ---
    # 1. 成本指标仅在成功任务上计算
    successful_data = final_data[final_data['service_hit'] == 1]
    cost_metrics = successful_data.groupby(['algorithm', 'client_id'])[
        [energy_col, delay_col, privacy_col]
    ].mean()

    # 2. 成功率在所有任务上计算
    success_rate_metrics = final_data.groupby(['algorithm', 'client_id'])['service_hit'].mean().rename('success_rate')

    # 3. 合并成本与成功率
    # --- 核心修改: 使用 'outer' join 保证所有 client-algorithm 对都被包含
    client_metrics = pd.merge(cost_metrics, success_rate_metrics, on=['algorithm', 'client_id'], how='outer').reset_index()

    # --- 计算最终统一代价 (除以成功率以施加惩罚) ---
    # 为避免除以零，将成功率为0的情况替换为一个极小值
    epsilon = 1e-9
    client_metrics['success_rate'] = client_metrics['success_rate'].replace(0, epsilon)

    base_cost = (
        client_metrics[energy_col] +
        client_metrics[privacy_col] +
        (client_metrics[delay_col] - tau).clip(lower=0)
    )
    client_metrics['unified_cost'] = base_cost / client_metrics['success_rate']

    # --- 核心修改: 对于没有成功任务的客户端(unified_cost为NaN)，将其成本设为无穷大
    client_metrics['unified_cost'] = client_metrics['unified_cost'].fillna(np.inf)


    # 选取算法并准备绘图（包含学习 + 非学习）
    present_algos = client_metrics['algorithm'].unique()
    all_available_algos = _ordered_algorithms(pd.Series(present_algos))

    if args.algorithms:
        # 如果用户指定了算法列表，则使用用户指定的
        algos_to_plot = [alg for alg in all_available_algos if alg in args.algorithms]
        print(f"✅ 根据用户指定，将为以下算法绘制统一代价图: {[ALGORITHM_CONFIG.get(a, {}).get('label', a) for a in algos_to_plot]}")
    else:
        # 否则，使用所有可用的算法
        algos_to_plot = all_available_algos
    
    if not algos_to_plot:
        print("⚠️  没有可用于绘制统一代价热力图的算法数据")
        return

    # 绘图网格布局
    num_algos = len(algos_to_plot)
    if num_algos <= 3:
        fig, axes = plt.subplots(1, num_algos, figsize=(8 * num_algos, 7), squeeze=False)
        axes = axes.flatten()
    elif num_algos <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
    elif num_algos <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        axes = axes.flatten()
    elif num_algos <= 8:
        fig, axes = plt.subplots(2, 4, figsize=(28, 12))
        axes = axes.flatten()
    elif num_algos <= 9:
        fig, axes = plt.subplots(3, 3, figsize=(22, 16))
        axes = axes.flatten()
    else:
        rows = int(np.ceil(np.sqrt(num_algos)))
        cols = int(np.ceil(num_algos / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)
        axes = axes.flatten()

    cbar_label = f'Unified Cost\n(Energy+Privacy+DelayPenalty) / SuccessRate'
    # 顶部大标题已移除

    # --- 核心修正：聚焦缩放策略 ---
    # 仅根据当前绘制的算法来确定颜色范围，以增强它们之间的可比性
    values_for_scaling = client_metrics[client_metrics['algorithm'].isin(algos_to_plot)]['unified_cost']

    # 构造用于相对对比的透视表（client_id × algorithm）
    cost_pivot = client_metrics.pivot_table(index='client_id', columns='algorithm', values='unified_cost', aggfunc='mean')
    baseline_alg = getattr(args, 'baseline', None)

    # 根据模式准备绘图所需的每算法-每客户端数值
    mode = getattr(args, 'unified_mode', 'absolute')
    # 预定义每种模式的颜色条标签与色图
    if mode == 'delta':
        plot_label = 'Δ Cost vs Baseline'
        cmap_name = 'RdBu_r'
        eps = 1e-9
        # 计算所有要绘制算法相对基线的差值
        if baseline_alg not in cost_pivot.columns:
            print(f"⚠️ 基线算法 {baseline_alg} 不在数据中，回退为 absolute 模式")
            mode = 'absolute'
        else:
            baseline_series = cost_pivot[baseline_alg]
            # 收集用于缩放的全部差值
            all_deltas = []
            for alg in algos_to_plot:
                if alg in cost_pivot.columns:
                    deltas = cost_pivot[alg] - baseline_series
                    all_deltas.append(deltas)
            if all_deltas:
                values_for_scaling = pd.concat(all_deltas, axis=0).dropna()
    elif mode == 'ratio':
        plot_label = 'Relative Change vs Baseline (%)'
        cmap_name = 'RdBu_r'
        eps = 1e-9
        if baseline_alg not in cost_pivot.columns:
            print(f"⚠️ 基线算法 {baseline_alg} 不在数据中，回退为 absolute 模式")
            mode = 'absolute'
        else:
            baseline_series = cost_pivot[baseline_alg].replace(0, eps)
            all_ratios = []
            for alg in algos_to_plot:
                if alg in cost_pivot.columns:
                    ratios = (cost_pivot[alg] - baseline_series) / baseline_series * 100.0
                    all_ratios.append(ratios)
            if all_ratios:
                values_for_scaling = pd.concat(all_ratios, axis=0).dropna()
    elif mode == 'rank':
        plot_label = 'Rank (lower is better)'
        cmap_name = 'YlGnBu'
        # 预计算每客户端的排名（按代价升序）
        ranks_pivot = cost_pivot.rank(axis=1, method='min', ascending=True)
        # 缩放范围为[1, 算法数]
        values_for_scaling = ranks_pivot.stack()
    else:
        plot_label = cbar_label
        cmap_name = 'YlGnBu'

    # 打印所有算法的统计数据，以供全面了解
    print("\n--- Unified Cost Statistics per Algorithm (Successful Tasks Only) ---")
    for algo in algos_to_plot:
        algo_costs = client_metrics[client_metrics['algorithm'] == algo]['unified_cost']
        if not algo_costs.empty:
            print(f"  - {algo:<20}: Min={algo_costs.min():.2f}, Mean={algo_costs.mean():.2f}, Max={algo_costs.max():.2f}")
    print("-------------------------------------------------------------------\n")

    # --- 不同模式的颜色缩放策略 ---
    if mode in ['delta', 'ratio']:
        # 以0为中心的对称发散色带，使用95%分位的绝对值作为上限
        if values_for_scaling.empty:
            vlim = 1.0
        else:
            q = values_for_scaling.quantile(0.95)
            if mode == 'delta':
                vlim = max(abs(values_for_scaling.min()), abs(q))
            else:  # ratio
                vlim = max(abs(values_for_scaling.min()), abs(q))
            if vlim <= 0:
                vlim = values_for_scaling.abs().max()
            if vlim <= 0:
                vlim = 1.0
        vmin, vmax = -vlim, vlim
        power_norm = None  # 线性
        print(f"   - 使用发散色带放大小差异: [{vmin:.2f}, {vmax:.2f}] (center=0)")
    elif mode == 'rank':
        vmin, vmax = 1, max(2, len(algos_to_plot))
        power_norm = None
        print(f"   - 使用离散排名色带: [1, {vmax}] (1为最佳)")
    else:
        # absolute 模式：鲁棒聚焦缩放（线性）
        vmin = values_for_scaling.min()
        vmax = values_for_scaling.quantile(0.95)
        if vmin >= vmax: # Fallback
            vmax = values_for_scaling.max()
            if vmin >= vmax:
                vmax = vmin + 1.0
        power_norm = PowerNorm(gamma=1.0, vmin=vmin, vmax=vmax)
        print(f"   - 颜色范围已通过鲁棒聚焦缩放进行优化: [{vmin:.2f}, {vmax:.2f}] with gamma=1.0")

    # 遍历每个算法并绘制其热力图
    for i, algo in enumerate(algos_to_plot):
        ax = axes[i]
        alg_data = client_metrics[client_metrics['algorithm'] == algo]
        
        num_clients = alg_data['client_id'].max() + 1
        grid_size = int(np.ceil(np.sqrt(num_clients)))
        
        heatmap_data = np.full((grid_size, grid_size), np.nan)
        annot_grid = np.full((grid_size, grid_size), '', dtype=object)

        # 根据模式取值
        for _, row in alg_data.iterrows():
            client_id = int(row['client_id'])
            row_idx, col_idx = divmod(client_id, grid_size)
            if mode == 'delta' and baseline_alg in cost_pivot.columns:
                base_val = cost_pivot.loc[client_id, baseline_alg]
                val = row['unified_cost'] - base_val
                heatmap_data[row_idx, col_idx] = val
                annot_grid[row_idx, col_idx] = str(client_id)
            elif mode == 'ratio' and baseline_alg in cost_pivot.columns:
                base_val = cost_pivot.loc[client_id, baseline_alg]
                base_val = base_val if base_val != 0 else 1e-9
                val = (row['unified_cost'] - base_val) / base_val * 100.0
                heatmap_data[row_idx, col_idx] = val
                annot_grid[row_idx, col_idx] = str(client_id)
            elif mode == 'rank':
                if algo in cost_pivot.columns and client_id in cost_pivot.index:
                    val = int(ranks_pivot.loc[client_id, algo])
                    heatmap_data[row_idx, col_idx] = val
                    annot_grid[row_idx, col_idx] = str(client_id)
            else:
                heatmap_data[row_idx, col_idx] = row['unified_cost']
                annot_grid[row_idx, col_idx] = str(client_id)

        # 使用单次绘制（颜色条每子图一个），注释稍后手动添加

        # --- 核心修改：手动绘制注释，以根据背景色动态调整文本颜色 ---
        # 1. 确定归一化函数
        norm_obj = power_norm
        if norm_obj is None:
            norm_obj = plt.Normalize(vmin=vmin, vmax=vmax)

        # 2. 绘制不带注释的热力图
        sns.heatmap(heatmap_data, ax=ax, annot=False, cmap=cmap_name,
                         cbar_kws={'shrink': 0.8, 'pad': 0.035}, norm=norm_obj if power_norm else None,
                         vmin=vmin if not power_norm else None, vmax=vmax if not power_norm else None,
                         linewidths=.3, linecolor='gray', square=True)
        
        # 若为 rank 模式，固定颜色条刻度为 1..N，避免出现 -1/0 等异常刻度
        if mode == 'rank' and ax.collections:
            try:
                cbar = ax.collections[0].colorbar
                if cbar is not None:
                    vmax_local = max(2, len(algos_to_plot))
                    ticks = np.arange(1, vmax_local + 1, 1)
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels([str(t) for t in ticks])
            except Exception:
                pass

        # 依赖 seaborn 默认范围，避免手动设置导致边缘裁切

        # 设置颜色条刻度字号
        if ax.collections and hasattr(ax.collections[0], 'colorbar') and ax.collections[0].colorbar:
            ax.collections[0].colorbar.ax.tick_params(labelsize=16)
        
        # 3. 获取颜色映射并手动添加注释
        cmap_obj = plt.get_cmap(cmap_name)
        for r in range(heatmap_data.shape[0]):
            for c in range(heatmap_data.shape[1]):
                if annot_grid[r, c] != '':
                    val = heatmap_data[r, c]
                    
                    if np.isinf(val):
                        # 任务全失败的客户端，其背景色为最深色，使用白色文本
                        text_color = 'white'
                    else:
                        # 根据背景亮度决定文本颜色
                        rgba = cmap_obj(norm_obj(val))
                        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        text_color = 'white' if luminance < 0.5 else 'black'

                    ax.text(c + 0.5, r + 0.59, annot_grid[r, c],
                            ha='center', va='center', color=text_color, size=16)

        ax.set_title(ALGORITHM_CONFIG.get(algo, {}).get('label', algo), fontsize=18)
        ax.set_xticks([]); ax.set_yticks([])

    for i in range(num_algos, len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # 增加子图间距（对当前图生效）
    fig.subplots_adjust(hspace=0.11, wspace=-0.34)
    
    # 保存图表
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dpi_save = int(getattr(args, 'dpi_save', 600) or 600)
    seed_suffix = get_seed_suffix(seed_list)
    filename_base = "heatmap_distribution_unified_cost"
    if mode in ['delta', 'ratio', 'rank']:
        filename_base += f"_{mode}"
        if mode in ['delta', 'ratio'] and baseline_alg:
            filename_base += f"_vs_{baseline_alg}"
    
    plt.savefig(output_path / f"{filename_base}{seed_suffix}.png", dpi=dpi_save, bbox_inches='tight')
    plt.savefig(output_path / f"{filename_base}{seed_suffix}.pdf", bbox_inches='tight')
    
    if getattr(args, 'show', False):
        plt.show()
    else:
        plt.close(fig)
    
    print(f"📊 保存统一代价分布热力图: {output_path / f'{filename_base}{seed_suffix}.png'}")

    # 若开启单图导出，则将每个子图单独保存为PNG(600dpi)
    if getattr(args, 'save_individual', False):
        indiv_dir = output_path / 'individual'
        indiv_dir.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        from matplotlib.transforms import Bbox
        for i, alg in enumerate(algos_to_plot):
            ax_i = axes[i]
            bb = ax_i.get_tightbbox(renderer)
            cbar = None
            try:
                cbar = ax_i.collections[0].colorbar
            except Exception:
                cbar = None
            if cbar is not None and hasattr(cbar, 'ax'):
                bb = Bbox.union([bb, cbar.ax.get_tightbbox(renderer)])
            bb_inches = bb.transformed(fig.dpi_scale_trans.inverted())
            fname = f"{filename_base}_{alg}{seed_suffix}.png".replace(' ', '_')
            fig.savefig(indiv_dir / fname, dpi=600, bbox_inches=bb_inches)
            print(f"🖼️ 已导出单图: {indiv_dir / fname}")
 
    if getattr(args, 'show', False):
        plt.show()
    else:
        plt.close(fig)
    
    print(f"📊 保存统一代价分布热力图: {output_path / f'{filename_base}{seed_suffix}.png'}")

def main():
    parser = argparse.ArgumentParser(description="生成收敛性分析图表")
    parser.add_argument(
        '--data-dir',
        default='analysis',
        help='聚合数据目录 (默认: analysis)'
    )
    parser.add_argument(
        '--base-exp-dir',
        type=str,
        default=None,
        help='指定包含所有实验结果的根目录，脚本将自动聚合数据并绘图。'
    )
    parser.add_argument(
        '--single-run-dir',
        type=str,
        default=None,
        help='指定单个实验运行的目录以生成热力图，此选项优先于 --data-dir'
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help='种子列表，逗号或空格分隔，如: "0,1,2" 或 "0 1 2"'
    )
    parser.add_argument(
        '--sub-dir',
        type=str,
        default=None,
        help='子目录名称，用于指定种子组合，如: "seeds_0_1_2"'
    )
    parser.add_argument(
        '--figures',
        nargs='+',
        choices=['curves', 'curves_4_metrics', 'comparison', 'metrics', 'metrics_curves', 'table', 'heatmaps', 'unified_cost', 'all'],
        default=['all'],
        help='要生成的图表类型'
    )
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=None,
        help='要绘制的算法列表，空格分隔，例如: h_mappo_l ippo。如果未提供，则绘制所有找到的算法。'
    )
    parser.add_argument(
        '--smooth', type=int, default=0,
        help='收敛曲线滑动平均窗口(步数)，0表示不平滑'
    )
    parser.add_argument(
        '--ci', action='store_true',
        help='误差带使用95%%置信区间(近似)，默认使用标准差'
    )
    parser.add_argument(
        '--tail-n', dest='tail_n', type=int, default=50,
        help='最终对比使用的尾段更新数，默认50'
    )
    parser.add_argument(
        '--latency-threshold', type=float, default=None,
        help='时延阈值；若不提供则从配置读取，默认3.0'
    )
    parser.add_argument(
        '--colorblind', action='store_true',
        help='使用色盲友好调色板'
    )
    parser.add_argument(
        '--max-points', type=int, default=0,
        help='收敛曲线可视化最大点数(下采样)，0表示不下采样'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='显示绘图窗口（默认不显示，仅保存）'
    )
    parser.add_argument(
        '--paper', action='store_true',
        help='启用论文风格的字号与线宽'
    )
    parser.add_argument(
        '--legend-cols', type=int, default=1,
        help='图例列数（默认1）'
    )
    parser.add_argument(
        '--dpi-save', type=int, default=300,
        help='导出PNG分辨率DPI（默认300）'
    )
    # 统一代价对比增强选项
    parser.add_argument(
        '--unified-mode', choices=['absolute', 'delta', 'ratio', 'rank'], default='absolute',
        help='统一代价热力图的数值模式：absolute(绝对值)、delta(与基线差值)、ratio(相对基线百分比)、rank(名次)'
    )
    parser.add_argument(
        '--baseline', type=str, default='h_mappo_l',
        help='delta/ratio/rank 模式下使用的基线算法（默认: h_mappo_l）'
    )
    parser.add_argument(
        '--save-individual', action='store_true',
        help='同时导出每个子图为单独的PNG(600dpi)，保存在输出目录的 individual 子文件夹下'
    )
    
    args = parser.parse_args()

    # 确定时延阈值 tau 的来源
    if args.latency_threshold is None:
        # 从配置文件读取，失败则用默认值
        args.latency_threshold = _load_latency_threshold(3.0)
    
    print(f"🔧 使用时延阈值 (Latency Threshold τ̄): {args.latency_threshold:.2f} s")

    # 如果提供了种子列表但没有指定子目录，自动构造子目录名称
    seed_list = parse_seed_list(args.seeds)
    if seed_list and not args.sub_dir:
        args.sub_dir = f"seeds_{'_'.join(map(str, seed_list))}"
    elif seed_list and args.sub_dir:
        print(f"⚠️  同时指定了种子列表和子目录，将使用子目录: {args.sub_dir}")
    elif args.seeds and not seed_list:
        print(f"❌ 无效的种子列表格式: {args.seeds}")
        return

    # 配色（可选色盲友好）
    if args.colorblind:
        palette = sns.color_palette('colorblind', n_colors=len(ALGORITHM_CONFIG))
        for (alg, cfg), col in zip(ALGORITHM_CONFIG.items(), palette):
            cfg['color'] = col
    
    data = {}
    output_dir = None
    seed_list = parse_seed_list(args.seeds)

    # --- 新增：聚合与绘图一体化模式 ---
    if args.base_exp_dir:
        base_path = Path(args.base_exp_dir)
        print(f"🔥 进入聚合与绘图模式，根目录: {base_path}")
        aggregated_df = aggregate_per_client_data(base_path, args.algorithms)
        if aggregated_df is not None:
            data = {'convergence': {'raw': aggregated_df}}
            # --- 核心修改：根据算法组合动态确定输出子目录 ---
            if args.algorithms:
                # 根据用户指定的算法列表生成目录名
                # 对算法名称进行排序，确保 'a vs b' 和 'b vs a' 的目录名一致
                sorted_algs = sorted(args.algorithms)
                algo_suffix = "_vs_".join(sorted_algs)
            else:
                # 如果绘制所有算法，则使用 'all_algorithms'
                algo_suffix = "all_algorithms"
            output_dir = Path('analysis/analysis_convergence/heatmap') / algo_suffix
        else:
            print("❌ 自动聚合数据失败，无法继续。")
            return

    # --- 模式二：处理单次运行目录 ---
    elif args.single_run_dir:
        single_run_path = Path(args.single_run_dir)
        print(f"📊 从单次运行目录加载数据: {single_run_path}")
        per_client_file = single_run_path / "per_client_metrics.csv"
        
        if not per_client_file.exists():
            print(f"❌ 在 {single_run_path} 中未找到 per_client_metrics.csv")
            return
            
        try:
            df = pd.read_csv(per_client_file)
            # 从目录名中提取算法
            alg_name = "unknown"
            for known_alg in ALGORITHM_CONFIG.keys():
                if f"__{known_alg}__" in single_run_path.name or single_run_path.name.startswith(f"{known_alg}__"):
                     alg_name = known_alg
                     break
            df['algorithm'] = alg_name
            print(f"   - 自动识别算法为: {alg_name}")
            
            data = {'convergence': {'raw': df}}
            output_dir = single_run_path / 'figures'
        except Exception as e:
            print(f"❌ 加载 {per_client_file} 失败: {e}")
            return
            
    # --- 模式三：处理预先聚合好的数据目录 ---
    else:
        # 加载数据
        data_path = args.data_dir
        if args.sub_dir:
            data_path = f"{args.data_dir}/{args.sub_dir}"
            print(f"📊 从子目录加载数据: {data_path}")
        else:
            print(f"📊 从目录加载数据: {data_path}")

        if seed_list:
            print(f"   种子组合: {seed_list}")

        data = load_convergence_data(args.data_dir, args.sub_dir)
        
        # 确定输出目录
        data_path_obj = Path(args.data_dir)
        if args.sub_dir:
            data_path_obj = data_path_obj / args.sub_dir
        output_dir = data_path_obj / 'figures'

    if not data:
        print("❌ 没有可用的收敛性数据")
        return

    # 只有在非 base_exp_dir 模式下才打印，因为 base_exp_dir 模式的 output_dir 已在前面确定
    if not args.base_exp_dir:
        print(f"📁 图表将保存在: {output_dir}")
    else:
        # 对于 base_exp_dir 模式，明确打印新的动态目录
        print(f"📁 图表将保存在: {output_dir}")
    
    # 生成图表
    if 'all' in args.figures or 'curves' in args.figures:
        plot_convergence_curves(data, output_dir, args, seed_list)
    
    if 'all' in args.figures or 'curves_4_metrics' in args.figures:
        plot_convergence_curves_four_metrics(data, output_dir, args, seed_list)

    if 'all' in args.figures or 'comparison' in args.figures:
        plot_final_comparison(data, output_dir, args, seed_list)
    
    if 'all' in args.figures or 'metrics' in args.figures:
        plot_metrics_comparison(data, output_dir, args, seed_list)

    if 'all' in args.figures or 'metrics_curves' in args.figures:
        plot_metrics_convergence_curves(data, output_dir, args, seed_list)

    if 'all' in args.figures or 'table' in args.figures:
        generate_summary_table(data, output_dir, args, seed_list)
    
    if 'all' in args.figures or 'heatmaps' in args.figures:
        plot_distribution_heatmaps(data, output_dir, args, seed_list)
        
    if 'all' in args.figures or 'unified_cost' in args.figures:
        plot_unified_cost_heatmap(data, output_dir, args, seed_list)
    
    print(f"🎉 收敛性图表生成完成! 保存在 {output_dir}")

if __name__ == "__main__":
    main()


