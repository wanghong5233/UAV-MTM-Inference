#!/usr/bin/env python3
"""
权重对 h_mappo_l 算法收敛性影响的分析脚本。

该脚本旨在：
1.  从 `experiment_result_sensitivity/energy_privacy_weights/` 目录加载数据。
2.  仅针对 'h_mappo_l' 算法，聚合所有种子的收敛数据。
3.  绘制 1x2 的收敛曲线图，对比不同能量/隐私权重比下的能耗和隐私成本。
4.  使用 sensitivity 分析中的 'vibrant' 调色板以保持风格一致。
5.  将图表保存在 `analysis/` 目录下。
6.  支持通过命令行参数过滤显示的权重。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings('ignore')

# --- Matplotlib 全局样式设置 (参考 sensitivity 脚本) ---

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置高质量绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.figsize'] = (18, 7)  # 1x2 布局

# 论文级配色与背景
plt.rcParams['axes.facecolor'] = '#f2f3f5'
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#c9c9c9'
plt.rcParams['grid.alpha'] = 0.35

# --- 字体尺寸统一放大 ---
BASE_FONT_SIZE = 20
plt.rcParams.update({
    'font.size': BASE_FONT_SIZE,            # 默认字体大小
    'axes.titlesize': BASE_FONT_SIZE + 2,   # 子图标题大小
    'axes.labelsize': BASE_FONT_SIZE,       # x、y轴标签大小
    'xtick.labelsize': BASE_FONT_SIZE - 2,  # x轴刻度标签大小
    'ytick.labelsize': BASE_FONT_SIZE - 2,  # y轴刻度标签大小
    'legend.fontsize': BASE_FONT_SIZE - 4,  # 图例字体大小
    'figure.titlesize': BASE_FONT_SIZE + 4, # Figure总标题大小 (suptitle)
})

# --- 配色方案 (参考 sensitivity 脚本) ---

# 使用 vibrant 调色板（高对比度，无纯黑，与 sensitivity 脚本统一）
VIBRANT_PALETTE = [
    '#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e',
    '#8c564b', '#17becf', '#7f7f7f', '#bcbd22'
]


def load_data_by_weight(data_dir: Path):
    """
    加载不同权重下 h_mappo_l 算法的所有种子数据。

    Args:
        data_dir (Path): 包含各个权重子目录的根目录。

    Returns:
        dict: 一个字典，键是权重值（字符串），值是该权重下所有种子数据的 DataFrame 列表。
    """
    print(f"🔍 开始从 {data_dir} 扫描数据...")
    all_data = {}
    if not data_dir.exists():
        print(f"❌ 错误：数据目录不存在: {data_dir}")
        return all_data

    for weight_dir in data_dir.iterdir():
        if not weight_dir.is_dir():
            continue

        try:
            # 验证目录名是否为浮点数
            float(weight_dir.name)
        except ValueError:
            print(f"  - 跳过非权重目录: {weight_dir.name}")
            continue

        algo_dir = weight_dir / 'h_mappo_l'
        if not algo_dir.exists() or not algo_dir.is_dir():
            continue

        seed_dfs = []
        for seed_dir in algo_dir.glob('seed_*'):
            metrics_file = seed_dir / 'metrics.csv'
            if metrics_file.exists():
                try:
                    df = pd.read_csv(metrics_file)
                    seed_dfs.append(df)
                except Exception as e:
                    print(f"    ⚠️ 加载失败: {metrics_file} - {e}")

        if seed_dfs:
            all_data[weight_dir.name] = seed_dfs
            print(f"  ✅ 读取权重 {weight_dir.name}: {len(seed_dfs)} 个种子")
    
    if not all_data:
        print("❌ 未找到任何有效的 h_mappo_l 算法数据。请检查目录结构。")

    return all_data


def process_and_summarize_data(raw_data: dict):
    """
    对每个权重的数据进行处理，计算均值和标准差。

    Args:
        raw_data (dict): load_data_by_weight 返回的原始数据。

    Returns:
        dict: 一个字典，键是权重值，值是包含均值和标准差的汇总 DataFrame。
    """
    summary_data = {}
    print("\n🔄 正在处理和聚合数据...")
    for weight, seed_dfs in raw_data.items():
        if not seed_dfs:
            continue

        # 使用 seed 编号作为 key 来合并所有 DataFrame
        combined_df = pd.concat(seed_dfs, keys=range(len(seed_dfs)), names=['seed_id', 'original_index'])

        # 按 'update' 步骤分组，并计算跨种子的均值和标准差
        grouped = combined_df.groupby('update')
        mean_df = grouped.mean()
        std_df = grouped.std()
        
        # 为了避免列名冲突，添加后缀
        summary = mean_df.join(std_df, lsuffix='_mean', rsuffix='_std')
        summary_data[weight] = summary.reset_index()
        print(f"  - 聚合权重 {weight} 的数据")
        
    return summary_data


def _moving_average(x: pd.Series, window: int) -> pd.Series:
    """计算滑动平均值"""
    if window is None or window <= 1:
        return x
    return x.rolling(window=window, min_periods=1, center=False).mean()


def plot_convergence_curves(summary_data: dict, output_dir: Path, args):
    """
    绘制能耗和隐私的收敛性曲线图。

    Args:
        summary_data (dict): process_and_summarize_data 返回的汇总数据。
        output_dir (Path): 图表输出目录。
        args: argparse 解析后的参数。
    """
    if not summary_data:
        print("⚠️ 汇总数据为空，跳过绘图。")
        return
    
    DEFAULT_PRIVACY_WEIGHT = 5.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax1, ax2 = axes

    # 待绘制的指标配置 (ax, metric_base_name, ylabel)
    metrics_to_plot = [
        (ax1, 'avg_energy_j', 'Energy (J)'),
        (ax2, 'avg_privacy_cost', 'Privacy Cost'),
    ]

    # 按权重值排序，确保图例和颜色分配一致
    sorted_weights = sorted(summary_data.keys(), key=float)

    # 根据参数过滤要显示的权重
    if args.use_filter:
        if not args.weights_to_show:
            print("⚠️ --use-filter 已启用但 --weights-to-show 为空，将不显示任何曲线。")
            sorted_weights = []
        else:
            # 将命令行传入的权重转换为字符串进行匹配
            weights_to_show_set = set(map(str, args.weights_to_show))
            original_count = len(sorted_weights)
            sorted_weights = [w for w in sorted_weights if w in weights_to_show_set]
            print(f"⚖️ 已启用权重过滤: 从 {original_count} 个权重中筛选出 {len(sorted_weights)} 个 -> {sorted_weights}")
            if not sorted_weights:
                print("⚠️ 过滤后没有匹配到任何权重，请检查 --weights-to-show 参数。")
                return

    colors = VIBRANT_PALETTE

    print("\n🎨 开始绘制收敛曲线...")

    for ax, metric_base_name, ylabel in metrics_to_plot:
        mean_col = f"{metric_base_name}_mean"
        std_col = f"{metric_base_name}_std"
        
        for i, weight in enumerate(sorted_weights):
            weight_data = summary_data[weight]
            config_color = colors[i % len(colors)]
            
            # 计算权重比用于图例显示
            try:
                energy_weight = float(weight)
                ratio = energy_weight / DEFAULT_PRIVACY_WEIGHT
                # 格式化标签以获得更清晰的显示
                if ratio == int(ratio):
                    label_text = f'Ratio = {int(ratio)}'
                else:
                    label_text = f'Ratio = {ratio:.2f}'
            except ValueError:
                label_text = f'Weight = {weight}' # Fallback

            mean_vals = weight_data[mean_col]
            std_vals = weight_data[std_col]
            
            # 平滑处理
            mean_vals = _moving_average(mean_vals, args.smooth)
            std_vals = _moving_average(std_vals, args.smooth)
            
            upd = weight_data['update']
            
            # 绘图
            ax.plot(upd, mean_vals, label=label_text, color=config_color, linewidth=2.0)
            ax.fill_between(upd, (mean_vals - std_vals).fillna(0), (mean_vals + std_vals).fillna(0),
                            alpha=0.15, color=config_color)

        ax.set_xlabel('Training Iteration', fontsize=plt.rcParams['axes.labelsize'], labelpad=10)
        ax.set_ylabel(ylabel, fontsize=plt.rcParams['axes.labelsize'])
        ax.grid(True, which='both', alpha=0.35)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', which='major', labelsize=plt.rcParams['xtick.labelsize'])

    # 创建统一的图例
    handles, labels = ax1.get_legend_handles_labels()
    # 将图例移至图表下方居中
    fig.legend(handles, labels, bbox_to_anchor=(0.51, 0.12), loc='upper center', ncol=len(sorted_weights), fontsize=plt.rcParams['legend.fontsize'], frameon=True, fancybox=True, shadow=True)
    
    # 对于学术论文图，通常在正文caption中添加标题，而不是在图本身中
    # plt.suptitle('Impact of Energy/Privacy Weight Ratio on Convergence of HC-MAPPO-L', y=1.01)
    
    # 调整布局，为下方的图例和标题留出空间
    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) 

    # 保存图表
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path_pdf = output_dir / "h_mappo_l_energy_privacy_convergence_by_weight.pdf"
    save_path_png = output_dir / "h_mappo_l_energy_privacy_convergence_by_weight.png"
    
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.savefig(save_path_png, bbox_inches='tight')
    
    if args.show:
        plt.show()
    else:
        plt.close()

    print(f"📊 图表已保存至: {save_path_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="为 h_mappo_l 算法绘制不同能量/隐私权重比下的能耗与隐私收敛曲线。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 假设此脚本最终位于 'scripts' 目录下，项目根目录是其父目录
    project_root = Path(__file__).parent.parent.resolve()

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=project_root / 'experiment_result_sensitivity' / 'energy_privacy_weights',
        help='包含权重子目录的数据根目录。'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=project_root / 'analysis' / 'figures_convergence_by_weight',
        help='图表输出目录。'
    )
    parser.add_argument(
        '--smooth',
        type=int,
        default=10,
        help='收敛曲线的滑动平均窗口大小，0表示不平滑。'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='显示绘图窗口而不是仅保存文件。'
    )
    parser.add_argument(
        '--use-filter',
        action='store_true',
        help='启用权重过滤器，仅绘制 --weights-to-show 中指定的权重。'
    )
    parser.add_argument(
        '--weights-to-show',
        nargs='+',
        type=str,
        default=['1.0', '3.0', '4.0', '10.0', '25.0'],
        help='当 --use-filter 启用时，要显示的特定权重列表 (例如: 1.0 3.0 5.0)。'
    )
    args = parser.parse_args()

    # 执行流程
    raw_data = load_data_by_weight(args.data_dir)
    if raw_data:
        summary_data = process_and_summarize_data(raw_data)
        plot_convergence_curves(summary_data, args.output_dir, args)
        print("\n🎉 分析完成!")


if __name__ == "__main__":
    main()
