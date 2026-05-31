#!/usr/bin/env python3
"""
为敏感性分析中的单个种子实验绘制收敛曲线并显示。

此脚本接收一个指向 `metrics.csv` 文件的路径作为参数，
然后生成一个包含四个核心指标（用户奖励、延迟、能耗、隐私成本）
的收敛曲线图，并直接在窗口中显示该图表，而不是保存到文件。

这对于快速、逐一检查不同参数下每个独立实验的收敛情况非常有用。
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 切换到更稳定的TkAgg后端，以改善显示效果和避免变形
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm

# --- 配置 ---

# 动态选择中文字体；若系统缺失中文字体则回退到英文标签
CANDIDATE_CN_FONTS = [
    'Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'DengXian', 'Arial Unicode MS'
]
AVAILABLE_FONTS = {f.name for f in fm.fontManager.ttflist}
CHOSEN_CN_FONT = next((f for f in CANDIDATE_CN_FONTS if f in AVAILABLE_FONTS), None)

if CHOSEN_CN_FONT:
    plt.rcParams['font.sans-serif'] = [CHOSEN_CN_FONT, 'DejaVu Sans']
    USE_ENGLISH_LABELS = False
else:
    # 无可用中文字体，启用英文标签以避免“口口口”
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    USE_ENGLISH_LABELS = True

# 强制使用英文标签/标题（按用户要求）
USE_ENGLISH_LABELS = True

plt.rcParams['axes.unicode_minus'] = False

# 屏幕显示使用较合适的DPI，避免窗口巨大导致的挤压与变形
plt.rcParams['figure.dpi'] = 120
# 保存DPI不重要（本脚本不保存），保持默认即可

# 设置高质量绘图风格
sns.set_style("whitegrid")

# 仅用于检查算法类别（不强制限制绘图）
LEARNING_ALGOS = {'h_mappo_l', 'mappo_no_constraint', 'ippo', 'hc_ippo_l', 'lru_avg'}


def _fit_and_center_window(fig, max_ratio: float = 0.90, y_offset_pct: float = 0.05):
    """
    缩放画布以适配屏幕，并将窗口置于屏幕中央再上移一点避免底部任务栏遮挡。
    - 计算窗口装饰(标题栏/边框/工具栏)的额外占用，确保画布不被裁剪
    - 保持图像纵横比，不变形
    - max_ratio: 窗口尺寸不超过屏幕该比例
    - y_offset_pct: 在居中的基础上向上偏移的屏幕高度百分比
    """
    try:
        manager = plt.get_current_fig_manager()
        win = manager.window  # Tk 窗口
        fig.canvas.draw()  # 先渲染一次，尺寸才可用
        win.update_idletasks()

        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()

        # 当前窗口与画布尺寸
        canvas = fig.canvas.get_tk_widget()
        cur_win_w = max(win.winfo_width(), win.winfo_reqwidth())
        cur_win_h = max(win.winfo_height(), win.winfo_reqheight())
        cur_canvas_w = max(canvas.winfo_width(), canvas.winfo_reqwidth())
        cur_canvas_h = max(canvas.winfo_height(), canvas.winfo_reqheight())

        # 窗口装饰额外占用尺寸（标题栏/边框/工具栏）
        extra_w = max(0, cur_win_w - cur_canvas_w)
        extra_h = max(0, cur_win_h - cur_canvas_h)

        # 允许的最大窗口/画布尺寸
        allow_win_w = int(screen_w * max_ratio)
        allow_win_h = int(screen_h * max_ratio)
        allow_canvas_w = max(100, allow_win_w - extra_w)
        allow_canvas_h = max(100, allow_win_h - extra_h)

        # 以像素为单位获取当前图大小
        dpi = fig.dpi
        fw_in, fh_in = fig.get_size_inches()
        fig_w_px = int(fw_in * dpi)
        fig_h_px = int(fh_in * dpi)

        # 若图像超出可用画布范围，则按比例缩放
        scale = min(1.0, allow_canvas_w / fig_w_px, allow_canvas_h / fig_h_px)
        if scale < 1.0:
            fig.set_size_inches(fw_in * scale, fh_in * scale, forward=True)
            fig.canvas.draw()
            win.update_idletasks()
            fig_w_px = int(fig.get_size_inches()[0] * dpi)
            fig_h_px = int(fig.get_size_inches()[1] * dpi)

        # 最终窗口尺寸 = 图像像素 + 装饰
        final_w = min(allow_win_w, fig_w_px + extra_w)
        final_h = min(allow_win_h, fig_h_px + extra_h)

        # 居中并上移一定像素（按屏幕高度百分比）
        x = (screen_w - final_w) // 2
        y_center = (screen_h - final_h) // 2
        y_offset = int(screen_h * max(0.0, min(0.2, y_offset_pct)))
        y = max(0, y_center - y_offset)
        win.geometry(f"{final_w}x{final_h}+{x}+{y}")
    except Exception:
        # 非 TkAgg 或访问失败时静默忽略
        pass


def _schedule_post_show_fit(fig, max_ratio: float = 0.90, y_offset_pct: float = 0.05):
    """在窗口创建后(工具栏/装饰已就绪)再次自适应缩放与居中上移，避免闪一下后被裁剪。"""
    try:
        manager = plt.get_current_fig_manager()
        win = manager.window
        win.after(120, lambda: _fit_and_center_window(fig, max_ratio, y_offset_pct))
        win.after(420, lambda: _fit_and_center_window(fig, max_ratio, y_offset_pct))
    except Exception:
        pass

# --- 核心功能 ---

def moving_average(series: pd.Series, window: int) -> pd.Series:
    """计算滑动平均值以平滑曲线。"""
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()

def plot_single_run_convergence(csv_path: Path, smooth_window: int):
    """
    为单个实验（一个 metrics.csv 文件）绘制并显示收敛曲线图。
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"  - ⚠️  跳过空文件: {csv_path}")
            return
    except Exception as e:
        print(f"  - ❌ 读取文件失败: {csv_path} ({e})")
        return

    # 创建 2x2 子图，使用 constrained_layout 并调整尺寸以获得更好的屏幕显示比例
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ((ax1, ax2), (ax3, ax4)) = axes
    
    # 从文件路径中提取信息用于标题
    try:
        parts = csv_path.parts
        seed = parts[-2]
        algorithm = parts[-3]
        param_value = parts[-4]
        param_name = parts[-5]
        # 英文标题
        title = (f"Alg: {algorithm} ({seed})\n"
                 f"Param: {param_name} = {param_value}")
    except IndexError:
        title = f"Convergence\n{csv_path.parent.name}"

    fig.suptitle(title, fontsize=14, fontweight='bold')

    updates = df.index

    # --- 绘制各个子图 ---
    # 1. 用户奖励 (User Reward)
    if 'avg_user_reward' in df.columns:
        reward = moving_average(df['avg_user_reward'], smooth_window)
        ax1.plot(updates, reward, color='#1f77b4', linewidth=2)
        ax1.set_title('User Reward', fontsize=12)
        ax1.set_ylabel('User Reward')
    else:
        ax1.text(0.5, 0.5, 'No user reward data', ha='center', va='center', transform=ax1.transAxes)

    # 2. 平均延迟 (Average Delay)
    if 'avg_delay_s' in df.columns:
        delay = moving_average(df['avg_delay_s'], smooth_window)
        ax2.plot(updates, delay, color='#ff7f0e', linewidth=2)
        ax2.set_title('Average Delay', fontsize=12)
        ax2.set_ylabel('Delay (s)')
    else:
        ax2.text(0.5, 0.5, 'No delay data', ha='center', va='center', transform=ax2.transAxes)

    # 3. 能耗 (Energy Consumption)
    if 'avg_energy_j' in df.columns:
        energy = moving_average(df['avg_energy_j'], smooth_window)
        ax3.plot(updates, energy, color='#2ca02c', linewidth=2)
        ax3.set_title('Energy Consumption', fontsize=12)
        ax3.set_ylabel('Energy (J)')
    else:
        ax3.text(0.5, 0.5, 'No energy data', ha='center', va='center', transform=ax3.transAxes)

    # 4. 隐私成本 (Privacy Cost)
    if 'avg_privacy_cost' in df.columns:
        privacy = moving_average(df['avg_privacy_cost'], smooth_window)
        ax4.plot(updates, privacy, color='#d62728', linewidth=2)
        ax4.set_title('Privacy Cost', fontsize=12)
        ax4.set_ylabel('Privacy Cost')
    else:
        ax4.text(0.5, 0.5, 'No privacy data', ha='center', va='center', transform=ax4.transAxes)

    # --- 统一设置和显示 ---
    for ax in axes.flatten():
        ax.set_xlabel('Training Update')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=9)

    # 预适配 + 显示后再次适配并上移，避免出现“先正常后被裁剪”的闪变
    _fit_and_center_window(fig, max_ratio=0.90, y_offset_pct=0.06)
    _schedule_post_show_fit(fig, max_ratio=0.90, y_offset_pct=0.06)

    print("  -> 正在打开绘图窗口...")
    plt.show()
    plt.close(fig)


def main():
    """
    主函数，解析命令行参数并启动绘图流程。
    """
    parser = argparse.ArgumentParser(
        description="为敏感性分析中的单个种子实验绘制收敛曲线并直接显示。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="使用示例:\npython %(prog)s experiment_result_sensitivity/num_users/20/h_mappo_l/seed_0/metrics.csv --smooth 20"
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='要绘制收敛曲线的 metrics.csv 文件路径。'
    )
    parser.add_argument(
        '--smooth',
        type=int,
        default=10,
        help='滑动平均窗口大小，用于平滑曲线。\n设置为1则不进行平滑。(默认: 10)'
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.is_file():
        print(f"❌ 错误: 文件 '{csv_path}' 不存在或不是一个文件。")
        return

    # 检查算法是否为学习型算法（仅提示，不阻止绘图）
    try:
        algorithm_name = csv_path.parts[-3]
        if algorithm_name not in LEARNING_ALGOS:
            print(f"⚠️  提示: 算法 '{algorithm_name}' 不是学习型算法，但仍将为您绘图。")
    except IndexError:
        pass

    # 显示当前字体信息
    print(f"ℹ️  Using font: {plt.rcParams['font.sans-serif'][0]} (labels set to English)")

    print(f"📊 正在为文件 '{csv_path}' 生成收敛曲线图...")
    plot_single_run_convergence(csv_path, args.smooth)
    print("\n🎉 图表窗口已关闭。")


if __name__ == '__main__':
    main()
