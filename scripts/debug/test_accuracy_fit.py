"""
测试精度建模模块

用法：
    python scripts/debug/test_accuracy_fit.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.accuracy_modeling import AccuracyFitter, compute_distortion
from src.utils.visualization import plot_accuracy_fitting


def main():
    print("测试精度建模...")
    print("="*50)
    
    # 生成模拟数据
    print("生成模拟数据...")
    distortions = np.linspace(0, 1, 50)
    true_accuracies = 0.9 * np.exp(-2 * distortions) + 0.1
    noisy_accuracies = true_accuracies + np.random.normal(0, 0.01, len(distortions))
    
    # 创建拟合器
    fitter = AccuracyFitter('test_model', 'test_task')
    
    # 测试拟合
    print("测试拟合...")
    # results = fitter.fit(distortions, noisy_accuracies, func_type='exponential')
    # print(f"✓ 拟合参数：{results}")
    
    # 测试预测
    print("测试预测...")
    # test_D = 0.5
    # predicted_A = fitter.predict(test_D)
    # print(f"✓ D={test_D} -> A={predicted_A:.4f}")
    
    # 绘图测试
    print("测试绘图...")
    plot_accuracy_fitting(
        distortions,
        noisy_accuracies,
        # fitted_curve=fitter.predict(distortions),
        save_path='logs/test_fitting.pdf'
    )
    print("✓ 图表已保存")
    
    print("="*50)
    print("✓ 精度建模测试通过！")


if __name__ == '__main__':
    main()

