"""
精度建模模块（核心贡献）

提供量化失真与任务精度关系的建模与拟合：
- 量化/反量化
- 失真计算
- 精度拟合
- 代理模型
"""

from .quantization import uniform_quantize, uniform_dequantize
from .distortion import compute_distortion
from .fit_accuracy import AccuracyFitter
from .fit_surrogate import SurrogateFitter
from .hooks import FeatureHook

__all__ = [
    'uniform_quantize',
    'uniform_dequantize',
    'compute_distortion',
    'AccuracyFitter',
    'SurrogateFitter',
    'FeatureHook',
]

