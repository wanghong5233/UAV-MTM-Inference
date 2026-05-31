"""
MTL模型定义模块

支持三种多任务学习模型：
- Split Network (Hard Parameter Sharing)
- MTAN (Multi-Task Attention Network)
- Cross-Stitch Networks
"""

from .base_mtl import BaseMTLModel
from .split_network import SplitNetwork
from .segnet_split import SplitSegNet
from .segnet_dense import DenseSegNet
from .segnet_cross import CrossStitchSegNet
from .mtan import MTAN
from .crossstitch import CrossStitchNetwork

__all__ = [
    'BaseMTLModel',
    'SplitNetwork',
    'SplitSegNet',
    'DenseSegNet',
    'CrossStitchSegNet',
    'MTAN',
    'CrossStitchNetwork',
]

