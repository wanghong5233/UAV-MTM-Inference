"""
数据加载模块
"""

from .nyuv2 import NYUv2Dataset, create_nyuv2_dataloader

__all__ = ['NYUv2Dataset', 'create_nyuv2_dataloader']
