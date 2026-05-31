"""
PyTorch Hooks 模块

用于提取中间层特征。
"""

import torch
import torch.nn as nn
from typing import Dict, List


class FeatureHook:
    """
    特征提取Hook
    
    在模型的指定层注册Hook，提取中间特征。
    """
    
    def __init__(self):
        """初始化Hook"""
        self.features = {}
        self.handles = []
    
    def register_hooks(self, model: nn.Module, layer_names: List[str]):
        """
        注册Hook到指定层
        
        Args:
            model: PyTorch模型
            layer_names: 需要提取特征的层名称列表
        """
        # TODO: 实现Hook注册
        # 1. 遍历模型找到对应层
        # 2. 注册forward hook
        # 3. 保存handle用于后续删除
        
        pass
    
    def _forward_hook(self, layer_name: str):
        """
        创建前向Hook函数
        
        Args:
            layer_name: 层名称
        
        Returns:
            hook函数
        """
        def hook(module, input, output):
            self.features[layer_name] = output.detach()
        return hook
    
    def get_features(self, layer_name: str) -> torch.Tensor:
        """
        获取指定层的特征
        
        Args:
            layer_name: 层名称
        
        Returns:
            特征张量
        """
        if layer_name not in self.features:
            raise KeyError(f"Layer '{layer_name}' not found in features.")
        return self.features[layer_name]
    
    def get_all_features(self) -> Dict[str, torch.Tensor]:
        """
        获取所有提取的特征
        
        Returns:
            特征字典
        """
        return self.features
    
    def clear(self):
        """清空已提取的特征"""
        self.features.clear()
    
    def remove_hooks(self):
        """移除所有Hook"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.features.clear()

