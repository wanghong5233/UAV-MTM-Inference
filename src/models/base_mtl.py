"""
Abstract base class for multi-task models.

Defines unified interfaces for multi-task learning models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class BaseMTLModel(ABC, nn.Module):
    """
    Abstract base class for multi-task learning models.
    
    Core interfaces:
    - forward: Forward propagation
    - get_split_points: Get available split points
    - partition: Model partitioning
    - get_feature_size: Get intermediate feature dimensions
    """
    
    def __init__(self, config: Dict):
        """
        Initialize MTL model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.model_name = config.get('name', 'unknown')
        self.num_tasks = config.get('architecture', {}).get('num_tasks', 3)
        self.task_names = config.get('architecture', {}).get('task_names', [])
        
    @abstractmethod
    def forward(self, x: torch.Tensor, split_point: int = -1) -> Dict[str, torch.Tensor]:
        """
        Forward propagation.
        
        Args:
            x: Input tensor [B, C, H, W]
            split_point: Split point index (-1 for full inference)
        
        Returns:
            Output dictionary:
            - If split_point=-1: {'task1': output1, 'task2': output2, ...}
            - If split_point>=0: {'features': intermediate_features}
        """
        pass
    
    @abstractmethod
    def get_split_points(self) -> List[str]:
        """
        Get all available split points.
        
        Returns:
            List of split point names (e.g., ['layer1.1', 'layer2.1', ...])
        """
        pass
    
    @abstractmethod
    def partition(self, split_point: int) -> Tuple[nn.Module, nn.Module]:
        """
        Partition the model into two parts.
        
        Args:
            split_point: Split point index
        
        Returns:
            (head_model, tail_model)
            - head_model: Part before the split point
            - tail_model: Part after the split point
        """
        pass
    
    @abstractmethod
    def get_feature_size(self, split_point: int) -> Tuple[int, ...]:
        """
        Get feature dimensions at specified split point.
        
        Args:
            split_point: Split point index
        
        Returns:
            Feature size tuple (C, H, W)
        """
        pass
    
    def get_num_parameters(self) -> int:
        """
        Get total number of model parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_flops(self, input_size: Tuple[int, ...]) -> float:
        """
        Estimate model FLOPs.
        
        Args:
            input_size: Input size (B, C, H, W)
        
        Returns:
            Estimated FLOPs
        """
        # TODO: Implement FLOPs computation
        # Can use thop or ptflops library
        return 0.0

    def get_split_flops(self, input_resolution: Tuple[int, int, int]) -> Dict[str, float]:
        """
        Estimate the incremental FLOPs associated with each split point.

        Args:
            input_resolution: Input resolution (C, H, W)

        Returns:
            Mapping from split-point name to incremental FLOPs.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement split-point FLOPs profiling."
        )

