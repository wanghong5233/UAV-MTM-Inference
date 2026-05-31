"""
失真计算模块

计算量化前后特征的失真度（NMSE）。
"""

import torch
import torch.nn.functional as F


def compute_distortion(
    original_features: torch.Tensor,
    quantized_features: torch.Tensor,
    metric: str = 'nmse'
) -> float:
    """
    计算失真度
    
    Args:
        original_features: 原始特征
        quantized_features: 量化后的特征
        metric: 失真度量（nmse, mse, psnr）
    
    Returns:
        distortion: 失真值
    """
    if metric == 'nmse':
        return compute_nmse(original_features, quantized_features)
    elif metric == 'mse':
        return compute_mse(original_features, quantized_features)
    elif metric == 'psnr':
        return compute_psnr(original_features, quantized_features)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_nmse(
    original: torch.Tensor,
    quantized: torch.Tensor
) -> float:
    """
    计算归一化均方误差（NMSE）
    
    NMSE = ||x - x_q||^2 / ||x||^2
    
    Args:
        original: 原始特征
        quantized: 量化特征
    
    Returns:
        nmse: NMSE值
    """
    # TODO: 实现NMSE计算
    # nmse = torch.norm(original - quantized) ** 2 / torch.norm(original) ** 2
    
    pass


def compute_mse(
    original: torch.Tensor,
    quantized: torch.Tensor
) -> float:
    """
    计算均方误差（MSE）
    
    Args:
        original: 原始特征
        quantized: 量化特征
    
    Returns:
        mse: MSE值
    """
    # TODO: 实现MSE计算
    
    pass


def compute_psnr(
    original: torch.Tensor,
    quantized: torch.Tensor
) -> float:
    """
    计算峰值信噪比（PSNR）
    
    Args:
        original: 原始特征
        quantized: 量化特征
    
    Returns:
        psnr: PSNR值 (dB)
    """
    # TODO: 实现PSNR计算
    
    pass

