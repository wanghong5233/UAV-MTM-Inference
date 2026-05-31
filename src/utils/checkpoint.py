"""
Checkpoint管理模块

保存和加载模型权重。
"""

import torch
from pathlib import Path
from typing import Dict, Any


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = None,
    metrics: Dict = None,
    path: str = 'checkpoint.pth'
):
    """
    保存checkpoint
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        metrics: 评估指标
        path: 保存路径
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # 确保目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, path)
    print(f"[INFO] Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    加载checkpoint
    
    Args:
        path: checkpoint路径
        model: 模型
        optimizer: 优化器
        device: 设备
    
    Returns:
        checkpoint字典
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"[INFO] Checkpoint loaded from {path}")
    
    return checkpoint

