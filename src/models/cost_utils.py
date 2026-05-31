"""
Utility helpers for split-point compute profiling.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out:
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    return None


def _kernel_area(kernel_size) -> int:
    if isinstance(kernel_size, tuple):
        return int(kernel_size[0]) * int(kernel_size[1])
    return int(kernel_size) * int(kernel_size)


def _leaf_module_flops(module: nn.Module, inp, out) -> float:
    out_tensor = _to_tensor(out)
    if out_tensor is None:
        return 0.0

    if isinstance(module, nn.Conv2d):
        batch, out_channels, out_h, out_w = out_tensor.shape
        kernel_mul = _kernel_area(module.kernel_size) * (module.in_channels // module.groups)
        bias_ops = 1 if module.bias is not None else 0
        return float(batch * out_channels * out_h * out_w * (2 * kernel_mul + bias_ops))

    if isinstance(module, nn.Linear):
        batch = out_tensor.shape[0] if out_tensor.ndim > 1 else 1
        out_features = out_tensor.shape[-1]
        bias_ops = 1 if module.bias is not None else 0
        return float(batch * out_features * (2 * module.in_features + bias_ops))

    if isinstance(module, nn.BatchNorm2d):
        # Inference-time affine normalization: subtract, scale, and shift.
        return float(4 * out_tensor.numel())

    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
        return float(out_tensor.numel())

    if isinstance(module, nn.Sigmoid):
        return float(4 * out_tensor.numel())

    if isinstance(module, nn.MaxPool2d):
        comps = max(_kernel_area(module.kernel_size) - 1, 1)
        return float(out_tensor.numel() * comps)

    if isinstance(module, nn.MaxUnpool2d):
        return float(out_tensor.numel())

    if isinstance(module, (nn.Identity, nn.Dropout, nn.Dropout2d)):
        return 0.0

    return 0.0


def profile_module(module: nn.Module, *inputs):
    """Run a module once and estimate the executed FLOPs."""
    total_flops = 0.0
    handles = []

    def hook_fn(mod, mod_inp, mod_out):
        nonlocal total_flops
        total_flops += _leaf_module_flops(mod, mod_inp, mod_out)

    for submodule in module.modules():
        if submodule is module:
            if len(list(submodule.children())) == 0:
                handles.append(submodule.register_forward_hook(hook_fn))
            continue
        if len(list(submodule.children())) == 0:
            handles.append(submodule.register_forward_hook(hook_fn))

    with torch.no_grad():
        output = module(*inputs)

    for handle in handles:
        handle.remove()

    return output, float(total_flops)


def elementwise_flops(tensor: torch.Tensor, ops_per_element: float = 1.0) -> float:
    return float(tensor.numel()) * float(ops_per_element)


def weighted_sum(tensors: Sequence[torch.Tensor], weights: Sequence[torch.Tensor | float]):
    """Compute a weighted sum and estimate scalar-mul plus add costs."""
    if len(tensors) != len(weights):
        raise ValueError("tensors and weights must have the same length.")
    if not tensors:
        raise ValueError("weighted_sum requires at least one tensor.")

    out = None
    flops = 0.0
    for idx, (tensor, weight) in enumerate(zip(tensors, weights)):
        cur = tensor * weight
        flops += elementwise_flops(tensor, 1.0)
        if idx == 0:
            out = cur
        else:
            out = out + cur
            flops += elementwise_flops(tensor, 1.0)
    return out, float(flops)


def bilinear_interpolate(x: torch.Tensor, scale_factor=None, size=None, align_corners: bool = True):
    out = F.interpolate(x, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=align_corners)
    # Bilinear interpolation uses four neighboring samples and a small fixed amount of arithmetic.
    flops = elementwise_flops(out, 7.0)
    return out, float(flops)


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12):
    out = x / torch.norm(x, p=2, dim=dim, keepdim=True).clamp_min(eps)
    # Approximate square + sum + sqrt + divide.
    flops = elementwise_flops(x, 5.0)
    return out, float(flops)
