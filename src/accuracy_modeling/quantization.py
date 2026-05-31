"""
量化/反量化模块

实现中间特征的均匀量化（论文建模中的 compression）。
"""

import torch
from typing import Tuple, Dict, Optional


def uniform_quantize(
    features: torch.Tensor,
    bit_width: int,
    per_channel: bool = False,
    quant_mode: str = "dynamic",
    calib_params: Optional[Dict] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    均匀量化（对应论文中的 r^(p) 量化级别）
    
    Args:
        features: 输入特征 [B, C, H, W]
        bit_width: 量化比特宽度（任意正整数，例如 32, 16, 8, 4）
        per_channel: 是否按通道量化（默认 False，即 per-tensor）
        quant_mode: 量化范围模式（dynamic=每次按特征自适应 min/max；fixed=使用离线校准的固定 min/max）
        calib_params: 固定量化参数（仅 quant_mode=fixed 时需要），包含 {'f_min': ..., 'f_max': ...}
    
    Returns:
        dequantized_features: 反量化后的特征（float32，带量化噪声）
        quant_params: 量化参数字典 {'scale': ..., 'f_min': ..., 'f_max': ..., 'bit_width': ...}
    """
    # 约定：32-bit 视为全精度（不量化），直接返回原特征
    if bit_width >= 32:
        return features, {
            'bit_width': bit_width,
            'skipped': True,
            'scale': 1.0,
            'f_min': 0.0,
        }
    if bit_width <= 0:
        raise ValueError(f"bit_width must be positive, got {bit_width}")

    # 计算量化范围
    n_levels = 2 ** bit_width
    qmin, qmax = 0, n_levels - 1
    
    # 计算 min/max（dynamic 或 fixed）
    if quant_mode not in {"dynamic", "fixed"}:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}")

    if quant_mode == "fixed":
        if calib_params is None or "f_min" not in calib_params or "f_max" not in calib_params:
            raise ValueError("Fixed quantization requires calib_params with keys: f_min, f_max")
        f_min = torch.as_tensor(calib_params["f_min"], device=features.device, dtype=features.dtype)
        f_max = torch.as_tensor(calib_params["f_max"], device=features.device, dtype=features.dtype)
    else:
        if per_channel:
            # 按通道计算 min/max [B, C, H, W] -> [1, C, 1, 1]
            f_min = features.amin(dim=(0, 2, 3), keepdim=True)
            f_max = features.amax(dim=(0, 2, 3), keepdim=True)
        else:
            # 全局 min/max
            f_min = features.min()
            f_max = features.max()
    
    # 计算 scale（量化步长）
    scale = (f_max - f_min) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)  # 避免除零
    
    # 量化：x → q（整数）
    # 标准公式：q = round((x - f_min) / scale)
    quantized = torch.clamp(
        torch.round((features - f_min) / scale),
        qmin,
        qmax
    )
    
    # 反量化：q → x̃（浮点数，带精度损失）
    # 标准公式：x̃ = q * scale + f_min
    dequantized = quantized * scale + f_min
    
    quant_params = {
        'scale': scale,
        'f_min': f_min,
        'f_max': f_max,
        'bit_width': bit_width,
        'qmin': qmin,
        'qmax': qmax,
    }
    
    return dequantized, quant_params


def uniform_dequantize(
    quantized_features: torch.Tensor,
    quant_params: Dict
) -> torch.Tensor:
    """
    反量化（对应论文中接收端的解码）
    
    Args:
        quantized_features: 量化后的特征（离散整数值或其 float32 表示）
        quant_params: 量化参数（由 uniform_quantize 返回）
    
    Returns:
        dequantized_features: 反量化后的特征
    """
    scale = quant_params['scale']
    f_min = quant_params.get('f_min', 0.0)
    
    # 标准反量化：x̃ = q * scale + f_min
    dequantized = quantized_features * scale + f_min
    
    return dequantized


def apply_group_quantization(
    model: torch.nn.Module,
    features_dict: Dict[str, torch.Tensor],
    group_config: Dict[str, int],
    per_channel: bool = False,
    quant_mode: str = "dynamic",
    calib_stats: Optional[Dict[str, Dict]] = None
) -> Dict[str, torch.Tensor]:
    """
    对压缩组应用量化（用于敏感度标定）
    
    Args:
        model: MTL 模型
        features_dict: 中间特征字典 {'split_point_name': tensor, ...}
        group_config: 压缩组量化配置 {'group_name': bit_width, ...}
            例如：{'enc_stage_0': 4, 'dec_stage_3': 8, 'task1_att': 16, ...}
        per_channel: 是否按通道量化（默认 False，即 per-tensor）
        quant_mode: 量化范围模式（dynamic 或 fixed）
        calib_stats: 固定量化的校准统计（仅 quant_mode=fixed 时需要）
    
    Returns:
        quantized_features_dict: 量化后的特征字典
    
    示例：
        # 标定敏感度 S_g^(p)：只量化组 g 到 bit_width p
        group_config = {'enc_stage_0': 4}  # 只量化 enc_stage_0 到 4-bit
        quantized_feats = apply_group_quantization(model, feats, group_config)
        # 用量化后的特征跑完整推理，测精度 A^(g,p)
        # S_g^(p) = A_full - A^(g,p)
    """
    # 获取模型的压缩组定义
    if hasattr(model, 'get_compression_groups'):
        compression_groups = model.get_compression_groups()
    else:
        raise AttributeError("Model does not define compression groups.")
    
    quantized_dict = {}
    
    for split_name, feat in features_dict.items():
        # 检查该 split_point 是否属于需要量化的组
        quantize_this = False
        target_bit_width = 32  # 默认全精度（不量化）
        
        for group_name, bit_width in group_config.items():
            if group_name in compression_groups:
                if split_name in compression_groups[group_name]:
                    quantize_this = True
                    target_bit_width = bit_width
                    break
        
        # 32-bit 视为全精度：不量化；其余 bit-width 视为量化
        if quantize_this and target_bit_width < 32:
            # 量化该特征（quantize + dequantize）
            calib_params = None
            if quant_mode == "fixed":
                if calib_stats is None or split_name not in calib_stats:
                    raise ValueError(f"Missing calibration stats for split point: {split_name}")
                calib_params = calib_stats[split_name]
            quantized_feat, _ = uniform_quantize(
                feat,
                target_bit_width,
                per_channel,
                quant_mode=quant_mode,
                calib_params=calib_params
            )
            quantized_dict[split_name] = quantized_feat
        else:
            # 保持全精度
            quantized_dict[split_name] = feat
    
    return quantized_dict
