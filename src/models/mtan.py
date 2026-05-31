"""
MTAN (Multi-Task Attention Network) 模型实现

基于 mtan-reference/im2im_pred/model_segnet_mtan.py，保持权重兼容性，
同时支持论文建模的切分、中间特征导出和量化。

关键扩展：
1. 支持 forward(..., split_point) 在指定切分点停止并返回中间特征
2. 定义压缩组 (compression groups) 对应论文中的 E^cmp
3. 实现 BaseMTLModel 接口以适配 UAV 仿真框架
"""

import copy
from typing import Dict, List, Tuple, Optional
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_mtl import BaseMTLModel
from .cost_utils import bilinear_interpolate, elementwise_flops, profile_module


class MTANSegNet(nn.Module):
    """
    MTAN SegNet 核心网络结构（与 mtan-reference 完全一致）
    
    这个类只负责网络定义和前向传播，不包含切分逻辑。
    切分逻辑由外层的 MTAN 类管理。
    """
    
    def __init__(self):
        super(MTANSegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(3):
            if j < 2:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        # expose attention-masked features as explicit modules for quantization hooks
        # one Identity per task (3) and stage (5), so each call is unique
        self.att_enc_out = nn.ModuleList(
            [nn.ModuleList([nn.Identity() for _ in range(5)]) for _ in range(3)]
        )
        self.att_dec_out = nn.ModuleList(
            [nn.ModuleList([nn.Identity() for _ in range(5)]) for _ in range(3)]
        )

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, x):
        """
        标准前向传播（与 mtan-reference 完全一致）
        
        Returns:
            predictions: [t1_pred, t2_pred, t3_pred]
            logsigma: 不确定性权重参数
        """
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][1] = self.att_enc_out[i][j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][1] = self.att_enc_out[i][j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                    atten_decoder[i][j][2] = self.att_dec_out[i][j](atten_decoder[i][j][2])
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                    atten_decoder[i][j][2] = self.att_dec_out[i][j](atten_decoder[i][j][2])

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma


class MTAN(BaseMTLModel):
    """
    MTAN 模型适配层（继承 BaseMTLModel）
    
    功能：
    1. 包装 MTANSegNet 核心网络
    2. 支持切分点导出（用于仿真量化敏感度标定）
    3. 支持加载 mtan-reference 预训练权重
    4. 定义压缩组（对应论文建模）
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 核心网络（与 mtan-reference 完全一致）
        self.core_net = MTANSegNet()
        
        # 任务名称映射
        self.task_names = config.get('architecture', {}).get('task_names', 
                                                             ['semantic_segmentation', 'depth_estimation', 'surface_normal'])
        
        # 定义切分点（用于论文建模的压缩组）
        # 按照论文：Shared Encoder/Decoder + Task-specific Attention modules
        self._define_split_points()
        
        # 定义压缩组（对应论文中的 E^cmp）
        self._define_compression_groups()

        input_resolution = config.get('architecture', {}).get('input_resolution', [3, 288, 288])
        self.feature_shapes = self._compute_feature_shapes(tuple(input_resolution))
    
    def _define_split_points(self):
        """
        定义切分点列表（对应论文中可量化的特征边）
        
        切分点命名规则：
        - shared_encoder_{i}: Shared Encoder 第 i 层输出
        - shared_decoder_{i}: Shared Decoder 第 i 层输出
        - task{k}_att_enc_{i}: Task k Attention Encoder 第 i 层输出
        - task{k}_att_dec_{i}: Task k Attention Decoder 第 i 层输出
        """
        self.split_points = []
        
        # Shared Encoder 切分点 (5层)
        for i in range(5):
            self.split_points.append(f'shared_encoder_{i}')
        
        # Shared Decoder 切分点 (5层)
        for i in range(5):
            self.split_points.append(f'shared_decoder_{i}')
        
        # Task-specific Attention 切分点 (3 tasks × 5 encoder + 5 decoder)
        for task_id in range(3):
            for i in range(5):
                self.split_points.append(f'task{task_id}_att_enc_{i}')
            for i in range(5):
                self.split_points.append(f'task{task_id}_att_dec_{i}')
    
    def _define_compression_groups(self):
        """
        定义压缩组（对应论文中的 E^cmp）
        
        论文建模：将相似深度/统计特性的边归为一组。
        这里采用“按网络阶段分组”的细粒度方案：
        - 每个 shared encoder / decoder 阶段各自单独成组（区分不同分辨率 / 语义深度）
        - 三个任务的 attention 分支按 encoder/decoder 分开成组（区分 task-private 语义与深度）
        
        这样一共得到 5（encoder）+5（decoder）+3×2（task-att enc/dec）=16 个压缩组，
        在显著增加压缩动作自由度的同时，仍保持可控的离线标定成本。
        """
        compression_groups: Dict[str, List[str]] = {}

        # 1) Shared encoder：按深度分成 5 组，对应 5 个分辨率/语义阶段
        for i in range(5):
            group_name = f'enc_stage_{i}'  # 例如 enc_stage_0, enc_stage_1, ...
            compression_groups[group_name] = [f'shared_encoder_{i}']

        # 2) Shared decoder：同理按深度分成 5 组
        for i in range(5):
            group_name = f'dec_stage_{i}'  # 例如 dec_stage_0, dec_stage_1, ...
            compression_groups[group_name] = [f'shared_decoder_{i}']

        # 3) 三个任务的 attention 分支：按任务与深度分组（encoder / decoder 分开）
        #    任务 0 / 1 / 2 分别对应：semantic / depth / normal
        for task_id in range(3):
            compression_groups[f'task{task_id}_att_enc'] = [f'task{task_id}_att_enc_{i}' for i in range(5)]
            compression_groups[f'task{task_id}_att_dec'] = [f'task{task_id}_att_dec_{i}' for i in range(5)]

        self.compression_groups = compression_groups
    
    def _get_module_for_split_point(self, split_name: str) -> nn.Module:
        """
        建立 split_point 名字到实际 nn.Module 的映射
        
        用于 Hook 机制提取/量化中间特征。
        
        Args:
            split_name: split_point 名称（如 'shared_encoder_0'）
        
        Returns:
            对应的 nn.Module（如 self.core_net.conv_block_enc[0]）
        
        Raises:
            ValueError: 如果 split_name 无法映射到 Module
        """
        # Shared Encoder: conv_block_enc[i] 的输出（pooling 之前）
        if split_name.startswith('shared_encoder_'):
            idx = int(split_name.split('_')[-1])
            if 0 <= idx < 5:
                return self.core_net.conv_block_enc[idx]
        
        # Shared Decoder: conv_block_dec[i] 的输出
        elif split_name.startswith('shared_decoder_'):
            idx = int(split_name.split('_')[-1])
            if 0 <= idx < 5:
                # 注意：在 core_net.forward() 中 decoder 是按 [-1, -2, ...] 的顺序调用，
                # 这里令 shared_decoder_0 表示“最深的第一个 decoder stage”（bottleneck 之后）。
                return self.core_net.conv_block_dec[-idx - 1]
        
        # Task Attention Encoder: attention-masked features after multiplication
        elif split_name.startswith('task') and '_att_enc_' in split_name:
            parts = split_name.split('_')
            task_id = int(parts[0].replace('task', ''))  # task0 -> 0
            layer_idx = int(parts[-1])  # task0_att_enc_3 -> 3
            if 0 <= task_id < 3 and 0 <= layer_idx < 5:
                return self.core_net.att_enc_out[task_id][layer_idx]
        
        # Task Attention Decoder: attention-masked features after multiplication
        elif split_name.startswith('task') and '_att_dec_' in split_name:
            parts = split_name.split('_')
            task_id = int(parts[0].replace('task', ''))
            layer_idx = int(parts[-1])
            if 0 <= task_id < 3 and 0 <= layer_idx < 5:
                return self.core_net.att_dec_out[task_id][layer_idx]
        
        raise ValueError(f"Cannot map split_point '{split_name}' to a nn.Module. "
                         f"Check _get_module_for_split_point() implementation.")
    
    def extract_features(self, x: torch.Tensor, split_points: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        使用 Hook 机制提取指定 split_point 的中间特征
        
        Args:
            x: 输入张量 [B, 3, H, W]
            split_points: 要提取的 split_point 列表（默认提取所有）
        
        Returns:
            features_dict: {split_name: feature_tensor, ...}
        
        示例：
            features = model.extract_features(x, ['shared_encoder_0', 'shared_encoder_1'])
            # features = {'shared_encoder_0': tensor([...]), 'shared_encoder_1': tensor([...])}
        """
        if split_points is None:
            split_points = self.split_points
        
        features = {}
        handles = []

        def make_hook(name: str):
            def hook(_m, _inp, out):
                features[name] = out.detach().clone()
            return hook

        for split_name in split_points:
            try:
                module = self._get_module_for_split_point(split_name)
            except ValueError:
                continue
            handles.append(module.register_forward_hook(make_hook(split_name)))
        
        # 执行一次 forward
        with torch.no_grad():
            _ = self.forward(x)
        
        # 移除所有 Hook
        for h in handles:
            h.remove()
        
        return features
    
    def apply_quantization_to_group(
        self, 
        group_name: str, 
        bit_width: int,
        per_channel: bool = False,
        quant_mode: str = "dynamic",
        calib_stats: Optional[Dict[str, Dict]] = None
    ) -> List:
        """
        对指定压缩组的所有层注册量化 Hook（用于敏感度标定）
        
        Args:
            group_name: 压缩组名称（如 'enc_stage_0'）
            bit_width: 量化比特宽度（如 32, 16, 8, 4；其中 bw >= 32 表示全精度不量化）
            per_channel: 是否按通道量化（默认 False，即 per-tensor）
            quant_mode: 量化范围模式（dynamic 或 fixed）
            calib_stats: 固定量化的校准统计（仅 quant_mode=fixed 时需要）
        
        Returns:
            hook_handles: Hook 句柄列表（用完后需要手动移除）
        
        使用示例：
            # 标定 enc_stage_0 在 4-bit 量化下的精度
            handles = model.apply_quantization_to_group('enc_stage_0', bit_width=4)
            acc_quantized = evaluate(model, dataloader)  # 跑推理测精度
            for h in handles:
                h.remove()  # 移除 Hook
            S_g_p = acc_full - acc_quantized  # 敏感度
        """
        from ..accuracy_modeling.quantization import uniform_quantize
        
        if group_name in self.compression_groups:
            split_points = self.compression_groups[group_name]
        else:
            # Backward-compatible alias: task{k}_att = task{k}_att_enc ∪ task{k}_att_dec
            m_alias = re.fullmatch(r"task([0-2])_att", group_name)
            if m_alias:
                k = int(m_alias.group(1))
                split_points = (
                    [f"task{k}_att_enc_{i}" for i in range(5)]
                    + [f"task{k}_att_dec_{i}" for i in range(5)]
                )
            else:
                raise ValueError(f"Unknown compression group: {group_name}")
        if quant_mode not in {"dynamic", "fixed"}:
            raise ValueError(f"Unsupported quant_mode: {quant_mode}")
        if quant_mode == "fixed":
            if calib_stats is None:
                raise ValueError("Fixed quantization requires calib_stats.")
            missing = [sp for sp in split_points if sp not in calib_stats]
            if missing:
                raise ValueError(f"Missing calibration stats for split points: {missing}")
        handles = []
        
        # 对该组的所有 split_point 注册量化 Hook
        for split_name in split_points:
            try:
                module = self._get_module_for_split_point(split_name)
                
                def make_quant_hook(bw, pc, split_key):
                    def quant_hook(m, inp, out):
                        # 32-bit 视为全精度：不量化
                        if bw >= 32:
                            return out

                        calib_params = None
                        if quant_mode == "fixed":
                            calib_params = calib_stats[split_key]
                        quantized, _ = uniform_quantize(
                            out,
                            bw,
                            pc,
                            quant_mode=quant_mode,
                            calib_params=calib_params
                        )
                        return quantized
                    return quant_hook

                handle = module.register_forward_hook(make_quant_hook(bit_width, per_channel, split_name))
                handles.append(handle)
            except ValueError:
                # 跳过无法映射的 split_point
                continue
        
        return handles
    
    def forward(self, x: torch.Tensor, split_point: int = -1) -> Dict[str, torch.Tensor]:
        """
        前向传播（支持切分点导出）
        
        Args:
            x: 输入张量 [B, 3, H, W]
            split_point: 切分点索引
                - split_point = -1: 完整推理，返回所有任务输出
                - split_point >= 0: 在指定切分点停止，返回中间特征
        
        Returns:
            - 如果 split_point = -1: {'seg': ..., 'depth': ..., 'normal': ...}
            - 如果 split_point >= 0: {'features': 中间特征字典, 'metadata': 辅助信息}
        """
        if split_point == -1:
            # 完整推理
            preds, logsigma = self.core_net(x)
            return {
                'seg': preds[0],      # [B, 13, H, W]
                'depth': preds[1],    # [B, 1, H, W]
                'normal': preds[2],   # [B, 3, H, W]
            }
        else:
            # 切分点推理（用于精度敏感度标定）
            # TODO: 实现分阶段前向传播，在指定 split_point 停止
            # 这里先返回占位符，完整实现需要重构 forward 为分阶段执行
            raise NotImplementedError(
                "Split-point inference will be implemented after validating full inference. "
                "For quantization sensitivity profiling, use the full forward pass with "
                "post-hoc feature extraction via hooks."
            )
    
    def get_split_points(self) -> List[str]:
        """获取所有切分点名称"""
        return self.split_points.copy()
    
    def get_compression_groups(self) -> Dict[str, List[str]]:
        """获取压缩组定义"""
        return self.compression_groups.copy()
    
    def partition(self, split_point: int) -> Tuple[nn.Module, nn.Module]:
        """
        模型分割（用于论文中的 segment partition）
        
        注意：MTAN 的 forward 逻辑复杂（多分支、多依赖），
        直接分割需要重构为分阶段执行。这里先返回 NotImplemented，
        实际使用时通过 forward(..., split_point) 导出中间特征。
        """
        raise NotImplementedError(
            "Direct partition is not supported for MTAN due to complex branching. "
            "Use forward(..., split_point) for feature extraction instead."
        )
    
    def get_feature_size(self, split_point: int) -> Tuple[int, ...]:
        """
        获取指定切分点的特征尺寸
        
        注意：需要运行一次 dummy forward 来推断尺寸
        """
        split_name = self._split_name(split_point)
        return self.feature_shapes[split_name]

    def get_split_flops(self, input_resolution: Tuple[int, int, int]) -> Dict[str, float]:
        c, h, w = input_resolution
        x = torch.zeros(1, c, h, w)
        flops: Dict[str, float] = {}

        with torch.no_grad():
            g_encoder = [[None, None] for _ in range(5)]
            g_decoder = [[None, None] for _ in range(5)]
            g_maxpool = [None] * 5
            g_upsampl = [None] * 5
            indices = [None] * 5
            pool_flops = [0.0] * 5

            for i in range(5):
                stage_flops = pool_flops[i - 1] if i > 0 else 0.0
                enc_input = x if i == 0 else g_maxpool[i - 1]
                g_encoder[i][0], extra = profile_module(self.core_net.encoder_block[i], enc_input)
                stage_flops += extra
                g_encoder[i][1], extra = profile_module(self.core_net.conv_block_enc[i], g_encoder[i][0])
                stage_flops += extra
                flops[f'shared_encoder_{i}'] = float(stage_flops)
                (g_maxpool[i], indices[i]), pool_flops[i] = profile_module(self.core_net.down_sampling, g_encoder[i][1])

            for i in range(5):
                stage_flops = pool_flops[-1] if i == 0 else 0.0
                dec_input = g_maxpool[-1] if i == 0 else g_decoder[i - 1][1]
                g_upsampl[i], extra = profile_module(self.core_net.up_sampling, dec_input, indices[-i - 1])
                stage_flops += extra
                g_decoder[i][0], extra = profile_module(self.core_net.decoder_block[-i - 1], g_upsampl[i])
                stage_flops += extra
                g_decoder[i][1], extra = profile_module(self.core_net.conv_block_dec[-i - 1], g_decoder[i][0])
                stage_flops += extra
                flops[f'shared_decoder_{i}'] = float(stage_flops)

            att_enc_out = [[None] * 5 for _ in range(3)]
            att_enc_next = [[None] * 5 for _ in range(3)]
            att_enc_next_flops = [[0.0] * 5 for _ in range(3)]
            for task_id in range(3):
                for i in range(5):
                    stage_flops = att_enc_next_flops[task_id][i - 1] if i > 0 else 0.0
                    if i == 0:
                        att_input = g_encoder[i][0]
                    else:
                        att_input = torch.cat((g_encoder[i][0], att_enc_next[task_id][i - 1]), dim=1)
                    att_mask, extra = profile_module(self.core_net.encoder_att[task_id][i], att_input)
                    stage_flops += extra
                    att_enc_out[task_id][i] = att_mask * g_encoder[i][1]
                    stage_flops += elementwise_flops(att_enc_out[task_id][i], 1.0)
                    flops[f'task{task_id}_att_enc_{i}'] = float(stage_flops)

                    enc_next, extra = profile_module(self.core_net.encoder_block_att[i], att_enc_out[task_id][i])
                    (att_enc_next[task_id][i], _), extra_pool = profile_module(self.core_net.down_sampling, enc_next)
                    att_enc_next_flops[task_id][i] = float(extra + extra_pool)

            att_dec_out = [[None] * 5 for _ in range(3)]
            for task_id in range(3):
                for i in range(5):
                    stage_flops = 0.0
                    dec_input = att_enc_next[task_id][-1] if i == 0 else att_dec_out[task_id][i - 1]
                    dec_interp, extra = bilinear_interpolate(dec_input, scale_factor=2, align_corners=True)
                    stage_flops += extra
                    dec_proj, extra = profile_module(self.core_net.decoder_block_att[-i - 1], dec_interp)
                    stage_flops += extra
                    att_input = torch.cat((g_upsampl[i], dec_proj), dim=1)
                    att_mask, extra = profile_module(self.core_net.decoder_att[task_id][-i - 1], att_input)
                    stage_flops += extra
                    att_dec_out[task_id][i] = att_mask * g_decoder[i][1]
                    stage_flops += elementwise_flops(att_dec_out[task_id][i], 1.0)
                    flops[f'task{task_id}_att_dec_{i}'] = float(stage_flops)

        return flops

    def _split_name(self, split_point: int) -> str:
        if split_point < 0 or split_point >= len(self.split_points):
            raise ValueError(
                f"Invalid split_point={split_point}. "
                f"Valid range: [0, {len(self.split_points) - 1}]"
            )
        return self.split_points[split_point]

    def _compute_feature_shapes(self, input_resolution: Tuple[int, int, int]) -> Dict[str, Tuple[int, ...]]:
        c, h, w = input_resolution
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feats = self.extract_features(dummy)
        return {name: tuple(feat.shape[1:]) for name, feat in feats.items()}
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """
        加载 mtan-reference 预训练权重
        
        Args:
            checkpoint_path: .pth 文件路径（例如 best_model_equal_standard.pth）
            strict: 是否严格匹配权重键名
        """
        import warnings
        import sys
        import types

        warnings.filterwarnings("ignore", category=FutureWarning)

        # ------------------------------------------------------------------
        # Compatibility shim for checkpoints saved with NumPy 2.x internals.
        #
        # Some legacy checkpoints pickle numpy objects whose module paths are
        # like `numpy._core.multiarray` / `numpy._core._multiarray_umath`.
        # In NumPy 1.x, these live under `numpy.core.*`.
        #
        # To avoid forcing users to rebuild conda environments (numba/tensorboardx
        # conflicts are common), we inject aliases into sys.modules before loading.
        # ------------------------------------------------------------------
        try:
            import numpy as _np  # noqa: F401
            import numpy.core.multiarray as _np_multiarray

            # `_multiarray_umath` may not exist as a public module in all numpy builds,
            # so we guard it.
            try:
                import numpy.core._multiarray_umath as _np_multiarray_umath  # type: ignore
            except Exception:
                _np_multiarray_umath = None

            if "numpy._core" not in sys.modules:
                pkg = types.ModuleType("numpy._core")
                pkg.__path__ = []  # mark as package
                sys.modules["numpy._core"] = pkg

            sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
            if _np_multiarray_umath is not None:
                sys.modules.setdefault("numpy._core._multiarray_umath", _np_multiarray_umath)
        except Exception:
            # If numpy itself is broken, let torch.load raise a clearer error later.
            pass

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # mtan-reference 保存格式可能包含多个键
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 加载到核心网络
        missing_keys, unexpected_keys = self.core_net.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            print(f"[WARN] Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"[WARN] Unexpected keys in checkpoint: {unexpected_keys}")
        
        print(f"[INFO] Loaded pretrained MTAN weights from {checkpoint_path}")
        
        return {
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
        }
