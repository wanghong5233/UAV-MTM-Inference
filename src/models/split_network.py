"""
Split Network (Hard Parameter Sharing) baseline implementation.

Design goals:
1. Decouple encoder/decoder/partitioning logic for easy model replacement.
2. Fully compatible with BaseMTLModel interface for accuracy fitting and simulation.
3. Configuration-driven to reduce maintenance cost.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .base_mtl import BaseMTLModel


class UpsampleBlock(nn.Module):
    """Basic upsampling block: bilinear interpolation followed by two conv layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class TaskDecoder(nn.Module):
    """Task decoder: recovers task-specific outputs from shared features."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: List[int],
        out_channels: int,
        output_resolution: Tuple[int, int],
    ):
        super().__init__()
        blocks = []
        prev_channels = in_channels
        for ch in mid_channels:
            blocks.append(UpsampleBlock(prev_channels, ch))
            prev_channels = ch
        self.body = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Conv2d(prev_channels, out_channels, kernel_size=1)
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        if self.output_resolution:
            x = F.interpolate(
                x,
                size=self.output_resolution,
                mode="bilinear",
                align_corners=False,
            )
        return self.head(x)


class StageSequential(nn.Module):
    """Sequential execution of encoder stages for independent deployment after partitioning."""

    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TailWithDecoders(nn.Module):
    """Tail module after partitioning: continues encoder and outputs task results."""

    def __init__(self, encoder_tail: List[nn.Module], task_heads: nn.ModuleDict):
        super().__init__()
        self.encoder_tail = StageSequential(encoder_tail) if encoder_tail else nn.Identity()
        self.task_heads = task_heads

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.encoder_tail(x)
        return {task: head(x) for task, head in self.task_heads.items()}


class SplitNetwork(BaseMTLModel):
    """
    ResNet shared backbone + task-specific decoders.
    Supports stage-wise partitioning for feature extraction, reusable in partitioning and compression experiments.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        arch_cfg = config.get("architecture", {})
        self.output_resolution = tuple(arch_cfg.get("output_resolution", [288, 288]))
        self.input_resolution = tuple(
            arch_cfg.get("input_resolution", config.get("input_resolution", [3, 288, 288]))
        )
        decoder_channels = arch_cfg.get("decoder_channels", [256, 128, 64, 32])
        backbone_name = arch_cfg.get("backbone", "resnet34")
        pretrained = arch_cfg.get("pretrained", True)

        # 1. Shared encoder
        self.encoder_stages, self.encoder_out_channels = self._build_encoder(
            backbone_name, pretrained
        )
        self.split_points = list(self.encoder_stages.keys())

        # 2. Task heads (default: seg/depth/normal, overridable via config)
        task_cfg = arch_cfg.get("task_heads", {})
        if not task_cfg:
            num_classes = arch_cfg.get("num_classes", 13)
            task_cfg = {
                "seg": {"out_channels": num_classes},
                "depth": {"out_channels": 1},
                "normal": {"out_channels": 3},
            }
        self.task_heads = nn.ModuleDict()
        for task_name, head_params in task_cfg.items():
            head_channels = head_params.get("channels", decoder_channels)
            out_channels = head_params["out_channels"]
            self.task_heads[task_name] = TaskDecoder(
                in_channels=self.encoder_out_channels,
                mid_channels=head_channels,
                out_channels=out_channels,
                output_resolution=self.output_resolution,
            )

        # 3. Record feature shapes at each split point
        self.feature_shapes = self._compute_feature_shapes(self.input_resolution)

    # ------------------------------------------------------------------
    # BaseMTLModel interface implementation
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, split_point: int = -1) -> Dict[str, torch.Tensor]:
        features = self._run_encoder(x)
        if split_point >= 0:
            split_name = self._split_name(split_point)
            return {"features": features[split_name]}

        encoder_output = features[self.split_points[-1]]
        return {task: head(encoder_output) for task, head in self.task_heads.items()}
    
    def get_split_points(self) -> List[str]:
        return self.split_points.copy()
    
    def partition(self, split_point: int) -> Tuple[nn.Module, nn.Module]:
        split_name = self._split_name(split_point)
        split_idx = self.split_points.index(split_name)

        encoder_names = self.split_points
        head_modules = [copy.deepcopy(self.encoder_stages[name]) for name in encoder_names[: split_idx + 1]]
        tail_encoder_modules = [copy.deepcopy(self.encoder_stages[name]) for name in encoder_names[split_idx + 1 :]]
        task_heads_copy = nn.ModuleDict(
            {task: copy.deepcopy(head) for task, head in self.task_heads.items()}
        )

        head = StageSequential(head_modules)
        tail = TailWithDecoders(tail_encoder_modules, task_heads_copy)
        return head, tail
    
    def get_feature_size(self, split_point: int) -> Tuple[int, ...]:
        split_name = self._split_name(split_point)
        return self.feature_shapes[split_name]

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _split_name(self, split_point: int) -> str:
        if split_point < 0 or split_point >= len(self.split_points):
            raise ValueError(
                f"Invalid split_point={split_point}. "
                f"Valid range: [0, {len(self.split_points) - 1}]"
            )
        return self.split_points[split_point]

    def _run_encoder(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        out = x
        for name, module in self.encoder_stages.items():
            out = module(out)
            outputs[name] = out
        return outputs

    def _build_encoder(self, backbone_name: str, pretrained: bool):
        backbone_name = backbone_name.lower()
        if backbone_name not in {"resnet18", "resnet34", "resnet50"}:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        backbone = self._load_resnet(backbone_name, pretrained)

        stages = OrderedDict()
        stages["stem"] = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        stages["layer1"] = backbone.layer1
        stages["layer2"] = backbone.layer2
        stages["layer3"] = backbone.layer3
        stages["layer4"] = backbone.layer4

        encoder_out_channels = backbone.fc.in_features
        return nn.ModuleDict(stages), encoder_out_channels

    def _load_resnet(self, name: str, pretrained: bool):
        constructor = getattr(models, name)
        if pretrained:
            weights_attr = f"{name.capitalize()}_Weights"
            weights = getattr(models, weights_attr, None)
            if weights is not None:
                return constructor(weights=weights.IMAGENET1K_V1)
            return constructor(pretrained=True)

        if "weights" in constructor.__code__.co_varnames:
            return constructor(weights=None)
        return constructor(pretrained=False)

    def _compute_feature_shapes(self, input_resolution: Tuple[int, int, int]) -> Dict[str, Tuple[int, ...]]:
        c, h, w = input_resolution
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            features = self._run_encoder(dummy)
        return {name: tuple(feat.shape[1:]) for name, feat in features.items()}


