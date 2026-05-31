"""
SegNet Split (Hard Parameter Sharing) model adapted from mtan-reference.

Goal:
1) Match the original architecture for checkpoint compatibility.
2) Provide split points and compression groups for feature quantization.
3) Implement BaseMTLModel interfaces for profiling and simulation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_mtl import BaseMTLModel
from .cost_utils import profile_module


class SegNetSplitCore(nn.Module):
    """Exact architecture mirror of mtan-reference/im2im_pred/model_segnet_split.py."""

    def __init__(self, split_type: str = "standard", num_classes: int = 13):
        super().__init__()
        if split_type not in {"standard", "wide", "deep"}:
            raise ValueError(f"Unsupported split_type: {split_type}")
        self.split_type = split_type

        if split_type == "wide":
            filt = [64, 128, 256, 512, 1024]
        else:
            filt = [64, 128, 256, 512, 512]

        self.class_nb = num_classes

        # encoder / decoder blocks (same names as reference)
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filt[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filt[0], filt[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filt[i], filt[i + 1]]))
            self.decoder_block.append(self.conv_layer([filt[i + 1], filt[i]]))

        # conv blocks
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filt[0], filt[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filt[0], filt[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filt[i + 1], filt[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filt[i], filt[i]]))
            else:
                self.conv_block_enc.append(
                    nn.Sequential(
                        self.conv_layer([filt[i + 1], filt[i + 1]]),
                        self.conv_layer([filt[i + 1], filt[i + 1]]),
                    )
                )
                self.conv_block_dec.append(
                    nn.Sequential(
                        self.conv_layer([filt[i], filt[i]]),
                        self.conv_layer([filt[i], filt[i]]),
                    )
                )

        # task heads
        self.pred_task1 = nn.Sequential(
            nn.Conv2d(in_channels=filt[0], out_channels=filt[0], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=filt[0], out_channels=self.class_nb, kernel_size=1, padding=0),
        )
        self.pred_task2 = nn.Sequential(
            nn.Conv2d(in_channels=filt[0], out_channels=filt[0], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=filt[0], out_channels=1, kernel_size=1, padding=0),
        )
        self.pred_task3 = nn.Sequential(
            nn.Conv2d(in_channels=filt[0], out_channels=filt[0], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=filt[0], out_channels=3, kernel_size=1, padding=0),
        )

        # pooling / unpooling
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

    def conv_layer(self, channel: List[int]) -> nn.Module:
        if self.split_type == "deep":
            return nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        return nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # encoder
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        # decoder
        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # task heads
        t1_pred = F.log_softmax(self.pred_task1(g_decoder[i][1]), dim=1)
        t2_pred = self.pred_task2(g_decoder[i][1])
        t3_pred = self.pred_task3(g_decoder[i][1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma


class SplitSegNet(BaseMTLModel):
    """
    Split SegNet model wrapper with split points and compression groups.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        arch_cfg = config.get("architecture", {})
        split_type = arch_cfg.get("split_type", "standard")
        num_classes = arch_cfg.get("num_classes", 13)

        self.core_net = SegNetSplitCore(split_type=split_type, num_classes=num_classes)
        self._define_split_points()
        self._define_compression_groups()

        input_resolution = arch_cfg.get("input_resolution", [3, 288, 288])
        self.feature_shapes = self._compute_feature_shapes(tuple(input_resolution))

    # ------------------------------------------------------------------
    # BaseMTLModel interface
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, split_point: int = -1) -> Dict[str, torch.Tensor]:
        if split_point >= 0:
            split_name = self._split_name(split_point)
            feats = self.extract_features(x, split_points=[split_name])
            return {"features": feats[split_name]}

        preds, _ = self.core_net(x)
        return {
            "seg": preds[0],
            "depth": preds[1],
            "normal": preds[2],
        }

    def get_split_points(self) -> List[str]:
        return self.split_points.copy()

    def partition(self, split_point: int):
        raise NotImplementedError(
            "Direct partition is not implemented for SplitSegNet; use extract_features or hooks."
        )

    def get_feature_size(self, split_point: int) -> Tuple[int, ...]:
        split_name = self._split_name(split_point)
        return self.feature_shapes[split_name]

    def get_split_flops(self, input_resolution: Tuple[int, int, int]) -> Dict[str, float]:
        c, h, w = input_resolution
        x = torch.zeros(1, c, h, w)
        flops: Dict[str, float] = {}

        with torch.no_grad():
            enc_b = [None] * 5
            enc_pool = [None] * 5
            enc_indices = [None] * 5
            pool_flops = [0.0] * 5

            for i in range(5):
                stage_flops = pool_flops[i - 1] if i > 0 else 0.0
                enc_input = x if i == 0 else enc_pool[i - 1]
                enc_a, extra = profile_module(self.core_net.encoder_block[i], enc_input)
                stage_flops += extra
                flops[f"shared_encoder_{i}_a"] = float(stage_flops)

                enc_b[i], extra = profile_module(self.core_net.conv_block_enc[i], enc_a)
                flops[f"shared_encoder_{i}_b"] = float(extra)

                (enc_pool[i], enc_indices[i]), pool_flops[i] = profile_module(
                    self.core_net.down_sampling, enc_b[i]
                )

            prev_dec_b = None
            for i in range(5):
                stage_flops = pool_flops[-1] if i == 0 else 0.0
                unpool_input = enc_pool[-1] if i == 0 else prev_dec_b
                dec_up, extra = profile_module(
                    self.core_net.up_sampling, unpool_input, enc_indices[-i - 1]
                )
                stage_flops += extra

                dec_a, extra = profile_module(self.core_net.decoder_block[-i - 1], dec_up)
                stage_flops += extra
                flops[f"shared_decoder_{i}_a"] = float(stage_flops)

                prev_dec_b, extra = profile_module(self.core_net.conv_block_dec[-i - 1], dec_a)
                flops[f"shared_decoder_{i}_b"] = float(extra)

        return flops

    # ------------------------------------------------------------------
    # Split points & compression groups
    # ------------------------------------------------------------------
    def _define_split_points(self):
        self.split_points = []
        for i in range(5):
            self.split_points.append(f"shared_encoder_{i}_a")  # encoder_block[i]
            self.split_points.append(f"shared_encoder_{i}_b")  # conv_block_enc[i]
        for i in range(5):
            # decoder index 0 is the deepest stage (consistent with MTAN convention)
            self.split_points.append(f"shared_decoder_{i}_a")  # decoder_block[-i-1]
            self.split_points.append(f"shared_decoder_{i}_b")  # conv_block_dec[-i-1]

    def _define_compression_groups(self):
        compression_groups: Dict[str, List[str]] = {}
        for i in range(5):
            compression_groups[f"enc_stage_{i}"] = [
                f"shared_encoder_{i}_a",
                f"shared_encoder_{i}_b",
            ]
        for i in range(5):
            compression_groups[f"dec_stage_{i}"] = [
                f"shared_decoder_{i}_a",
                f"shared_decoder_{i}_b",
            ]
        self.compression_groups = compression_groups

    def get_compression_groups(self) -> Dict[str, List[str]]:
        return self.compression_groups.copy()

    # ------------------------------------------------------------------
    # Feature extraction & quantization hooks
    # ------------------------------------------------------------------
    def _get_module_for_split_point(self, split_name: str) -> nn.Module:
        if split_name.startswith("shared_encoder_"):
            parts = split_name.split("_")
            idx = int(parts[2])
            suffix = parts[3]
            if 0 <= idx < 5:
                if suffix == "a":
                    return self.core_net.encoder_block[idx]
                if suffix == "b":
                    return self.core_net.conv_block_enc[idx]
        elif split_name.startswith("shared_decoder_"):
            parts = split_name.split("_")
            idx = int(parts[2])
            suffix = parts[3]
            if 0 <= idx < 5:
                if suffix == "a":
                    return self.core_net.decoder_block[-idx - 1]
                if suffix == "b":
                    return self.core_net.conv_block_dec[-idx - 1]
        raise ValueError(f"Cannot map split_point '{split_name}' to a nn.Module.")

    def extract_features(
        self, x: torch.Tensor, split_points: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        if split_points is None:
            split_points = self.split_points

        features: Dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(name: str):
            def hook(_m, _inp, out):
                features[name] = out.detach().clone()
            return hook

        for split_name in split_points:
            module = self._get_module_for_split_point(split_name)
            handles.append(module.register_forward_hook(make_hook(split_name)))

        with torch.no_grad():
            _ = self.core_net(x)

        for h in handles:
            h.remove()

        return features

    def apply_quantization_to_group(
        self,
        group_name: str,
        bit_width: int,
        per_channel: bool = False,
        quant_mode: str = "dynamic",
        calib_stats: Optional[Dict[str, Dict]] = None,
    ) -> List:
        from ..accuracy_modeling.quantization import uniform_quantize

        if group_name not in self.compression_groups:
            raise ValueError(f"Unknown compression group: {group_name}")
        if quant_mode not in {"dynamic", "fixed"}:
            raise ValueError(f"Unsupported quant_mode: {quant_mode}")

        split_points = self.compression_groups[group_name]
        if quant_mode == "fixed":
            if calib_stats is None:
                raise ValueError("Fixed quantization requires calib_stats.")
            missing = [sp for sp in split_points if sp not in calib_stats]
            if missing:
                raise ValueError(f"Missing calibration stats for split points: {missing}")

        handles = []

        def make_quant_hook(bw: int, pc: bool, split_key: str):
            def quant_hook(_m, _inp, out):
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
                    calib_params=calib_params,
                )
                return quantized
            return quant_hook

        for split_name in split_points:
            module = self._get_module_for_split_point(split_name)
            handles.append(module.register_forward_hook(make_quant_hook(bit_width, per_channel, split_name)))

        return handles

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
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
        import warnings
        import sys
        import types

        warnings.filterwarnings("ignore", category=FutureWarning)

        # Compatibility shim for checkpoints saved with NumPy 2.x internals.
        try:
            import numpy as _np  # noqa: F401
            import numpy.core.multiarray as _np_multiarray

            try:
                import numpy.core._multiarray_umath as _np_multiarray_umath  # type: ignore
            except Exception:
                _np_multiarray_umath = None

            if "numpy._core" not in sys.modules:
                pkg = types.ModuleType("numpy._core")
                pkg.__path__ = []
                sys.modules["numpy._core"] = pkg

            sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
            if _np_multiarray_umath is not None:
                sys.modules.setdefault("numpy._core._multiarray_umath", _np_multiarray_umath)
        except Exception:
            pass

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        missing_keys, unexpected_keys = self.core_net.load_state_dict(state_dict, strict=strict)
        if missing_keys:
            print(f"[WARN] Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"[WARN] Unexpected keys in checkpoint: {unexpected_keys}")

        print(f"[INFO] Loaded pretrained SplitSegNet weights from {checkpoint_path}")
        return {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }

