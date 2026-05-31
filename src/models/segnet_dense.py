"""
SegNet Dense (Global-to-Task Skip Connections) model adapted from mtan-reference.

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


class SegNetDenseCore(nn.Module):
    """Exact architecture mirror of mtan-reference/im2im_pred/model_segnet_dense.py."""

    def __init__(self, num_classes: int = 13):
        super().__init__()
        filt = [64, 128, 256, 512, 512]
        self.class_nb = num_classes

        # shared encoder/decoder
        self.encoder_block = nn.ModuleList(
            [self.conv_layer([3, filt[0], filt[0]], bottle_neck=True)]
        )
        self.decoder_block = nn.ModuleList(
            [self.conv_layer([filt[0], filt[0], self.class_nb], bottle_neck=True)]
        )

        # task-specific encoder/decoder (dense connections)
        self.encoder_block_t = nn.ModuleList(
            [nn.ModuleList([self.conv_layer([3, filt[0], filt[0]], bottle_neck=True)])]
        )
        self.decoder_block_t = nn.ModuleList(
            [nn.ModuleList([self.conv_layer([2 * filt[0], 2 * filt[0], filt[0]], bottle_neck=True)])]
        )

        for i in range(4):
            if i == 0:
                self.encoder_block.append(
                    self.conv_layer([filt[i], filt[i + 1], filt[i + 1]], bottle_neck=True)
                )
                self.decoder_block.append(
                    self.conv_layer([filt[i + 1], filt[i], filt[i]], bottle_neck=True)
                )
            else:
                self.encoder_block.append(
                    self.conv_layer([filt[i], filt[i + 1], filt[i + 1]], bottle_neck=False)
                )
                self.decoder_block.append(
                    self.conv_layer([filt[i + 1], filt[i], filt[i]], bottle_neck=False)
                )

        for j in range(3):
            if j < 2:
                self.encoder_block_t.append(
                    nn.ModuleList([self.conv_layer([3, filt[0], filt[0]], bottle_neck=True)])
                )
                self.decoder_block_t.append(
                    nn.ModuleList([self.conv_layer([2 * filt[0], 2 * filt[0], filt[0]], bottle_neck=True)])
                )
            for i in range(4):
                if i == 0:
                    self.encoder_block_t[j].append(
                        self.conv_layer([2 * filt[i], filt[i + 1], filt[i + 1]], bottle_neck=True)
                    )
                    self.decoder_block_t[j].append(
                        self.conv_layer([2 * filt[i + 1], filt[i], filt[i]], bottle_neck=True)
                    )
                else:
                    self.encoder_block_t[j].append(
                        self.conv_layer([2 * filt[i], filt[i + 1], filt[i + 1]], bottle_neck=False)
                    )
                    self.decoder_block_t[j].append(
                        self.conv_layer([2 * filt[i + 1], filt[i], filt[i]], bottle_neck=False)
                    )

        # pooling / unpooling
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # task heads
        self.pred_task1 = self.conv_layer([filt[0], self.class_nb], bottle_neck=True, pred_layer=True)
        self.pred_task2 = self.conv_layer([filt[0], 1], bottle_neck=True, pred_layer=True)
        self.pred_task3 = self.conv_layer([filt[0], 3], bottle_neck=True, pred_layer=True)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel: List[int], bottle_neck: bool, pred_layer: bool = False) -> nn.Module:
        if bottle_neck:
            if not pred_layer:
                return nn.Sequential(
                    nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channel[1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channel[2]),
                    nn.ReLU(inplace=True),
                )
            return nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )

        return nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        encoder_conv, decoder_conv, encoder_samp, decoder_samp, indices = ([0] * 5 for _ in range(5))
        encoder_conv_t, decoder_conv_t, encoder_samp_t, decoder_samp_t, indices_t = ([0] * 3 for _ in range(5))
        for i in range(3):
            encoder_conv_t[i], decoder_conv_t[i], encoder_samp_t[i], decoder_samp_t[i], indices_t[i] = (
                [0] * 5 for _ in range(5)
            )

        # shared encoder
        for i in range(5):
            if i == 0:
                encoder_conv[i] = self.encoder_block[i](x)
                encoder_samp[i], indices[i] = self.down_sampling(encoder_conv[i])
            else:
                encoder_conv[i] = self.encoder_block[i](encoder_samp[i - 1])
                encoder_samp[i], indices[i] = self.down_sampling(encoder_conv[i])

        # shared decoder
        for i in range(5):
            if i == 0:
                decoder_samp[i] = self.up_sampling(encoder_samp[-1], indices[-1])
                decoder_conv[i] = self.decoder_block[-i - 1](decoder_samp[i])
            else:
                decoder_samp[i] = self.up_sampling(decoder_conv[i - 1], indices[-i - 1])
                decoder_conv[i] = self.decoder_block[-i - 1](decoder_samp[i])

        # task-specific branches
        for j in range(3):
            for i in range(5):
                if i == 0:
                    encoder_conv_t[j][i] = self.encoder_block_t[j][i](x)
                    encoder_samp_t[j][i], indices_t[j][i] = self.down_sampling(encoder_conv_t[j][i])
                else:
                    encoder_conv_t[j][i] = self.encoder_block_t[j][i](
                        torch.cat((encoder_samp_t[j][i - 1], encoder_samp[i - 1]), dim=1)
                    )
                    encoder_samp_t[j][i], indices_t[j][i] = self.down_sampling(encoder_conv_t[j][i])

            for i in range(5):
                if i == 0:
                    decoder_samp_t[j][i] = self.up_sampling(encoder_samp_t[j][-1], indices_t[j][-1])
                    decoder_conv_t[j][i] = self.decoder_block_t[j][-i - 1](
                        torch.cat((decoder_samp_t[j][i], decoder_samp[i]), dim=1)
                    )
                else:
                    decoder_samp_t[j][i] = self.up_sampling(decoder_conv_t[j][i - 1], indices_t[j][-i - 1])
                    decoder_conv_t[j][i] = self.decoder_block_t[j][-i - 1](
                        torch.cat((decoder_samp_t[j][i], decoder_samp[i]), dim=1)
                    )

        t1_pred = F.log_softmax(self.pred_task1(decoder_conv_t[0][-1]), dim=1)
        t2_pred = self.pred_task2(decoder_conv_t[1][-1])
        t3_pred = self.pred_task3(decoder_conv_t[2][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma


class DenseSegNet(BaseMTLModel):
    """Dense SegNet wrapper with split points and compression groups."""

    def __init__(self, config: Dict):
        super().__init__(config)
        arch_cfg = config.get("architecture", {})
        num_classes = arch_cfg.get("num_classes", 13)

        self.core_net = SegNetDenseCore(num_classes=num_classes)
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
        return {"seg": preds[0], "depth": preds[1], "normal": preds[2]}

    def get_split_points(self) -> List[str]:
        return self.split_points.copy()

    def partition(self, split_point: int):
        raise NotImplementedError(
            "Direct partition is not implemented for DenseSegNet; use extract_features or hooks."
        )

    def get_feature_size(self, split_point: int) -> Tuple[int, ...]:
        split_name = self._split_name(split_point)
        return self.feature_shapes[split_name]

    def get_split_flops(self, input_resolution: Tuple[int, int, int]) -> Dict[str, float]:
        c, h, w = input_resolution
        x = torch.zeros(1, c, h, w)
        flops: Dict[str, float] = {}

        with torch.no_grad():
            shared_conv = [None] * 5
            shared_pool = [None] * 5
            shared_indices = [None] * 5
            shared_pool_flops = [0.0] * 5

            for i in range(5):
                stage_flops = shared_pool_flops[i - 1] if i > 0 else 0.0
                shared_input = x if i == 0 else shared_pool[i - 1]
                shared_conv[i], extra = profile_module(self.core_net.encoder_block[i], shared_input)
                stage_flops += extra
                flops[f"shared_encoder_{i}"] = float(stage_flops)

                (shared_pool[i], shared_indices[i]), shared_pool_flops[i] = profile_module(
                    self.core_net.down_sampling, shared_conv[i]
                )

            shared_dec_samp = [None] * 5
            shared_dec_conv = [None] * 5
            for i in range(5):
                stage_flops = shared_pool_flops[-1] if i == 0 else 0.0
                unpool_input = shared_pool[-1] if i == 0 else shared_dec_conv[i - 1]
                shared_dec_samp[i], extra = profile_module(
                    self.core_net.up_sampling, unpool_input, shared_indices[-i - 1]
                )
                stage_flops += extra

                shared_dec_conv[i], extra = profile_module(self.core_net.decoder_block[-i - 1], shared_dec_samp[i])
                stage_flops += extra
                flops[f"shared_decoder_{i}"] = float(stage_flops)

            for task_id in range(3):
                task_conv = [None] * 5
                task_pool = [None] * 5
                task_indices = [None] * 5
                task_pool_flops = [0.0] * 5

                for i in range(5):
                    stage_flops = task_pool_flops[i - 1] if i > 0 else 0.0
                    if i == 0:
                        task_input = x
                    else:
                        task_input = torch.cat((task_pool[i - 1], shared_pool[i - 1]), dim=1)
                    task_conv[i], extra = profile_module(self.core_net.encoder_block_t[task_id][i], task_input)
                    stage_flops += extra
                    flops[f"task{task_id}_enc_{i}"] = float(stage_flops)

                    (task_pool[i], task_indices[i]), task_pool_flops[i] = profile_module(
                        self.core_net.down_sampling, task_conv[i]
                    )

                task_dec_conv = [None] * 5
                for i in range(5):
                    stage_flops = task_pool_flops[-1] if i == 0 else 0.0
                    unpool_input = task_pool[-1] if i == 0 else task_dec_conv[i - 1]
                    task_dec_samp, extra = profile_module(
                        self.core_net.up_sampling, unpool_input, task_indices[-i - 1]
                    )
                    stage_flops += extra

                    task_input = torch.cat((task_dec_samp, shared_dec_samp[i]), dim=1)
                    task_dec_conv[i], extra = profile_module(
                        self.core_net.decoder_block_t[task_id][-i - 1], task_input
                    )
                    stage_flops += extra
                    flops[f"task{task_id}_dec_{i}"] = float(stage_flops)

        return flops

    # ------------------------------------------------------------------
    # Split points & compression groups
    # ------------------------------------------------------------------
    def _define_split_points(self):
        self.split_points = []
        for i in range(5):
            self.split_points.append(f"shared_encoder_{i}")
        for i in range(5):
            self.split_points.append(f"shared_decoder_{i}")
        for task_id in range(3):
            for i in range(5):
                self.split_points.append(f"task{task_id}_enc_{i}")
            for i in range(5):
                self.split_points.append(f"task{task_id}_dec_{i}")

    def _define_compression_groups(self):
        compression_groups: Dict[str, List[str]] = {}
        for i in range(5):
            compression_groups[f"enc_stage_{i}"] = [f"shared_encoder_{i}"]
        for i in range(5):
            compression_groups[f"dec_stage_{i}"] = [f"shared_decoder_{i}"]
        for task_id in range(3):
            compression_groups[f"task{task_id}_enc"] = [f"task{task_id}_enc_{i}" for i in range(5)]
            compression_groups[f"task{task_id}_dec"] = [f"task{task_id}_dec_{i}" for i in range(5)]
        self.compression_groups = compression_groups

    def get_compression_groups(self) -> Dict[str, List[str]]:
        return self.compression_groups.copy()

    # ------------------------------------------------------------------
    # Feature extraction & quantization hooks
    # ------------------------------------------------------------------
    def _get_module_for_split_point(self, split_name: str) -> nn.Module:
        if split_name.startswith("shared_encoder_"):
            idx = int(split_name.split("_")[-1])
            if 0 <= idx < 5:
                return self.core_net.encoder_block[idx]
        elif split_name.startswith("shared_decoder_"):
            idx = int(split_name.split("_")[-1])
            if 0 <= idx < 5:
                return self.core_net.decoder_block[-idx - 1]
        elif split_name.startswith("task"):
            parts = split_name.split("_")
            task_id = int(parts[0].replace("task", ""))
            branch = parts[1]
            idx = int(parts[2])
            if 0 <= task_id < 3 and 0 <= idx < 5:
                if branch == "enc":
                    return self.core_net.encoder_block_t[task_id][idx]
                if branch == "dec":
                    return self.core_net.decoder_block_t[task_id][-idx - 1]
        raise ValueError(f"Cannot map split_point '{split_name}' to a nn.Module.")

    def extract_features(self, x: torch.Tensor, split_points: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
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

        if group_name in self.compression_groups:
            split_points = self.compression_groups[group_name]
        else:
            # Backward-compatible alias: task{k}_branch = task{k}_enc ∪ task{k}_dec
            if group_name.startswith("task") and group_name.endswith("_branch"):
                try:
                    k = int(group_name.replace("task", "").replace("_branch", ""))
                except Exception as e:
                    raise ValueError(f"Unknown compression group: {group_name}") from e
                split_points = [f"task{k}_enc_{i}" for i in range(5)] + [f"task{k}_dec_{i}" for i in range(5)]
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

        print(f"[INFO] Loaded pretrained DenseSegNet weights from {checkpoint_path}")
        return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

