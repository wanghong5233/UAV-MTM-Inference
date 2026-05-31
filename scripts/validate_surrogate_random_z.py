"""
Validate the sensitivity-aware additive surrogate with random multi-group Z.

Goal:
  Compare surrogate-predicted accuracy A_pred vs. true accuracy A_true under
  jointly-quantized configurations (multiple groups quantized at once).

Paper Modeling Reference (main.tex Eq. 18):
  A_hat(t) = A_full - sum_g theta_g(t) * sum_p z_{g,p}(t) * S_g^{(p)}

  In this script, we assume theta_g = 1 for all groups (all edges transmitted),
  which is the worst-case scenario. This is valid for verifying the additive
  property of the surrogate model.

Plot:
  Scatter (x=A_pred, y=A_true) with y=x reference line, best-fit line, R^2, MAE.

Run (PowerShell, for paper figure - 200 configs):
  python scripts/validate_surrogate_random_z.py --model mtan --sensitivity "results\accuracy_modeling\mtan\sensitivity_mtan_fixed.pkl" --dataset "data\nyuv2" --checkpoint "..\mtan-reference\im2im_pred\checkpoints\mtan\best_model_equal_standard.pth" --device cuda --batch_size 4 --num_workers 0 --pin_memory 0 --num_configs 200 --num_corners 4 --max_batches 50 --quant_mode fixed

Quick test (20 configs, auto output path):
  python scripts/validate_surrogate_random_z.py --model mtan --sensitivity "results\accuracy_modeling\mtan\sensitivity_mtan_fixed.pkl" --dataset "data\nyuv2" --checkpoint "..\mtan-reference\im2im_pred\checkpoints\mtan\best_model_equal_standard.pth" --device cuda --batch_size 4 --num_workers 0 --pin_memory 0 --num_configs 20 --num_corners 4 --max_batches 50 --quant_mode fixed
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_mtl import BaseMTLModel
from src.models.mtan import MTAN
from src.models.segnet_split import SplitSegNet
from src.models.segnet_dense import DenseSegNet
from src.models.segnet_cross import CrossStitchSegNet
from src.data.nyuv2 import create_nyuv2_dataloader
from src.accuracy_modeling.aggregate_accuracy import compute_aggregate_accuracy_relative


# =========================
# Metrics (copied from calibrate_sensitivity.py to ensure consistency)
# =========================

class ConfMatrix(object):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self) -> Tuple[float, float]:
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item(), acc.item()


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    denom = torch.nonzero(binary_mask, as_tuple=False).size(0)
    return (torch.sum(abs_err) / denom).item(), (torch.sum(rel_err) / denom).item()


def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error)


def evaluate_mtan(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> dict:
    model.eval()
    model.to(device)

    conf_mat = ConfMatrix(num_classes=13)

    total_depth_abs = 0.0
    total_normal_mean = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if max_batches is not None and i >= max_batches:
                break

            images = batch["image"].to(device)
            seg_gt = batch["semantic"].to(device)
            depth_gt = batch["depth"].to(device)   # [B, 1, H, W]
            normal_gt = batch["normal"].to(device)

            test_pred, _ = model.core_net(images)

            seg_pred = test_pred[0]
            depth_pred = test_pred[1]
            normal_pred = test_pred[2]

            conf_mat.update(seg_pred.argmax(1).flatten(), seg_gt.flatten())

            abs_err, _ = depth_error(depth_pred, depth_gt)
            total_depth_abs += abs_err

            mean_err, _ = normal_error(normal_pred, normal_gt)
            total_normal_mean += mean_err

            num_batches += 1

    seg_miou, _ = conf_mat.get_metrics()
    depth_abs_err = total_depth_abs / max(1, num_batches)
    normal_mean_err = total_normal_mean / max(1, num_batches)

    return {
        "seg_miou": seg_miou,
        "depth_abs_err": depth_abs_err,
        "normal_mean": normal_mean_err,
    }


def build_model(args) -> BaseMTLModel:
    if args.model == "mtan":
        config = {"architecture": {"task_names": ["semantic", "depth", "normal"]}}
        model = MTAN(config)
    elif args.model == "split":
        config = {
            "architecture": {
                "task_names": ["semantic", "depth", "normal"],
                "split_type": args.split_type,
                "num_classes": 13,
                "input_resolution": [3, 288, 288],
            }
        }
        model = SplitSegNet(config)
    elif args.model == "dense":
        config = {
            "architecture": {
                "task_names": ["semantic", "depth", "normal"],
                "num_classes": 13,
                "input_resolution": [3, 288, 288],
            }
        }
        model = DenseSegNet(config)
    elif args.model == "cross":
        config = {
            "architecture": {
                "task_names": ["semantic", "depth", "normal"],
                "num_classes": 13,
                "input_resolution": [3, 288, 288],
            }
        }
        model = CrossStitchSegNet(config)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.checkpoint is None:
        if args.model == "mtan":
            args.checkpoint = "../mtan-reference/im2im_pred/checkpoints/mtan/best_model_equal_standard.pth"
        elif args.model == "split":
            args.checkpoint = "../mtan-reference/im2im_pred/checkpoints/split_network/best_model_equal_standard.pth"
        elif args.model == "dense":
            args.checkpoint = "../mtan-reference/im2im_pred/checkpoints/dense/best_model_equal_standard.pth"
        elif args.model == "cross":
            args.checkpoint = "../mtan-reference/im2im_pred/checkpoints/crossstitch/best_model_equal_standard.pth"

    model.load_pretrained(args.checkpoint)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Validate surrogate with random Z")
    parser.add_argument("--sensitivity", type=str, default=None, help="Path to sensitivity .pkl (default by model + quant_mode)")
    parser.add_argument("--model", type=str, default="mtan", choices=["mtan", "split", "dense", "cross"], help="Model type")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (default by model)")
    parser.add_argument("--split_type", type=str, default="standard", choices=["standard", "wide", "deep"],
                        help="Split SegNet type (only for model=split)")
    parser.add_argument("--dataset", type=str, default="data/nyuv2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--pin_memory", type=int, default=0, choices=[0, 1])
    parser.add_argument("--per_channel", type=int, default=-1, choices=[-1, 0, 1],
                        help="Quantization mode: 1=per-channel, 0=per-tensor, -1=follow sensitivity metadata (default)")
    parser.add_argument("--quant_mode", type=str, default="auto", choices=["auto", "dynamic", "fixed"],
                        help="Range mode: auto=follow sensitivity metadata, dynamic=per-batch min/max, fixed=calibrated min/max")
    parser.add_argument("--num_configs", type=int, default=200, help="Number of random Z configs to evaluate (recommend 100-200 for paper)")
    parser.add_argument("--num_corners", type=int, default=4,
                        help="Include first N corner configs: all-32/all-16/all-8/all-4 (0-4). "
                             "If num_configs<=4 and num_corners=4, then configs are NOT random.")
    parser.add_argument("--max_batches", type=int, default=50, help="Evaluate on first N batches (set -1 to use full val)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path (default: results/accuracy_modeling/{model}/validation_{model}_{quant_mode}.csv)")
    parser.add_argument("--output_fig", type=str, default=None,
                        help="Output figure path (default: results/accuracy_modeling/{model}/validation_{model}_{quant_mode}.pdf)")
    parser.add_argument(
        "--calibration",
        type=str,
        default="none",
        choices=["none", "affine", "through1"],
        help=(
            "Optional post-hoc calibration to reduce systematic bias. "
            "none: no calibration; "
            "affine: fit A ≈ k*Â + b; "
            "through1: fit A ≈ k*Â + (1-k) to enforce A(Â=1)=1."
        ),
    )
    parser.add_argument(
        "--calib_frac",
        type=float,
        default=0.0,
        help=(
            "Fraction of configs used to fit calibration (0=fit on all configs; recommended 0.5 for hold-out evaluation). "
            "Only used when --calibration != none."
        ),
    )
    parser.add_argument(
        "--save_calibration",
        type=str,
        default=None,
        help="Save calibration parameters to JSON (default: results/accuracy_modeling/{model}/calibration_{model}_{quant_mode}.json).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.sensitivity is None:
        # If quant_mode=auto, default to fixed (recommended in the paper) and still read metadata from the file.
        default_mode = args.quant_mode if args.quant_mode in {"fixed", "dynamic"} else "fixed"
        args.sensitivity = f"results/accuracy_modeling/{args.model}/sensitivity_{args.model}_{default_mode}.pkl"

    sens_path = Path(args.sensitivity)
    if not sens_path.exists():
        raise FileNotFoundError(f"Sensitivity file not found: {sens_path}")

    with open(sens_path, "rb") as f:
        sens_data = pickle.load(f)

    A_full = float(sens_data["full_precision_acc"])
    sens_table: Dict[str, Dict[int, float]] = sens_data["sensitivity"]
    groups = list(sens_table.keys())

    meta = sens_data.get("metadata", {}) or {}
    agg_mode = meta.get("accuracy_aggregation", None)
    if agg_mode is not None and agg_mode != "relative_degradation_v1":
        print(f"⚠ Warning: sensitivity file aggregation={agg_mode} (expected relative_degradation_v1).")

    weights_list = meta.get("accuracy_weights", [1.0, 1.0, 1.0])
    try:
        agg_weights = (float(weights_list[0]), float(weights_list[1]), float(weights_list[2]))
    except Exception:
        agg_weights = (1.0, 1.0, 1.0)

    reference_metrics = meta.get("reference_metrics", None) or sens_data.get("baseline_metrics", None)
    if reference_metrics is None:
        raise ValueError("Cannot find reference metrics in sensitivity file (metadata.reference_metrics or baseline_metrics).")

    sens_per_channel = meta.get("per_channel", False)
    if args.per_channel == -1:
        per_channel = bool(sens_per_channel)
    else:
        per_channel = bool(args.per_channel)
    if bool(per_channel) != bool(sens_per_channel):
        print(f"⚠ Warning: per_channel={per_channel} differs from sensitivity metadata per_channel={sens_per_channel}.")

    meta_quant_mode = meta.get("quant_mode", "dynamic")
    if args.quant_mode == "auto":
        quant_mode = meta_quant_mode
    else:
        quant_mode = args.quant_mode
    if quant_mode not in {"dynamic", "fixed"}:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}")
    calib_stats = None
    if quant_mode == "fixed":
        calib_stats = meta.get("calibration_stats", None)
        if calib_stats is None:
            raise ValueError(
                "Fixed quantization requires calibration_stats in sensitivity file. "
                "Please re-run calibrate_sensitivity.py with --quant_mode fixed."
            )

    # Bit-width choices: prioritize paper levels
    bw_choices = [32, 16, 8, 4]

    # Build configs (include a few corner cases for coverage)
    configs = []
    corner = [
        {g: 32 for g in groups},
        {g: 16 for g in groups},
        {g: 8 for g in groups},
        {g: 4 for g in groups},
    ]
    num_corners = int(max(0, min(4, args.num_corners, args.num_configs)))
    for c in corner[:num_corners]:
        configs.append(c)
    while len(configs) < args.num_configs:
        cfg = {g: random.choice(bw_choices) for g in groups}
        configs.append(cfg)

    # Load model + dataloader
    model = build_model(args)

    dataloader = create_nyuv2_dataloader(
        root=args.dataset,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        augmentation=False,
        pin_memory=bool(args.pin_memory),
    )

    max_batches = None if args.max_batches is None or args.max_batches < 0 else int(args.max_batches)

    rows = []
    A_preds = []
    A_trues = []

    print("=" * 70)
    print("Surrogate validation (random multi-group Z)")
    print("=" * 70)
    print(f"A_full (from sensitivity file): {A_full:.6f}")
    print(f"num_groups: {len(groups)} | num_configs: {len(configs)} (corners={num_corners}, random={len(configs) - num_corners})")
    print(f"accuracy_aggregation: {agg_mode or 'unknown'} | weights={agg_weights}")
    print(f"quantization_mode: {'per-channel' if per_channel else 'per-tensor'} | range_mode={quant_mode}")
    print(f"eval: batch_size={args.batch_size}, max_batches={max_batches}, device={args.device}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # CRITICAL FIX: Compute reference_metrics on the SAME subset used for validation
    # Otherwise, using reference from full dataset causes A_true != 1 for all-32 config
    # -------------------------------------------------------------------------
    print("\n[Pre-step] Computing full-precision baseline on validation subset...")
    baseline_metrics_local = evaluate_mtan(model, dataloader, device=args.device, max_batches=max_batches)
    reference_metrics_local = {
        "seg_miou": float(baseline_metrics_local["seg_miou"]),
        "depth_abs_err": float(baseline_metrics_local["depth_abs_err"]),
        "normal_mean": float(baseline_metrics_local["normal_mean"]),
    }
    print(f"  Local baseline (on {max_batches} batches): seg_miou={reference_metrics_local['seg_miou']:.4f}, "
          f"depth_err={reference_metrics_local['depth_abs_err']:.4f}, normal_err={reference_metrics_local['normal_mean']:.2f}")
    print("=" * 70)

    for idx, cfg in enumerate(configs, 1):
        # Predict with theta_g = 1 (all transmitted) => A_pred = A_full - sum_g S_g^(bw_g)
        degradation = 0.0
        for g, bw in cfg.items():
            if bw >= 32:
                continue
            degradation += float(sens_table[g].get(int(bw), 0.0))
        A_pred = float(np.clip(A_full - degradation, 0.0, 1.0))

        # Apply joint quantization hooks (skip bw>=32)
        handles = []
        for g, bw in cfg.items():
            if int(bw) >= 32:
                continue
            handles.extend(model.apply_quantization_to_group(
                g,
                bit_width=int(bw),
                per_channel=per_channel,
                quant_mode=quant_mode,
                calib_stats=calib_stats,
            ))

        metrics = evaluate_mtan(model, dataloader, device=args.device, max_batches=max_batches)
        agg_true = compute_aggregate_accuracy_relative(
            metrics=metrics,
            reference_metrics=reference_metrics_local,  # Use LOCAL baseline from same subset
            weights=agg_weights,
        )
        A_true = float(agg_true["A"])

        for h in handles:
            h.remove()

        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        abs_err = abs(A_true - A_pred)
        n32 = sum(1 for _g, _bw in cfg.items() if int(_bw) == 32)
        n16 = sum(1 for _g, _bw in cfg.items() if int(_bw) == 16)
        n8 = sum(1 for _g, _bw in cfg.items() if int(_bw) == 8)
        n4 = sum(1 for _g, _bw in cfg.items() if int(_bw) == 4)
        corner_tag = ""
        if idx <= num_corners:
            corner_tag = ["all-32", "all-16", "all-8", "all-4"][idx - 1]
            corner_tag = f" | corner={corner_tag}"
        rows.append(
            {
                "idx": idx,
                "A_pred": A_pred,
                "A_true": A_true,
                "abs_err": abs_err,
                "config_json": json.dumps(cfg, ensure_ascii=False),
            }
        )
        A_preds.append(A_pred)
        A_trues.append(A_true)

        print(
            f"[{idx:03d}/{len(configs):03d}] "
            f"A_pred={A_pred:.6f} | A_true={A_true:.6f} | abs_err={abs_err:.6f} | "
            f"bw_counts(32/16/8/4)={n32}/{n16}/{n8}/{n4}{corner_tag}"
        )

    A_preds = np.asarray(A_preds, dtype=np.float64)
    A_trues = np.asarray(A_trues, dtype=np.float64)
    errors = A_trues - A_preds

    # Core statistics
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_abs_err = float(np.max(np.abs(errors)))
    mean_err = float(np.mean(errors))  # Bias
    std_err = float(np.std(errors))

    # R^2 (coefficient of determination)
    ss_res = float(np.sum((A_trues - A_preds) ** 2))
    ss_tot = float(np.sum((A_trues - np.mean(A_trues)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Pearson correlation
    if len(A_preds) >= 2:
        pearson_r = float(np.corrcoef(A_preds, A_trues)[0, 1])
    else:
        pearson_r = float("nan")

    # Degradation range
    deg_pred = A_full - A_preds
    deg_true = A_full - A_trues
    deg_range_pred = (float(deg_pred.min()), float(deg_pred.max()))
    deg_range_true = (float(deg_true.min()), float(deg_true.max()))

    # -----------------------------
    # Optional post-hoc calibration
    # -----------------------------
    cal_method = str(args.calibration)
    cal_k, cal_b = 1.0, 0.0
    calib_idx = np.arange(len(A_preds), dtype=np.int64)
    test_idx = np.arange(len(A_preds), dtype=np.int64)
    A_preds_cal = A_preds.copy()

    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        err = (y_true - y_pred).astype(np.float64)
        mae_ = float(np.mean(np.abs(err)))
        rmse_ = float(np.sqrt(np.mean(err ** 2)))
        max_abs_ = float(np.max(np.abs(err)))
        mean_err_ = float(np.mean(err))
        std_err_ = float(np.std(err))
        ss_res_ = float(np.sum((y_true - y_pred) ** 2))
        ss_tot_ = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        r2_ = float(1.0 - ss_res_ / ss_tot_) if ss_tot_ > 0 else float("nan")
        return {
            "mae": mae_,
            "rmse": rmse_,
            "max_abs_err": max_abs_,
            "mean_err": mean_err_,
            "std_err": std_err_,
            "r2": r2_,
        }

    raw_all = _compute_metrics(A_trues, A_preds)
    raw_test = None
    cal_all = None
    cal_test = None

    if cal_method != "none":
        idx_all = np.arange(len(A_preds), dtype=np.int64)
        rng = np.random.RandomState(args.seed)
        rng.shuffle(idx_all)
        if 0.0 < float(args.calib_frac) < 1.0:
            n_cal = int(round(float(args.calib_frac) * len(idx_all)))
            n_cal = max(2, min(len(idx_all) - 1, n_cal))
            calib_idx = idx_all[:n_cal]
            test_idx = idx_all[n_cal:]
        else:
            calib_idx = idx_all
            test_idx = idx_all

        x_cal = A_preds[calib_idx]
        y_cal = A_trues[calib_idx]

        if cal_method == "affine":
            cal_k, cal_b = np.polyfit(x_cal, y_cal, 1)
            cal_k, cal_b = float(cal_k), float(cal_b)
        elif cal_method == "through1":
            # Constrained affine calibration that enforces (Â=1) -> (A=1).
            # Equivalent to scaling predicted degradation with zero intercept.
            x0 = (x_cal - 1.0).astype(np.float64)
            y0 = (y_cal - 1.0).astype(np.float64)
            denom = float(np.dot(x0, x0))
            if denom <= 1e-12:
                cal_k = 1.0
            else:
                cal_k = float(np.dot(x0, y0) / denom)
            cal_b = float(1.0 - cal_k)
        else:
            raise ValueError(f"Unsupported calibration method: {cal_method}")

        A_preds_cal = np.clip(cal_k * A_preds + cal_b, 0.0, 1.0)

        raw_test = _compute_metrics(A_trues[test_idx], A_preds[test_idx])
        cal_all = _compute_metrics(A_trues, A_preds_cal)
        cal_test = _compute_metrics(A_trues[test_idx], A_preds_cal[test_idx])

        if args.save_calibration is None:
            args.save_calibration = f"results/accuracy_modeling/{args.model}/calibration_{args.model}_{quant_mode}.json"
        cal_path = Path(args.save_calibration)
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "quant_mode": quant_mode,
                    "per_channel": bool(per_channel),
                    "method": cal_method,
                    "k": float(cal_k),
                    "b": float(cal_b),
                    "calib_frac": float(args.calib_frac),
                    "n_total": int(len(A_preds)),
                    "n_calib": int(len(calib_idx)),
                    "n_test": int(len(test_idx)),
                    "metrics_raw_all": raw_all,
                    "metrics_raw_test": raw_test,
                    "metrics_cal_all": cal_all,
                    "metrics_cal_test": cal_test,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # 动态设置默认输出路径（如果未指定）
    if args.output_csv is None:
        args.output_csv = f"results/accuracy_modeling/{args.model}/validation_{args.model}_{quant_mode}.csv"
    if args.output_fig is None:
        args.output_fig = f"results/accuracy_modeling/{args.model}/validation_{args.model}_{quant_mode}.pdf"
    
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    # Add calibrated columns (identity if calibration disabled)
    for i in range(len(rows)):
        rows[i]["A_pred_cal"] = float(A_preds_cal[i])
        rows[i]["abs_err_cal"] = float(abs(float(rows[i]["A_true"]) - float(A_preds_cal[i])))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["idx", "A_pred", "A_pred_cal", "A_true", "abs_err", "abs_err_cal", "config_json"],
        )
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Configurations evaluated: {len(A_preds)}")
    print(f"  - Corner cases: {num_corners}")
    print(f"  - Random configs: {len(A_preds) - num_corners}")
    print()
    print("Prediction Error Metrics:")
    print(f"  MAE (Mean Abs Error)    : {mae:.6f}")
    print(f"  RMSE (Root Mean Sq Err) : {rmse:.6f}")
    print(f"  Max Abs Error           : {max_abs_err:.6f}")
    print(f"  Mean Error (Bias)       : {mean_err:+.6f}")
    print(f"  Std Error               : {std_err:.6f}")
    print()
    print("Goodness of Fit:")
    print(f"  R^2 (Coeff of Determ)   : {r2:.6f}")
    print(f"  Pearson Correlation     : {pearson_r:.6f}")
    print()
    print("Degradation Range:")
    print(f"  Predicted: [{deg_range_pred[0]:.4f}, {deg_range_pred[1]:.4f}]")
    print(f"  True:      [{deg_range_true[0]:.4f}, {deg_range_true[1]:.4f}]")
    print()
    print(f"CSV saved to: {out_csv}")
    if cal_method != "none":
        print("\n" + "-" * 70)
        print("Post-hoc Calibration (for bias correction)")
        print("-" * 70)
        print(f"  method: {cal_method} | k={cal_k:.4f}, b={cal_b:+.4f}")
        if 0.0 < float(args.calib_frac) < 1.0:
            print(f"  split : calib={len(calib_idx)} | test={len(test_idx)} (calib_frac={float(args.calib_frac):.2f})")
        else:
            print(f"  split : fit on all configs (n={len(A_preds)})")
        if raw_test is not None and cal_test is not None:
            print("  metrics (test subset):")
            print(f"    - raw: MAE={raw_test['mae']:.4f}, R^2={raw_test['r2']:.4f}")
            print(f"    - cal: MAE={cal_test['mae']:.4f}, R^2={cal_test['r2']:.4f}")
        if cal_all is not None:
            print("  metrics (all configs):")
            print(f"    - cal: MAE={cal_all['mae']:.4f}, R^2={cal_all['r2']:.4f}")
        if args.save_calibration is not None:
            print(f"  calibration JSON: {args.save_calibration}")

    # Plot (for paper - TMC style, black-white friendly)
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator

        out_fig = Path(args.output_fig)
        out_fig.parent.mkdir(parents=True, exist_ok=True)

        # IEEE/TMC publication style settings
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "axes.linewidth": 0.8,
        })

        # Single column IEEE figure: 3.5 inches wide
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        # Data range
        mn = float(min(A_preds.min(), A_trues.min()))
        mx = float(max(A_preds.max(), A_trues.max()))
        margin = 0.005
        xs = np.linspace(mn - margin, mx + margin, 200)

        # Ideal line (y = x) - thick black
        ax.plot(xs, xs, "k-", linewidth=1.8, label="Ideal", zorder=3)

        # Best-fit line - dashed red
        if len(A_preds) >= 2:
            k, b = np.polyfit(A_preds, A_trues, 1)
            ax.plot(xs, k * xs + b, "r--", linewidth=1.5,
                    label=f"Fit ($y={k:.2f}x{b:+.2f}$)", zorder=2)

        # Scatter plot - simple gray with black edge for print clarity
        ax.scatter(
            A_preds, A_trues,
            c="lightsteelblue",
            s=18,
            alpha=0.7,
            edgecolors="gray",
            linewidths=0.3,
            zorder=1,
            label="Samples"
        )

        # Axis limits
        ax.set_xlim(mn - margin, mx + margin)
        ax.set_ylim(mn - margin, mx + margin)
        ax.set_aspect("equal", adjustable="box")

        # Labels (use \hat{A} for predicted, A for true)
        ax.set_xlabel(r"Predicted Accuracy $\hat{A}$", fontsize=10)
        ax.set_ylabel(r"True Accuracy $A$", fontsize=10)

        # Minor ticks for precision
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='both', direction='in', top=True, right=True)

        # Legend - upper left, compact
        ax.legend(loc="upper left", frameon=True, framealpha=0.95, 
                  edgecolor="gray", fancybox=False)

        # Statistics annotation - lower right corner
        stats_text = (
            f"$n = {len(A_preds)}$\n"
            f"$R^2 = {r2:.4f}$\n"
            f"MAE $= {mae:.4f}$"
        )
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                fontsize=8.5, verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                         edgecolor="gray", linewidth=0.8, alpha=0.9))

        plt.tight_layout(pad=0.3)

        # Save PDF (vector, best for paper)
        plt.savefig(out_fig, dpi=300, bbox_inches="tight", format="pdf")
        print(f"Figure (PDF) saved to: {out_fig}")

        # Also save high-res PNG
        png_path = out_fig.with_suffix(".png")
        plt.savefig(png_path, dpi=600, bbox_inches="tight")
        print(f"Figure (PNG) saved to: {png_path}")

        # Save EPS for some journals
        eps_path = out_fig.with_suffix(".eps")
        plt.savefig(eps_path, dpi=300, bbox_inches="tight", format="eps")
        print(f"Figure (EPS) saved to: {eps_path}")

        plt.close()

        # Calibrated parity plot (optional)
        if cal_method != "none":
            out_fig_cal = out_fig.with_name(out_fig.stem + "_cal" + out_fig.suffix)

            # Prefer plotting the test subset when hold-out split is enabled
            if 0.0 < float(args.calib_frac) < 1.0 and len(test_idx) >= 2 and cal_test is not None:
                x_plot = A_preds_cal[test_idx]
                y_plot = A_trues[test_idx]
                r2_plot = float(cal_test["r2"])
                mae_plot = float(cal_test["mae"])
            else:
                x_plot = A_preds_cal
                y_plot = A_trues
                r2_plot = float(cal_all["r2"]) if cal_all is not None else float("nan")
                mae_plot = float(cal_all["mae"]) if cal_all is not None else float("nan")

            fig, ax = plt.subplots(figsize=(3.5, 3.5))

            mn = float(min(x_plot.min(), y_plot.min()))
            mx = float(max(x_plot.max(), y_plot.max()))
            margin = 0.005
            xs = np.linspace(mn - margin, mx + margin, 200)

            ax.plot(xs, xs, "k-", linewidth=1.8, label="Ideal", zorder=3)

            if len(x_plot) >= 2:
                k, b = np.polyfit(x_plot, y_plot, 1)
                ax.plot(xs, k * xs + b, "r--", linewidth=1.5,
                        label=f"Fit ($y={k:.2f}x{b:+.2f}$)", zorder=2)

            ax.scatter(
                x_plot, y_plot,
                c="lightsteelblue",
                s=18,
                alpha=0.7,
                edgecolors="gray",
                linewidths=0.3,
                zorder=1,
                label="Samples"
            )

            ax.set_xlim(mn - margin, mx + margin)
            ax.set_ylim(mn - margin, mx + margin)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"Calibrated Prediction $\hat{A}_{\mathrm{cal}}$", fontsize=10)
            ax.set_ylabel(r"True Accuracy $A$", fontsize=10)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which='both', direction='in', top=True, right=True)

            ax.legend(loc="upper left", frameon=True, framealpha=0.95,
                      edgecolor="gray", fancybox=False, title=f"cal={cal_method}")

            stats_text = (
                f"$n = {len(x_plot)}$\n"
                f"$R^2 = {r2_plot:.4f}$\n"
                f"MAE $= {mae_plot:.4f}$"
            )
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                    fontsize=8.5, verticalalignment="bottom", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="gray", linewidth=0.8, alpha=0.9))

            plt.tight_layout(pad=0.3)

            plt.savefig(out_fig_cal, dpi=300, bbox_inches="tight", format="pdf")
            print(f"Figure (calibrated, PDF) saved to: {out_fig_cal}")

            png_path = out_fig_cal.with_suffix(".png")
            plt.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Figure (calibrated, PNG) saved to: {png_path}")

            eps_path = out_fig_cal.with_suffix(".eps")
            plt.savefig(eps_path, dpi=300, bbox_inches="tight", format="eps")
            print(f"Figure (calibrated, EPS) saved to: {eps_path}")

            plt.close()

    except Exception as e:
        import traceback
        print(f"(Skip plotting) matplotlib not available or failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

