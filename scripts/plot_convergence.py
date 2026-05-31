"""
Plot academic-quality RL convergence figures for TMC paper.

Reads metrics_long.csv from each run and generates:
  1. Main 2x2 paper figure  (--main_only)
  2. Per-weight comparison  (4 models, 1x4)
  3. Per-model comparison   (4 weights, 1x4)

Recommended usage:
  - Final paper figure with error band:
      --log_dir logs/training_cloud/multi_seed
  - Single-seed weight-selection figures:
      --single_seed_log_dir logs/training_cloud/data_for_select_weights

When multiple seeded runs are found for the same `(model, weight)` pair
(e.g., `split_w19_s42`, `split_w19_s123`, `split_w19_s456`, ...),
the script automatically aggregates them into:
  - mean convergence curve
  - shaded uncertainty band (std / sem / 95% CI)

Each figure is saved in both PDF (paper) and PNG (preview) formats.

Usage:
    # Main paper figure, w19 preset, first 1600 iterations:
    python scripts/plot_convergence.py \\
        --log_dir logs/training_cloud --main_only \\
        --main_weight w19 --max_iter 1600 --style ieee

    # Try shaded panel background (analysis only, not for TMC submission):
    python scripts/plot_convergence.py \\
        --log_dir logs/training_cloud --main_only \\
        --main_weight w19 --max_iter 1600 --style ieee_shaded

    # Grayscale (for B&W printing check):
    python scripts/plot_convergence.py \\
        --log_dir logs/training_cloud --main_only \\
        --main_weight w19 --max_iter 1600 --style grayscale

    # All auxiliary plots:
    python scripts/plot_convergence.py --log_dir logs/training_cloud --smooth 10
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Base rcParams applied before any style ───────────────────────────────────
_BASE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 800,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.55,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.45,
}

# ── Style presets ─────────────────────────────────────────────────────────────
# Each style dict overrides rcParams and provides extra visual parameters.
STYLES: dict[str, dict[str, Any]] = {
    # Default for TMC paper submission: clean white background, low-saturation colors.
    "ieee": {
        "rc": {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.alpha": 0.22,
            "grid.color": "#BFC4C9",
        },
        "panel_bg": None,           # No panel background tint
        "colors": {
            "split": "#3B6FB6",
            "mtan":  "#C44E52",
            "dense": "#2A9D8F",
            "cross": "#6C5B7B",
        },
        "linestyles": {
            "split": "-",
            "mtan":  "--",
            "dense": "-.",
            "cross": (0, (3.0, 1.2, 1.0, 1.2)),
        },
        "markers": {
            "split": "o",
            "mtan":  "s",
            "dense": "^",
            "cross": "D",
        },
        "linewidth": 1.55,
        "markersize": 3.0,
        "markerfacecolor": "white",
        "markeredgewidth": 0.85,
        "band_alpha": 0.18,
    },
    # With subtle panel background — useful for analysis/slides, NOT for TMC submission.
    "ieee_shaded": {
        "rc": {
            "axes.facecolor": "#F5F6F7",
            "figure.facecolor": "white",
            "grid.alpha": 0.30,
            "grid.color": "#D0D4D8",
        },
        "panel_bg": "#F5F6F7",
        "colors": {
            "split": "#3B6FB6",
            "mtan":  "#C44E52",
            "dense": "#2A9D8F",
            "cross": "#6C5B7B",
        },
        "linestyles": {
            "split": "-",
            "mtan":  "--",
            "dense": "-.",
            "cross": (0, (3.0, 1.2, 1.0, 1.2)),
        },
        "markers": {
            "split": "o",
            "mtan":  "s",
            "dense": "^",
            "cross": "D",
        },
        "linewidth": 1.55,
        "markersize": 3.0,
        "markerfacecolor": "white",
        "markeredgewidth": 0.85,
        "band_alpha": 0.20,
    },
    # Grayscale — for checking B&W print readability.
    "grayscale": {
        "rc": {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.alpha": 0.30,
            "grid.color": "#AAAAAA",
        },
        "panel_bg": None,
        "colors": {
            "split": "#000000",
            "mtan":  "#555555",
            "dense": "#888888",
            "cross": "#BBBBBB",
        },
        "linestyles": {
            "split": "-",
            "mtan":  "--",
            "dense": "-.",
            "cross": ":",
        },
        "markers": {
            "split": "o",
            "mtan":  "s",
            "dense": "^",
            "cross": "D",
        },
        "linewidth": 1.6,
        "markersize": 3.5,
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
        "band_alpha": 0.16,
    },
}

# ── Fixed display metadata ────────────────────────────────────────────────────
MODEL_DISPLAY = {
    "split": "Hard-Split",
    "mtan":  "MTAN",
    "cross": "Cross-Stitch",
    "dense": "Dense-Soft",
}
MODEL_ORDER = ["split", "mtan", "dense", "cross"]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

WEIGHT_DISPLAY = {
    "w19": r"$\omega$=(0.1, 0.9)",
    "w28": r"$\omega$=(0.2, 0.8)",
    "w37": r"$\omega$=(0.3, 0.7)",
    "w55": r"$\omega$=(0.5, 0.5)",
}
WEIGHT_COLORS = {
    "w19": "#d62728",
    "w28": "#ff7f0e",
    "w37": "#2ca02c",
    "w55": "#1f77b4",
}

METRIC_CFG = {
    "metrics/reward":   {"label": "Average Reward",
                         "ylabel": "Reward"},
    "metrics/delay":    {"label": "Inference Delay (s)",
                         "ylabel": "Delay (s)"},
    "metrics/energy":   {"label": r"Energy per Request $E_{\mathrm{req}}$ (J)",
                         "ylabel": "Energy (J)"},
    "metrics/accuracy": {"label": "Inference Accuracy",
                         "ylabel": "Accuracy"},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def apply_style(style_name: str) -> dict[str, Any]:
    """Apply rcParams for chosen style and return the style config dict."""
    if style_name not in STYLES:
        raise ValueError(f"Unknown style '{style_name}'. Choose from: {list(STYLES)}")
    cfg = STYLES[style_name]
    mpl.rcParams.update(_BASE_RC)
    mpl.rcParams.update(cfg["rc"])
    return dict(cfg)


def save_figure(fig: plt.Figure, base_path: Path, formats: tuple[str, ...] = (".pdf", ".png")) -> None:
    for suffix in formats:
        p = base_path.with_suffix(suffix)
        fig.savefig(p)
        print(f"  Saved: {p}")


def load_metric(csv_path: Path, metric_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    sub = df[df["key"] == metric_key][["step", "value"]].copy()
    return sub.sort_values("step").reset_index(drop=True)


def clip_metric(sub: pd.DataFrame, max_iter: int | None) -> pd.DataFrame:
    if max_iter is None:
        return sub
    return sub[sub["step"] <= max_iter].reset_index(drop=True)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()


def parse_run_tag(run_name: str) -> tuple[str, str, str | None] | None:
    """
    Parse run names like:
      - split_w19
      - split_w19_s123
      - split_w19_proposed_s123              (algo segment in the middle)
    """
    m = re.match(r"^(split|mtan|cross|dense)_(w\d+)(?:_s(\d+))?$", run_name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.match(r"^(split|mtan|cross|dense)_(w\d+)_.+_s(\d+)$", run_name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def discover_runs(log_dir: Path) -> dict:
    """
    Discover all runs and group them by (model, weight).

    Returns:
        {
            ("split", "w19"): [Path(.../split_w19/metrics_long.csv),
                               Path(.../split_w19_s123/metrics_long.csv),
                               ...],
            ...
        }
    """
    runs_raw = defaultdict(list)
    for csv_path in sorted(log_dir.rglob("metrics_long.csv")):
        parsed = parse_run_tag(csv_path.parent.name)
        if parsed is None:
            continue
        model, weight, seed = parsed
        runs_raw[(model, weight)].append((seed, csv_path))

    runs = {}
    for key, items in sorted(runs_raw.items()):
        seeded = sorted([p for seed, p in items if seed is not None])
        legacy = sorted([p for seed, p in items if seed is None])
        if seeded:
            if legacy:
                print(
                    f"  [note] Ignoring {len(legacy)} legacy unseeded run(s) for "
                    f"{key[0]}/{key[1]} because seeded runs are available."
                )
            runs[key] = seeded
        else:
            runs[key] = legacy
    return runs


def load_metric_aggregate(
    csv_paths: list[Path],
    metric_key: str,
    smooth_w: int,
    max_iter: int | None,
) -> pd.DataFrame:
    """
    Aggregate one metric across multiple runs of the same model-weight pair.

    We align runs on the common step range (inner join) so that the plotted
    mean/std band always corresponds to the same PPO iterations across seeds.
    """
    series_list = []
    for idx, csv_path in enumerate(sorted(csv_paths)):
        sub = clip_metric(load_metric(csv_path, metric_key), max_iter)
        if sub.empty:
            continue
        y = smooth(sub["value"].to_numpy(), smooth_w)
        s = pd.Series(y, index=sub["step"].to_numpy(), name=f"run_{idx}")
        s = s[~s.index.duplicated(keep="last")]
        series_list.append(s)

    if not series_list:
        return pd.DataFrame(columns=["step", "mean", "std", "count"])

    aligned = pd.concat(series_list, axis=1, join="inner").sort_index()
    aligned = aligned.dropna(how="any")
    if aligned.empty:
        return pd.DataFrame(columns=["step", "mean", "std", "count"])

    return pd.DataFrame(
        {
            "step": aligned.index.to_numpy(dtype=int),
            "mean": aligned.mean(axis=1).to_numpy(),
            "std": aligned.std(axis=1, ddof=1).fillna(0.0).to_numpy(),
            "count": aligned.count(axis=1).to_numpy(dtype=int),
        }
    )


def ordered_models_for_weight(runs: dict, weight_tag: str) -> list[str]:
    return [m for m in MODEL_ORDER if (m, weight_tag) in runs]


def style_axis(ax: plt.Axes, style_cfg: dict) -> None:
    """Apply per-axis cosmetics consistent with chosen style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    # y-margin only; x-margin is handled via xlim to prevent marker clipping
    ax.margins(y=0.12)
    if style_cfg.get("panel_bg"):
        ax.set_facecolor(style_cfg["panel_bg"])


def compute_band_radius(std: np.ndarray, count: np.ndarray, band_mode: str) -> np.ndarray | None:
    n = np.maximum(count.astype(float), 1.0)
    if band_mode == "none":
        return None
    if band_mode == "std":
        return std
    if band_mode == "sem":
        return std / np.sqrt(n)
    if band_mode == "ci95":
        return 1.96 * std / np.sqrt(n)
    raise ValueError(f"Unknown band mode: {band_mode}")


def band_bounds(metric_key: str, center: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = center - radius
    upper = center + radius
    if metric_key in {"metrics/delay", "metrics/energy", "metrics/accuracy"}:
        lower = np.maximum(lower, 0.0)
    if metric_key == "metrics/accuracy":
        upper = np.minimum(upper, 1.0)
    return lower, upper


def add_shared_legend(fig, handles, labels, position: str, ncol: int) -> list:
    """Place a single shared legend; return tight_layout rect."""
    common = dict(ncol=ncol, frameon=False, columnspacing=1.3, handlelength=2.2)
    if position == "bottom":
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.01), **common)
        return [0, 0.06, 1, 1.0]
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), **common)
    return [0, 0, 1, 0.93]


# ── Main 2×2 paper figure ─────────────────────────────────────────────────────

def plot_tmc_main_figure(
    runs: dict,
    weight_tag: str,
    out_dir: Path,
    smooth_w: int,
    style_cfg: dict,
    max_iter: int | None = None,
    legend_position: str = "top",
    band_mode: str = "std",
    save_formats: tuple[str, ...] = (".pdf", ".png"),
    file_stem: str = "fig_convergence_main",
) -> None:
    """2×2 convergence figure — one weight, all four models."""
    metrics = list(METRIC_CFG.keys())
    models = ordered_models_for_weight(runs, weight_tag)
    if not models:
        print(f"  [warn] No runs found for weight={weight_tag}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.40))
    axes = axes.flatten()

    for idx, (ax, mk) in enumerate(zip(axes, metrics)):
        cfg = METRIC_CFG[mk]
        for model in models:
            agg = load_metric_aggregate(runs[(model, weight_tag)], mk, smooth_w, max_iter)
            if agg.empty:
                continue
            x = agg["step"].to_numpy()
            y = agg["mean"].to_numpy()
            y_std = agg["std"].to_numpy()
            y_count = agg["count"].to_numpy()
            run_count = int(np.max(y_count)) if len(y_count) else 0
            n = len(x)
            markevery = max(n // 6, 1)
            color = style_cfg["colors"].get(model, "gray")
            band_radius = compute_band_radius(y_std, y_count, band_mode)
            if run_count > 1 and band_radius is not None:
                lower, upper = band_bounds(mk, y, band_radius)
                ax.fill_between(
                    x,
                    lower,
                    upper,
                    color=color,
                    alpha=style_cfg.get("band_alpha", 0.13),
                    linewidth=0.0,
                    zorder=1,
                )
            ax.plot(
                x, y,
                color=color,
                linestyle=style_cfg["linestyles"].get(model, "-"),
                marker=style_cfg["markers"].get(model),
                markevery=markevery,
                markersize=style_cfg["markersize"],
                markerfacecolor=style_cfg["markerfacecolor"],
                markeredgewidth=style_cfg["markeredgewidth"],
                linewidth=style_cfg["linewidth"],
                alpha=0.97,
                solid_capstyle="round",
                dash_capstyle="round",
                label=MODEL_DISPLAY.get(model, model),
                clip_on=True,
                zorder=2,
            )

        ax.set_title(f"{PANEL_LABELS[idx]}  {cfg['label']}", pad=5.0)
        ax.set_ylabel(cfg["ylabel"])
        ax.set_xlabel("Training Iteration")

        # Lock only left boundary; right auto-scales with y.margins, ensuring
        # the last marker symbol is never clipped at x = max_iter.
        if max_iter is not None:
            ax.set_xlim(left=0, right=max_iter * 1.04)
            # With larger fonts, 200-step ticks overlap; use 400-step + endpoint.
            ax.set_xticks([0, 400, 800, 1200, max_iter])
        else:
            ax.set_xlim(left=0)

        style_axis(ax, style_cfg)

    handles, labels = axes[0].get_legend_handles_labels()
    rect = add_shared_legend(fig, handles, labels, legend_position, len(models))
    fig.tight_layout(rect=rect, h_pad=1.0, w_pad=0.8)

    save_figure(fig, out_dir / file_stem, formats=save_formats)
    plt.close(fig)


# ── Auxiliary plots ───────────────────────────────────────────────────────────

def plot_convergence_by_weight(
    runs: dict,
    weight_tag: str,
    out_dir: Path,
    smooth_w: int,
    style_cfg: dict,
    max_iter: int | None = None,
    legend_position: str = "top",
    band_mode: str = "std",
    save_formats: tuple[str, ...] = (".pdf", ".png"),
    file_stem: str | None = None,
) -> None:
    """1×4 figure: 4 metrics, 4 model curves for one weight."""
    metrics = list(METRIC_CFG.keys())
    models = ordered_models_for_weight(runs, weight_tag)
    if not models:
        return

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.85))
    for ax, mk in zip(axes, metrics):
        cfg = METRIC_CFG[mk]
        for model in models:
            agg = load_metric_aggregate(runs[(model, weight_tag)], mk, smooth_w, max_iter)
            if agg.empty:
                continue
            x = agg["step"].to_numpy()
            y = agg["mean"].to_numpy()
            y_std = agg["std"].to_numpy()
            y_count = agg["count"].to_numpy()
            run_count = int(np.max(y_count)) if len(y_count) else 0
            color = style_cfg["colors"].get(model, "gray")
            band_radius = compute_band_radius(y_std, y_count, band_mode)
            if run_count > 1 and band_radius is not None:
                lower, upper = band_bounds(mk, y, band_radius)
                ax.fill_between(
                    x,
                    lower,
                    upper,
                    color=color,
                    alpha=style_cfg.get("band_alpha", 0.13),
                    linewidth=0.0,
                )
            ax.plot(x, y,
                    color=color,
                    linestyle=style_cfg["linestyles"].get(model, "-"),
                    linewidth=style_cfg["linewidth"],
                    label=MODEL_DISPLAY.get(model, model))
        ax.set_xlabel("Iteration")
        ax.set_ylabel(cfg["ylabel"])
        if max_iter is not None:
            ax.set_xlim(left=0, right=max_iter * 1.04)
        style_axis(ax, style_cfg)

    handles, labels = axes[0].get_legend_handles_labels()
    rect = add_shared_legend(fig, handles, labels, legend_position, len(models))
    fig.tight_layout(rect=rect)
    save_figure(fig, out_dir / (file_stem or f"convergence_{weight_tag}"), formats=save_formats)
    plt.close(fig)


def plot_convergence_by_model(
    runs: dict,
    model: str,
    out_dir: Path,
    smooth_w: int,
    style_cfg: dict,
    max_iter: int | None = None,
    legend_position: str = "top",
    band_mode: str = "std",
    save_formats: tuple[str, ...] = (".pdf", ".png"),
    file_stem: str | None = None,
) -> None:
    """1×4 figure: 4 metrics, 4 weight curves for one model."""
    metrics = list(METRIC_CFG.keys())
    weights = sorted({w for m, w in runs if m == model})
    if not weights:
        return

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.85))
    for ax, mk in zip(axes, metrics):
        cfg = METRIC_CFG[mk]
        for wt in weights:
            agg = load_metric_aggregate(runs[(model, wt)], mk, smooth_w, max_iter)
            if agg.empty:
                continue
            x = agg["step"].to_numpy()
            y = agg["mean"].to_numpy()
            y_std = agg["std"].to_numpy()
            y_count = agg["count"].to_numpy()
            run_count = int(np.max(y_count)) if len(y_count) else 0
            color = WEIGHT_COLORS.get(wt, "gray")
            band_radius = compute_band_radius(y_std, y_count, band_mode)
            if run_count > 1 and band_radius is not None:
                lower, upper = band_bounds(mk, y, band_radius)
                ax.fill_between(
                    x,
                    lower,
                    upper,
                    color=color,
                    alpha=style_cfg.get("band_alpha", 0.13),
                    linewidth=0.0,
                )
            ax.plot(x, y,
                    color=color,
                    linewidth=style_cfg["linewidth"],
                    label=WEIGHT_DISPLAY.get(wt, wt))
        ax.set_xlabel("Iteration")
        ax.set_ylabel(cfg["ylabel"])
        if max_iter is not None:
            ax.set_xlim(left=0, right=max_iter * 1.04)
        style_axis(ax, style_cfg)

    handles, labels = axes[0].get_legend_handles_labels()
    rect = add_shared_legend(fig, handles, labels, legend_position, len(weights))
    fig.tight_layout(rect=rect)
    save_figure(fig, out_dir / (file_stem or f"convergence_{model}"), formats=save_formats)
    plt.close(fig)


def export_single_seed_weight_pngs(
    runs: dict,
    out_dir: Path,
    smooth_w: int,
    style_cfg: dict,
    max_iter: int | None = None,
    legend_position: str = "top",
) -> None:
    """Export one single-seed PNG per weight for weight-selection analysis."""
    weights = sorted({w for _, w in runs})
    if not weights:
        print("  [warn] No single-seed runs found for weight-selection export.")
        return

    root = out_dir / "single_seed_weights"
    for wt in weights:
        wt_out = root / wt
        wt_out.mkdir(parents=True, exist_ok=True)
        plot_convergence_by_weight(
            runs=runs,
            weight_tag=wt,
            out_dir=wt_out,
            smooth_w=smooth_w,
            style_cfg=style_cfg,
            max_iter=max_iter,
            legend_position=legend_position,
            band_mode="none",
            save_formats=(".png",),
            file_stem=f"convergence_{wt}_single_seed",
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TMC-grade RL convergence figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log_dir",        type=str, default="logs/training_cloud")
    parser.add_argument("--out_dir",        type=str, default="results/figures")
    parser.add_argument("--smooth",         type=int, default=10,
                        help="Moving-average window")
    parser.add_argument("--max_iter",       type=int, default=None,
                        help="Only plot steps <= max_iter")
    parser.add_argument("--band_mode",      type=str, default="std",
                        choices=["std", "sem", "ci95", "none"],
                        help="Shaded band type across seeds")
    parser.add_argument("--band_alpha",     type=float, default=None,
                        help="Override shaded band transparency, e.g. 0.22")
    parser.add_argument("--style",          type=str, default="ieee",
                        choices=list(STYLES),
                        help="Visual style preset")
    parser.add_argument("--legend_position", choices=["top", "bottom"], default="top")
    parser.add_argument("--main_weight",    type=str, default="w19",
                        help="Weight tag for the 2×2 paper figure")
    parser.add_argument("--main_only",      action="store_true",
                        help="Only generate the 2×2 main figure and exit")
    parser.add_argument("--weights",        type=str, default=None,
                        help="Aux plot: only this weight tag")
    parser.add_argument("--model",          type=str, default=None,
                        help="Aux plot: only this model")
    parser.add_argument("--single_seed_log_dir", type=str, default=None,
                        help="Optional log root for exporting single-seed weight-selection PNGs")
    parser.add_argument("--save_single_seed_weight_pngs", action="store_true",
                        help="Export one PNG per weight from single_seed_log_dir")
    args = parser.parse_args()

    style_cfg = apply_style(args.style)
    if args.band_alpha is not None:
        style_cfg["band_alpha"] = float(args.band_alpha)

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if log_dir.name == "training_cloud" and (log_dir / "multi_seed").exists() and (log_dir / "data_for_select_weights").exists():
        print(
            "  [note] Mixed root detected. Prefer --log_dir logs/training_cloud/multi_seed "
            "for final shaded figures, and --single_seed_log_dir "
            "logs/training_cloud/data_for_select_weights for single-seed selection plots."
        )

    runs = discover_runs(log_dir)
    total_csv = sum(len(v) for v in runs.values())
    print(f"Found {len(runs)} model-weight groups / {total_csv} run files in {log_dir}")
    for k, v in sorted(runs.items()):
        print(f"  {k[0]:6s}  {k[1]:4s}  -> {len(v)} run(s)")

    kw = dict(smooth_w=args.smooth, style_cfg=style_cfg,
              max_iter=args.max_iter, legend_position=args.legend_position,
              band_mode=args.band_mode)

    print(f"\nGenerating figures  [style={args.style}  smooth={args.smooth}"
          f"  max_iter={args.max_iter}  band={args.band_mode}]")

    plot_tmc_main_figure(runs, args.main_weight, out_dir, **kw)

    if args.save_single_seed_weight_pngs:
        single_seed_log_dir = Path(args.single_seed_log_dir or "logs/training_cloud/data_for_select_weights")
        print(f"\nGenerating single-seed weight-selection PNGs from {single_seed_log_dir}")
        single_seed_runs = discover_runs(single_seed_log_dir)
        export_single_seed_weight_pngs(
            runs=single_seed_runs,
            out_dir=out_dir,
            smooth_w=args.smooth,
            style_cfg=style_cfg,
            max_iter=args.max_iter,
            legend_position=args.legend_position,
        )

    if args.main_only:
        print(f"\nDone. Figures saved to {out_dir}/")
        return

    all_weights = sorted({w for _, w in runs})
    all_models  = sorted({m for m, _ in runs})

    for wt in ([args.weights] if args.weights else all_weights):
        plot_convergence_by_weight(runs, wt, out_dir, **kw)

    for model in ([args.model] if args.model else all_models):
        plot_convergence_by_model(runs, model, out_dir, **kw)

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
