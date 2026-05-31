"""
Plot TMC-grade per-model bar figures for the E2 cross-model main table.

For each of the four DNN models (split / mtan / dense / cross), this script
reads five JSON files from `--eval_dir` and produces a single-column figure
with two stacked subplots:

    top    : mean inference delay  across the 5 algorithms (with std error bar)
    bottom : mean energy per request across the 5 algorithms (with std error bar)

Algorithm color/hatch palette is shared across all four figures so the
reader gets a consistent algo → color mapping.

Usage:
    python scripts/plot_cross_model_e2.py \
        --eval_dir results/eval/e2_cross_model_w19 \
        --out_dir  results/figures/e2_cross_model \
        --style    ieee
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ── Base rcParams (consistent with plot_convergence.py) ──────────────────────
_BASE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "figure.dpi": 150,
    "savefig.dpi": 800,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linestyle": "--",
    "grid.linewidth": 0.45,
}


STYLES: dict[str, dict[str, Any]] = {
    "ieee": {
        "rc": {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.alpha": 0.25,
            "grid.color": "#BFC4C9",
        },
        # Algorithm palette: shared across all 4 model figures
        # so that color ↔ algorithm mapping is stable for the reader.
        "colors": {
            "local_only":             "#8C8C8C",  # gray
            "single_split":           "#5B8FF9",  # blue
            "partition_routing_only": "#E07B00",  # orange
            "mlp_ppo":                "#B45ABD",  # purple
            "proposed":               "#C44E52",  # red  (ours, highlighted)
        },
        "hatches": {
            "local_only":             "//",
            "single_split":           "\\\\",
            "partition_routing_only": "++",
            "mlp_ppo":                "xx",
            "proposed":               "",       # solid — make "ours" stand out
        },
        "edgecolor": "black",
        "bar_linewidth": 0.6,
    },
    "grayscale": {
        "rc": {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.alpha": 0.30,
            "grid.color": "#AAAAAA",
        },
        "colors": {
            "local_only":             "#DDDDDD",
            "single_split":           "#BBBBBB",
            "partition_routing_only": "#888888",
            "mlp_ppo":                "#555555",
            "proposed":               "#222222",
        },
        "hatches": {
            "local_only":             "//",
            "single_split":           "\\\\",
            "partition_routing_only": "++",
            "mlp_ppo":                "xx",
            "proposed":               "",
        },
        "edgecolor": "black",
        "bar_linewidth": 0.8,
    },
}


MODEL_DISPLAY = {
    "split": "Hard-Split",
    "mtan":  "MTAN",
    "dense": "Dense-Soft",
    "cross": "Cross-Stitch",
}
MODEL_ORDER = ["split", "mtan", "dense", "cross"]

ALGO_DISPLAY = {
    "local_only":             "Local",
    "single_split":           "S-Split",
    "partition_routing_only": "P+R",
    "mlp_ppo":                "MLP-PPO",
    "proposed":               "Proposed",
}
ALGO_ORDER = [
    "local_only",
    "single_split",
    "partition_routing_only",
    "mlp_ppo",
    "proposed",
]

METRICS = [
    {
        "key":   "mean_delay",
        "std":   "std_delay",
        "title": "(a) Inference Delay",
        "ylabel": "Delay (s)",
        "value_fmt": "{:.2f}",
    },
    {
        "key":   "mean_energy",
        "std":   "std_energy",
        "title": "(b) Energy per Request",
        "ylabel": r"$E_{\mathrm{req}}$ (J)",
        "value_fmt": "{:.2f}",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def apply_style(style_name: str) -> dict[str, Any]:
    if style_name not in STYLES:
        raise ValueError(f"Unknown style '{style_name}'. Choose from: {list(STYLES)}")
    cfg = STYLES[style_name]
    mpl.rcParams.update(_BASE_RC)
    mpl.rcParams.update(cfg["rc"])
    return dict(cfg)


def load_one(eval_dir: Path, model: str, weight_tag: str, algo: str) -> dict | None:
    p = eval_dir / f"{model}_{weight_tag}_{algo}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_model(eval_dir: Path, model: str, weight_tag: str) -> dict[str, dict]:
    """Return {algo: payload_dict} for one model across all 5 algorithms."""
    out: dict[str, dict] = {}
    for algo in ALGO_ORDER:
        payload = load_one(eval_dir, model, weight_tag, algo)
        if payload is None:
            print(f"  [warn] Missing JSON: {model}/{weight_tag}/{algo}")
            continue
        out[algo] = payload
    return out


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)


def draw_metric_bars(
    ax: plt.Axes,
    payloads: dict[str, dict],
    metric_key: str,
    std_key: str,
    style_cfg: dict,
    title: str,
    ylabel: str,
    value_fmt: str,
    annotate: bool,
) -> None:
    algos = [a for a in ALGO_ORDER if a in payloads]
    means = np.array([float(payloads[a].get(metric_key, np.nan)) for a in algos])
    # Note: std_key intentionally ignored. Mixing tail-iter PPO std with
    # deterministic-baseline episode std on the same bar group is misleading;
    # we render bare bars and discuss numerical robustness in the text.
    _ = std_key

    xs = np.arange(len(algos))
    colors = [style_cfg["colors"].get(a, "#888") for a in algos]
    hatches = [style_cfg["hatches"].get(a, "") for a in algos]

    bars = ax.bar(
        xs, means,
        color=colors,
        edgecolor=style_cfg.get("edgecolor", "black"),
        linewidth=style_cfg.get("bar_linewidth", 0.6),
        width=0.68,
    )
    # Hatching: apply after creation so face color stays.
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)

    ax.set_xticks(xs)
    ax.set_xticklabels([ALGO_DISPLAY.get(a, a) for a in algos], rotation=0)
    ax.set_ylabel(ylabel)
    # IEEE/ACM convention: subfigure labels go below the panel.
    ax.set_xlabel(title, labelpad=8.0)

    # Headroom for value annotations.
    ymax = float(np.max(means)) if len(means) else 1.0
    ax.set_ylim(0, ymax * 1.18)

    if annotate:
        for x, m in zip(xs, means):
            ax.text(
                x, m + ymax * 0.012,
                value_fmt.format(m),
                ha="center", va="bottom",
                fontsize=7.0,
                color="#222",
            )

    style_axis(ax)


def plot_one_model(
    eval_dir: Path,
    out_dir: Path,
    model: str,
    weight_tag: str,
    style_cfg: dict,
    save_formats: tuple[str, ...] = (".pdf", ".png"),
    annotate: bool = True,
    figsize: tuple[float, float] = (3.5, 3.55),
    show_model_title: bool = False,
    model_title_position: str = "bottom",
) -> None:
    payloads = collect_model(eval_dir, model, weight_tag)
    if not payloads:
        print(f"  [skip] {model}: no JSON found.")
        return

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for ax, mcfg in zip(axes, METRICS):
        draw_metric_bars(
            ax=ax,
            payloads=payloads,
            metric_key=mcfg["key"],
            std_key=mcfg["std"],
            style_cfg=style_cfg,
            title=mcfg["title"],
            ylabel=mcfg["ylabel"],
            value_fmt=mcfg["value_fmt"],
            annotate=annotate,
        )

    # Model label on the figure. Default: do not draw (let LaTeX subcaption
    # below the figure handle the model name to avoid duplication).
    if show_model_title:
        if model_title_position == "bottom":
            fig.text(
                0.5, -0.02,
                MODEL_DISPLAY.get(model, model),
                ha="center", va="top",
                fontsize=10.0, fontweight="bold",
            )
        else:
            fig.suptitle(
                MODEL_DISPLAY.get(model, model),
                y=1.005, fontsize=10.0, fontweight="bold",
            )
    fig.tight_layout(h_pad=1.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    file_stem = out_dir / f"fig_e2_cross_model_{model}"
    for suf in save_formats:
        path = file_stem.with_suffix(suf)
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def export_summary_csv(eval_dir: Path, out_dir: Path, weight_tag: str) -> None:
    """Write a 4×5 table (model × algo) of mean_delay and mean_energy for record-keeping."""
    rows = []
    for model in MODEL_ORDER:
        for algo in ALGO_ORDER:
            payload = load_one(eval_dir, model, weight_tag, algo)
            if payload is None:
                continue
            rows.append({
                "model": model,
                "algo":  algo,
                "mean_delay":  payload.get("mean_delay"),
                "std_delay":   payload.get("std_delay", 0.0),
                "mean_energy": payload.get("mean_energy"),
                "std_energy":  payload.get("std_energy", 0.0),
                "mean_reward": payload.get("mean_reward"),
                "mean_accuracy": payload.get("mean_accuracy"),
            })
    if not rows:
        return
    import csv
    csv_path = out_dir / "tab_e2_cross_model_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot E2 per-model bar figures (5 algorithms × delay/energy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval_dir", type=str,
                        default="results/eval/e2_cross_model_w19",
                        help="Directory containing per-(model,algo) JSON files")
    parser.add_argument("--out_dir",  type=str,
                        default="results/figures/e2_cross_model")
    parser.add_argument("--weight_tag", type=str, default="w19")
    parser.add_argument("--style",   type=str, default="ieee", choices=list(STYLES))
    parser.add_argument("--no_annotate", action="store_true",
                        help="Disable on-bar value labels (cleaner look)")
    parser.add_argument("--fig_width",   type=float, default=3.5,
                        help="Figure width in inches (3.5 = IEEE single column)")
    parser.add_argument("--fig_height",  type=float, default=3.55)
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Subset of models to plot (default: all 4)")
    parser.add_argument("--show_model_title", action="store_true",
                        help="Render the model name on the figure "
                             "(default: off; let LaTeX subcaption handle it)")
    parser.add_argument("--model_title_position", choices=["top", "bottom"],
                        default="bottom",
                        help="When --show_model_title is set, where to place it")
    args = parser.parse_args()

    style_cfg = apply_style(args.style)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)

    if not eval_dir.exists():
        raise SystemExit(f"[ERROR] eval_dir not found: {eval_dir}")

    models = args.models or MODEL_ORDER
    print(f"E2 cross-model bar figures   eval_dir={eval_dir}  style={args.style}")
    print(f"  weight_tag={args.weight_tag}   models={models}")
    print()

    for model in models:
        if model not in MODEL_DISPLAY:
            print(f"  [skip] Unknown model: {model}")
            continue
        plot_one_model(
            eval_dir=eval_dir,
            out_dir=out_dir,
            model=model,
            weight_tag=args.weight_tag,
            style_cfg=style_cfg,
            annotate=not args.no_annotate,
            figsize=(args.fig_width, args.fig_height),
            show_model_title=args.show_model_title,
            model_title_position=args.model_title_position,
        )

    export_summary_csv(eval_dir, out_dir, args.weight_tag)
    print(f"\nDone. Figures + summary CSV in {out_dir}/")


if __name__ == "__main__":
    main()
