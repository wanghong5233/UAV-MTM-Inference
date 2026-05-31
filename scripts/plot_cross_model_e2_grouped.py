"""
Plot a single grouped bar figure for the E2 cross-model comparison.

Layout:
    figure*  -- two horizontally arranged subplots
        (a) Inference Delay (s)             (b) Energy per Request (J)
        x = backbone groups  (Hard-Split, MTAN, Dense-Soft, Cross-Stitch)
        each group contains 5 bars, one per algorithm, color/hatch shared.

This is a complement to plot_cross_model_e2.py. The latter produces 4 separate
per-backbone figures (Layout B), while this script produces one figure that
shows all 4 backbones at once (Layout A), which scales better in a paper.

Usage:
    python scripts/plot_cross_model_e2_grouped.py \
        --eval_dir results/eval/e2_cross_model_w19 \
        --out_dir  results/figures/e2_cross_model \
        --style    ieee
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Reuse styling tables from the per-backbone script so colors / hatches
# stay consistent across figures.
from plot_cross_model_e2 import (
    STYLES, MODEL_DISPLAY, MODEL_ORDER, ALGO_DISPLAY, ALGO_ORDER,
    apply_style, collect_model, style_axis,
)


METRICS = [
    {
        "key":   "mean_delay",
        "title": "(a) Inference Delay",
        "ylabel": "Delay (s)",
        "value_fmt": "{:.1f}",
    },
    {
        "key":   "mean_energy",
        "title": "(b) Energy per Request",
        "ylabel": r"$E_{\mathrm{req}}$ (J)",
        "value_fmt": "{:.1f}",
    },
]


def collect_all(eval_dir: Path, weight_tag: str) -> dict[str, dict[str, dict]]:
    """Return nested dict: {model: {algo: payload}}."""
    out: dict[str, dict[str, dict]] = {}
    for m in MODEL_ORDER:
        payloads = collect_model(eval_dir, m, weight_tag)
        if payloads:
            out[m] = payloads
        else:
            print(f"  [warn] no JSON for backbone {m}; skipping")
    return out


def collect_from_csv(csv_path: Path) -> dict[str, dict[str, dict]]:
    """Load the same nested structure from tab_e2_cross_model_summary.csv."""
    out: dict[str, dict[str, dict]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"].strip()
            algo = row["algo"].strip()
            payload: dict[str, Any] = {}
            for k, v in row.items():
                if k in {"model", "algo"}:
                    continue
                try:
                    payload[k] = float(v)
                except (TypeError, ValueError):
                    payload[k] = v
            out.setdefault(model, {})[algo] = payload
    return out


def draw_grouped(
    ax: plt.Axes,
    data: dict[str, dict[str, dict]],
    metric_key: str,
    style_cfg: dict,
    title: str,
    ylabel: str,
    value_fmt: str,
    annotate_mode: str,
    show_xlabel_top: bool = True,
) -> None:
    models = [m for m in MODEL_ORDER if m in data]
    algos = ALGO_ORDER  # fixed order, all 5

    n_groups = len(models)
    n_bars   = len(algos)

    bar_width = 0.90 / n_bars
    group_spacing = 1.18
    group_centers = np.arange(n_groups) * group_spacing

    # gather values: shape (n_bars, n_groups)
    vals = np.full((n_bars, n_groups), np.nan)
    for j, m in enumerate(models):
        for i, a in enumerate(algos):
            if a in data[m]:
                vals[i, j] = float(data[m][a].get(metric_key, np.nan))

    colors = [style_cfg["colors"].get(a, "#888") for a in algos]
    hatches = [style_cfg["hatches"].get(a, "") for a in algos]

    bar_handles = []
    for i, a in enumerate(algos):
        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(
            group_centers + offset,
            vals[i],
            bar_width * 0.98,
            color=colors[i],
            edgecolor=style_cfg.get("edgecolor", "black"),
            linewidth=style_cfg.get("bar_linewidth", 0.6),
            label=ALGO_DISPLAY.get(a, a),
        )
        for b in bars:
            if hatches[i]:
                b.set_hatch(hatches[i])
        bar_handles.append(bars)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models])
    ax.set_ylabel(ylabel)
    if show_xlabel_top:
        ax.set_xlabel(title, labelpad=8.0)

    finite = vals[np.isfinite(vals)]
    ymax = float(finite.max()) if finite.size else 1.0
    ax.set_ylim(0, ymax * 1.18)

    if annotate_mode != "none":
        # Exact values are reported in Table~\ref{tab:main_results}; figure
        # labels should emphasize the proposed method rather than cluttering
        # every bar. This avoids unavoidable overlaps such as Cross-Stitch
        # Local/S-Split, which have identical values.
        annot_fs = 5.8
        for i, a in enumerate(algos):
            if annotate_mode == "ours" and a != "proposed":
                continue
            offset = (i - (n_bars - 1) / 2) * bar_width
            for j, v in enumerate(vals[i]):
                if not np.isfinite(v):
                    continue
                if annotate_mode == "best":
                    # Lower is better for both delay and energy.
                    col = vals[:, j]
                    if abs(v - np.nanmin(col)) > 1e-9:
                        continue
                ax.text(
                    group_centers[j] + offset,
                    v + ymax * 0.018,
                    value_fmt.format(v),
                    ha="center", va="bottom",
                    fontsize=annot_fs,
                    fontweight="bold" if a == "proposed" else "normal",
                    color="#222",
                )

    style_axis(ax)
    return bar_handles


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 grouped bar figure (one row, two metrics).")
    parser.add_argument("--eval_dir", type=str, default="results/eval/e2_cross_model_w19")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional tab_e2_cross_model_summary.csv fallback input")
    parser.add_argument("--out_dir",  type=str, required=True)
    parser.add_argument("--style",    type=str, default="ieee", choices=list(STYLES.keys()))
    parser.add_argument("--weight_tag", type=str, default="w19")
    parser.add_argument("--no_annotate", action="store_true",
                        help="Deprecated alias for --annotate_mode none")
    parser.add_argument("--annotate_mode", choices=["all", "ours", "best", "none"], default="none",
                        help="Bar value labels to draw. Use 'none' for the paper figure and 'all' only for debugging.")
    parser.add_argument("--fig_width",  type=float, default=3.5,
                        help="Width in inches (IEEE column width ≈ 3.5)")
    parser.add_argument("--fig_height", type=float, default=4.4,
                        help="Total height for 2 stacked panels")
    parser.add_argument("--orientation", choices=["row", "col"], default="col",
                        help="row = 1x2 side-by-side; col = 2x1 stacked")
    args = parser.parse_args()

    style_cfg = apply_style(args.style)

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.out_dir)
    csv_path = Path(args.csv) if args.csv else out_dir / "tab_e2_cross_model_summary.csv"

    if eval_dir.exists():
        data = collect_all(eval_dir, args.weight_tag)
        source_desc = str(eval_dir)
    elif csv_path.exists():
        data = collect_from_csv(csv_path)
        source_desc = str(csv_path)
    else:
        raise SystemExit(f"[ERROR] neither eval_dir nor csv exists: eval_dir={eval_dir}, csv={csv_path}")

    annotate_mode = "none" if args.no_annotate else args.annotate_mode
    if not data:
        raise SystemExit(f"[ERROR] No E2 data found from {source_desc}")

    print(f"E2 grouped bar figure   source={source_desc}  style={args.style}")
    print(f"  weight_tag={args.weight_tag}   backbones present={list(data.keys())}")

    if args.orientation == "row":
        nrows, ncols = 1, 2
        fig_size = (args.fig_width, args.fig_height)
    else:
        nrows, ncols = 2, 1
        fig_size = (args.fig_width, args.fig_height)
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)
    axes = np.atleast_1d(axes).flatten()
    handles_first = None
    for ax, mcfg in zip(axes, METRICS):
        h = draw_grouped(
            ax=ax,
            data=data,
            metric_key=mcfg["key"],
            style_cfg=style_cfg,
            title=mcfg["title"],
            ylabel=mcfg["ylabel"],
            value_fmt=mcfg["value_fmt"],
            annotate_mode=annotate_mode,
        )
        if handles_first is None:
            handles_first = h

    # One legend at the very top for both panels.
    legend_handles = [handles_first[i][0] for i in range(len(ALGO_ORDER))]
    legend_labels  = [ALGO_DISPLAY.get(a, a) for a in ALGO_ORDER]
    if args.orientation == "row":
        legend_y, rect_top = 1.02, 0.94
    else:
        # 2x1 stacked: legend is denser, sits above the upper panel
        legend_y, rect_top = 1.005, 0.96
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        ncol=len(ALGO_ORDER),
        bbox_to_anchor=(0.5, legend_y),
        frameon=False,
        fontsize=8.0,
        handlelength=1.6,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, rect_top))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_e2_grouped"
    for suf in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suf))
        print(f"  Saved: {stem.with_suffix(suf)}")
    plt.close(fig)
    print(f"\nDone. Grouped figure in {out_dir}/")


if __name__ == "__main__":
    main()
