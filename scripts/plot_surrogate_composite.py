"""Render the surrogate-validation figure as ONE composite vector PDF.

This reproduces the previously hand-assembled LaTeX layout (a shared top legend,
a 2x2 grid of parity panels, per-panel (a)-(d) subcaptions, and a bottom-right
stats inset) as a single high-resolution figure, so it can be inserted directly
with \\includegraphics[width=\\columnwidth]{...} instead of being assembled in
LaTeX with \\subfloat.

Panel data and the through-1 calibration EXACTLY match
scripts/plot_surrogate_validation.py:
  (a) Hard-Split   : split  CSV, through1 calibration, calib_frac=0.5, test split
  (b) MTAN         : mtan   CSV, raw prediction
  (c) Dense-Soft   : dense  CSV, through1 calibration, calib_frac=0.5, test split
  (d) Cross-Stitch : cross  CSV, raw prediction

Usage:
    python scripts/plot_surrogate_composite.py \
        --results_dir results/accuracy_modeling \
        --output results/accuracy_modeling/validation_surrogate_composite.pdf

By default also writes ``<stem>_nolegend.<ext>`` without the top legend banner.
Use ``--single legend`` or ``--single no_legend`` to emit only one variant.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

PANELS = [
    dict(tag="(a) Hard-Split.", csv="split/validation_split.csv", cal=True,
         xlabel=r"Calibrated Prediction $\hat{A}_{\mathrm{cal}}$"),
    dict(tag="(b) MTAN.", csv="mtan/validation_mtan_fixed.csv", cal=False,
         xlabel=r"Predicted Accuracy $\hat{A}$"),
    dict(tag="(c) Dense-Soft.", csv="dense/validation_dense_fixed.csv", cal=True,
         xlabel=r"Calibrated Prediction $\hat{A}_{\mathrm{cal}}$"),
    dict(tag="(d) Cross-Stitch.", csv="cross/validation_cross.csv", cal=False,
         xlabel=r"Predicted Accuracy $\hat{A}$"),
]

LEFT, RIGHT = 0.105, 0.985
TOP_WITH_LEGEND = 0.92
TOP_NO_LEGEND = 0.98


def _load_csv(path: Path):
    A_pred, A_true = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            A_pred.append(float(row["A_pred"]))
            A_true.append(float(row["A_true"]))
    return np.asarray(A_pred, dtype=np.float64), np.asarray(A_true, dtype=np.float64)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    err = (y_true - y_pred).astype(np.float64)
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return r2, mae


def _through1(A_pred, A_true, calib_frac=0.5, seed=42):
    """Through-(1,1) affine calibration; returns calibrated preds and held-out test idx."""
    idx = np.arange(len(A_pred), dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(idx)
    n_cal = max(2, min(len(idx) - 1, int(round(calib_frac * len(idx)))))
    calib_idx, test_idx = idx[:n_cal], idx[n_cal:]
    xc, yc = A_pred[calib_idx], A_true[calib_idx]
    x0, y0 = xc - 1.0, yc - 1.0
    denom = float(np.dot(x0, x0))
    k = float(np.dot(x0, y0) / denom) if denom > 1e-12 else 1.0
    b = 1.0 - k
    A_cal = np.clip(k * A_pred + b, 0.0, 1.0)
    return A_cal, test_idx


def _variant_path(output: Path, with_legend: bool) -> Path:
    if with_legend:
        return output
    return output.with_name(f"{output.stem}_nolegend{output.suffix}")


def _add_top_legend(fig, handles, *, LEGEND_FS: float) -> tuple:
    """Return (frame, legend) artists for the shared top banner."""
    from matplotlib.patches import Rectangle

    H_INSET = 0.072
    LEGEND_Y = TOP_WITH_LEGEND + 0.015
    inner_w = RIGHT - LEFT - 2 * H_INSET
    _probe_kw = dict(
        fontsize=LEGEND_FS, handlelength=1.2, handletextpad=0.5,
        columnspacing=1.6, borderpad=0.55,
    )
    _inner_kw = dict(
        fontsize=LEGEND_FS, handlelength=1.2, handletextpad=0.5,
        columnspacing=1.6, borderpad=0.30,
    )
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP_WITH_LEGEND, bottom=0.075,
                        wspace=0.30, hspace=0.34)
    probe = fig.legend(handles=handles, ncol=3, loc="center",
                       frameon=True, **_probe_kw)
    fig.canvas.draw()
    bbox = probe.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()
    )
    legend_h = bbox.height
    probe.remove()
    frame = Rectangle(
        (LEFT, LEGEND_Y), RIGHT - LEFT, legend_h,
        transform=fig.transFigure, facecolor="white",
        edgecolor="gray", linewidth=0.8, clip_on=False, zorder=5,
    )
    fig.add_artist(frame)
    leg = fig.legend(
        handles=handles, ncol=3,
        loc="center", mode="expand", alignment="center",
        bbox_to_anchor=(LEFT + H_INSET, LEGEND_Y, inner_w, legend_h),
        frameon=False, **_inner_kw,
    )
    return frame, leg


def render_composite(
    results_dir: Path,
    output: Path,
    *,
    with_legend: bool,
    calib_frac: float = 0.5,
    seed: int = 42,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator

    LABEL_FS = 17
    TICK_FS = 14
    ANNOT_FS = 13.5
    SUBCAP_FS = 18
    LEGEND_FS = 14
    IDEAL_LW = 2.0
    FIT_LW = 1.8
    SCATTER_KW = dict(c="#2f6db5", s=30, alpha=0.85,
                      edgecolors="#0d2747", linewidths=0.45, zorder=1)
    STATS_BBOX = dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="gray", linewidth=0.8, alpha=0.9)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": TICK_FS,
        "axes.labelsize": LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "axes.linewidth": 1.0,
    })

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 7.2))

    for ax, panel in zip(axes.flat, PANELS):
        A_pred, A_true = _load_csv(results_dir / panel["csv"])
        if panel["cal"]:
            A_cal, test_idx = _through1(A_pred, A_true, calib_frac, seed)
            xx, yy = A_cal[test_idx], A_true[test_idx]
        else:
            xx, yy = A_pred, A_true

        r2, mae = _metrics(yy, xx)
        n = len(xx)

        mn = float(min(xx.min(), yy.min()))
        mx = float(max(xx.max(), yy.max()))
        margin = 0.005
        line = np.linspace(mn - margin, mx + margin, 200)

        ax.plot(line, line, "k-", lw=IDEAL_LW, zorder=3)
        if n >= 2:
            kf, bf = np.polyfit(xx, yy, 1)
            ax.plot(line, kf * line + bf, "r--", lw=FIT_LW, zorder=2)
        ax.scatter(xx, yy, **SCATTER_KW)

        ax.set_xlim(mn - margin, mx + margin)
        ax.set_ylim(mn - margin, mx + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(panel["xlabel"], fontsize=LABEL_FS)
        ax.set_ylabel(r"True Accuracy $A$", fontsize=LABEL_FS)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="both", direction="in", top=True, right=True)

        stats = f"$n = {n}$\n$R^2 = {r2:.4f}$\nMAE $= {mae:.4f}$"
        ax.text(0.97, 0.03, stats, transform=ax.transAxes, fontsize=ANNOT_FS,
                va="bottom", ha="right", bbox=STATS_BBOX)
        ax.text(0.5, -0.30, panel["tag"], transform=ax.transAxes,
                fontsize=SUBCAP_FS, ha="center", va="top")

    extra_artists = ()
    if with_legend:
        handles = [
            Line2D([0], [0], color="black", lw=IDEAL_LW, label="Ideal"),
            Line2D([0], [0], color="red", lw=FIT_LW, ls="--", label="Fit"),
            Line2D([0], [0], color="none", marker="o", markersize=7,
                   markerfacecolor="#2f6db5", markeredgecolor="#0d2747",
                   markeredgewidth=0.5, label="Samples"),
        ]
        extra_artists = _add_top_legend(fig, handles, LEGEND_FS=LEGEND_FS)
    else:
        fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP_NO_LEGEND, bottom=0.075,
                            wspace=0.30, hspace=0.34)

    out = _variant_path(output, with_legend)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kw = dict(bbox_inches="tight", pad_inches=0.03)
    if extra_artists:
        save_kw["bbox_extra_artists"] = extra_artists
    fig.savefig(out, format="pdf", **save_kw)
    fig.savefig(out.with_suffix(".png"), dpi=600, **save_kw)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results/accuracy_modeling")
    ap.add_argument("--output", type=str,
                    default="results/accuracy_modeling/validation_surrogate_composite.pdf")
    ap.add_argument("--calib_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--single", choices=("both", "legend", "no_legend"), default="both",
        help="Emit both variants (default), or only one.",
    )
    args = ap.parse_args()

    res = Path(args.results_dir)
    output = Path(args.output)
    variants = {
        "legend": True,
        "no_legend": False,
    }
    if args.single == "both":
        to_render = list(variants.items())
    else:
        to_render = [(args.single, variants[args.single])]

    for label, with_legend in to_render:
        out = render_composite(
            res, output,
            with_legend=with_legend,
            calib_frac=args.calib_frac,
            seed=args.seed,
        )
        print(f"OK composite saved ({label}): {out}")
        print(f"OK composite saved ({label}): {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
