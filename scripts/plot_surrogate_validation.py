"""
Quick script to re-plot surrogate validation figure from saved CSV data.

Usage:
    # Re-plot existing validation CSV
    python scripts/plot_surrogate_validation.py --csv results\\accuracy_modeling\\split\\validation_split.csv --output results\\accuracy_modeling\\split\\validation_split_fixed.pdf --mode accuracy

    # Optional: post-hoc calibration (fit on 50%, report/plot on remaining 50%)
    python scripts/plot_surrogate_validation.py --csv results\\accuracy_modeling\\split\\validation_split.csv --output results\\accuracy_modeling\\split\\validation_split_fixed.pdf --mode accuracy --calibration through1 --calib_frac 0.5
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np


def _make_legend_strip(out_path: str) -> None:
    """Render a shared legend strip spanning ~one IEEE column (2-panel row width)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    })

    IDEAL_LW = 1.8
    FIT_LW = 1.6
    handles = [
        Line2D([0], [0], color="black", lw=IDEAL_LW, label="Ideal"),
        Line2D([0], [0], color="red", lw=FIT_LW, ls="--", label="Fit"),
        Line2D(
            [0], [0], color="none", marker="o", markersize=5.5,
            markerfacecolor="#2f6db5", markeredgecolor="#0d2747",
            markeredgewidth=0.5, label="Samples",
        ),
    ]

    # Slightly inset axes so the frame matches the visible 2-panel row (not full column).
    fig_w, fig_h = 3.45, 0.30
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.04, 0.0, 0.92, 1.0])
    ax.set_axis_off()
    ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        mode="expand",
        frameon=True,
        framealpha=1.0,
        edgecolor="gray",
        fancybox=False,
        fontsize=9.5,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=1.0,
        borderpad=0.42,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, pad_inches=0.02)
    plt.close(fig)
    print(f"OK Legend strip saved to: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/exp_surrogate_validation/surrogate_validation.csv")
    parser.add_argument("--output", type=str, default="results/exp_surrogate_validation/surrogate_validation.pdf")
    parser.add_argument(
        "--mode",
        type=str,
        default="degradation",
        choices=["accuracy", "degradation"],
        help="Plot mode: 'accuracy' plots (A_hat vs A); 'degradation' plots (Delta A_hat vs Delta A).",
    )
    parser.add_argument(
        "--A_full",
        type=float,
        default=1.0,
        help="A_full for degradation mode (default 1.0 under relative normalization).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used for calibration split).")
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
            "Fraction of samples used to fit calibration (0=fit on all samples; recommended 0.5 for hold-out evaluation). "
            "Only used when --calibration != none."
        ),
    )
    parser.add_argument(
        "--output_cal",
        type=str,
        default=None,
        help="Output path for calibrated figure (default: <output>_cal.pdf). Only used when --calibration != none.",
    )
    parser.add_argument(
        "--save_calibration",
        type=str,
        default=None,
        help="Save calibration parameters to JSON (default: alongside CSV as calibration_<name>.json).",
    )
    parser.add_argument(
        "--write_csv",
        type=str,
        default=None,
        help="Write a calibrated CSV with added columns A_pred_cal and abs_err_cal (default: disabled).",
    )
    parser.add_argument(
        "--legend",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help=(
            "Per-panel legend: 'auto' draws the Ideal/Fit/Samples legend inside each panel; "
            "'none' omits it (recommended for multi-panel figures with a shared legend described in the caption)."
        ),
    )
    parser.add_argument(
        "--legend_only",
        type=str,
        default=None,
        help=(
            "If set, generate ONLY a standalone horizontal legend strip (Ideal/Fit/Samples) "
            "to this path and exit. Used as a shared legend above a multi-panel figure."
        ),
    )
    args = parser.parse_args()

    if args.legend_only is not None:
        _make_legend_strip(args.legend_only)
        return

    # Read CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Backward-compatible fallback: older runs saved to results/surrogate_validation.csv
        legacy = Path("results/surrogate_validation.csv")
        if str(args.csv).replace("\\", "/") == "results/exp_surrogate_validation/surrogate_validation.csv" and legacy.exists():
            print(f"⚠ CSV not found at default path: {csv_path}")
            print(f"  → Fallback to legacy CSV: {legacy}")
            csv_path = legacy
        else:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    A_preds = []
    A_trues = []
    rows = []
    fieldnames = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)
            A_preds.append(float(row["A_pred"]))
            A_trues.append(float(row["A_true"]))

    A_preds = np.asarray(A_preds, dtype=np.float64)
    A_trues = np.asarray(A_trues, dtype=np.float64)

    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        err = (y_true - y_pred).astype(np.float64)
        mae_ = float(np.mean(np.abs(err)))
        rmse_ = float(np.sqrt(np.mean(err ** 2)))
        ss_res_ = float(np.sum(err ** 2))
        ss_tot_ = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        r2_ = float(1.0 - ss_res_ / ss_tot_) if ss_tot_ > 0 else float("nan")
        return {"mae": mae_, "rmse": rmse_, "r2": r2_}

    # Calibration is always fitted in ACCURACY space (A_pred -> A_true).
    cal_method = str(args.calibration)
    cal_k, cal_b = 1.0, 0.0
    calib_idx = np.arange(len(A_preds), dtype=np.int64)
    test_idx = np.arange(len(A_preds), dtype=np.int64)
    A_preds_cal = A_preds.copy()

    raw_all_acc = _compute_metrics(A_trues, A_preds)
    raw_test_acc = None
    cal_all_acc = None
    cal_test_acc = None

    if cal_method != "none":
        idx_all = np.arange(len(A_preds), dtype=np.int64)
        rng = np.random.RandomState(int(args.seed))
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
            # Enforce (Â=1) -> (A=1): b = 1 - k
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

        raw_test_acc = _compute_metrics(A_trues[test_idx], A_preds[test_idx])
        cal_all_acc = _compute_metrics(A_trues, A_preds_cal)
        cal_test_acc = _compute_metrics(A_trues[test_idx], A_preds_cal[test_idx])

        # Save calibration JSON
        if args.save_calibration is None:
            name = csv_path.stem
            if name.startswith("validation_"):
                name = name[len("validation_"):]
            args.save_calibration = str(csv_path.parent / f"calibration_{name}.json")
        cal_path = Path(args.save_calibration)
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "csv": str(csv_path).replace("\\", "/"),
                    "method": cal_method,
                    "k": float(cal_k),
                    "b": float(cal_b),
                    "calib_frac": float(args.calib_frac),
                    "n_total": int(len(A_preds)),
                    "n_calib": int(len(calib_idx)),
                    "n_test": int(len(test_idx)),
                    "metrics_raw_all_accuracy": raw_all_acc,
                    "metrics_raw_test_accuracy": raw_test_acc,
                    "metrics_cal_all_accuracy": cal_all_acc,
                    "metrics_cal_test_accuracy": cal_test_acc,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if args.write_csv is not None:
            out_csv = Path(args.write_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out_fields = list(fieldnames or [])
            if "A_pred_cal" not in out_fields:
                out_fields.append("A_pred_cal")
            if "abs_err_cal" not in out_fields:
                out_fields.append("abs_err_cal")
            for i, r in enumerate(rows):
                r["A_pred_cal"] = f"{A_preds_cal[i]:.12f}"
                r["abs_err_cal"] = f"{abs(A_trues[i] - A_preds_cal[i]):.12f}"
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=out_fields)
                w.writeheader()
                w.writerows(rows)

    # Prepare plot arrays (raw)
    if args.mode == "accuracy":
        x = A_preds
        y = A_trues
        x_label = r"Predicted Accuracy $\hat{A}$"
        y_label = r"True Accuracy $A$"
        mn0 = float(min(x.min(), y.min()))
        mx0 = float(max(x.max(), y.max()))
    else:
        # Degradation: Delta A = A_full - A
        A_full = float(args.A_full)
        x = (A_full - A_preds)
        y = (A_full - A_trues)
        x_label = r"Predicted Degradation $\Delta\hat{A}$"
        y_label = r"True Degradation $\Delta A$"
        mn0 = 0.0
        mx0 = float(max(x.max(), y.max()))

    raw_plot_metrics = _compute_metrics(y_true=y, y_pred=x)
    print(f"Loaded {len(A_preds)} samples from {csv_path}")
    print(f"mode={args.mode} | MAE = {raw_plot_metrics['mae']:.6f}, RMSE = {raw_plot_metrics['rmse']:.6f}, R² = {raw_plot_metrics['r2']:.6f}")
    if cal_method != "none":
        print("-" * 60)
        print(f"calibration={cal_method} | k={cal_k:.4f}, b={cal_b:+.4f} | calib_frac={float(args.calib_frac):.2f}")
        if raw_test_acc is not None and cal_test_acc is not None:
            print(f"  (accuracy, test) raw: MAE={raw_test_acc['mae']:.4f}, R²={raw_test_acc['r2']:.4f}")
            print(f"  (accuracy, test) cal: MAE={cal_test_acc['mae']:.4f}, R²={cal_test_acc['r2']:.4f}")
        if args.save_calibration is not None:
            print(f"  calibration JSON: {args.save_calibration}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    out_fig = Path(args.output)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    # Style tuned for 2x2 subfigure layout in IEEE single column.
    # Each panel is shown at ~0.47\columnwidth (~1.65 in), i.e. downscaled ~0.69x
    # from the generated size below; fonts/markers are enlarged to stay legible
    # after that downscaling.
    FIGSIZE = (2.45, 2.45)
    LABEL_FS = 13
    TICK_FS = 11
    LEGEND_FS = 10.5
    ANNOT_FS = 10.0
    IDEAL_LW = 2.2
    FIT_LW = 2.0
    # Saturated marker so the scatter stays prominent after downscaling.
    SCATTER_KW = dict(
        c="#2f6db5",
        s=34,
        alpha=0.85,
        edgecolors="#0d2747",
        linewidths=0.45,
        zorder=1,
        label="Samples",
    )
    # Compact stats box so it stays in the empty corner without covering lines.
    STATS_BBOX = dict(
        boxstyle="round,pad=0.28",
        facecolor="white",
        edgecolor="gray",
        linewidth=0.8,
        alpha=0.9,
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": TICK_FS,
        "axes.labelsize": LABEL_FS,
        "axes.titlesize": LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": LEGEND_FS,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "axes.linewidth": 1.0,
    })

    SAVE_PAD = 0.10  # extra margin so long axis labels are not clipped in 2x2 LaTeX layout
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")

    # Data range
    if args.mode == "degradation":
        mn = 0.0
        mx = float(mx0)
        margin = 0.0
    else:
        mn = float(mn0)
        mx = float(mx0)
        margin = 0.005
    xs = np.linspace(mn - margin, mx + margin, 200)

    # Ideal line (y = x) - thick black
    ax.plot(xs, xs, "k-", linewidth=IDEAL_LW, label="Ideal", zorder=3)

    # Best-fit line - dashed red
    if len(A_preds) >= 2:
        k_fit, b_fit = np.polyfit(x, y, 1)
        ax.plot(
            xs,
            k_fit * xs + b_fit,
            "r--",
            linewidth=FIT_LW,
            label="Fit",
            zorder=2,
        )

    # Scatter plot - saturated blue for visibility after downscaling
    ax.scatter(x, y, **SCATTER_KW)

    # Axis
    ax.set_xlim(mn - margin, mx + margin)
    ax.set_ylim(mn - margin, mx + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label, fontsize=LABEL_FS)
    ax.set_ylabel(y_label, fontsize=LABEL_FS)

    # Ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", direction="in", top=True, right=True)

    # Legend - upper left, compact (omitted when a shared legend is used)
    if str(args.legend) != "none":
        ax.legend(
            loc="upper left", frameon=True, framealpha=0.95, edgecolor="gray", fancybox=False,
            handlelength=1.4, handletextpad=0.5, borderpad=0.35, labelspacing=0.3, borderaxespad=0.4,
        )

    # Stats annotation - lower right corner (3 lines, same as validate_surrogate_random_z.py)
    stats_text = (
        f"$n = {len(A_preds)}$\n"
        f"$R^2 = {raw_plot_metrics['r2']:.4f}$\n"
        f"MAE $= {raw_plot_metrics['mae']:.4f}$"
    )
    ax.text(
        0.97,
        0.03,
        stats_text,
        transform=ax.transAxes,
        fontsize=ANNOT_FS,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=STATS_BBOX,
    )

    # Save (pad_inches prevents x/y label clipping when panels are placed flush in LaTeX)
    plt.savefig(out_fig, dpi=300, bbox_inches="tight", pad_inches=SAVE_PAD, format="pdf")
    print(f"✓ PDF saved to: {out_fig}")

    png_path = out_fig.with_suffix(".png")
    plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=SAVE_PAD)
    print(f"✓ PNG saved to: {png_path}")

    eps_path = out_fig.with_suffix(".eps")
    plt.savefig(eps_path, dpi=300, bbox_inches="tight", pad_inches=SAVE_PAD, format="eps")
    print(f"✓ EPS saved to: {eps_path}")

    plt.close()

    # Optional calibrated figure
    if cal_method != "none":
        if args.output_cal is None:
            out_fig_cal = out_fig.with_name(out_fig.stem + "_cal" + out_fig.suffix)
        else:
            out_fig_cal = Path(args.output_cal)
        out_fig_cal.parent.mkdir(parents=True, exist_ok=True)

        # Choose what to plot for calibrated case: test subset when hold-out split is enabled
        if 0.0 < float(args.calib_frac) < 1.0 and len(test_idx) >= 2:
            idx_plot = test_idx
        else:
            idx_plot = np.arange(len(A_preds_cal), dtype=np.int64)

        if args.mode == "accuracy":
            x_cal_plot = A_preds_cal[idx_plot]
            y_cal_plot = A_trues[idx_plot]
            mn = float(min(x_cal_plot.min(), y_cal_plot.min()))
            mx = float(max(x_cal_plot.max(), y_cal_plot.max()))
            margin = 0.005
            x_label_cal = r"Calibrated Prediction $\hat{A}_{\mathrm{cal}}$"
            y_label_cal = r"True Accuracy $A$"
        else:
            A_full = float(args.A_full)
            x_cal_plot = (A_full - A_preds_cal[idx_plot])
            y_cal_plot = (A_full - A_trues[idx_plot])
            mn = 0.0
            mx = float(max(x_cal_plot.max(), y_cal_plot.max()))
            margin = 0.0
            x_label_cal = r"Calibrated Pred. Degradation $\Delta\hat{A}_{\mathrm{cal}}$"
            y_label_cal = r"True Degradation $\Delta A$"

        cal_plot_metrics = _compute_metrics(y_true=y_cal_plot, y_pred=x_cal_plot)
        xs = np.linspace(mn - margin, mx + margin, 200)

        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        ax.plot(xs, xs, "k-", linewidth=IDEAL_LW, label="Ideal", zorder=3)
        if len(x_cal_plot) >= 2:
            k_fit, b_fit = np.polyfit(x_cal_plot, y_cal_plot, 1)
            ax.plot(
                xs,
                k_fit * xs + b_fit,
                "r--",
                linewidth=FIT_LW,
                label="Fit",
                zorder=2,
            )
        ax.scatter(x_cal_plot, y_cal_plot, **SCATTER_KW)
        ax.set_xlim(mn - margin, mx + margin)
        ax.set_ylim(mn - margin, mx + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(x_label_cal, fontsize=LABEL_FS)
        ax.set_ylabel(y_label_cal, fontsize=LABEL_FS)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="both", direction="in", top=True, right=True)
        if str(args.legend) != "none":
            ax.legend(
                loc="upper left", frameon=True, framealpha=0.95, edgecolor="gray", fancybox=False,
                handlelength=1.4, handletextpad=0.5, borderpad=0.35, labelspacing=0.3, borderaxespad=0.4,
            )

        stats_text = (
            f"$n = {len(x_cal_plot)}$\n"
            f"$R^2 = {cal_plot_metrics['r2']:.4f}$\n"
            f"MAE $= {cal_plot_metrics['mae']:.4f}$"
        )
        ax.text(
            0.97,
            0.03,
            stats_text,
            transform=ax.transAxes,
            fontsize=ANNOT_FS,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=STATS_BBOX,
        )
        plt.savefig(out_fig_cal, dpi=300, bbox_inches="tight", pad_inches=SAVE_PAD, format="pdf")
        print(f"✓ Calibrated PDF saved to: {out_fig_cal}")
        png_path = out_fig_cal.with_suffix(".png")
        plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=SAVE_PAD)
        print(f"✓ Calibrated PNG saved to: {png_path}")
        eps_path = out_fig_cal.with_suffix(".eps")
        plt.savefig(eps_path, dpi=300, bbox_inches="tight", pad_inches=SAVE_PAD, format="eps")
        print(f"✓ Calibrated EPS saved to: {eps_path}")
        plt.close()


if __name__ == "__main__":
    main()
