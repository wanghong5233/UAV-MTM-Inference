"""
Plot TMC-grade Pareto fronts for E2 from tail-summary JSON files.

Each point corresponds to one (algorithm, weight) pair, where the point
coordinates are taken from the tail-mean statistics of the training logs.
By default, the script generates a clean single-panel Pareto figure suitable
for IEEE TMC manuscript use.

Example:
    python scripts/plot_pareto_e2.py \
        --input_dir results/pareto_tail50/e2_pareto_7w_i1000 \
        --out_dir results/figures/pareto_e2 \
        --style ieee \
        --tail_iters 50
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plot_convergence import STYLES, apply_style, save_figure

WEIGHT_ORDER = ["w19", "w28", "w37", "w55", "w73", "w82", "w91"]
WEIGHT_DISPLAY = {
    "w19": r"$\omega$=(0.1, 0.9)",
    "w28": r"$\omega$=(0.2, 0.8)",
    "w37": r"$\omega$=(0.3, 0.7)",
    "w55": r"$\omega$=(0.5, 0.5)",
    "w73": r"$\omega$=(0.7, 0.3)",
    "w82": r"$\omega$=(0.8, 0.2)",
    "w91": r"$\omega$=(0.9, 0.1)",
}

ALGO_DISPLAY = {
    "proposed": "Proposed",
    "partition_routing_only": "PR-Only",
    "mlp_ppo": "MLP-PPO",
    "local_only": "Local-Only",
    "single_split": "Single-Split",
}
ALGO_ORDER = [
    "proposed",
    "partition_routing_only",
    "mlp_ppo",
    "single_split",
    "local_only",
]

ANNOTATION_OFFSETS = {
    "proposed": (4, 5),
    "partition_routing_only": (4, -9),
    "mlp_ppo": (-14, 5),
    "single_split": (4, 5),
    "local_only": (4, -9),
}


@dataclass(frozen=True)
class ParetoPoint:
    model: str
    algorithm: str
    weight_tag: str
    mean_delay: float
    mean_energy: float
    mean_accuracy: float | None
    num_iterations: int
    tail_count_delay: int
    tail_count_energy: int
    source_file: Path


def algorithm_visuals(style_name: str) -> dict[str, dict[str, Any]]:
    if style_name == "grayscale":
        return {
            "proposed": {
                "color": "#000000",
                "linestyle": "-",
                "marker": "o",
                "linewidth": 1.30,
                "markersize": 3.5,
                "markerfacecolor": "#000000",
            },
            "partition_routing_only": {
                "color": "#565656",
                "linestyle": "--",
                "marker": "s",
                "linewidth": 1.10,
                "markersize": 3.2,
                "markerfacecolor": "white",
            },
            "mlp_ppo": {
                "color": "#8C8C8C",
                "linestyle": "-.",
                "marker": "^",
                "linewidth": 1.10,
                "markersize": 3.2,
                "markerfacecolor": "white",
            },
            "single_split": {
                "color": "#B0B0B0",
                "linestyle": ":",
                "marker": "D",
                "linewidth": 1.00,
                "markersize": 3.0,
                "markerfacecolor": "white",
            },
            "local_only": {
                "color": "#C7C7C7",
                "linestyle": (0, (3.0, 1.1)),
                "marker": "v",
                "linewidth": 1.00,
                "markersize": 3.0,
                "markerfacecolor": "white",
            },
        }

    return {
        "proposed": {
            "color": "#C44E52",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 1.30,
            "markersize": 3.5,
            "markerfacecolor": "#C44E52",
        },
        "partition_routing_only": {
            "color": "#3B6FB6",
            "linestyle": "--",
            "marker": "s",
            "linewidth": 1.10,
            "markersize": 3.2,
            "markerfacecolor": "white",
        },
        "mlp_ppo": {
            "color": "#2A9D8F",
            "linestyle": "-.",
            "marker": "^",
            "linewidth": 1.10,
            "markersize": 3.2,
            "markerfacecolor": "white",
        },
        "single_split": {
            "color": "#6C5B7B",
            "linestyle": ":",
            "marker": "D",
            "linewidth": 1.00,
            "markersize": 3.0,
            "markerfacecolor": "white",
        },
        "local_only": {
            "color": "#E59B02",
            "linestyle": (0, (3.0, 1.1)),
            "marker": "v",
            "linewidth": 1.00,
            "markersize": 3.0,
            "markerfacecolor": "white",
        },
    }


def parse_run_tag(run_tag: str) -> tuple[str, str, str] | None:
    parts = run_tag.split("_")
    if len(parts) < 3:
        return None
    model = parts[0]
    weight_tag = parts[1]
    algorithm = "_".join(parts[2:])
    if not weight_tag.startswith("w"):
        return None
    return model, weight_tag, algorithm


def parse_summary_json(path: Path) -> ParetoPoint | None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stem = path.stem
    run_tag = stem.split("__", 1)[-1]
    parsed = parse_run_tag(run_tag)
    if parsed is None:
        return None

    model, weight_tag, algorithm = parsed
    delay = data.get("mean_delay")
    energy = data.get("mean_energy")
    accuracy = data.get("mean_accuracy")
    num_iterations = int(data.get("num_iterations_observed") or 0)
    tail_count_delay = int(data.get("tail_count_delay") or 0)
    tail_count_energy = int(data.get("tail_count_energy") or 0)

    if not isinstance(delay, (int, float)) or not math.isfinite(float(delay)):
        return None
    if not isinstance(energy, (int, float)) or not math.isfinite(float(energy)):
        return None
    if accuracy is not None and (not isinstance(accuracy, (int, float)) or not math.isfinite(float(accuracy))):
        accuracy = None

    return ParetoPoint(
        model=model,
        algorithm=algorithm,
        weight_tag=weight_tag,
        mean_delay=float(delay),
        mean_energy=float(energy),
        mean_accuracy=float(accuracy) if accuracy is not None else None,
        num_iterations=num_iterations,
        tail_count_delay=tail_count_delay,
        tail_count_energy=tail_count_energy,
        source_file=path,
    )


def load_points(input_dir: Path) -> list[ParetoPoint]:
    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No summary JSON files found in: {input_dir}")

    points: list[ParetoPoint] = []
    skipped: list[str] = []
    for path in json_paths:
        point = parse_summary_json(path)
        if point is None:
            skipped.append(path.name)
            continue
        points.append(point)

    if not points:
        raise ValueError(f"No valid Pareto points parsed from: {input_dir}")

    if skipped:
        print(f"[warn] Skipped {len(skipped)} JSON file(s) with unexpected naming or values.")
        for name in skipped[:10]:
            print(f"  - {name}")

    return points


def validate_points(
    points: list[ParetoPoint],
    expected_algorithms: list[str] | None,
    expected_weights: list[str] | None,
    expected_model: str | None,
) -> None:
    if expected_model is not None:
        models = sorted({p.model for p in points})
        if models != [expected_model]:
            raise ValueError(
                f"Expected only model '{expected_model}', but found models: {models}"
            )

    if expected_algorithms is None or expected_weights is None:
        return

    observed = {(p.algorithm, p.weight_tag) for p in points}
    missing = [
        (algo, wt)
        for algo in expected_algorithms
        for wt in expected_weights
        if (algo, wt) not in observed
    ]
    if missing:
        msg = ", ".join([f"{algo}/{wt}" for algo, wt in missing[:10]])
        raise ValueError(f"Missing expected Pareto points: {msg}")


def weight_rank(weight_tag: str) -> int:
    try:
        return WEIGHT_ORDER.index(weight_tag)
    except ValueError:
        return 10_000


def style_axis(ax: plt.Axes, style_cfg: dict[str, Any]) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8, width=0.6)
    if style_cfg.get("panel_bg"):
        ax.set_facecolor(style_cfg["panel_bg"])


def choose_annotation_label(weight_tag: str, fmt: str) -> str:
    if fmt == "omega":
        return WEIGHT_DISPLAY.get(weight_tag, weight_tag)
    return weight_tag


def annotate_selected_points(
    ax: plt.Axes,
    algorithm: str,
    points: list[ParetoPoint],
    mode: str,
    label_format: str,
) -> None:
    if mode == "none" or not points:
        return

    if mode == "endpoints":
        selected = [points[0]] if len(points) == 1 else [points[0], points[-1]]
    else:
        selected = points

    dx, dy = ANNOTATION_OFFSETS.get(algorithm, (4, 4))
    for point in selected:
        ax.annotate(
            choose_annotation_label(point.weight_tag, label_format),
            xy=(point.mean_delay, point.mean_energy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.0,
            ha="left",
            va="bottom",
        )


def compute_limits(values: np.ndarray, pad_ratio: float) -> tuple[float, float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = max(vmax - vmin, 1e-12)
    pad = span * pad_ratio
    return vmin - pad, vmax + pad


def plot_pareto_front(
    points: list[ParetoPoint],
    out_dir: Path,
    style_name: str,
    annotate_weights: str,
    annotation_format: str,
    legend_position: str,
    file_stem: str,
) -> None:
    style_cfg = apply_style(style_name)
    visuals = algorithm_visuals(style_name)

    grouped: dict[str, list[ParetoPoint]] = {}
    for algorithm in ALGO_ORDER:
        algo_points = [p for p in points if p.algorithm == algorithm]
        if algo_points:
            grouped[algorithm] = sorted(algo_points, key=lambda p: weight_rank(p.weight_tag))

    fig, ax = plt.subplots(figsize=(3.48, 2.80))
    handles = []
    labels = []

    for algorithm, algo_points in grouped.items():
        vis = visuals[algorithm]
        x = np.array([p.mean_delay for p in algo_points], dtype=float)
        y = np.array([p.mean_energy for p in algo_points], dtype=float)

        # Local-Only: all 7 weights collapse to the same point → draw as single marker
        is_single_point = (np.ptp(x) < 1e-6 and np.ptp(y) < 1e-6)

        if is_single_point:
            handle = ax.scatter(
                [x[0]], [y[0]],
                color=vis["color"],
                marker=vis["marker"],
                s=vis["markersize"] ** 2 * 1.5,
                edgecolors=vis["color"],
                facecolors=vis.get("markerfacecolor", vis["color"]),
                linewidths=0.8,
                zorder=4,
                label=ALGO_DISPLAY.get(algorithm, algorithm),
            )
            handles.append(handle)
        else:
            line, = ax.plot(
                x,
                y,
                color=vis["color"],
                linestyle=vis["linestyle"],
                linewidth=vis["linewidth"],
                marker=vis["marker"],
                markersize=vis["markersize"],
                markerfacecolor=vis["markerfacecolor"],
                markeredgecolor=vis["color"],
                markeredgewidth=0.70,
                solid_capstyle="round",
                dash_capstyle="round",
                zorder=3 if algorithm == "proposed" else 2,
                label=ALGO_DISPLAY.get(algorithm, algorithm),
            )
            handles.append(line)
        labels.append(ALGO_DISPLAY.get(algorithm, algorithm))
        annotate_selected_points(ax, algorithm, algo_points, annotate_weights, annotation_format)

    all_delay = np.array([p.mean_delay for p in points], dtype=float)
    all_energy = np.array([p.mean_energy for p in points], dtype=float)
    xlim = compute_limits(all_delay, pad_ratio=0.08)
    ylim = compute_limits(all_energy, pad_ratio=0.08)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Inference Delay (s)", fontsize=8.0)
    ax.set_ylabel(r"Energy per Request $E_{\mathrm{req}}$ (J)", fontsize=8.0)
    ax.tick_params(axis="both", labelsize=7.0)
    style_axis(ax, style_cfg)

    if legend_position == "top":
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
            columnspacing=1.0,
            handlelength=1.6,
            fontsize=7.0,
            borderaxespad=0.0,
        )
        rect = [0, 0, 1, 0.88]
    else:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            frameon=False,
            columnspacing=1.0,
            handlelength=1.6,
            fontsize=7.0,
            borderaxespad=0.0,
        )
        rect = [0, 0.10, 1, 1.0]

    fig.tight_layout(rect=rect)
    save_figure(fig, out_dir / file_stem, formats=(".pdf", ".png"))
    plt.close(fig)


def print_summary(points: list[ParetoPoint]) -> None:
    print(f"Loaded {len(points)} Pareto points.")
    print("Algorithms:")
    for algorithm in ALGO_ORDER:
        subset = [p for p in points if p.algorithm == algorithm]
        if not subset:
            continue
        weights = ", ".join([p.weight_tag for p in sorted(subset, key=lambda p: weight_rank(p.weight_tag))])
        print(f"  - {ALGO_DISPLAY.get(algorithm, algorithm)}: {len(subset)} point(s) [{weights}]")

    tail_delay = sorted({p.tail_count_delay for p in points})
    tail_energy = sorted({p.tail_count_energy for p in points})
    observed_iters = sorted({p.num_iterations for p in points})
    print(f"Tail count (delay): {tail_delay}")
    print(f"Tail count (energy): {tail_energy}")
    print(f"Observed iterations: {observed_iters}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TMC-grade Pareto fronts from E2 tail-summary JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/pareto_tail50/e2_pareto_7w_i1000",
        help="Directory containing tail-summary JSON files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/figures/pareto_e2",
        help="Output directory for PDF/PNG figures.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="ieee",
        choices=list(STYLES),
        help="Visual style preset.",
    )
    parser.add_argument(
        "--annotate_weights",
        type=str,
        default="none",
        choices=["none", "endpoints", "all"],
        help="Whether to annotate preference weights on the front.",
    )
    parser.add_argument(
        "--annotation_format",
        type=str,
        default="tag",
        choices=["tag", "omega"],
        help="Format used when annotate_weights is enabled.",
    )
    parser.add_argument(
        "--legend_position",
        type=str,
        default="top",
        choices=["top", "bottom"],
        help="Shared legend location.",
    )
    parser.add_argument(
        "--file_stem",
        type=str,
        default="fig_pareto_e2_mtan",
        help="Base filename (without extension).",
    )
    parser.add_argument(
        "--tail_iters",
        type=int,
        default=50,
        help="Expected tail length for sanity checking only.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require the expected 3 algorithms × 7 weights to be present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(input_dir)
    if args.strict:
        validate_points(
            points,
            expected_algorithms=["proposed", "partition_routing_only", "mlp_ppo", "single_split", "local_only"],
            expected_weights=WEIGHT_ORDER,
            expected_model="mtan",
        )

    unexpected_delay_tail = sorted({p.tail_count_delay for p in points if p.tail_count_delay != args.tail_iters})
    unexpected_energy_tail = sorted({p.tail_count_energy for p in points if p.tail_count_energy != args.tail_iters})
    if unexpected_delay_tail or unexpected_energy_tail:
        print("[warn] Some points do not match the requested tail length.")
        print(f"  delay tail counts  : {sorted({p.tail_count_delay for p in points})}")
        print(f"  energy tail counts : {sorted({p.tail_count_energy for p in points})}")

    print_summary(points)
    plot_pareto_front(
        points=points,
        out_dir=out_dir,
        style_name=args.style,
        annotate_weights=args.annotate_weights,
        annotation_format=args.annotation_format,
        legend_position=args.legend_position,
        file_stem=args.file_stem,
    )
    print(f"Done. Figure saved to {out_dir}/")


if __name__ == "__main__":
    main()
