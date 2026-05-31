"""
Plot TMC-grade sweep figures for E3..E7.

Generates a 2x2 figure (Reward / Delay / Energy / Accuracy vs swept parameter)
for whichever sweep tag is present in the eval directory.

Supported sweeps (auto-detected by tag, can be filtered with --sweeps):
  arrival_rate   tag prefix "lambda"  -> fig_robustness_arrival_rate.{pdf,png}
  swarm_size     tag prefix "uav"     -> fig_robustness_swarm_size.{pdf,png}
  area           tag prefix "area"    -> fig_sensitivity_area.{pdf,png}
  taskdens       tag prefix "td"      -> fig_sensitivity_taskdens.{pdf,png}
  maxrange       tag prefix "mr"      -> fig_sensitivity_maxrange.{pdf,png}

Example:
    python scripts/plot_robustness_e5.py \
        --eval_dir results/eval/e5_area_w19 \
        --out_dir results/figures/sensitivity_area \
        --sweeps area \
        --style ieee
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from plot_convergence import STYLES, apply_style, save_figure

# ── Algorithm display names & order (consistent with E2) ─────────────────────
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


def algorithm_visuals(style_name: str) -> Dict[str, Dict[str, Any]]:
    """Line styles consistent with E2 Pareto figure."""
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


def style_axis(ax, style_cfg: dict):
    """Apply axis styling consistent with E2."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.tick_params(direction="out", length=2.8, width=0.6)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)


def load_e5_data(eval_dir: Path, sweep: str) -> Dict[str, Dict[float, Dict[str, float]]]:
    """
    Load E5 JSON files for a given sweep type.

    Returns:
        {algo: {param_value: {"delay": ..., "energy": ..., "accuracy": ...}}}
    """
    # Deterministic baselines whose decisions don't depend on swarm size:
    # task UAV hardware is pinned (min==max in source_profile), so no
    # special handling is needed — kept as comment for documentation.

    data: Dict[str, Dict[float, Dict[str, float]]] = {}

    for jf in sorted(eval_dir.glob("*.json")):
        name = jf.stem  # e.g. mtan_w19_proposed_lambda0_5 or mtan_w19_local_only_uav10

        if sweep == "arrival_rate":
            m = re.search(r"lambda(\d+)_(\d+)$", name) or re.search(r"lambda(\d+)$", name)
            if not m:
                continue
            param_val = float(f"{m.group(1)}.{m.group(2)}") if m.lastindex == 2 else float(m.group(1))
        elif sweep == "swarm_size":
            m = re.search(r"uav(\d+)$", name)
            if not m:
                continue
            param_val = float(m.group(1))
        elif sweep == "area":
            m = re.search(r"area(\d+)$", name)
            if not m:
                continue
            param_val = float(m.group(1))
        elif sweep == "taskdens":
            m = re.search(r"td(\d+)_(\d+)$", name) or re.search(r"td(\d+)$", name)
            if not m:
                continue
            param_val = float(f"{m.group(1)}.{m.group(2)}") if m.lastindex == 2 else float(m.group(1))
        elif sweep == "maxrange":
            m = re.search(r"mr(\d+)$", name)
            if not m:
                continue
            param_val = float(m.group(1))
        else:
            continue

        # Determine algorithm
        algo = None
        for a in ALGO_ORDER:
            if f"_{a}_" in name or name.endswith(f"_{a}"):
                algo = a
                break
        if algo is None:
            continue

        with open(jf) as f:
            d = json.load(f)

        delay = d.get("mean_delay")
        energy = d.get("mean_energy")
        accuracy = d.get("mean_accuracy")
        reward = d.get("mean_reward")
        mean_length = d.get("mean_length", None)

        delay_std = d.get("std_delay")
        energy_std = d.get("std_energy")
        accuracy_std = d.get("std_accuracy")
        reward_std = d.get("std_reward")

        # Fix: if reward is episode-total (deterministic baseline old format),
        # normalize to per-step
        if reward is not None and mean_length is not None and mean_length > 1 and abs(reward) > 10:
            reward = reward / mean_length
            if reward_std is not None:
                reward_std = reward_std / mean_length

        if delay is None or energy is None:
            continue

        data.setdefault(algo, {})[param_val] = {
            "delay": delay,
            "energy": energy,
            "accuracy": accuracy,
            "reward": reward,
            "delay_std": delay_std,
            "energy_std": energy_std,
            "accuracy_std": accuracy_std,
            "reward_std": reward_std,
        }

    return data


def plot_robustness_figure(
    data: Dict[str, Dict[float, Dict[str, float]]],
    sweep: str,
    out_dir: Path,
    style_name: str = "ieee",
):
    """Plot one robustness figure with 2x2 subplots:
    (a) weighted objective  (b) accuracy  (c) delay  (d) energy."""
    apply_style(style_name)
    style_cfg = STYLES.get(style_name, STYLES["ieee"])

    vis = algorithm_visuals(style_name)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(3.48, 2.80),
        constrained_layout=True,
    )
    ax_reward, ax_delay = axes[0]
    ax_energy, ax_acc = axes[1]

    # Determine tick labels from first available algo
    first_algo = next((a for a in ALGO_ORDER if a in data), None)
    all_params = sorted(data[first_algo].keys()) if first_algo else []
    x_pos = list(range(len(all_params)))

    for algo in ALGO_ORDER:
        if algo not in data:
            continue
        points = data[algo]
        params = sorted(points.keys())
        delays = [points[p]["delay"] for p in params]
        energies = [points[p]["energy"] for p in params]
        accuracies = [points[p].get("accuracy", 1.0) or 1.0 for p in params]
        rewards = [points[p].get("reward", 0.0) for p in params]

        xp = list(range(len(params)))
        v = vis.get(algo, {})
        common = dict(
            label=ALGO_DISPLAY.get(algo, algo),
            color=v.get("color", "gray"),
            linestyle=v.get("linestyle", "-"),
            marker=v.get("marker", "o"),
            linewidth=v.get("linewidth", 1.0),
            markersize=v.get("markersize", 3.0),
            markerfacecolor=v.get("markerfacecolor", "white"),
            markeredgewidth=0.70,
            markeredgecolor=v.get("color", "gray"),
        )

        # Note: Single-seed deterministic evaluation; we intentionally do not
        # render error bars/bands. The std fields in JSON come from heterogeneous
        # sources (PPO tail iterations vs deterministic episode replays) and
        # mixing them on the same axis would be misleading.

        ax_reward.plot(xp, rewards, **common)
        ax_acc.plot(xp, accuracies, **common)
        ax_delay.plot(xp, delays, **common)
        ax_energy.plot(xp, energies, **common)

    # X-axis ticks
    if sweep == "arrival_rate":
        xlabel = r"Arrival Rate $\lambda$ (tasks/s)"
        file_stem = "fig_robustness_arrival_rate"
        tick_labels = [str(v) for v in all_params]
    elif sweep == "swarm_size":
        xlabel = "Number of UAVs"
        file_stem = "fig_robustness_swarm_size"
        tick_labels = [str(int(v)) for v in all_params]
    elif sweep == "area":
        xlabel = r"Deployment Area Side $L$ (m)"
        file_stem = "fig_sensitivity_area"
        tick_labels = [str(int(v)) for v in all_params]
    elif sweep == "taskdens":
        xlabel = r"Average Tasks per Request $\bar{D}$"
        file_stem = "fig_sensitivity_taskdens"
        tick_labels = [f"{v:g}" for v in all_params]
    elif sweep == "maxrange":
        xlabel = r"Link Maximum Range $d_{\max}$ (m)"
        file_stem = "fig_sensitivity_maxrange"
        tick_labels = [str(int(v)) for v in all_params]
    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    subtitles = [
        "(a) Average Reward",
        "(b) Inference Delay (s)",
        r"(c) Energy per Request $E_{\mathrm{req}}$ (J)",
        "(d) Inference Accuracy",
    ]
    ylabels = [
        "Reward",
        "Delay (s)",
        r"$E_{\mathrm{req}}$ (J)",
        "Accuracy",
    ]

    # Auto-thin x tick labels when they would otherwise overlap.
    n = len(tick_labels)
    avg_label_chars = (
        sum(len(t) for t in tick_labels) / max(n, 1) if tick_labels else 0
    )
    needs_thin = (avg_label_chars * n) > 30 and n >= 6
    if needs_thin:
        keep = set(range(0, n, 2))
        keep.add(n - 1)  # always show the right endpoint
        # If keeping the last index makes it adjacent to a previously-kept
        # one (e.g. n=10 keeps {0,2,4,6,8,9} where 8 and 9 collide),
        # drop the previously-kept neighbour to restore spacing.
        if (n - 2) in keep and n >= 2:
            keep.discard(n - 2)
        thinned = [t if i in keep else "" for i, t in enumerate(tick_labels)]
    else:
        thinned = tick_labels

    for idx, ax in enumerate([ax_reward, ax_delay, ax_energy, ax_acc]):
        ax.set_xticks(x_pos)
        ax.set_xticklabels(thinned)
        ax.set_xlim(-0.3, len(x_pos) - 0.7)
        ax.set_xlabel(xlabel, fontsize=7.0)
        ax.set_ylabel(ylabels[idx], fontsize=7.0)
        ax.tick_params(axis="both", labelsize=6.0)
        ax.set_title(subtitles[idx], fontsize=7.5, pad=3)
        style_axis(ax, style_cfg)

    # Shared legend at top
    handles, labels = ax_reward.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.4,
        fontsize=6.5,
        borderaxespad=0.0,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir / file_stem, formats=(".pdf", ".png"))
    print(f"  Saved: {out_dir / file_stem}.pdf")
    print(f"  Saved: {out_dir / file_stem}.png")
    plt.close(fig)


SUPPORTED_SWEEPS = ("arrival_rate", "swarm_size", "area", "taskdens", "maxrange")


def main():
    parser = argparse.ArgumentParser(description="Plot sweep figures for E3..E7")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Directory containing sweep eval JSON files")
    parser.add_argument("--out_dir", type=str, default="results/figures/sweep",
                        help="Output directory for figures")
    parser.add_argument("--sweeps", type=str, nargs="+", default=None,
                        choices=list(SUPPORTED_SWEEPS),
                        help="Sweep types to plot (default: auto-detect all present in eval_dir)")
    parser.add_argument("--style", type=str, default="ieee",
                        choices=list(STYLES.keys()),
                        help="Visual style preset")
    parser.add_argument("--exclude_x", type=float, nargs="+", default=None,
                        help="Drop these sweep parameter values from the plot "
                             "(e.g. --exclude_x 24 to remove the 24-UAV point). "
                             "JSON files on disk are untouched.")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)

    sweeps = list(args.sweeps) if args.sweeps else list(SUPPORTED_SWEEPS)

    for sweep in sweeps:
        print(f"Loading {sweep} sweep data from {eval_dir}/...")
        data = load_e5_data(eval_dir, sweep)
        if not data:
            print(f"  (no files matched the {sweep} tag, skipping)")
            print()
            continue
        if args.exclude_x:
            removed = 0
            for algo in list(data.keys()):
                for v in list(data[algo].keys()):
                    if any(abs(v - float(x)) < 1e-6 for x in args.exclude_x):
                        del data[algo][v]
                        removed += 1
            print(f"  excluded x values {args.exclude_x} -> dropped {removed} points")
        for algo, points in data.items():
            print(f"  {ALGO_DISPLAY.get(algo, algo)}: {len(points)} points")
        plot_robustness_figure(data, sweep, out_dir, args.style)
        print()

    print(f"Done. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
