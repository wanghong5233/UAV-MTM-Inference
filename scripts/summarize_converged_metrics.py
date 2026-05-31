"""
Summarize converged metrics from a training run without loading checkpoints.

This script reads `metrics_long.csv` and computes tail statistics for the
paper-facing metrics recorded online during training.
"""

import argparse
import csv
import json
import math
import statistics
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


KEY_MAP = {
    "metrics/reward": "reward",
    "metrics/delay": "delay",
    "metrics/energy": "energy",
    "metrics/accuracy": "accuracy",
    "metrics/comm_volume": "comm_volume",
    "metrics/peak_rho": "peak_rho",
    "metrics_debug/energy_compute": "energy_compute",
    "metrics_debug/energy_comm": "energy_comm",
}


def load_config(config_path: str) -> Dict[str, Any]:
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path_obj}")

    with open(config_path_obj, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if "base" in config:
        base_configs = config.pop("base")
        if not isinstance(base_configs, list):
            base_configs = [base_configs]

        merged: Dict[str, Any] = {}
        for base_path in base_configs:
            if not Path(base_path).is_absolute():
                base_path = config_path_obj.parent / base_path
            base_config = load_config(str(base_path))
            merged = merge_configs(merged, base_config)
        config = merge_configs(merged, config)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize tail metrics from metrics_long.csv")
    parser.add_argument("--config", type=str, default=None, help="Experiment config path")
    parser.add_argument("--run_tag", type=str, default=None, help="Run tag used by train.py")
    parser.add_argument("--run_group", type=str, default=None,
                        help="Optional run group under logs/training/, e.g. e2_pareto")
    parser.add_argument("--run_dir", type=str, default=None, help="Run directory containing metrics_long.csv")
    parser.add_argument("--tail_iters", type=int, default=100, help="Number of final iterations to average")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    return parser.parse_args()


def resolve_run_dir(config_path: str, run_tag: str, run_group: str = None) -> Path:
    config = load_config(config_path)
    experiment_name = config.get("experiment", {}).get("name", "default")
    run_root = Path("logs") / "training"
    if run_group:
        run_root = run_root / str(run_group)
    return run_root / experiment_name / run_tag


def load_series(csv_path: Path) -> Dict[str, List[Tuple[int, float]]]:
    series: Dict[str, List[Tuple[int, float]]] = {key: [] for key in KEY_MAP}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("key", "")
            if key not in series:
                continue
            try:
                step = int(float(row["step"]))
                value = float(row["value"])
            except Exception:
                continue
            if math.isfinite(value):
                series[key].append((step, value))
    return series


def summarize_tail(values: List[Tuple[int, float]], tail_iters: int) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "tail_count": 0,
            "last_step": None,
            "last_value": None,
            "mean": None,
            "std": None,
        }

    tail = values[-tail_iters:] if tail_iters > 0 else values
    tail_values = [v for _, v in tail]
    std = statistics.pstdev(tail_values) if len(tail_values) > 1 else 0.0
    return {
        "count": len(values),
        "tail_count": len(tail_values),
        "last_step": values[-1][0],
        "last_value": values[-1][1],
        "mean": statistics.fmean(tail_values),
        "std": std,
    }


def build_summary(run_dir: Path, tail_iters: int) -> Dict:
    csv_path = run_dir / "metrics_long.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics_long.csv not found: {csv_path}")

    series = load_series(csv_path)
    summary = {
        "source": "training_tail_mean",
        "run_dir": str(run_dir),
        "metrics_csv": str(csv_path),
        "tail_iterations_requested": int(tail_iters),
    }

    reward_series = series.get("metrics/reward", [])
    summary["num_iterations_observed"] = len(reward_series)

    for raw_key, alias in KEY_MAP.items():
        stats = summarize_tail(series.get(raw_key, []), tail_iters=tail_iters)
        summary[f"mean_{alias}"] = stats["mean"]
        summary[f"std_{alias}"] = stats["std"]
        summary[f"last_{alias}"] = stats["last_value"]
        summary[f"last_step_{alias}"] = stats["last_step"]
        summary[f"count_{alias}"] = stats["count"]
        summary[f"tail_count_{alias}"] = stats["tail_count"]

    return summary


def main() -> None:
    args = parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        if not args.config or not args.run_tag:
            raise ValueError("Either --run_dir or both --config and --run_tag must be provided")
        run_dir = resolve_run_dir(args.config, args.run_tag, args.run_group)

    summary = build_summary(run_dir=run_dir, tail_iters=int(args.tail_iters))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Summary saved to {output_path}")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
