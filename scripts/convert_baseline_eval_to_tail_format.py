"""
Convert deterministic baseline evaluation JSON files to the same format
as learning algorithm tail-summary JSON files, so that plot_pareto_e2.py
can read all algorithms from a single directory.

Usage:
    python scripts/convert_baseline_eval_to_tail_format.py \
        --input_dir results/eval/e2_pareto_localdet_50ep \
        --output_dir results/pareto_tail50/e2_pareto_7w_i1000
"""

import argparse
import json
from pathlib import Path


# Map from baseline eval JSON field names to tail-summary field names
FIELD_MAP = {
    "mean_delay_per_step": "mean_delay",
    "mean_energy_per_step": "mean_energy",
    "mean_accuracy": "mean_accuracy",
    "std_reward": "std_reward",
    "mean_reward": "mean_reward",
    "num_episodes": "num_episodes",
}

# Experiment name prefix used in the output filename
EXPERIMENT_NAMES = {
    "local_only": "baseline_local_only_tmc_stable",
    "single_split": "baseline_single_split_tmc_stable",
}


def parse_run_tag(filename_stem: str) -> tuple[str, str, str] | None:
    """Parse 'mtan_w91_single_split' -> ('mtan', 'w91', 'single_split')."""
    parts = filename_stem.split("_")
    if len(parts) < 3 or not parts[1].startswith("w"):
        return None
    model = parts[0]
    weight_tag = parts[1]
    algorithm = "_".join(parts[2:])
    return model, weight_tag, algorithm


def convert_one(src: Path, dst: Path) -> None:
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {
        "source": "deterministic_baseline_eval",
        "original_file": str(src),
        "num_iterations_observed": int(data.get("num_episodes", 0)),
        "tail_iterations_requested": int(data.get("num_episodes", 0)),
        "mean_reward": data.get("mean_reward"),
        "std_reward": data.get("std_reward"),
        "mean_delay": data.get("mean_delay_per_step"),
        "std_delay": 0.0,
        "tail_count_delay": int(data.get("num_episodes", 0)),
        "mean_energy": data.get("mean_energy_per_step"),
        "std_energy": 0.0,
        "tail_count_energy": int(data.get("num_episodes", 0)),
        "mean_accuracy": data.get("mean_accuracy"),
        "std_accuracy": 0.0,
        "tail_count_accuracy": int(data.get("num_episodes", 0)),
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  {src.name}  ->  {dst.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert baseline eval JSON to tail-summary format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/eval/e2_pareto_localdet_50ep",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pareto_tail50/e2_pareto_7w_i1000",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No JSON files found in {input_dir}")
        return

    converted = 0
    for src in json_files:
        parsed = parse_run_tag(src.stem)
        if parsed is None:
            print(f"  [skip] {src.name} (unexpected naming)")
            continue

        model, weight_tag, algorithm = parsed
        exp_name = EXPERIMENT_NAMES.get(algorithm)
        if exp_name is None:
            print(f"  [skip] {src.name} (unknown algorithm '{algorithm}')")
            continue

        # Output filename matches learning algo convention:
        #   <experiment_name>__<model>_<weight>_<algorithm>.json
        dst_name = f"{exp_name}__{model}_{weight_tag}_{algorithm}.json"
        dst = output_dir / dst_name
        convert_one(src, dst)
        converted += 1

    print(f"\nConverted {converted} file(s) to {output_dir}/")


if __name__ == "__main__":
    main()
