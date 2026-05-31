"""
Compare saved evaluation JSON files for paper baselines.

Usage:
    python scripts/compare_algorithms.py \
        --result joint_prc=results/eval/split_joint_prc.json \
        --result local_only=results/eval/split_local_only.json \
        --result single_split=results/eval/split_single_split.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare saved evaluation results")
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Labelled result in the form label=path/to/result.json; can be repeated",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/tables/comparison.csv",
        help="Path to save the aggregated CSV table",
    )
    args = parser.parse_args()

    rows = {}
    for item in args.result:
        if "=" not in item:
            raise ValueError(f"Invalid --result entry: {item}. Expected label=path.")
        label, raw_path = item.split("=", 1)
        result_path = Path(raw_path)
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_path}")
        rows[label] = _load_json(result_path)
        print(f"Loaded {label}: {result_path}")

    df = pd.DataFrame(rows).T
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"\nSaved comparison table to {output_path}")


if __name__ == "__main__":
    main()

