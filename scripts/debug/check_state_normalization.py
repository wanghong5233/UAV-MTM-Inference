"""
Check observation normalization ranges and out-of-bound ratios.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_loader import load_config
from src.env import UAVEnv


def stat(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "oob_ratio": float(np.mean((arr < 0.0) | (arr > 1.0))),
    }


def main():
    cfg = load_config("configs/experiments/main_gnn_ppo_tmc_stable.yaml")
    cfg["device"] = "cpu"
    env = UAVEnv(cfg)

    keys = [
        "model_nodes",
        "model_adj",
        "uav_nodes",
        "uav_adj",
        "uav_rate",
        "task_popularity",
        "arrival_rate",
        "preference",
        "prev_partition",
        "prev_routing",
        "prev_compression",
    ]

    data = {k: [] for k in keys}
    for ep in range(8):
        obs, _ = env.reset(seed=ep)
        for _ in range(20):
            for k in keys:
                data[k].append(np.asarray(obs[k], dtype=np.float32).reshape(-1))
            obs, r, d, tr, info = env.step(env.action_space.sample())
            if d or tr:
                break

    print("=" * 100)
    print("State Normalization Check ([0,1] range)")
    print("=" * 100)
    for k in keys:
        x = np.concatenate(data[k], axis=0)
        s = stat(x)
        print(
            f"{k:>16} | min={s['min']:.4f} max={s['max']:.4f} "
            f"mean={s['mean']:.4f} std={s['std']:.4f} oob={s['oob_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
