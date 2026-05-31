"""
Check NaN/Inf values in UAVEnv observations and metrics.
"""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_loader import load_config
from src.env import UAVEnv


def has_bad(arr) -> bool:
    arr = np.asarray(arr)
    return np.isnan(arr).any() or np.isinf(arr).any()


def main():
    cfg = load_config("configs/experiments/main_gnn_ppo.yaml")
    env = UAVEnv(cfg)
    obs, _ = env.reset(seed=0)
    for k, v in obs.items():
        if has_bad(v):
            print("BAD_RESET", k, np.nanmin(v), np.nanmax(v))
            return

    for i in range(20):
        act = env.action_space.sample()
        obs, rew, done, trunc, info = env.step(act)
        for k, v in obs.items():
            if has_bad(v):
                print("BAD_STEP", i, k, np.nanmin(v), np.nanmax(v))
                return
        if np.isnan(rew) or np.isinf(rew):
            print("BAD_REWARD", i, rew)
            return
        if done or trunc:
            obs, _ = env.reset(seed=0)

    print("ENV_NUMERIC_OK")


if __name__ == "__main__":
    main()
