"""
Check reward raw/signal distributions for scaling sanity.

This script verifies:
1) Raw environment rewards are preserved (for logging/plotting).
2) Training reward signals are bounded and roughly zero-centered (window standardization).
3) Clip ratio is low (signal not heavily truncated).
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_loader import load_config
from src.env import UAVEnv
from src.core import AgentRegistry, Trainer
import src.agents  # noqa: F401


def summarize(x: np.ndarray) -> dict:
    if x.size == 0:
        return {}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p01": float(np.percentile(x, 1)),
        "p50": float(np.percentile(x, 50)),
        "p99": float(np.percentile(x, 99)),
        "pos_ratio": float(np.mean(x > 0)),
        "neg_ratio": float(np.mean(x < 0)),
    }


def main():
    cfg = load_config("configs/experiments/main_gnn_ppo_tmc_stable.yaml")
    cfg["device"] = "cpu"
    # For debug speed, shorten warmup so we can inspect both phases quickly.
    cfg["reward"]["signal_scaling"]["warmup_episodes"] = 10

    env = UAVEnv(cfg)
    agent = AgentRegistry.create(cfg["algorithm"]["name"], env, cfg)
    trainer = Trainer(agent=agent, config=cfg, logger=None, evaluator=None)

    raw_warmup = []
    sig_warmup = []
    clip_warmup = []
    raw_main = []
    sig_main = []
    clip_main = []

    num_episodes = 30
    steps_per_ep = 40
    warmup_episodes = int(cfg["reward"]["signal_scaling"]["warmup_episodes"])
    target_main = float(cfg["reward"]["signal_scaling"]["target_abs_max"])
    target_warmup = float(cfg["reward"]["signal_scaling"]["warmup_target_abs_max"])

    for ep in range(num_episodes):
        trainer.episode_count = ep
        obs, _ = env.reset(seed=ep)
        for _ in range(steps_per_ep):
            a = env.action_space.sample()
            obs, r_raw, d, tr, info = env.step(a)
            r_sig, clipped = trainer._scale_reward_for_update(float(r_raw))
            if ep < warmup_episodes:
                raw_warmup.append(float(r_raw))
                sig_warmup.append(float(r_sig))
                clip_warmup.append(float(clipped))
            else:
                raw_main.append(float(r_raw))
                sig_main.append(float(r_sig))
                clip_main.append(float(clipped))
            if d or tr:
                break

    raw_warmup = np.asarray(raw_warmup, dtype=np.float64)
    sig_warmup = np.asarray(sig_warmup, dtype=np.float64)
    clip_warmup = np.asarray(clip_warmup, dtype=np.float64)
    raw_main = np.asarray(raw_main, dtype=np.float64)
    sig_main = np.asarray(sig_main, dtype=np.float64)
    clip_main = np.asarray(clip_main, dtype=np.float64)

    print("=" * 100)
    print("Reward Scaling Distribution Check")
    print("=" * 100)
    print(f"mode={trainer.reward_signal_scaling_mode}, warmup_episodes={warmup_episodes}")
    print(f"target_abs_warmup={target_warmup}, target_abs_main={target_main}")
    print()

    print("[Raw Reward] first 20 samples:")
    print(np.array2string(raw_main[:20], precision=4, separator=", "))
    print()

    print("[Warmup Phase] raw stats:")
    print(summarize(raw_warmup))
    print("[Warmup Phase] signal stats:")
    print(summarize(sig_warmup))
    print(f"[Warmup Phase] clip_ratio={float(np.mean(clip_warmup)):.4f}")
    print(f"[Warmup Phase] bound_check max_abs={float(np.max(np.abs(sig_warmup))):.4f}")
    print()

    print("[Main Phase] raw stats:")
    print(summarize(raw_main))
    print("[Main Phase] signal stats:")
    print(summarize(sig_main))
    print(f"[Main Phase] clip_ratio={float(np.mean(clip_main)):.4f}")
    print(f"[Main Phase] bound_check max_abs={float(np.max(np.abs(sig_main))):.4f}")
    print(f"[Main Phase] near_zero_mean={float(np.mean(sig_main)):.4f}")
    print(
        "[Main Phase] side_balance="
        f"pos={float(np.mean(sig_main > 0)):.3f}, neg={float(np.mean(sig_main < 0)):.3f}"
    )


if __name__ == "__main__":
    main()
