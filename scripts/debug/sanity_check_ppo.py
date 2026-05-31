"""
Minimal sanity check for UAVEnv + GNN-PPO training pipeline.

This script runs a tiny rollout and one PPO update to catch runtime errors
in state/action/reward/data-flow integration.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_loader import load_config
from src.env import UAVEnv
from src.core import AgentRegistry
import src.agents  # noqa: F401


def main():
    cfg = load_config("configs/experiments/main_gnn_ppo.yaml")
    cfg["device"] = "cpu"
    cfg["algorithm"]["ppo"]["n_steps"] = 8
    cfg["algorithm"]["ppo"]["n_epochs"] = 2
    cfg["algorithm"]["ppo"]["batch_size"] = 4
    cfg["training"]["max_steps_per_episode"] = 4

    env = UAVEnv(cfg)
    agent = AgentRegistry.create(cfg["algorithm"]["name"], env, cfg)

    state, _ = env.reset(seed=0)
    for _ in range(8):
        action = agent.select_action(state, deterministic=False)
        next_state, reward, done, truncated, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done or truncated, info)
        state = next_state
        if done or truncated:
            state, _ = env.reset(seed=0)

    metrics = agent.update()
    print("SANITY_CHECK_OK", metrics)


if __name__ == "__main__":
    main()
