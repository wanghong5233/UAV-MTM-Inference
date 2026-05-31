"""
Short smoke training for UAVEnv + GNN-PPO.

Runs a few episodes and checks for finite losses/parameters.
"""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_loader import load_config
from src.env import UAVEnv
from src.core import AgentRegistry, Trainer
import src.agents  # noqa: F401


def check_finite_params(module: torch.nn.Module) -> bool:
    for p in module.parameters():
        if not torch.isfinite(p).all():
            return False
    return True


def main():
    cfg = load_config("configs/experiments/main_gnn_ppo.yaml")
    cfg["device"] = "cpu"
    cfg["training"]["num_episodes"] = 3
    cfg["training"]["max_steps_per_episode"] = 20
    cfg["training"]["eval_interval"] = 1000
    cfg["training"]["save_interval"] = 1000
    cfg["algorithm"]["ppo"]["n_steps"] = 20
    cfg["algorithm"]["ppo"]["n_epochs"] = 2
    cfg["algorithm"]["ppo"]["batch_size"] = 10

    env = UAVEnv(cfg)
    agent = AgentRegistry.create(cfg["algorithm"]["name"], env, cfg)
    trainer = Trainer(agent=agent, config=cfg, logger=None, evaluator=None)
    trainer.train()

    if not check_finite_params(agent):
        raise RuntimeError("Non-finite PPO parameters detected after smoke training.")

    print("SMOKE_TRAIN_OK")


if __name__ == "__main__":
    main()
