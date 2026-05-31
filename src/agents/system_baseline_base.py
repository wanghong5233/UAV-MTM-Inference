"""
Shared helpers for deterministic system baselines.
"""

from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import numpy as np

from src.core import BaseAgent


class SystemBaselineAgent(BaseAgent):
    """Utility base class for non-learning baselines."""

    def __init__(self, env: gym.Env, config: Dict):
        super().__init__(env, config)
        self.num_nodes = int(env.observation_space["model_nodes"].shape[0])
        self.num_groups = int(env.observation_space["prev_compression"].shape[0])
        self.source_uav = int(getattr(env, "source_uav", 0))
        bit_widths = list(getattr(env, "bit_widths", [32, 16, 8, 4]))
        self.full_precision_idx = int(np.argmax(np.asarray(bit_widths, dtype=np.int32))) if bit_widths else 0

    def _zero_partition(self) -> np.ndarray:
        return np.zeros(self.num_nodes - 1, dtype=np.int32)

    def _full_precision_compression(self) -> np.ndarray:
        return np.full(self.num_groups, self.full_precision_idx, dtype=np.int32)

    def _pad_block_routing(self, routing_blocks: np.ndarray) -> np.ndarray:
        routing_full = np.full(self.num_nodes, self.source_uav, dtype=np.int32)
        if routing_blocks.size > 0:
            routing_full[: routing_blocks.shape[0]] = routing_blocks.astype(np.int32)
        return routing_full

    def _score_candidate(self, action: Dict) -> Dict:
        return self.env.evaluate_action_candidate(action)

    def _is_better(self, candidate: Dict, incumbent: Optional[Dict]) -> bool:
        if incumbent is None:
            return True
        cand_reward = float(candidate["reward"])
        inc_reward = float(incumbent["reward"])
        if abs(cand_reward - inc_reward) > 1e-9:
            return cand_reward > inc_reward

        cand_info = candidate["info"]
        inc_info = incumbent["info"]
        cand_delay = float(cand_info.get("delay", 0.0))
        inc_delay = float(inc_info.get("delay", 0.0))
        if abs(cand_delay - inc_delay) > 1e-9:
            return cand_delay < inc_delay

        cand_energy = float(cand_info.get("energy", 0.0))
        inc_energy = float(inc_info.get("energy", 0.0))
        if abs(cand_energy - inc_energy) > 1e-9:
            return cand_energy < inc_energy

        cand_acc = float(cand_info.get("accuracy", 0.0))
        inc_acc = float(inc_info.get("accuracy", 0.0))
        return cand_acc > inc_acc

    def update(self, batch=None) -> Dict[str, float]:
        return {}

    def save(self, path: str):
        return None

    def load(self, path: str):
        return None
