"""
Conventional single-split computing baseline.
"""

from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import numpy as np

from src.core import AgentRegistry
from .system_baseline_base import SystemBaselineAgent


@AgentRegistry.register("single_split")
class SingleSplitAgent(SystemBaselineAgent):
    """
    Restrict deployment to one optional cut:
    source UAV executes the prefix, one worker UAV executes the suffix.
    """

    def __init__(self, env: gym.Env, config: Dict):
        super().__init__(env, config)
        policy_cfg = config.get("algorithm", {}).get("policy", {})
        self.include_local_fallback = bool(policy_cfg.get("include_local_fallback", False))
        self.allow_source_worker = bool(policy_cfg.get("allow_source_worker", False))

    def _local_action(self) -> Dict:
        return {
            "partition": self._zero_partition(),
            "routing": np.full(self.num_nodes, self.source_uav, dtype=np.int32),
            "compression": self._full_precision_compression(),
        }

    def select_action(self, state: Dict, deterministic: bool = False) -> Dict:
        mandatory = np.asarray(self.env.model_graph.mandatory_cuts, dtype=bool)
        optional_cut_indices = [idx for idx in range(self.num_nodes - 1) if not mandatory[idx]]
        worker_candidates = list(range(int(self.env.num_uavs)))
        if not self.allow_source_worker:
            worker_candidates = [u for u in worker_candidates if u != self.source_uav]

        if not optional_cut_indices or not worker_candidates:
            return self._local_action()

        compression = self._full_precision_compression()
        best_eval: Optional[Dict] = None
        best_action: Optional[Dict] = None

        for cut_idx in optional_cut_indices:
            partition = self._zero_partition()
            partition[cut_idx] = 1
            block_graph = self.env.model_graph.build_blocks(partition, self.env.task_popularity)
            first_remote_block = int(block_graph.node_to_block[cut_idx + 1])

            for worker_uav in worker_candidates:
                routing_blocks = np.full(block_graph.num_blocks, worker_uav, dtype=np.int32)
                routing_blocks[:first_remote_block] = self.source_uav
                action = {
                    "partition": partition.copy(),
                    "routing": self._pad_block_routing(routing_blocks),
                    "compression": compression.copy(),
                }
                candidate_eval = self._score_candidate(action)
                if self._is_better(candidate_eval, best_eval):
                    best_eval = candidate_eval
                    best_action = action

        if self.include_local_fallback:
            local_action = self._local_action()
            local_eval = self._score_candidate(local_action)
            if self._is_better(local_eval, best_eval):
                best_action = local_action

        return best_action if best_action is not None else self._local_action()
