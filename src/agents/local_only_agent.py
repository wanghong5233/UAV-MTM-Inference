"""
Local-only system baseline.
"""

from __future__ import annotations

from typing import Dict

import gymnasium as gym
import numpy as np

from src.core import AgentRegistry
from .system_baseline_base import SystemBaselineAgent


@AgentRegistry.register("local_only")
class LocalOnlyAgent(SystemBaselineAgent):
    """Execute the entire inference flow on the source UAV."""

    def __init__(self, env: gym.Env, config: Dict):
        super().__init__(env, config)

    def select_action(self, state: Dict, deterministic: bool = False) -> Dict:
        partition = self._zero_partition()
        routing = np.full(self.num_nodes, self.source_uav, dtype=np.int32)
        compression = self._full_precision_compression()
        return {
            "partition": partition,
            "routing": routing,
            "compression": compression,
        }
