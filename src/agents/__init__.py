"""
Agent module.

All agents inherit BaseAgent and are registered in AgentRegistry.
"""

from src.core import AgentRegistry

# Import all agents to trigger registry
from .gnn_ppo import GNNPPO
from .mlp_ppo import MLPPPO
from .pr_no_compression_ppo import PRNoCompressionPPO
from .local_only_agent import LocalOnlyAgent
from .single_split_agent import SingleSplitAgent

# Print registry summary
print(f"[INFO] Registered agents: {AgentRegistry.list_agents()}")

__all__ = [
    'GNNPPO',
    'MLPPPO',
    'PRNoCompressionPPO',
    'LocalOnlyAgent',
    'SingleSplitAgent',
]

