"""
Core framework module.

Provides the agent-environment interface:
- BaseAgent: abstract agent interface
- AgentRegistry: agent registry
- Trainer: training loop
- Evaluator: evaluation logic
"""

from .base_agent import BaseAgent
from .agent_factory import AgentRegistry
from .trainer import Trainer
from .evaluator import Evaluator

__all__ = [
    'BaseAgent',
    'AgentRegistry',
    'Trainer',
    'Evaluator',
]

