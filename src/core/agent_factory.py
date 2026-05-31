"""
Agent registry (factory pattern).

Creates agent instances by name.
"""

from typing import Dict, Type
import gymnasium as gym
from .base_agent import BaseAgent


class AgentRegistry:
    """Registry for agent classes."""
    
    _registry: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an agent by name."""
        def decorator(agent_cls: Type[BaseAgent]):
            if name in cls._registry:
                print(f"[WARN] Algorithm '{name}' already registered, overwriting.")
            cls._registry[name] = agent_cls
            return agent_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, env: gym.Env, config: Dict) -> BaseAgent:
        """Create an agent instance by name."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown algorithm: '{name}'. "
                f"Available algorithms: {available}"
            )
        
        agent_cls = cls._registry[name]
        return agent_cls(env, config)
    
    @classmethod
    def list_agents(cls):
        """List all registered agents."""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an agent name is registered."""
        return name in cls._registry

