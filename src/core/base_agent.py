"""
Abstract agent base class.

Defines the unified interface for all algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import gymnasium as gym


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        """Initialize the agent."""
        self.env = env
        self.config = config
        requested_device = str(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[WARN] Requested device '{requested_device}' is unavailable. Falling back to CPU.")
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        
        # Algorithm name
        self.name = config.get('algorithm', {}).get('name', 'unknown')
        
    @abstractmethod
    def select_action(self, state: Dict, deterministic: bool = False) -> Dict:
        """Select an action (used for training and evaluation)."""
        pass
    
    @abstractmethod
    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        """Update agent parameters and return training metrics."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model weights."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model weights."""
        pass
    
    def store_transition(self, state, action, reward, next_state, done, info=None):
        """Store a transition (for off-policy agents)."""
        pass
    
    def reset(self):
        """Reset internal states if needed."""
        pass
    
    def train_mode(self):
        """Set training mode."""
        pass
    
    def eval_mode(self):
        """Set evaluation mode."""
        pass

