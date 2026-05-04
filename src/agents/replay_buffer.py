"""
Experience Replay Buffer for multi-agent DQN training.
Stores graph-level transitions (all agents' data per timestep).
"""

import random
import numpy as np
from collections import deque
from typing import Any, Dict, Tuple


class ReplayBuffer:
    """
    Replay buffer storing full graph transitions.

    Each transition contains:
    - states: all agents' observations [num_agents, obs_dim]
    - actions: all agents' actions [num_agents]
    - rewards: all agents' rewards [num_agents]
    - next_states: all agents' next observations [num_agents, obs_dim]
    - done: episode termination flag
    """

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        done: bool,
    ):
        """Store a graph-level transition."""
        self.buffer.append((
            states.copy(),
            actions.copy(),
            rewards.copy(),
            next_states.copy(),
            done,
        ))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.

        Returns:
            states: [batch, num_agents, obs_dim]
            actions: [batch, num_agents]
            rewards: [batch, num_agents]
            next_states: [batch, num_agents, obs_dim]
            dones: [batch]
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize replay buffer for checkpointing/resume."""
        return {
            "capacity": self.buffer.maxlen,
            "data": list(self.buffer),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore replay buffer from serialized state."""
        capacity = state.get("capacity", self.buffer.maxlen)
        data = state.get("data", [])
        self.buffer = deque(data, maxlen=capacity)
