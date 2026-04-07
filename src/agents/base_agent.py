"""Base agent abstraction for sequential RL experiments.

All agents share the same interface so the training loop in
run_sequential_tasks.py can treat them uniformly.
"""

from __future__ import annotations

import abc
import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    """Two-hidden-layer Q-network used by all DQN-based agents."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Return activations of the second hidden layer (for probing)."""
        h = torch.relu(self.net[0](x))
        h = torch.relu(self.net[2](h))
        return h


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer."""

    def __init__(self, capacity: int = 20_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def sample_list(self, n: int) -> list:
        """Return a list of n random transitions (for reservoir sampling)."""
        return random.sample(list(self.buffer), min(n, len(self.buffer)))

    def extend(self, transitions: list):
        """Add a list of transitions."""
        for t in transitions:
            self.buffer.append(t)

    def clear(self):
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class BaseAgent(abc.ABC):
    """Abstract base class for all sequential RL agents.

    Subclasses must implement ``optimize`` and may override
    ``on_task_switch`` for CRL-specific logic (e.g., computing
    Fisher matrices for EWC).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 20_000,
        target_update: int = 25,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 300.0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.q_net = QNet(obs_dim, n_actions, hidden_dim)
        self.target_net = QNet(obs_dim, n_actions, hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self._global_step = 0

    # ── Action selection ────────────────────────────────────

    def select_action(self, obs: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            return self.q_net(torch.FloatTensor(obs).unsqueeze(0)).argmax(1).item()

    def epsilon(self, global_episode: int) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -global_episode / self.eps_decay
        )

    # ── Training ────────────────────────────────────────────

    def store_transition(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    @abc.abstractmethod
    def optimize(self) -> float | None:
        """Run one gradient step. Returns loss value or None."""
        ...

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── CRL hooks ───────────────────────────────────────────

    def on_task_switch(self, task_idx: int, task_name: str):
        """Called at task boundaries. Override for CRL methods."""
        pass

    def clear_buffer(self):
        self.buffer.clear()

    # ── Representation access ───────────────────────────────

    def get_hidden_activations(self, obs: np.ndarray) -> np.ndarray:
        """Get second-layer hidden activations for a batch of observations."""
        with torch.no_grad():
            t = torch.FloatTensor(obs) if obs.ndim == 2 else torch.FloatTensor(obs).unsqueeze(0)
            h = self.q_net.hidden_activations(t)
        return h.numpy()

    # ── Serialization ───────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])

    def reinitialize(self):
        """Full re-initialization (for oracle agent)."""
        self.q_net = QNet(self.obs_dim, self.n_actions, self.hidden_dim)
        self.target_net = QNet(self.obs_dim, self.n_actions, self.hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.buffer.clear()
