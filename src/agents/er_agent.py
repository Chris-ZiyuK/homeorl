"""Experience Replay agent — retains a reservoir of old-task transitions.

At each task boundary, samples a fixed number of transitions from the
current buffer into a long-term reservoir. During future tasks,
training batches are composed of 50% current-task data and 50%
reservoir data, preventing the agent from fully forgetting old tasks.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from src.agents.base_agent import BaseAgent, ReplayBuffer


class ExperienceReplayAgent(BaseAgent):
    """DQN with reservoir-based experience replay from old tasks.

    Args:
        reservoir_size: maximum number of transitions kept from old tasks.
        reservoir_ratio: fraction of each training batch from the reservoir.
    """

    def __init__(self, *args, reservoir_size: int = 5000,
                 reservoir_ratio: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.reservoir = ReplayBuffer(reservoir_size)
        self.reservoir_ratio = reservoir_ratio

    def on_task_switch(self, task_idx: int, task_name: str):
        """Move a sample of current buffer into the reservoir."""
        if len(self.buffer) == 0:
            return
        # Reservoir sampling: keep a balanced sample from each past task
        n_to_keep = min(len(self.buffer), self.reservoir.buffer.maxlen // max(task_idx + 1, 1))
        transitions = self.buffer.sample_list(n_to_keep)
        self.reservoir.extend(transitions)

    def optimize(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        # Mix current and old data
        if len(self.reservoir) >= self.batch_size // 4:
            n_old = int(self.batch_size * self.reservoir_ratio)
            n_new = self.batch_size - n_old

            s_new, a_new, r_new, s2_new, d_new = self.buffer.sample(n_new)
            s_old, a_old, r_old, s2_old, d_old = self.reservoir.sample(n_old)

            s = torch.cat([s_new, s_old])
            a = torch.cat([a_new, a_old])
            r = torch.cat([r_new, r_old])
            s2 = torch.cat([s2_new, s2_old])
            d = torch.cat([d_new, d_old])
        else:
            s, a, r, s2, d = self.buffer.sample(self.batch_size)

        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            targets = r + self.gamma * self.target_net(s2).max(1)[0] * (1 - d)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
