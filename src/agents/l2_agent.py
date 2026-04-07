"""L2 regularization agent — simpler alternative to EWC.

At each task boundary, snapshots the current weights. During the next
task, adds an L2 penalty on the deviation from the snapshot. Unlike
EWC, this treats all weights equally (no Fisher weighting).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.agents.base_agent import BaseAgent


class L2Agent(BaseAgent):
    """DQN with L2 weight regularization toward previous task params.

    Args:
        l2_lambda: strength of the L2 penalty.
    """

    def __init__(self, *args, l2_lambda: float = 100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_lambda = l2_lambda
        self._old_params: dict[str, torch.Tensor] = {}

    def on_task_switch(self, task_idx: int, task_name: str):
        """Snapshot current params as the anchor for L2 penalty."""
        self._old_params = {
            name: param.data.clone()
            for name, param in self.q_net.named_parameters()
        }

    def _l2_penalty(self) -> torch.Tensor:
        if not self._old_params:
            return torch.tensor(0.0)
        penalty = torch.tensor(0.0)
        for name, param in self.q_net.named_parameters():
            if name in self._old_params:
                penalty += ((param - self._old_params[name]) ** 2).sum()
        return penalty

    def optimize(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            targets = r + self.gamma * self.target_net(s2).max(1)[0] * (1 - d)

        dqn_loss = nn.MSELoss()(q_values, targets)
        l2_loss = self._l2_penalty()
        total_loss = dqn_loss + self.l2_lambda * l2_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
