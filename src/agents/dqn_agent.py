"""Standard DQN agent — the vanilla baseline without CRL mechanisms."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.agents.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Vanilla DQN agent. Used for agents A-E in the original setup."""

    def optimize(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            targets = r + self.gamma * self.target_net(s2).max(1)[0] * (1 - d)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
