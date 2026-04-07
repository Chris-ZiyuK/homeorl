"""EWC agent — Elastic Weight Consolidation for continual RL.

After each task, computes the Fisher Information Matrix (diagonal
approximation) to identify which weights were important. During
subsequent tasks, adds a penalty for deviating from those weights.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting
in neural networks", PNAS 2017.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from src.agents.base_agent import BaseAgent


class EWCAgent(BaseAgent):
    """DQN with Elastic Weight Consolidation (EWC) regularization.

    Args:
        ewc_lambda: strength of the EWC penalty (higher = more protection
            of old weights, but less plasticity for new tasks).
        fisher_samples: number of buffer samples used to estimate the
            diagonal Fisher information matrix.
    """

    def __init__(self, *args, ewc_lambda: float = 400.0,
                 fisher_samples: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples

        # Accumulated across tasks (online EWC variant)
        self._fisher: dict[str, torch.Tensor] = {}
        self._old_params: dict[str, torch.Tensor] = {}
        self._task_count = 0

    def on_task_switch(self, task_idx: int, task_name: str):
        """Compute Fisher and snapshot params after finishing a task."""
        self._compute_fisher()
        self._snapshot_params()
        self._task_count += 1

    def _compute_fisher(self):
        """Diagonal Fisher ≈ E[∇log π(a|s)²] estimated from buffer."""
        if len(self.buffer) == 0:
            return

        n_samples = min(self.fisher_samples, len(self.buffer))
        s, a, r, s2, d = self.buffer.sample(n_samples)

        self.q_net.zero_grad()
        q_values = self.q_net(s)
        # Use the Q-values as log-likelihood proxies
        log_probs = torch.log_softmax(q_values, dim=1)
        selected_log_probs = log_probs.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = -selected_log_probs.mean()
        loss.backward()

        new_fisher = {}
        for name, param in self.q_net.named_parameters():
            if param.grad is not None:
                new_fisher[name] = param.grad.data.clone() ** 2

        # Online EWC: accumulate Fisher across tasks
        for name in new_fisher:
            if name in self._fisher:
                self._fisher[name] = (
                    self._fisher[name] * self._task_count + new_fisher[name]
                ) / (self._task_count + 1)
            else:
                self._fisher[name] = new_fisher[name]

    def _snapshot_params(self):
        """Save a copy of current params as the anchor."""
        self._old_params = {
            name: param.data.clone()
            for name, param in self.q_net.named_parameters()
        }

    def _ewc_penalty(self) -> torch.Tensor:
        """Compute the EWC regularization term."""
        penalty = torch.tensor(0.0)
        if not self._fisher:
            return penalty
        for name, param in self.q_net.named_parameters():
            if name in self._fisher:
                penalty += (
                    self._fisher[name] * (param - self._old_params[name]) ** 2
                ).sum()
        return penalty

    def optimize(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            targets = r + self.gamma * self.target_net(s2).max(1)[0] * (1 - d)

        dqn_loss = nn.MSELoss()(q_values, targets)
        ewc_loss = self._ewc_penalty()
        total_loss = dqn_loss + self.ewc_lambda * ewc_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
