from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class CWHomeostasisConfig:
    alpha: float = 1.0
    beta: float = 1.0
    control_setpoint: float = 0.0


class CWHACEWrapper:
    """Minimal HACE-style reward wrapper for continuous-control tasks.

    This scaffold uses action magnitude as a proxy homeostatic drive:
      drive_t = ||a_t||_2
      r_homeo = beta * (|setpoint - drive_{t-1}| - |setpoint - drive_t|)
      r_total = alpha * r_task + r_homeo
    """

    def __init__(self, env, cfg: CWHomeostasisConfig):
        self._env = env
        self.cfg = cfg
        self._prev_drive: Optional[float] = None
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        self._prev_drive = None
        return out

    def step(self, action: Any):
        out = self._env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward_task, terminated, truncated, info = out
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward_task, done, info = out
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError(f"Unexpected step() return shape: {type(out)} {out}")

        info = dict(info or {})
        drive = float(np.linalg.norm(np.asarray(action, dtype=np.float32).ravel(), ord=2))
        if self._prev_drive is None:
            r_homeo = 0.0
        else:
            old = abs(float(self.cfg.control_setpoint) - float(self._prev_drive))
            new = abs(float(self.cfg.control_setpoint) - float(drive))
            r_homeo = float(self.cfg.beta) * (old - new)

        reward_total = float(self.cfg.alpha) * float(reward_task) + float(r_homeo)
        info["original_reward"] = float(reward_task)
        info["homeo_reward"] = float(r_homeo)
        info["homeo_drive"] = float(drive)
        self._prev_drive = drive
        return obs, reward_total, bool(terminated), bool(truncated), info
