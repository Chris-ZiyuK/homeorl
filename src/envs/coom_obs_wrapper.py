"""
COOM Observation Wrappers
========================
SB3's default CNN policies expect 3D image observations (H, W, C) or (C, H, W),
but COOM's default wrapper stack may return 4D observations, e.g.:

  (frame_stack, H, W, C)  == (4, 84, 84, 3)

This file provides a thin wrapper to flatten the frame stack into channels:

  (4, 84, 84, 3) -> (84, 84, 12)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class FlattenFrameStackToChannels:
    """Convert (T, H, W, C) observations to (H, W, T*C)."""

    def __init__(self, env):
        self._env = env
        self.action_space = getattr(env, "action_space", None)
        self.observation_space = self._infer_observation_space()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _infer_observation_space(self):
        space = getattr(self._env, "observation_space", None)
        try:
            import gymnasium as gym
        except Exception:
            gym = None
        if space is None or gym is None:
            return space

        if hasattr(space, "shape") and space.shape is not None and len(space.shape) == 4:
            t, h, w, c = space.shape
            new_shape = (h, w, t * c)

            # Gymnasium requires low/high shapes to match the Box shape.
            # If the underlying space uses array lows/highs, collapse to scalars.
            low = getattr(space, "low", 0)
            high = getattr(space, "high", 255)
            if isinstance(low, np.ndarray):
                low = float(np.min(low))
            if isinstance(high, np.ndarray):
                high = float(np.max(high))
            return gym.spaces.Box(low=low, high=high, shape=new_shape, dtype=space.dtype)
        return space

    @staticmethod
    def _convert(obs: Any) -> Any:
        # gymnasium FrameStack may return LazyFrames; np.asarray handles both ndarray and LazyFrames
        try:
            arr = np.asarray(obs)
        except Exception:
            return obs

        if isinstance(arr, np.ndarray) and arr.ndim == 4:
            # (T, H, W, C) -> (H, W, T*C)
            t, h, w, c = arr.shape
            return arr.transpose(1, 2, 0, 3).reshape(h, w, t * c)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            try:
                out = self._env.reset(seed=seed)
            except TypeError:
                out = self._env.reset()
        else:
            out = self._env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return self._convert(obs), info

    def step(self, action):
        out = self._env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return self._convert(obs), reward, terminated, truncated, info
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return self._convert(obs), reward, done, info
        return out

