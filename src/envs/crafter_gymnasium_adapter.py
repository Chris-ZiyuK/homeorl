"""
Gymnasium Adapter for Crafter
=============================
Thin adapter layer that wraps Crafter's native Env to be compatible with
gymnasium's interface (required by Stable-Baselines3).

Crafter native: obs = env.reset(); obs, reward, done, info = env.step(action)
Gymnasium:      obs, info = env.reset(); obs, reward, terminated, truncated, info = env.step(action)

Also bridges the action_space and observation_space to use proper
gymnasium.spaces objects (Crafter uses namedtuple stubs if gym not installed).
"""

from __future__ import annotations

import gymnasium
import numpy as np
from typing import Any, Dict, Optional, Tuple


class CrafterGymnasiumAdapter(gymnasium.Env):
    """Adapt crafter.Env (or wrapped Crafter env) to gymnasium.Env interface.

    This adapter enables use with Stable-Baselines3, which requires a proper
    gymnasium.Env with:
      - reset() → (obs, info)
      - step()  → (obs, reward, terminated, truncated, info)
      - observation_space: gymnasium.spaces.Box
      - action_space: gymnasium.spaces.Discrete

    Args:
        crafter_env: A crafter.Env instance (or any of our custom wrappers).
        max_episode_steps: If set, truncate episodes at this length.
                           Crafter's default is 10000.
    """

    metadata = {'render_modes': ['rgb_array']}

    def __init__(
        self,
        crafter_env,
        max_episode_steps: Optional[int] = None,
    ):
        super().__init__()
        self._env = crafter_env

        # Build proper gymnasium spaces
        crafter_obs_space = crafter_env.observation_space
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=crafter_obs_space.shape,
            dtype=np.uint8,
        )

        crafter_act_space = crafter_env.action_space
        self.action_space = gymnasium.spaces.Discrete(crafter_act_space.n)

        self._max_episode_steps = max_episode_steps
        self._step_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Note: Crafter seeds are set at env creation time. The `seed` arg
        here is accepted for API compatibility but has no effect.
        """
        obs = self._env.reset()
        self._step_count = 0
        info = {}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take one step in the environment.

        Returns:
            obs: RGB image (64, 64, 3)
            reward: Scalar reward (may include HACE component)
            terminated: True if agent died (health ≤ 0)
            truncated: True if max_episode_steps reached
            info: Dict containing inventory, achievements, vitals, etc.
        """
        obs, reward, done, info = self._env.step(action)
        self._step_count += 1

        # Decompose `done` into terminated vs truncated
        # Crafter's done = dead OR over(step_limit)
        health = info.get('inventory', {}).get('health', 0)
        if hasattr(info, 'get') and 'health' in info:
            health = info['health']
        elif 'inventory' in info:
            health = info['inventory'].get('health', 0)

        terminated = done and health <= 0  # Agent died
        truncated = done and health > 0    # Time limit

        # Also check our max_episode_steps
        if self._max_episode_steps and self._step_count >= self._max_episode_steps:
            truncated = True

        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        """Render the current frame."""
        if hasattr(self._env, 'render'):
            return self._env.render()
        return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def close(self):
        """Clean up resources."""
        pass  # Crafter doesn't have a close method

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying env."""
        return getattr(self._env, name)


# ── Convenience factory ─────────────────────────────────────────────────────

def make_gymnasium_crafter(
    agent_type: str = 'vanilla',
    alpha: float = 1.0,
    beta: float = 1.0,
    seed: int = 0,
    logdir: str = None,
    record: bool = False,
    max_episode_steps: int = None,
) -> CrafterGymnasiumAdapter:
    """Create a gymnasium-compatible Crafter env with HACE wrapper.

    This is the main entry point for SB3 training scripts.

    Args:
        agent_type: 'vanilla', 'hace', 'pure_homeo', 'health_only', 'naive_survival'
        alpha: Weight for original Crafter reward
        beta: Weight for HACE reward
        seed: Random seed
        logdir: Crafter recording directory
        record: Enable Crafter recording
        max_episode_steps: Override max steps per episode

    Returns:
        CrafterGymnasiumAdapter wrapping the configured Crafter env
    """
    from src.envs.crafter_hace_wrapper import make_crafter_env

    crafter_env = make_crafter_env(
        agent_type=agent_type,
        alpha=alpha,
        beta=beta,
        seed=seed,
        logdir=logdir,
        record=record,
    )

    return CrafterGymnasiumAdapter(crafter_env, max_episode_steps=max_episode_steps)
