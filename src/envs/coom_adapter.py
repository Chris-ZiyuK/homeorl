"""
COOM Gymnasium Adapter
=====================
Thin adapter layer that wraps COOM's environments to be compatible with
gymnasium's interface.

COOM provides environment builders under:
  - COOM.env.builder.make_env
  - COOM.env.continual.ContinualLearningEnv

This adapter is intentionally minimal: it keeps observations/rewards exactly
as COOM provides them, and only normalizes the API shape to:
  reset() -> (obs, info)
  step()  -> (obs, reward, terminated, truncated, info)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "gymnasium is required for COOM adapter. Install with: pip install gymnasium"
    ) from e


def _import_coom():
    try:
        import importlib

        coom_make_env = importlib.import_module("COOM.env.builder").make_env
        ContinualLearningEnv = importlib.import_module("COOM.env.continual").ContinualLearningEnv
        cfg = importlib.import_module("COOM.utils.config")
        Scenario = cfg.Scenario
        Sequence = cfg.Sequence
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "COOM is not installed or failed to import. "
            "Install with: pip install COOM\n"
            "If installation fails, ensure ViZDoom dependencies are installed for your OS."
        ) from e
    return coom_make_env, ContinualLearningEnv, Scenario, Sequence


@dataclass(frozen=True)
class COOMSequenceSpec:
    """Represents a COOM continual learning sequence (e.g., CO8, CD8)."""

    name: str


class COOMGymnasiumAdapter(gym.Env):
    """Adapt a COOM env to gymnasium.Env interface.

    COOM environments are typically already gym/gymnasium-like, but this adapter
    makes the interface consistent with the rest of this repo.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 35}

    def __init__(self, env, max_episode_steps: Optional[int] = None):
        super().__init__()
        self._env = env
        self._max_episode_steps = max_episode_steps
        self._step_count = 0

        # Best-effort space passthrough (COOM envs should have these).
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        self._step_count = 0

        if hasattr(self._env, "reset"):
            if seed is not None:
                try:
                    out = self._env.reset(seed=seed)
                except TypeError:
                    # Some COOM wrappers implement reset() without seed/options.
                    out = self._env.reset()
            else:
                out = self._env.reset()
        else:  # pragma: no cover
            raise RuntimeError("Underlying COOM env has no reset().")

        # COOM may return obs or (obs, info) depending on version.
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        out = self._env.step(action)

        # COOM should be gymnasium-style, but tolerate older gym-style.
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
        else:  # pragma: no cover
            raise RuntimeError(f"Unexpected step() return shape: {type(out)} {out}")

        self._step_count += 1
        if self._max_episode_steps is not None and self._step_count >= self._max_episode_steps:
            truncated = True

        return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})

    def render(self):
        if hasattr(self._env, "render"):
            return self._env.render()
        # Fallback if render is unavailable
        if self.observation_space is not None and hasattr(self.observation_space, "shape"):
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        return None

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_coom_env(
    env_id: Optional[str] = None,
    *,
    scenario: Optional[str] = None,
    seed: int = 0,
    max_episode_steps: Optional[int] = None,
    hace: bool = False,
    hace_alpha: float = 1.0,
    hace_beta: float = 1.0,
    health_setpoint: float = 100.0,
    stamina_setpoint: float = 100.0,
    stamina_source: str = "stamina",
    flatten_frame_stack: bool = True,
):
    """Create a single COOM environment.

    Use either:
      - env_id: a gym id like 'raise_the_roof-default-v0'
      - scenario: a COOM Scenario enum name like 'RAISE_THE_ROOF'
    """

    coom_make_env, _ContinualLearningEnv, Scenario, _Sequence = _import_coom()

    if env_id is None and scenario is None:
        raise ValueError("Provide either env_id (e.g. 'raise_the_roof-default-v0') or scenario (e.g. 'RAISE_THE_ROOF').")

    if scenario is not None and env_id is not None:
        raise ValueError("Provide only one of env_id or scenario.")

    if scenario is not None:
        try:
            scenario_enum = getattr(Scenario, scenario)
        except AttributeError as e:
            raise ValueError(f"Unknown COOM scenario '{scenario}'.") from e
        env = coom_make_env(scenario_enum)
    else:
        # Many gym registries accept gym.make(env_id), but COOM's make_env
        # expects Scenario. We attempt to route env_id to gymnasium if possible.
        try:
            import gymnasium as gymnasium

            env = gymnasium.make(env_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create env via gymnasium.make('{env_id}'). "
                "If your COOM version doesn't register gym ids, use --scenario instead."
            ) from e

    # Best-effort seed
    if hasattr(env, "reset"):
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

    if hace:
        from src.envs.coom_hace_wrapper import COOMHACEWrapper, HomeostasisConfig

        env = COOMHACEWrapper(
            env,
            HomeostasisConfig(
                alpha=hace_alpha,
                beta=hace_beta,
                health_setpoint=health_setpoint,
                stamina_setpoint=stamina_setpoint,
                stamina_source=stamina_source,
            ),
        )

    if flatten_frame_stack:
        from src.envs.coom_obs_wrapper import FlattenFrameStackToChannels

        env = FlattenFrameStackToChannels(env)

    return COOMGymnasiumAdapter(env, max_episode_steps=max_episode_steps)


def make_coom_sequence(sequence_name: str):
    """Create a COOM continual learning sequence env container.

    Returns:
      cl_env: ContinualLearningEnv instance (from COOM), which exposes `.tasks`.
    """

    _coom_make_env, ContinualLearningEnv, _Scenario, Sequence = _import_coom()

    try:
        seq_enum = getattr(Sequence, sequence_name)
    except AttributeError as e:
        raise ValueError(f"Unknown COOM sequence '{sequence_name}'.") from e
    return ContinualLearningEnv(seq_enum)

