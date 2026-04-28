from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
except Exception as e:  # pragma: no cover
    raise RuntimeError("gymnasium is required for CW adapter.") from e


def _import_metaworld():
    try:
        import metaworld
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "MetaWorld is not installed. Run: bash scripts/setup_cw_env.sh"
        ) from e
    return metaworld

def _to_gymnasium_space(space):
    """Convert old gym spaces from MetaWorld into gymnasium spaces for SB3 2.x."""
    if space is None:
        return None

    if isinstance(space, gym.Space):
        return space

    cls_name = space.__class__.__name__

    if cls_name == "Box" and hasattr(space, "low") and hasattr(space, "high"):
        return gym.spaces.Box(
            low=np.asarray(space.low, dtype=space.dtype),
            high=np.asarray(space.high, dtype=space.dtype),
            shape=getattr(space, "shape", None),
            dtype=space.dtype,
        )

    if cls_name == "Discrete" and hasattr(space, "n"):
        return gym.spaces.Discrete(int(space.n))

    if cls_name == "MultiDiscrete" and hasattr(space, "nvec"):
        return gym.spaces.MultiDiscrete(np.asarray(space.nvec))

    if cls_name == "MultiBinary" and hasattr(space, "n"):
        return gym.spaces.MultiBinary(space.n)

    raise TypeError(f"Unsupported space type for Gymnasium conversion: {type(space)}")

@dataclass(frozen=True)
class CWSequenceSpec:
    name: str
    tasks: Tuple[str, ...]
    steps_per_task: int = 1_000_000

    
class CWGymnasiumAdapter(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env, max_episode_steps: Optional[int] = None):
        super().__init__()
        self._env = env
        if max_episode_steps is None:
            max_episode_steps = int(getattr(env, "max_path_length", 200))
        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        self.observation_space = _to_gymnasium_space(getattr(env, "observation_space", None))
        self.action_space = _to_gymnasium_space(getattr(env, "action_space", None))

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        self._step_count = 0
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
        return obs, dict(info or {})

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        out = self._env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError(f"Unexpected step() return shape: {type(out)} {out}")

        self._step_count += 1
        if self._max_episode_steps is not None and self._step_count >= self._max_episode_steps:
            truncated = True

        info = dict(info or {})
        if "cw_success" not in info:
            for key in ("success", "is_success", "episode_success", "task_success"):
                if key in info:
                    try:
                        info["cw_success"] = float(info[key])
                    except Exception:
                        info["cw_success"] = 1.0 if bool(info[key]) else 0.0
                    break

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if hasattr(self._env, "render"):
            return self._env.render()
        if self.observation_space is not None and hasattr(self.observation_space, "shape"):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return None

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    def __getattr__(self, name: str):
        return getattr(self._env, name)


def make_cw_env(
    *,
    task_name: str,
    seed: int = 0,
    max_episode_steps: Optional[int] = None,
    hace: bool = False,
    hace_alpha: float = 1.0,
    hace_beta: float = 1.0,
):
    metaworld = _import_metaworld()

    try:
        ml1 = metaworld.ML1(task_name, seed=seed)
    except TypeError:
        ml1 = metaworld.ML1(task_name)

    env = ml1.train_classes[task_name]()
    env.set_task(ml1.train_tasks[0])

    try:
        env.seed(seed)
    except Exception:
        pass

    try:
        env.action_space.seed(seed)
    except Exception:
        pass

    try:
        env.observation_space.seed(seed)
    except Exception:
        pass

    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()

    if hace:
        from src.envs.cw_hace_wrapper import CWHACEWrapper, CWHomeostasisConfig

        env = CWHACEWrapper(
            env,
            CWHomeostasisConfig(alpha=float(hace_alpha), beta=float(hace_beta)),
        )

    return CWGymnasiumAdapter(env, max_episode_steps=max_episode_steps)


def make_cw_sequence(name: str, tasks: list[str], *, steps_per_task: int = 1_000_000) -> CWSequenceSpec:
    if not tasks:
        raise ValueError("CW sequence needs at least one task.")
    return CWSequenceSpec(name=str(name), tasks=tuple(str(t) for t in tasks), steps_per_task=int(steps_per_task))
