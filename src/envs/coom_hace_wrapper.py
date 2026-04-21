"""
COOM HACE Wrapper
================
Gymnasium-style wrapper that adds homeostatic drive-reduction rewards to COOM
environments.

This mirrors the gridworld HACE formulation:

  drive = Σ_i w_i * |setpoint_i - x_i|
  r_homeo = beta * (drive_old - drive_new)
  r_total = alpha * r_task + r_homeo

We target internal variables that are (usually) available in ViZDoom-based envs:
  - health
  - stamina (if present), otherwise a configurable proxy (armor/ammo)

Since COOM's exact `info` schema can vary by version, extraction is best-effort:
  1) Look for common keys in `info` (health/stamina/armor/ammo)
  2) Fall back to ViZDoom game variables if the env exposes a `game` object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class HomeostasisConfig:
    alpha: float = 1.0
    beta: float = 1.0
    health_setpoint: float = 100.0
    stamina_setpoint: float = 100.0
    health_weight: float = 1.0
    stamina_weight: float = 1.0
    stamina_source: str = "stamina"  # stamina|armor|ammo


class COOMHACEWrapper:
    """Adds homeostatic auxiliary reward while preserving gymnasium API."""

    def __init__(self, env, cfg: HomeostasisConfig):
        self._env = env
        self.cfg = cfg
        self._prev_drive: Optional[float] = None

        # Pass-through (best effort)
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
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

        signals = self._extract_signals(info)
        self._prev_drive = self._compute_drive(signals)
        return obs, info

    def step(self, action: Any):
        out = self._env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward_task, terminated, truncated, info = out
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward_task, done, info = out
            terminated, truncated = bool(done), False
        else:  # pragma: no cover
            raise RuntimeError(f"Unexpected step() return shape: {type(out)} {out}")

        info = dict(info or {})
        signals = self._extract_signals(info)
        drive = self._compute_drive(signals)

        # Drive reduction reward
        r_homeo = 0.0
        if self._prev_drive is not None and drive is not None:
            r_homeo = (self._prev_drive - drive) * float(self.cfg.beta)

        reward_total = float(self.cfg.alpha) * float(reward_task) + float(r_homeo)

        # Attach diagnostics
        info["original_reward"] = float(reward_task)
        info["homeo_reward"] = float(r_homeo)
        info["homeo_drive"] = None if drive is None else float(drive)
        if signals.get("health") is not None:
            info["homeo_health"] = float(signals["health"])
        if signals.get("stamina") is not None:
            info["homeo_stamina"] = float(signals["stamina"])
            info["homeo_stamina_source"] = self.cfg.stamina_source

        self._prev_drive = drive
        return obs, reward_total, bool(terminated), bool(truncated), info

    def _compute_drive(self, signals: Dict[str, Optional[float]]) -> Optional[float]:
        health = signals.get("health")
        stamina = signals.get("stamina")

        if health is None and stamina is None:
            return None

        d = 0.0
        if health is not None:
            d += float(self.cfg.health_weight) * abs(float(self.cfg.health_setpoint) - float(health))
        if stamina is not None:
            d += float(self.cfg.stamina_weight) * abs(float(self.cfg.stamina_setpoint) - float(stamina))
        return float(d)

    def _extract_signals(self, info: Dict[str, Any]) -> Dict[str, Optional[float]]:
        # 1) info keys (best effort)
        health = _to_float(info.get("health"))
        stamina_raw = None

        if self.cfg.stamina_source == "stamina":
            stamina_raw = info.get("stamina", info.get("energy", None))
        elif self.cfg.stamina_source == "armor":
            stamina_raw = info.get("armor", None)
        elif self.cfg.stamina_source == "ammo":
            stamina_raw = info.get("ammo", info.get("ammo2", None))

        stamina = _to_float(stamina_raw)

        # Sometimes nested dicts exist
        if health is None:
            inv = info.get("inventory") or info.get("stats") or {}
            if isinstance(inv, dict):
                health = _to_float(inv.get("health"))
                if stamina is None:
                    stamina = _to_float(inv.get(self.cfg.stamina_source))

        # 2) ViZDoom fallback if env exposes game object
        if health is None or (stamina is None and self.cfg.stamina_source == "stamina"):
            game = getattr(getattr(self._env, "unwrapped", self._env), "game", None)
            if game is not None:
                try:
                    import vizdoom as vzd  # type: ignore
                except Exception:
                    vzd = None
                if vzd is not None:
                    if health is None:
                        try:
                            health = _to_float(game.get_game_variable(vzd.GameVariable.HEALTH))
                        except Exception:
                            pass
                    if stamina is None:
                        # There is no universal STAMINA; try a few plausible proxies.
                        for var in (
                            getattr(vzd.GameVariable, "STAMINA", None),
                            getattr(vzd.GameVariable, "ARMOR", None) if self.cfg.stamina_source == "armor" else None,
                            getattr(vzd.GameVariable, "AMMO2", None) if self.cfg.stamina_source == "ammo" else None,
                            getattr(vzd.GameVariable, "AMMO", None) if self.cfg.stamina_source == "ammo" else None,
                        ):
                            if var is None:
                                continue
                            try:
                                stamina = _to_float(game.get_game_variable(var))
                                if stamina is not None:
                                    break
                            except Exception:
                                continue

        return {"health": health, "stamina": stamina}

