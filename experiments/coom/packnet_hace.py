from __future__ import annotations

from typing import Dict, Tuple

from CL.methods.packnet import PackNet_SAC


class PackNet_HACE_SAC(PackNet_SAC):
    """PackNet continual learning + HACE-style reward shaping.

    PackNet helps reduce forgetting by pruning/freezing weights per task.
    HACE adds an internal reward for reducing deviation from a health setpoint.
    """

    def __init__(
        self,
        regularize_critic: bool = False,
        retrain_steps: int = 0,
        hace_health_setpoint: float = 100.0,
        hace_internal_reward_scale: float = 0.01,
        **vanilla_sac_kwargs,
    ):
        super().__init__(
            regularize_critic=regularize_critic,
            retrain_steps=retrain_steps,
            **vanilla_sac_kwargs,
        )
        self.hace_health_setpoint = float(hace_health_setpoint)
        self.hace_internal_reward_scale = float(hace_internal_reward_scale)
        self._prev_health = None

    def on_env_reset(self, info: Dict) -> None:
        self._prev_health = self._read_health()

    def shape_reward(self, reward: float, info: Dict) -> Tuple[float, float]:
        health_now = self._read_health()
        if health_now is None:
            return reward, 0.0

        if self._prev_health is None:
            self._prev_health = health_now
            return reward, 0.0

        drive_prev = abs(float(self._prev_health) - self.hace_health_setpoint)
        drive_now = abs(float(health_now) - self.hace_health_setpoint)

        internal_reward = self.hace_internal_reward_scale * (drive_prev - drive_now)
        self._prev_health = health_now

        return reward + internal_reward, internal_reward

    def _read_health(self):
        try:
            from vizdoom import GameVariable
        except Exception:
            return None

        try:
            active_env = self.env.get_active_env()
            game = getattr(active_env, "game", None)
            if game is None:
                return None
            return float(game.get_game_variable(GameVariable.HEALTH))
        except Exception:
            return None

