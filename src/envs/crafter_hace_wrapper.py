"""
Crafter HACE Wrapper
====================
Lightweight wrappers that add homeostatic auxiliary rewards to the native
Crafter environment.

IMPORTANT: Crafter uses its own Env class (NOT gym/gymnasium). The wrappers
here use plain composition rather than gym.Wrapper inheritance to avoid
compatibility issues. They expose the same interface as crafter.Env:
    obs = env.reset()
    obs, reward, done, info = env.step(action)

Crafter's survival mechanics (from source code analysis):
  - food:   0-9, initial 9, consumed 1 per 25 ticks
  - drink:  0-9, initial 9, consumed 1 per 20 ticks
  - energy: 0-9, initial 9, consumed 1 per 30 ticks
  - health: 0-9, initial 9, degrades when ANY of food/drink/energy ≤ 0

health only recovers when ALL three vitals > 0. This means maintaining
food/drink/energy homeostasis is a PREREQUISITE for survival — exactly
the multi-dimensional version of HACE.

Wrapper variants:
  - CrafterHACEWrapper:       multi-dim HACE (food + drink + energy)
  - CrafterHealthOnlyWrapper: single-dim HACE (health only, for ablation)
  - CrafterNaiveSurvivalWrapper: raw bonus for high vitals (sham control)
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional


# ── Base wrapper for Crafter's native API ───────────────────────────────────

class CrafterWrapperBase:
    """Base class for wrapping crafter.Env without gym.Wrapper dependency.

    Delegates all attribute access to the underlying env, only overriding
    reset() and step().
    """

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        # Delegate attribute access to the underlying env
        return getattr(self._env, name)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space


# ── Multi-Dimensional HACE ──────────────────────────────────────────────────

class CrafterHACEWrapper(CrafterWrapperBase):
    """Add multi-dimensional homeostatic drive-reduction reward to Crafter.

    The homeostatic reward is the sum of drive reductions across
    food, drink, and energy:

        r_homeo = Σ_v (|setpoint_v - v_old| - |setpoint_v - v_new|)

    Positive when vitals move toward setpoint, negative when they
    deviate further.

    Args:
        env: A crafter.Env instance.
        alpha: Weight for the original Crafter reward. Set to 0 for pure
               homeostatic mode.
        beta: Weight for the homeostatic reward. Default 1.0.
        setpoints: Dict mapping vital name → target value.
                   Default: {'food': 9, 'drink': 9, 'energy': 9} (full).
        track_death_cause: If True, add death cause to info dict.
    """

    VITAL_NAMES = ('food', 'drink', 'energy')

    def __init__(
        self,
        env,
        alpha: float = 1.0,
        beta: float = 1.0,
        setpoints: Optional[Dict[str, float]] = None,
        track_death_cause: bool = True,
    ):
        super().__init__(env)
        self.alpha = alpha
        self.beta = beta
        self.setpoints = setpoints or {v: 9.0 for v in self.VITAL_NAMES}
        self.track_death_cause = track_death_cause

        # State tracking
        self._prev_vitals: Dict[str, float] = {}
        self._prev_health: float = 9.0
        self._step_count: int = 0
        self._vital_history: list = []

    def reset(self):
        obs = self._env.reset()
        self._prev_vitals = {v: 9.0 for v in self.VITAL_NAMES}
        self._prev_health = 9.0
        self._step_count = 0
        self._vital_history = []
        return obs

    def step(self, action):
        obs, reward_orig, done, info = self._env.step(action)
        self._step_count += 1

        # ── Extract current vitals from info ────────────────────────
        inventory = info.get('inventory', {})
        cur_vitals = {
            v: float(inventory.get(v, 0)) for v in self.VITAL_NAMES
        }
        cur_health = float(inventory.get('health', 0))

        # ── Compute multi-dim drive reduction ───────────────────────
        r_homeo = 0.0
        for v in self.VITAL_NAMES:
            if v in self.setpoints and v in self._prev_vitals:
                old_drive = abs(self.setpoints[v] - self._prev_vitals[v])
                new_drive = abs(self.setpoints[v] - cur_vitals[v])
                r_homeo += (old_drive - new_drive)

        # ── Combined reward ─────────────────────────────────────────
        reward_total = self.alpha * reward_orig + self.beta * r_homeo

        # ── Track vitals for analysis ───────────────────────────────
        vital_snapshot = {
            'step': self._step_count,
            'health': cur_health,
            **cur_vitals,
            'r_orig': reward_orig,
            'r_homeo': r_homeo,
            'r_total': reward_total,
        }
        self._vital_history.append(vital_snapshot)

        info['vitals'] = cur_vitals.copy()
        info['health'] = cur_health
        info['homeo_reward'] = r_homeo
        info['original_reward'] = reward_orig

        # ── Death cause analysis ────────────────────────────────────
        if done and self.track_death_cause:
            info['death_cause'] = _classify_death(cur_vitals, cur_health)
            info['episode_length'] = self._step_count
            info['vital_averages'] = self._compute_vital_averages()

        # ── Update state ────────────────────────────────────────────
        self._prev_vitals = cur_vitals
        self._prev_health = cur_health

        return obs, reward_total, done, info

    def _compute_vital_averages(self) -> Dict[str, float]:
        """Compute average vital values over the episode."""
        if not self._vital_history:
            return {}
        avg = {}
        for key in ['health', 'food', 'drink', 'energy']:
            values = [s[key] for s in self._vital_history]
            avg[f'avg_{key}'] = float(np.mean(values))
            avg[f'min_{key}'] = float(np.min(values))
        return avg


# ── Health-Only HACE (Ablation) ─────────────────────────────────────────────

class CrafterHealthOnlyWrapper(CrafterWrapperBase):
    """Single-variable HACE using only health.

    This serves as an ablation to show multi-dimensional HACE
    (food+drink+energy) is superior to single-variable HACE (health only).

    The health-only drive reduction:
        r_homeo = |setpoint - health_old| - |setpoint - health_new|
    """

    def __init__(
        self,
        env,
        alpha: float = 1.0,
        beta: float = 1.0,
        setpoint: float = 9.0,
        track_death_cause: bool = True,
    ):
        super().__init__(env)
        self.alpha = alpha
        self.beta = beta
        self.setpoint = setpoint
        self.track_death_cause = track_death_cause
        self._prev_health: float = 9.0
        self._step_count: int = 0

    def reset(self):
        obs = self._env.reset()
        self._prev_health = 9.0
        self._step_count = 0
        return obs

    def step(self, action):
        obs, reward_orig, done, info = self._env.step(action)
        self._step_count += 1

        cur_health = float(info.get('inventory', {}).get('health', 0))

        # Single-variable drive reduction
        old_drive = abs(self.setpoint - self._prev_health)
        new_drive = abs(self.setpoint - cur_health)
        r_homeo = old_drive - new_drive

        reward_total = self.alpha * reward_orig + self.beta * r_homeo

        info['homeo_reward'] = r_homeo
        info['original_reward'] = reward_orig

        if done and self.track_death_cause:
            vitals = {v: float(info.get('inventory', {}).get(v, 0))
                      for v in ('food', 'drink', 'energy')}
            info['death_cause'] = _classify_death(vitals, cur_health)
            info['episode_length'] = self._step_count

        self._prev_health = cur_health
        return obs, reward_total, done, info


# ── Naive Survival Reward (Sham Control) ────────────────────────────────────

class CrafterNaiveSurvivalWrapper(CrafterWrapperBase):
    """Sham control: gives raw bonus for keeping vitals high.

    Unlike HACE (which uses drive reduction — the CHANGE toward setpoint),
    this wrapper gives a per-step bonus proportional to current vital levels:

        r_naive = c * (food + drink + energy) / (3 * max_val)

    This tests whether it's the homeostatic formulation that matters,
    or simply "any signal correlated with survival."
    """

    def __init__(
        self,
        env,
        alpha: float = 1.0,
        bonus_coef: float = 0.1,
        track_death_cause: bool = True,
    ):
        super().__init__(env)
        self.alpha = alpha
        self.bonus_coef = bonus_coef
        self.track_death_cause = track_death_cause
        self._step_count = 0

    def reset(self):
        obs = self._env.reset()
        self._step_count = 0
        return obs

    def step(self, action):
        obs, reward_orig, done, info = self._env.step(action)
        self._step_count += 1

        inventory = info.get('inventory', {})
        food = float(inventory.get('food', 0))
        drink = float(inventory.get('drink', 0))
        energy = float(inventory.get('energy', 0))

        # Raw bonus proportional to vital levels (not drive reduction)
        r_naive = self.bonus_coef * (food + drink + energy) / 27.0

        reward_total = self.alpha * reward_orig + r_naive

        info['naive_reward'] = r_naive
        info['original_reward'] = reward_orig

        if done and self.track_death_cause:
            vitals = {'food': food, 'drink': drink, 'energy': energy}
            health = float(inventory.get('health', 0))
            info['death_cause'] = _classify_death(vitals, health)
            info['episode_length'] = self._step_count

        return obs, reward_total, done, info


# ── Shared Utilities ────────────────────────────────────────────────────────

def _classify_death(
    vitals: Dict[str, float], health: float
) -> str:
    """Classify the likely cause of death.

    Priority order: direct vital depletion → monster attack → timeout.
    """
    if health > 0:
        return 'timeout'

    depleted = [v for v in ('food', 'drink', 'energy') if vitals.get(v, 0) <= 0]

    if 'food' in depleted and 'drink' not in depleted and 'energy' not in depleted:
        return 'starved'
    elif 'drink' in depleted and 'food' not in depleted and 'energy' not in depleted:
        return 'dehydrated'
    elif 'energy' in depleted and 'food' not in depleted and 'drink' not in depleted:
        return 'exhausted'
    elif len(depleted) >= 2:
        return 'multiple_depletion'
    else:
        return 'combat_death'


# ── Factory function ────────────────────────────────────────────────────────

def make_crafter_env(
    agent_type: str = 'vanilla',
    alpha: float = 1.0,
    beta: float = 1.0,
    seed: int = 0,
    logdir: str = None,
    record: bool = True,
):
    """Create a Crafter environment with the specified reward configuration.

    Args:
        agent_type: One of 'vanilla', 'hace', 'pure_homeo',
                    'health_only', 'naive_survival'.
        alpha: Weight for original Crafter reward (auto 0 for pure_homeo).
        beta: Weight for HACE reward.
        seed: Random seed.
        logdir: Directory for Crafter's built-in recording.
        record: Whether to wrap with Crafter's Recorder.

    Returns:
        Wrapped Crafter environment.
    """
    import crafter

    env = crafter.Env(seed=seed, reward=True)

    if record and logdir:
        env = crafter.Recorder(
            env, logdir,
            save_stats=True,
            save_video=False,
            save_episode=False,
        )

    if agent_type == 'vanilla':
        # Use HACE wrapper with beta=0 for tracking only (no reward change)
        env = CrafterHACEWrapper(env, alpha=1.0, beta=0.0)
    elif agent_type == 'hace':
        env = CrafterHACEWrapper(env, alpha=alpha, beta=beta)
    elif agent_type == 'pure_homeo':
        env = CrafterHACEWrapper(env, alpha=0.0, beta=beta)
    elif agent_type == 'health_only':
        env = CrafterHealthOnlyWrapper(env, alpha=alpha, beta=beta)
    elif agent_type == 'naive_survival':
        env = CrafterNaiveSurvivalWrapper(env, alpha=alpha, bonus_coef=0.1)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    return env
