"""
Sequential gridworld tasks for homeostatic RL.

The same agent body is reused across three tasks:
  - reach:   enough energy to reach the exit directly
  - recharge: must detour through a food tile to survive
  - detour:  must recharge and avoid harmful hazard tiles

Two reward modes are supported:
  - "task":        dense shaping tailored to the active task
  - "homeostatic": shared internal drive-reduction reward
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np


TASK_INDEX = {
    "reach": 0,
    "recharge": 1,
    "hazard_reach": 2,
    "detour": 3,
    "tight_detour": 4,
}

TASK_SPECS = {
    "reach": {
        "grid_size": 5,
        "max_steps": 20,
        "initial_energy": 10,
        "energy_cap": 12,
        "step_cost": 1,
        "food_pos": None,
        "food_gain": 0,
        "hazards": (),
        "hazard_cost": 0,
    },
    "recharge": {
        "grid_size": 5,
        "max_steps": 24,
        "initial_energy": 6,
        "energy_cap": 12,
        "step_cost": 1,
        "food_pos": (2, 1),
        "food_gain": 6,
        "hazards": (),
        "hazard_cost": 0,
    },
    "hazard_reach": {
        "grid_size": 5,
        "max_steps": 24,
        "initial_energy": 10,
        "energy_cap": 12,
        "step_cost": 1,
        "food_pos": None,
        "food_gain": 0,
        "hazards": ((1, 2), (2, 2)),
        "hazard_cost": 3,
    },
    "detour": {
        "grid_size": 5,
        "max_steps": 26,
        "initial_energy": 7,
        "energy_cap": 13,
        "step_cost": 1,
        "food_pos": (2, 1),
        "food_gain": 6,
        "hazards": ((2, 2), (2, 3)),
        "hazard_cost": 3,
    },
    "tight_detour": {
        "grid_size": 6,
        "max_steps": 32,
        "initial_energy": 7,
        "energy_cap": 14,
        "step_cost": 1,
        "food_pos": (3, 1),
        "food_gain": 7,
        "hazards": ((2, 2), (2, 3)),
        "hazard_cost": 4,
    },
}


class SequentialHomeostasisEnv(gym.Env):
    """Minimal sequential-task gridworld with shared energy dynamics."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        task_name: str = "reach",
        reward_mode: str = "homeostatic",
        exit_bonus: float = 6.0,
        death_penalty: float = 6.0,
        progress_coef: float = 0.35,
        food_bonus: float = 1.5,
        hazard_penalty: float = 1.0,
        internal_coef: float = 1.0,
    ):
        super().__init__()
        if task_name not in TASK_SPECS:
            raise ValueError(f"Unknown task_name: {task_name}")
        if reward_mode not in {"task", "homeostatic", "eval"}:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")

        self.task_name = task_name
        self.reward_mode = reward_mode
        self.exit_bonus = exit_bonus
        self.death_penalty = death_penalty
        self.progress_coef = progress_coef
        self.food_bonus = food_bonus
        self.hazard_penalty = hazard_penalty
        self.internal_coef = internal_coef

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.5, shape=(13,), dtype=np.float32
        )
        self.obs_dim = 13
        self._dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self._load_task_spec()

    def _load_task_spec(self) -> None:
        spec = TASK_SPECS[self.task_name]
        self.grid_size = spec["grid_size"]
        self.max_steps = spec["max_steps"]
        self.initial_energy = float(spec["initial_energy"])
        self.energy_cap = float(spec["energy_cap"])
        self.step_cost = float(spec["step_cost"])
        self.food_pos = spec["food_pos"]
        self.food_gain = float(spec["food_gain"])
        self.hazards = tuple(spec["hazards"])
        self.hazard_cost = float(spec["hazard_cost"])
        self.setpoint = self.energy_cap

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_task_spec()
        self.agent_pos = (0, 0)
        self.exit_pos = (self.grid_size - 1, self.grid_size - 1)
        self.food_available = self.food_pos is not None
        self.energy = self.initial_energy
        self.steps = 0
        self.stats = {
            "success": False,
            "energy_depleted": False,
            "food_collected": False,
            "hazard_hits": 0,
            "energy_left": self.energy,
        }
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        terminated = False
        truncated = False
        reward = 0.0

        old_energy = self.energy
        old_target_dist = self._target_distance(self.agent_pos)

        dr, dc = self._dirs[action]
        nr = int(np.clip(self.agent_pos[0] + dr, 0, self.grid_size - 1))
        nc = int(np.clip(self.agent_pos[1] + dc, 0, self.grid_size - 1))
        self.agent_pos = (nr, nc)

        self.energy -= self.step_cost

        ate_food = False
        if self.food_available and self.agent_pos == self.food_pos:
            self.food_available = False
            self.energy = min(self.energy + self.food_gain, self.energy_cap)
            self.stats["food_collected"] = True
            ate_food = True

        hazard_hit = self.agent_pos in self.hazards
        if hazard_hit:
            self.energy -= self.hazard_cost
            self.stats["hazard_hits"] += 1

        self.energy = float(np.clip(self.energy, -999.0, self.energy_cap))

        if self.agent_pos == self.exit_pos:
            terminated = True
            self.stats["success"] = True

        if self.energy <= 0:
            terminated = True
            self.stats["energy_depleted"] = True

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        if self.reward_mode == "task":
            new_target_dist = self._target_distance(self.agent_pos)
            reward += self.progress_coef * (old_target_dist - new_target_dist)
            if ate_food:
                reward += self.food_bonus
            if hazard_hit:
                reward -= self.hazard_penalty
        elif self.reward_mode == "homeostatic":
            old_drive = abs(self.setpoint - old_energy)
            new_drive = abs(self.setpoint - self.energy)
            reward += self.internal_coef * (old_drive - new_drive)

        if self.stats["success"]:
            reward += self.exit_bonus
        if self.stats["energy_depleted"]:
            reward -= self.death_penalty

        self.stats["energy_left"] = self.energy
        return self._obs(), reward, terminated, truncated, {}

    def _target_distance(self, pos) -> int:
        if self.food_available and self.food_pos is not None:
            target = self.food_pos
        else:
            target = self.exit_pos
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

    def _obs(self):
        g = self.grid_size - 1
        food_r, food_c = self.food_pos if self.food_available and self.food_pos else (-1, -1)
        hazard_coords = list(self.hazards[:2])
        while len(hazard_coords) < 2:
            hazard_coords.append((-1, -1))

        obs = [
            self.agent_pos[0] / g,
            self.agent_pos[1] / g,
            self.exit_pos[0] / g,
            self.exit_pos[1] / g,
            food_r / g if food_r >= 0 else -0.25,
            food_c / g if food_c >= 0 else -0.25,
            1.0 if self.food_available else 0.0,
            self.energy / self.energy_cap,
            TASK_INDEX[self.task_name] / max(len(TASK_INDEX) - 1, 1),
            hazard_coords[0][0] / g if hazard_coords[0][0] >= 0 else -0.25,
            hazard_coords[0][1] / g if hazard_coords[0][1] >= 0 else -0.25,
            hazard_coords[1][0] / g if hazard_coords[1][0] >= 0 else -0.25,
            hazard_coords[1][1] / g if hazard_coords[1][1] >= 0 else -0.25,
        ]
        return np.array(obs, dtype=np.float32)

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.exit_pos[0]][self.exit_pos[1]] = "E"
        if self.food_available and self.food_pos is not None:
            grid[self.food_pos[0]][self.food_pos[1]] = "F"
        for hr, hc in self.hazards:
            grid[hr][hc] = "H"
        grid[self.agent_pos[0]][self.agent_pos[1]] = "@"

        lines = [
            f"task={self.task_name} reward={self.reward_mode} energy={self.energy:.1f} step={self.steps}"
        ]
        for row in grid:
            lines.append(" ".join(row))
        return "\n".join(lines)
