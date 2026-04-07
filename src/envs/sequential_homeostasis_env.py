"""
Sequential gridworld tasks for homeostatic RL.

Extended task catalog with 12 structurally diverse tasks supporting:
  - Multiple food sources
  - Walls (impassable cells)
  - Gate mechanism (exit locked until food collected)
  - Varied energy economics (step cost, capacity)
  - Multiple hazard configurations (up to 4)

Observation modes:
  - "full":   includes internal energy
  - "masked": hides internal energy to isolate reward effects

Reward modes:
  - "task":        dense task-specific shaping
  - "homeostatic": drive-reduction objective plus sparse terminal signals
  - "mixed":       task reward + weighted homeostatic reward (HACE)
  - "eval":        no dense reward, used only for clean rollouts/tests
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ── Task Index ──────────────────────────────────────────────
# Maps task name to integer index for the observation vector.

TASK_CATALOG = [
    "reach",
    "recharge",
    "hazard_cross",
    "detour",
    "wall_maze",
    "dual_food",
    "hazard_gauntlet",
    "conservation",
    "endurance",
    "collect_exit",
    "sprint",
    "gauntlet_refuel",
]

TASK_INDEX = {name: idx for idx, name in enumerate(TASK_CATALOG)}

# ── Task Specifications ─────────────────────────────────────
# Each task defines the grid layout, energy economics, and obstacles.
#
# New fields vs v1:
#   food_positions: list of (row, col) tuples (replaces single food_pos)
#   walls:          list of (row, col) tuples for impassable cells
#   gate_locked:    if True, exit is locked until ALL food is collected
#   hazards:        now supports up to 4 positions

TASK_SPECS = {
    # ── T1: reach ────────────────────────────────────────────
    # Simple navigation. Plenty of energy. Optional food.
    # Optimal path: 8 steps. Surplus: 7 direct, 14 with food.
    "reach": {
        "grid_size": 5,
        "max_steps": 24,
        "initial_energy": 15,
        "energy_cap": 20,
        "step_cost": 1,
        "food_positions": [(2, 0)],
        "food_gain": 7,
        "hazards": (),
        "hazard_cost": 0,
        "walls": (),
        "gate_locked": False,
    },

    # ── T2: recharge ─────────────────────────────────────────
    # Must detour through food to survive.
    # Food detour path: ~8 steps. Surplus with carryover: 5-10.
    "recharge": {
        "grid_size": 5,
        "max_steps": 26,
        "initial_energy": 8,
        "energy_cap": 18,
        "step_cost": 1,
        "food_positions": [(2, 1)],
        "food_gain": 10,
        "hazards": (),
        "hazard_cost": 0,
        "walls": (),
        "gate_locked": False,
    },

    # ── T3: hazard_cross ─────────────────────────────────────
    # Hazard avoidance with optional food.
    # Safe path: ~8-9 steps. Hazard penalty is significant.
    "hazard_cross": {
        "grid_size": 5,
        "max_steps": 26,
        "initial_energy": 12,
        "energy_cap": 18,
        "step_cost": 1,
        "food_positions": [(0, 3)],
        "food_gain": 7,
        "hazards": ((1, 2), (2, 2)),
        "hazard_cost": 3,
        "walls": (),
        "gate_locked": False,
    },

    # ── T4: detour ───────────────────────────────────────────
    # Food detour + hazard avoidance combined.
    # Optimal path through food avoiding hazards: ~8 steps.
    "detour": {
        "grid_size": 5,
        "max_steps": 28,
        "initial_energy": 9,
        "energy_cap": 18,
        "step_cost": 1,
        "food_positions": [(2, 1)],
        "food_gain": 10,
        "hazards": ((2, 2), (2, 3)),
        "hazard_cost": 3,
        "walls": (),
        "gate_locked": False,
    },

    # ── T5: wall_maze ────────────────────────────────────────
    # Navigation with impassable walls creating a maze-like path.
    # Agent must find the open corridor. Tests spatial planning.
    #
    # Layout (6x6):
    #   @ . . . . .
    #   . W W . . .
    #   . . W . W .
    #   . . . . W .
    #   . F . . . .
    #   . . . . . E
    "wall_maze": {
        "grid_size": 6,
        "max_steps": 34,
        "initial_energy": 14,
        "energy_cap": 22,
        "step_cost": 1,
        "food_positions": [(4, 1)],
        "food_gain": 9,
        "hazards": (),
        "hazard_cost": 0,
        "walls": ((1, 1), (1, 2), (2, 2), (2, 4), (3, 4)),
        "gate_locked": False,
    },

    # ── T6: dual_food ────────────────────────────────────────
    # Two food sources — agent must choose the more efficient one.
    # Food A at (1, 0): closer but smaller gain.
    # Food B at (3, 4): farther but larger gain.
    # Tests decision-making under energy constraints.
    "dual_food": {
        "grid_size": 6,
        "max_steps": 30,
        "initial_energy": 10,
        "energy_cap": 20,
        "step_cost": 1,
        "food_positions": [(1, 0), (3, 4)],
        "food_gain": 8,
        "hazards": ((2, 2),),
        "hazard_cost": 3,
        "walls": (),
        "gate_locked": False,
    },

    # ── T7: hazard_gauntlet ──────────────────────────────────
    # Dense hazard field with a narrow safe corridor.
    # 4 hazards blocking the direct path. Must navigate precisely.
    #
    # Layout (6x6):
    #   @ . . . . .
    #   . . H . . .
    #   . H . H . .
    #   . . . . H .
    #   . . . . . .
    #   . . . F . E
    "hazard_gauntlet": {
        "grid_size": 6,
        "max_steps": 30,
        "initial_energy": 14,
        "energy_cap": 20,
        "step_cost": 1,
        "food_positions": [(5, 3)],
        "food_gain": 8,
        "hazards": ((1, 2), (2, 1), (2, 3), (3, 4)),
        "hazard_cost": 4,
        "walls": (),
        "gate_locked": False,
    },

    # ── T8: conservation ─────────────────────────────────────
    # High initial energy but high step cost. Must be conservative.
    # Every wasted step is expensive. Tests energy-conservation behavior.
    "conservation": {
        "grid_size": 5,
        "max_steps": 20,
        "initial_energy": 20,
        "energy_cap": 24,
        "step_cost": 2,
        "food_positions": [(3, 1)],
        "food_gain": 12,
        "hazards": ((2, 2),),
        "hazard_cost": 4,
        "walls": (),
        "gate_locked": False,
    },

    # ── T9: endurance ────────────────────────────────────────
    # Large grid, low step cost, long path. Tests exploration stamina.
    # Need to find food along the way. Patience > speed.
    "endurance": {
        "grid_size": 7,
        "max_steps": 50,
        "initial_energy": 16,
        "energy_cap": 24,
        "step_cost": 1,
        "food_positions": [(2, 1)],
        "food_gain": 12,
        "hazards": ((2, 3), (4, 4)),
        "hazard_cost": 3,
        "walls": (),
        "gate_locked": False,
    },

    # ── T10: collect_exit ────────────────────────────────────
    # Exit is locked until the agent collects food first.
    # Must eat food to "unlock the gate", then navigate to exit.
    # Tests multi-objective planning: collect → then navigate.
    "collect_exit": {
        "grid_size": 6,
        "max_steps": 32,
        "initial_energy": 14,
        "energy_cap": 20,
        "step_cost": 1,
        "food_positions": [(1, 4)],
        "food_gain": 10,
        "hazards": ((2, 2), (3, 3)),
        "hazard_cost": 3,
        "walls": (),
        "gate_locked": True,
    },

    # ── T11: sprint ──────────────────────────────────────────
    # Very short step limit. Must find the most efficient path.
    # No room for exploration — pure efficiency optimization.
    "sprint": {
        "grid_size": 5,
        "max_steps": 12,
        "initial_energy": 14,
        "energy_cap": 18,
        "step_cost": 1,
        "food_positions": [(2, 2)],
        "food_gain": 7,
        "hazards": (),
        "hazard_cost": 0,
        "walls": (),
        "gate_locked": False,
    },

    # ── T12: gauntlet_refuel ─────────────────────────────────
    # Hardest task: large grid, multiple hazards, must collect food
    # midway through a hazardous corridor, then reach exit.
    # Integrates ALL skills: navigation, food, hazard, efficiency.
    #
    # Layout (7x7):
    #   @ . . . . . .
    #   . . H . . . .
    #   . . . . H . .
    #   . F1. . . . .
    #   . . H . . F2.
    #   . . . . H . .
    #   . . . . . . E
    "gauntlet_refuel": {
        "grid_size": 7,
        "max_steps": 50,
        "initial_energy": 16,
        "energy_cap": 24,
        "step_cost": 1,
        "food_positions": [(3, 1), (4, 5)],
        "food_gain": 10,
        "hazards": ((1, 2), (2, 4), (4, 2), (5, 4)),
        "hazard_cost": 4,
        "walls": (),
        "gate_locked": False,
    },
}


# ── Three Task Sequences ────────────────────────────────────

TASK_SEQUENCES = {
    # α: Energy-Transfer — energy management skills build cumulatively.
    # Homeostatic agents should show strongest forward transfer here.
    "alpha": [
        "reach", "recharge", "dual_food", "conservation", "endurance",
        "collect_exit", "detour", "wall_maze", "sprint", "gauntlet_refuel",
    ],

    # β: Safety-Transfer — safety/avoidance skills learned first,
    # then applied to resource tasks. Mixed transfer pattern.
    "beta": [
        "reach", "hazard_cross", "hazard_gauntlet", "wall_maze", "detour",
        "recharge", "sprint", "conservation", "collect_exit", "gauntlet_refuel",
    ],

    # γ: Interference — deliberately conflicting strategies.
    # e.g., conservation (be slow) → sprint (be fast).
    # Tests robustness to negative transfer.
    "gamma": [
        "conservation", "sprint", "hazard_gauntlet", "reach", "endurance",
        "dual_food", "hazard_cross", "collect_exit", "recharge", "wall_maze",
    ],

    # Legacy 5-task sequence for backward compatibility
    "legacy": [
        "reach", "recharge", "hazard_cross", "detour", "sprint",
    ],
}


# Maximum counts for observation padding
_MAX_FOOD = 2
_MAX_HAZARDS = 4
_OBS_DIM = 19


class SequentialHomeostasisEnv(gym.Env):
    """Sequential-task gridworld with shared energy dynamics.

    Supports walls, multi-food, gate mechanism, and
    up to 4 hazard positions for structurally diverse tasks.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        task_name: str = "reach",
        reward_mode: str = "homeostatic",
        observation_mode: str = "full",
        exit_bonus: float = 6.0,
        death_penalty: float = 6.0,
        progress_coef: float = 0.35,
        food_bonus: float = 1.5,
        hazard_penalty: float = 1.0,
        internal_coef: float = 1.0,
        task_reward_coef: float = 1.0,
        initial_energy_override: float | None = None,
    ):
        super().__init__()
        if task_name not in TASK_SPECS:
            raise ValueError(f"Unknown task_name: {task_name}")
        if reward_mode not in {"task", "homeostatic", "mixed", "eval"}:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")
        if observation_mode not in {"full", "masked"}:
            raise ValueError(f"Unknown observation_mode: {observation_mode}")

        self.task_name = task_name
        self.reward_mode = reward_mode
        self.observation_mode = observation_mode
        self.exit_bonus = exit_bonus
        self.death_penalty = death_penalty
        self.progress_coef = progress_coef
        self.food_bonus = food_bonus
        self.hazard_penalty = hazard_penalty
        self.internal_coef = internal_coef
        self.task_reward_coef = task_reward_coef
        self.initial_energy_override = initial_energy_override

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.5, shape=(_OBS_DIM,), dtype=np.float32
        )
        self.obs_dim = _OBS_DIM
        self._dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self._load_task_spec()

    # ── Task loading ────────────────────────────────────────

    def _load_task_spec(self) -> None:
        spec = TASK_SPECS[self.task_name]
        self.grid_size = spec["grid_size"]
        self.max_steps = spec["max_steps"]

        base_initial = float(spec["initial_energy"])
        if self.initial_energy_override is None:
            self.initial_energy = base_initial
        else:
            self.initial_energy = float(
                np.clip(self.initial_energy_override, 0.0, spec["energy_cap"])
            )

        self.energy_cap = float(spec["energy_cap"])
        self.step_cost = float(spec["step_cost"])
        self.food_gain = float(spec["food_gain"])
        self.hazard_cost = float(spec["hazard_cost"])
        self.setpoint = self.energy_cap

        # Multi-food support
        self._food_positions = list(spec["food_positions"])

        # Hazard support (up to 4)
        self._hazard_set = set(spec["hazards"])

        # Walls
        self._wall_set = set(spec.get("walls", ()))

        # Gate mechanism
        self._gate_locked_spec = spec.get("gate_locked", False)

    # ── Reset ───────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "initial_energy" in options:
            self.initial_energy_override = options["initial_energy"]
        self._load_task_spec()

        self.agent_pos = (0, 0)
        self.exit_pos = (self.grid_size - 1, self.grid_size - 1)

        # Food state: list of booleans indicating availability
        self.food_available = [True] * len(self._food_positions)

        # Gate state
        self.gate_locked = self._gate_locked_spec

        self.energy = self.initial_energy
        self.steps = 0
        self.stats = {
            "success": False,
            "energy_depleted": False,
            "food_collected": 0,
            "food_total": len(self._food_positions),
            "hazard_hits": 0,
            "start_energy": self.energy,
            "energy_left": self.energy,
        }
        return self._obs(), {}

    # ── Step ────────────────────────────────────────────────

    def step(self, action):
        self.steps += 1
        terminated = False
        truncated = False

        old_energy = self.energy
        old_target_dist = self._target_distance(self.agent_pos)

        # Movement with wall collision
        dr, dc = self._dirs[action]
        nr = int(np.clip(self.agent_pos[0] + dr, 0, self.grid_size - 1))
        nc = int(np.clip(self.agent_pos[1] + dc, 0, self.grid_size - 1))

        if (nr, nc) not in self._wall_set:
            self.agent_pos = (nr, nc)
        # else: agent stays in place (wall blocks movement)

        # Energy cost
        self.energy -= self.step_cost

        # Food collection
        ate_food = False
        for i, fpos in enumerate(self._food_positions):
            if self.food_available[i] and self.agent_pos == fpos:
                self.food_available[i] = False
                self.energy = min(self.energy + self.food_gain, self.energy_cap)
                self.stats["food_collected"] += 1
                ate_food = True

        # Gate: unlock when all food collected
        if self.gate_locked and all(not avail for avail in self.food_available):
            self.gate_locked = False

        # Hazard
        hazard_hit = self.agent_pos in self._hazard_set
        if hazard_hit:
            self.energy -= self.hazard_cost
            self.stats["hazard_hits"] += 1

        self.energy = float(np.clip(self.energy, -999.0, self.energy_cap))

        # Termination: exit reached (only if gate is open)
        if self.agent_pos == self.exit_pos and not self.gate_locked:
            terminated = True
            self.stats["success"] = True

        # Termination: energy depleted
        if self.energy <= 0:
            terminated = True
            self.stats["energy_depleted"] = True

        # Truncation: time out
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        # ── Reward computation ──────────────────────────────
        task_reward = 0.0
        homeostatic_reward = 0.0

        if self.reward_mode in {"task", "mixed"}:
            new_target_dist = self._target_distance(self.agent_pos)
            task_reward += self.progress_coef * (old_target_dist - new_target_dist)
            if ate_food:
                task_reward += self.food_bonus
            if hazard_hit:
                task_reward -= self.hazard_penalty

        if self.reward_mode in {"homeostatic", "mixed"}:
            old_drive = abs(self.setpoint - old_energy)
            new_drive = abs(self.setpoint - self.energy)
            homeostatic_reward += self.internal_coef * (old_drive - new_drive)

        reward = self.task_reward_coef * task_reward + homeostatic_reward

        if self.stats["success"]:
            reward += self.exit_bonus
        if self.stats["energy_depleted"]:
            reward -= self.death_penalty

        self.stats["energy_left"] = self.energy
        return self._obs(), reward, terminated, truncated, {}

    # ── Navigation target ───────────────────────────────────

    def _target_distance(self, pos) -> int:
        """Manhattan distance to the current navigation target.

        Target priority:
        1. If gate is locked → nearest uncollected food
        2. If any food is available → nearest uncollected food
        3. Otherwise → exit
        """
        available_food = [
            self._food_positions[i]
            for i in range(len(self._food_positions))
            if self.food_available[i]
        ]

        if available_food:
            # Choose nearest food
            distances = [
                abs(pos[0] - fp[0]) + abs(pos[1] - fp[1])
                for fp in available_food
            ]
            return min(distances)

        return abs(pos[0] - self.exit_pos[0]) + abs(pos[1] - self.exit_pos[1])

    # ── Observation ─────────────────────────────────────────

    def _obs(self):
        """Build 19-dimensional observation vector.

        Layout:
            [0-1]   agent position (row, col) / grid
            [2-3]   exit position / grid
            [4-6]   food 1: row, col, available
            [7-9]   food 2: row, col, available
            [10]    energy observation (or -1.0 if masked)
            [11]    task index / (N-1)
            [12-13] hazard 1: row, col
            [14-15] hazard 2: row, col
            [16-17] hazard 3: row, col
            [18]    gate locked (1.0 / 0.0)
        """
        g = max(self.grid_size - 1, 1)
        _ABSENT = -0.25

        # Food slots (padded to 2)
        food_obs = []
        for slot in range(_MAX_FOOD):
            if slot < len(self._food_positions):
                fp = self._food_positions[slot]
                avail = self.food_available[slot]
                food_obs.extend([
                    fp[0] / g if avail else _ABSENT,
                    fp[1] / g if avail else _ABSENT,
                    1.0 if avail else 0.0,
                ])
            else:
                food_obs.extend([_ABSENT, _ABSENT, 0.0])

        # Hazard slots (padded to 3 — we use 3 slots in obs for space)
        hazard_list = list(self._hazard_set)
        hazard_obs = []
        for slot in range(3):
            if slot < len(hazard_list):
                h = hazard_list[slot]
                hazard_obs.extend([h[0] / g, h[1] / g])
            else:
                hazard_obs.extend([_ABSENT, _ABSENT])

        obs = [
            self.agent_pos[0] / g,           # 0
            self.agent_pos[1] / g,           # 1
            self.exit_pos[0] / g,            # 2
            self.exit_pos[1] / g,            # 3
            *food_obs,                       # 4-9
            self._energy_obs(),              # 10
            TASK_INDEX.get(self.task_name, 0) / max(len(TASK_CATALOG) - 1, 1),  # 11
            *hazard_obs,                     # 12-17
            1.0 if self.gate_locked else 0.0,  # 18
        ]

        return np.array(obs, dtype=np.float32)

    def _energy_obs(self) -> float:
        if self.observation_mode == "masked":
            return -1.0
        return self.energy / self.energy_cap

    # ── Rendering ───────────────────────────────────────────

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.exit_pos[0]][self.exit_pos[1]] = "E" if not self.gate_locked else "X"
        for i, fp in enumerate(self._food_positions):
            if self.food_available[i]:
                grid[fp[0]][fp[1]] = "F"
        for hr, hc in self._hazard_set:
            grid[hr][hc] = "H"
        for wr, wc in self._wall_set:
            grid[wr][wc] = "W"
        grid[self.agent_pos[0]][self.agent_pos[1]] = "@"

        gate_str = " LOCKED" if self.gate_locked else ""
        lines = [
            f"task={self.task_name} reward={self.reward_mode} "
            f"energy={self.energy:.1f} step={self.steps}{gate_str}"
        ]
        for row in grid:
            lines.append(" ".join(row))
        return "\n".join(lines)
