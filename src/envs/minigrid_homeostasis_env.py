"""
MiniGrid Sequential Homeostasis Environment — 12-Task Version
=============================================================
12 structurally diverse tasks mirroring the gridworld alpha sequence.
Tasks progressively increase in difficulty via tighter energy margins
AND more complex layouts, with a gate mechanism for multi-objective pressure.

TASK SEQUENCE (MG_ALPHA_SEQUENCE)
----------------------------------
  T1  mg_reach          : empty room, navigate to goal
  T2  mg_recharge       : must collect food to survive
  T3  mg_hazard_reach   : lava blocks direct path
  T4  mg_detour         : food + lava combined
  T5  mg_conservation   : high step cost, must be efficient
  T6  mg_dual_food      : two food sources, choose wisely
  T7  mg_gate_collect   : exit locked until food collected
  T8  mg_hazard_gauntlet: dense lava field, narrow corridor
  T9  mg_wall_maze      : impassable walls force detour
  T10 mg_endurance      : large grid, long path, must refuel midway
  T11 mg_sprint         : very tight step limit, efficiency critical
  T12 mg_gauntlet_refuel: hardest — large grid, lava, two foods, gate

WHAT MAKES THIS HARDER THAN THE 5-TASK VERSION
-----------------------------------------------
- Energy margins shrink progressively: T1 has lots of slack, T12 has almost none
- Step costs increase on conservation/sprint tasks
- Gate mechanism forces specific ordering (food before exit)
- Wall mazes force longer paths without extra energy
- Dual food tests decision-making under scarcity
- T12 combines everything: navigation + food + hazard + gate + large grid

INTERFACE CONTRACT (identical to SequentialHomeostasisEnv)
----------------------------------------------------------
  Constructor : MiniGridHomeostasisEnv(task_name, reward_mode,
                                       observation_mode, ...)
  Attributes  : energy, energy_cap, agent_pos, food_available (list),
                _hazard_set (set of (r,c)), stats dict, obs_dim=19
  reward_mode : "task" | "homeostatic" | "mixed" | "eval"
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import minigrid  # noqa
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv

# ── Task catalog ──────────────────────────────────────────────────────────────

MG_TASK_CATALOG = [
    "mg_reach",
    "mg_recharge",
    "mg_hazard_reach",
    "mg_detour",
    "mg_conservation",
    "mg_dual_food",
    "mg_gate_collect",
    "mg_hazard_gauntlet",
    "mg_wall_maze",
    "mg_endurance",
    "mg_sprint",
    "mg_gauntlet_refuel",
]

MG_TASK_INDEX = {name: i for i, name in enumerate(MG_TASK_CATALOG)}

# ── Task specifications ───────────────────────────────────────────────────────
# Energy margins are deliberately tight to reward conservation.
# An agent that exits T1 with 5 energy will struggle at T2.
# An agent that exits T1 with 12 energy will have real margin.
#
# Fields:
#   grid_size   : side length of square room (interior = grid_size-2)
#   max_steps   : episode truncation
#   initial_energy : energy at task start (ignored in carryover mode)
#   energy_cap  : maximum energy (carryover is clipped to this)
#   step_cost   : energy lost per step
#   food_gain   : energy gained per food item collected
#   hazard_cost : energy lost on lava contact
#   n_food      : number of food items (keys)
#   n_lava      : number of lava tiles
#   gate_locked : exit locked until ALL food collected
#   wall_pattern: "none" | "corridor" | "maze"
#   food_gain_2 : gain for second food item (if different)

MG_TASK_SPECS = {
    # T1: Simple navigation. Generous energy. Agent learns to reach exit.
    # Tight enough that spending energy randomly hurts.
    "mg_reach": {
        "grid_size":     9,
        "max_steps":     60,
        "initial_energy": 18.0,
        "energy_cap":    20.0,
        "step_cost":     1.0,
        "food_gain":     0.0,
        "food_gain_2":   0.0,
        "hazard_cost":   0.0,
        "n_food":        0,
        "n_lava":        0,
        "gate_locked":   False,
        "wall_pattern":  "none",
        "description":   "Empty room. Navigate to goal.",
    },

    # T2: Must collect food to survive. Direct path costs ~14 steps.
    # Without food, agent runs out. With food, has ~12 energy to spare.
    "mg_recharge": {
        "grid_size":     9,
        "max_steps":     80,
        "initial_energy": 12.0,
        "energy_cap":    20.0,
        "step_cost":     1.0,
        "food_gain":     14.0,
        "food_gain_2":   0.0,
        "hazard_cost":   0.0,
        "n_food":        1,
        "n_lava":        0,
        "gate_locked":   False,
        "wall_pattern":  "none",
        "description":   "Collect food to survive long path.",
    },

    # T3: Lava blocks direct path. Must navigate around.
    # Optional food near lava. Hazard cost is significant.
    "mg_hazard_reach": {
        "grid_size":     9,
        "max_steps":     90,
        "initial_energy": 16.0,
        "energy_cap":    22.0,
        "step_cost":     1.0,
        "food_gain":     8.0,
        "food_gain_2":   0.0,
        "hazard_cost":   4.0,
        "n_food":        1,
        "n_lava":        3,
        "gate_locked":   False,
        "wall_pattern":  "none",
        "description":   "Lava blocks direct path. Optional food.",
    },

    # T4: Food + lava. Must detour through food AND avoid lava.
    # Combining both pressures. Tight margins.
    "mg_detour": {
        "grid_size":     9,
        "max_steps":     100,
        "initial_energy": 12.0,
        "energy_cap":    22.0,
        "step_cost":     1.0,
        "food_gain":     14.0,
        "food_gain_2":   0.0,
        "hazard_cost":   4.0,
        "n_food":        1,
        "n_lava":        3,
        "gate_locked":   False,
        "wall_pattern":  "none",
        "description":   "Must collect food AND avoid lava.",
    },

    # T5: High step cost. Every wasted move is expensive.
    # Tests whether agent learned efficiency from homeostatic objective.
    "mg_conservation": {
        "grid_size":     7,
        "max_steps":     30,
        "initial_energy": 20.0,
        "energy_cap":    24.0,
        "step_cost":     2.0,
        "food_gain":     10.0,
        "food_gain_2":   0.0,
        "hazard_cost":   6.0,
        "n_food":        1,
        "n_lava":        2,
        "gate_locked":   False,
        "wall_pattern":  "none",
        "description":   "High step cost. Efficiency critical.",
    },

    # T6: Two food sources. One closer (small gain), one farther (large gain).
    # Tests decision-making under energy constraints.
    "mg_dual_food": {
        "grid_size":     9,
        "max_steps":     90,
        "initial_energy": 10.0,
        "energy_cap":    22.0,
        "step_cost":     1.0,
        "food_gain":     8.0,    # near food
        "food_gain_2":   14.0,   # far food
        "hazard_cost":   3.0,
        "n_food":        2,
        "n_lava":        2,
        "gate_locked":   False,
        "wall_pattern":  "none",
        "description":   "Two food sources. Choose wisely.",
    },

    # T7: Exit locked until food collected. Multi-objective.
    # Agent MUST collect food before exit works.
    "mg_gate_collect": {
        "grid_size":     9,
        "max_steps":     100,
        "initial_energy": 14.0,
        "energy_cap":    22.0,
        "step_cost":     1.0,
        "food_gain":     12.0,
        "food_gain_2":   0.0,
        "hazard_cost":   4.0,
        "n_food":        1,
        "n_lava":        2,
        "gate_locked":   True,
        "wall_pattern":  "none",
        "description":   "Exit locked until food collected.",
    },

    # T8: Dense lava field. 5 lava tiles create narrow corridor.
    # Very punishing if agent stumbles into lava.
    "mg_hazard_gauntlet": {
        "grid_size":     9,
        "max_steps":     100,
        "initial_energy": 18.0,
        "energy_cap":    22.0,
        "step_cost":     1.0,
        "food_gain":     10.0,
        "food_gain_2":   0.0,
        "hazard_cost":   3.0,
        "n_food":        1,
        "n_lava":        5,
        "gate_locked":   False,
        "wall_pattern":  "none",
        "description":   "Dense lava. Navigate narrow corridor.",
    },

    # T9: Wall maze. Impassable walls force longer path.
    # Agent can't take direct route even if it has energy.
    "mg_wall_maze": {
        "grid_size":     9,
        "max_steps":     120,
        "initial_energy": 22.0,
        "energy_cap":    26.0,
        "step_cost":     1.0,
        "food_gain":     12.0,
        "food_gain_2":   0.0,
        "hazard_cost":   4.0,
        "n_food":        1,
        "n_lava":        2,
        "gate_locked":   False,
        "wall_pattern":  "corridor",
        "description":   "Walls force detour. Efficiency matters.",
    },

    # T10: Large grid, long path. Must refuel midway.
    # Tests stamina and planning over longer horizons.
    "mg_endurance": {
        "grid_size":     11,
        "max_steps":     140,
        "initial_energy": 16.0,
        "energy_cap":    24.0,
        "step_cost":     1.0,
        "food_gain":     14.0,
        "food_gain_2":   0.0,
        "hazard_cost":   4.0,
        "n_food":        1,
        "n_lava":        3,
        "gate_locked":   False,
        "wall_pattern":  "none",
        "description":   "Large grid. Must refuel midway.",
    },

    # T11: Very tight step limit. Pure efficiency.
    # No room for exploration. Agent must know the optimal path.
    "mg_sprint": {
        "grid_size":     7,
        "max_steps":     18,
        "initial_energy": 18.0,
        "energy_cap":    20.0,
        "step_cost":     1.0,
        "food_gain":     8.0,
        "food_gain_2":   0.0,
        "hazard_cost":   0.0,
        "n_food":        0,
        "n_lava":        0,
        "gate_locked":   False,
        "wall_pattern":  "none",
        "description":   "Tight step limit. Optimal path required.",
    },

    # T12: Hardest. Large grid + lava + two foods + gate.
    # Integrates ALL skills. Very tight margins.
    "mg_gauntlet_refuel": {
        "grid_size":     11,
        "max_steps":     160,
        "initial_energy": 14.0,
        "energy_cap":    26.0,
        "step_cost":     1.0,
        "food_gain":     12.0,
        "food_gain_2":   12.0,
        "hazard_cost":   3.0,
        "n_food":        2,
        "n_lava":        5,
        "gate_locked":   True,
        "wall_pattern":  "none",
        "description":   "All skills combined. Hardest task.",
    },
}

# Alpha sequence — energy management skills build cumulatively
MG_ALPHA_SEQUENCE = list(MG_TASK_CATALOG)

# Shorter sequence for quick runs
MG_LEGACY_SEQUENCE = [
    "mg_reach", "mg_recharge", "mg_hazard_reach",
    "mg_detour", "mg_sprint",
]

MG_TASK_SEQUENCE = [
    "mg_reach",
    "mg_recharge", 
    "mg_hazard_reach",
    "mg_detour",
    "mg_conservation",
    "mg_dual_food",
]  # default


# ── Custom MiniGrid layout ────────────────────────────────────────────────────

def _lava_positions(n_lava: int, width: int, height: int,
                    blocked: set) -> list[tuple[int,int]]:
    """Generate n_lava positions in a cluster around the midpoint."""
    mid_x, mid_y = width // 2, height // 2
    candidates = [
        (mid_x,     mid_y),
        (mid_x,     mid_y + 1),
        (mid_x - 1, mid_y),
        (mid_x + 1, mid_y),
        (mid_x,     mid_y - 1),
        (mid_x + 1, mid_y + 1),
        (mid_x - 1, mid_y - 1),
    ]
    result = []
    for lx, ly in candidates:
        if len(result) >= n_lava:
            break
        if (1 <= lx < width - 1 and 1 <= ly < height - 1
                and (lx, ly) not in blocked):
            result.append((lx, ly))
    return result


class HomeostasisMiniGridEnv(MiniGridEnv):
    """Fixed-layout MiniGrid env supporting all 12 task types."""

    def __init__(self, spec: dict):
        self._spec       = spec
        self._food_pos   = None    # (col, row) for first food
        self._food_pos2  = None    # (col, row) for second food
        self._lava_cells = []      # list of (col, row)
        self._wall_cells = []      # list of (col, row)
        self._goal_pos   = (0, 0)  # (col, row)

        mission_space = MissionSpace(mission_func=lambda: "reach the goal")
        super().__init__(
            mission_space=mission_space,
            grid_size=spec["grid_size"],
            max_steps=spec["max_steps"],
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Goal: bottom-right interior
        gx, gy = width - 2, height - 2
        self.put_obj(Goal(), gx, gy)
        self._goal_pos = (gx, gy)

        self._food_pos   = None
        self._food_pos2  = None
        self._lava_cells = []
        self._wall_cells = []

        blocked = {(1, 1), (gx, gy)}  # agent start + goal

        # Wall pattern
        if self._spec["wall_pattern"] == "corridor":
            # Vertical wall in middle with gap — forces detour
            mid_x = width // 2
            gap_y  = height // 2
            for wy in range(1, height - 1):
                if wy != gap_y:
                    self.put_obj(Wall(), mid_x, wy)
                    self._wall_cells.append((mid_x, wy))
                    blocked.add((mid_x, wy))

        # Lava
        lava_cells = _lava_positions(self._spec["n_lava"], width, height, blocked)
        for lx, ly in lava_cells:
            self.put_obj(Lava(), lx, ly)
            self._lava_cells.append((lx, ly))
            blocked.add((lx, ly))

        # Food 1: left-middle area
        if self._spec["n_food"] >= 1:
            fx, fy = 2, height // 2
            while (fx, fy) in blocked:
                fy += 1
            self.put_obj(Key("yellow"), fx, fy)
            self._food_pos = (fx, fy)
            blocked.add((fx, fy))

        # Food 2: right-upper area (farther, potentially larger gain)
        if self._spec["n_food"] >= 2:
            fx2, fy2 = width - 3, 2
            while (fx2, fy2) in blocked:
                fx2 -= 1
            self.put_obj(Key("blue"), fx2, fy2)
            self._food_pos2 = (fx2, fy2)

        self.agent_pos = np.array([1, 1])
        self.agent_dir = 0
        self.mission   = "reach the goal"


# ── Main wrapper ──────────────────────────────────────────────────────────────

class MiniGridHomeostasisEnv(gym.Env):
    """
    12-task MiniGrid wrapper. Drop-in replacement for SequentialHomeostasisEnv.
    Uses direct cardinal movement and automatic food pickup.
    Supports gate mechanism, walls, dual food, and variable step cost.
    """

    metadata = {"render_modes": ["ansi"]}
    obs_dim  = 19

    def __init__(
        self,
        task_name: str = "mg_reach",
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

        if task_name not in MG_TASK_SPECS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Valid: {list(MG_TASK_SPECS)}"
            )
        if reward_mode not in {"task", "homeostatic", "mixed", "eval"}:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")
        if observation_mode not in {"full", "masked"}:
            raise ValueError(f"Unknown observation_mode: {observation_mode}")

        self.task_name               = task_name
        self.reward_mode             = reward_mode
        self.observation_mode        = observation_mode
        self.exit_bonus              = exit_bonus
        self.death_penalty           = death_penalty
        self.progress_coef           = progress_coef
        self.food_bonus              = food_bonus
        self.hazard_penalty          = hazard_penalty
        self.internal_coef           = internal_coef
        self.task_reward_coef        = task_reward_coef
        self.initial_energy_override = initial_energy_override

        self._spec = MG_TASK_SPECS[task_name]
        self._load_spec()
        self._mg = HomeostasisMiniGridEnv(self._spec)

        self.action_space      = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.5, shape=(self.obs_dim,), dtype=np.float32
        )

        # Episode state
        self.energy           = self.initial_energy
        self.agent_pos        = (1, 1)
        self.food_available:  list = []   # list of (row,col) for any()
        self._hazard_set:     set  = set()
        self._wall_set:       set  = set()
        self._goal_pos        = (0, 0)
        self._food1_collected = False
        self._food2_collected = False
        self.gate_locked      = False
        self._step_count      = 0
        self.stats            = {}

    def _load_spec(self):
        s = self._spec
        self.initial_energy = float(
            np.clip(self.initial_energy_override, 0.0, s["energy_cap"])
        ) if self.initial_energy_override is not None \
          else float(s["initial_energy"])
        self.energy_cap = float(s["energy_cap"])
        self.setpoint   = self.energy_cap

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_spec()
        self._mg.reset(seed=seed)

        self.energy           = self.initial_energy
        self._food1_collected = False
        self._food2_collected = False
        self._step_count      = 0
        self.agent_pos        = (1, 1)
        self._mg.agent_pos    = np.array([1, 1])
        self.gate_locked      = self._spec["gate_locked"]

        # Goal (row, col)
        gc, gr         = self._mg._goal_pos
        self._goal_pos = (gr, gc)

        # Food positions (row, col)
        self.food_available = []
        self._food1_pos = None
        self._food2_pos = None

        if self._mg._food_pos is not None:
            fc, fr = self._mg._food_pos
            self._food1_pos = (fr, fc)
            self.food_available.append((fr, fc))

        if self._mg._food_pos2 is not None:
            fc2, fr2 = self._mg._food_pos2
            self._food2_pos = (fr2, fc2)
            self.food_available.append((fr2, fc2))

        # Hazard set (row, col)
        self._hazard_set = {(lr, lc) for lc, lr in self._mg._lava_cells}

        # Wall set (row, col)
        self._wall_set = {(wr, wc) for wc, wr in self._mg._wall_cells}

        self.stats = {
            "success":         False,
            "energy_depleted": False,
            "food_collected":  False,  # True if any food eaten
            "hazard_hits":     0,
            "energy_left":     self.energy,
            "start_energy":    self.energy,
        }

        return self._obs(), {}

    # ── step ──────────────────────────────────────────────────────────────────

    _DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def step(self, action: int):
        self._step_count += 1
        old_energy = self.energy
        old_drive  = abs(self.setpoint - self.energy)
        old_pos    = self.agent_pos
        sz         = self._spec["grid_size"]

        # Move (direct position manipulation)
        dr, dc = self._DELTAS[action]
        nr     = int(np.clip(self.agent_pos[0] + dr, 1, sz - 2))
        nc     = int(np.clip(self.agent_pos[1] + dc, 1, sz - 2))

        # Wall bounce
        if (nr, nc) in self._wall_set:
            nr, nc = self.agent_pos

        self.agent_pos    = (nr, nc)
        self._mg.agent_pos = np.array([nc, nr])

        # Energy cost (step_cost can be > 1 for conservation task)
        self.energy -= self._spec["step_cost"]

        # Food 1 pickup
        ate_food = False
        if (self._food1_pos is not None and not self._food1_collected
                and self.agent_pos == self._food1_pos):
            self._food1_collected = True
            self.food_available   = [p for p in self.food_available
                                     if p != self._food1_pos]
            self.energy = float(np.clip(
                self.energy + self._spec["food_gain"],
                -999.0, self.energy_cap
            ))
            self.stats["food_collected"] = True
            ate_food = True
            fc, fr = self._mg._food_pos
            self._mg.grid.set(fc, fr, None)

        # Food 2 pickup
        if (self._food2_pos is not None and not self._food2_collected
                and self.agent_pos == self._food2_pos):
            self._food2_collected = True
            self.food_available   = [p for p in self.food_available
                                     if p != self._food2_pos]
            self.energy = float(np.clip(
                self.energy + self._spec["food_gain_2"],
                -999.0, self.energy_cap
            ))
            self.stats["food_collected"] = True
            ate_food = True
            fc2, fr2 = self._mg._food_pos2
            self._mg.grid.set(fc2, fr2, None)

        # Gate: unlock when all food collected
        if self.gate_locked:
            all_collected = (
                (self._food1_pos is None or self._food1_collected) and
                (self._food2_pos is None or self._food2_collected)
            )
            if all_collected:
                self.gate_locked = False

        # Hazard
        on_hazard = self.agent_pos in self._hazard_set
        if on_hazard:
            self.energy -= self._spec["hazard_cost"]
            self.stats["hazard_hits"] += 1

        self.energy = float(np.clip(self.energy, -999.0, self.energy_cap))

        # Terminal conditions
        terminated = False
        at_goal    = (self.agent_pos == self._goal_pos)
        if at_goal and not self.gate_locked:
            self.stats["success"] = True
            terminated = True
        if self.energy <= 0 and not terminated:
            self.stats["energy_depleted"] = True
            terminated = True

        truncated = (
            self._step_count >= self._spec["max_steps"] and not terminated
        )
        self.stats["energy_left"] = max(self.energy, 0.0)

        # ── Reward ────────────────────────────────────────────────────────────
        reward    = 0.0
        new_drive = abs(self.setpoint - self.energy)

        if self.reward_mode in {"task", "mixed"}:
            # Target: nearest uncollected food if available, else goal
            if self.food_available:
                dists  = [abs(self.agent_pos[0]-fp[0]) +
                          abs(self.agent_pos[1]-fp[1])
                          for fp in self.food_available]
                target = self.food_available[int(np.argmin(dists))]
            else:
                target = self._goal_pos

            old_dist = abs(old_pos[0]-target[0]) + abs(old_pos[1]-target[1])
            new_dist = abs(self.agent_pos[0]-target[0]) + \
                       abs(self.agent_pos[1]-target[1])
            reward  += self.task_reward_coef * self.progress_coef * \
                       (old_dist - new_dist)
            if ate_food:
                reward += self.task_reward_coef * self.food_bonus
            if on_hazard:
                reward -= self.task_reward_coef * self.hazard_penalty

        if self.reward_mode in {"homeostatic", "mixed"}:
            reward += self.internal_coef * (old_drive - new_drive)

        if self.stats["success"]:
            reward += self.exit_bonus
        if self.stats["energy_depleted"]:
            reward -= self.death_penalty

        return self._obs(), reward, terminated, truncated, {}

    # ── Observation (obs_dim = 19) ────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        sz     = float(self._spec["grid_size"] - 1)
        ar, ac = self.agent_pos
        gr, gc = self._goal_pos

        # Food slots: use first available food, or sentinel
        if self.food_available:
            fr, fc    = self.food_available[0]
            food_flag = 1.0
        else:
            fr, fc    = -1, -1
            food_flag = 0.0

        # Hazard slots (up to 3)
        hazards = list(self._hazard_set)[:3]
        while len(hazards) < 3:
            hazards.append((-1, -1))

        # Local view signals
        try:
            mg_obs        = self._mg.gen_obs()
            img           = mg_obs["image"]
            has_food_near = float(np.any(img[:, :, 0] == 5) or
                                  np.any(img[:, :, 0] == 6))  # key types
            has_lava_near = float(np.any(img[:, :, 0] == 9))
        except Exception:
            has_food_near = 0.0
            has_lava_near = 0.0

        energy_obs = (self.energy / self.energy_cap) \
            if self.observation_mode == "full" else -1.0

        obs = np.array([
            ar / sz, ac / sz,                                      # 0-1  agent
            gr / sz, gc / sz,                                      # 2-3  goal
            fr / sz if fr >= 0 else -0.25,                        # 4    food r
            fc / sz if fc >= 0 else -0.25,                        # 5    food c
            food_flag,                                             # 6    food available
            float(np.clip(energy_obs, -1.0, 1.5)),                # 7    energy
            MG_TASK_INDEX[self.task_name] / max(len(MG_TASK_INDEX)-1, 1),  # 8 task id
            hazards[0][0]/sz if hazards[0][0] >= 0 else -0.25,   # 9
            hazards[0][1]/sz if hazards[0][0] >= 0 else -0.25,   # 10
            hazards[1][0]/sz if hazards[1][0] >= 0 else -0.25,   # 11
            hazards[1][1]/sz if hazards[1][0] >= 0 else -0.25,   # 12
            hazards[2][0]/sz if hazards[2][0] >= 0 else -0.25,   # 13
            hazards[2][1]/sz if hazards[2][0] >= 0 else -0.25,   # 14
            has_food_near,                                         # 15
            has_lava_near,                                         # 16
            float(self._food1_collected or self._food2_collected), # 17
            float(self.gate_locked),                               # 18
        ], dtype=np.float32)

        assert len(obs) == self.obs_dim, \
            f"obs length {len(obs)} != {self.obs_dim}"
        return obs

    def render(self):
        return (
            f"task={self.task_name}  E={self.energy:.1f}/{self.energy_cap:.1f}"
            f"  agent={self.agent_pos}  goal={self._goal_pos}"
            f"  food={self.food_available}  gate={'LOCKED' if self.gate_locked else 'open'}"
            f"  step={self._step_count}"
        )

    def close(self):
        self._mg.close()
        super().close()


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    print("=" * 65)
    print("MiniGridHomeostasisEnv — 12-task smoke test")
    print("=" * 65)

    passed = failed = 0

    for task_name in MG_TASK_CATALOG:
        for reward_mode in ("task", "homeostatic", "mixed", "eval"):
            for obs_mode in ("full", "masked"):
                try:
                    env = MiniGridHomeostasisEnv(
                        task_name=task_name,
                        reward_mode=reward_mode,
                        observation_mode=obs_mode,
                    )
                    obs, _ = env.reset(seed=42)
                    assert obs.shape == (env.obs_dim,), \
                        f"Shape {obs.shape}"
                    assert isinstance(env.food_available, list)
                    assert isinstance(env._hazard_set, set)

                    total_r = 0.0
                    for _ in range(60):
                        a = env.action_space.sample()
                        obs, r, term, trunc, _ = env.step(a)
                        total_r += r
                        if term or trunc:
                            break

                    print(f"  ok {task_name:<22} {reward_mode:<12} "
                          f"{obs_mode:<6} "
                          f"E={env.energy:5.1f}  r={total_r:+6.2f}  "
                          f"ok={env.stats['success']}  "
                          f"food={env.stats['food_collected']}")
                    env.close()
                    passed += 1

                except Exception as e:
                    print(f"  FAIL {task_name} {reward_mode} {obs_mode}: {e}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'='*65}")
    print(f"  Passed: {passed}   Failed: {failed}")
    print(f"  Total tasks: {len(MG_TASK_CATALOG)}")

    # Food pickup + gate test
    print("\n--- Gate mechanism test (mg_gate_collect) ---")
    env = MiniGridHomeostasisEnv("mg_gate_collect", "task", "full",
                                  initial_energy_override=14.0)
    obs, _ = env.reset(seed=0)
    print(f"  gate_locked={env.gate_locked}  "
          f"food={env.food_available}  "
          f"goal={env._goal_pos}")

    # Walk to food
    food_r, food_c = env.food_available[0] if env.food_available else (0,0)
    for step in range(50):
        ar, ac = env.agent_pos
        if ar < food_r: a = 1
        elif ar > food_r: a = 0
        elif ac < food_c: a = 3
        elif ac > food_c: a = 2
        else: a = 3
        obs, r, term, trunc, _ = env.step(a)
        if env.stats["food_collected"] and not env.gate_locked:
            print(f"  Food collected + gate unlocked at step {step+1}!")
            break
        if term or trunc:
            print(f"  Episode ended. food={env.stats['food_collected']} "
                  f"gate={env.gate_locked}")
            break
    env.close()

    print("\nPlug in with:")
    print("  from src.envs.minigrid_homeostasis_env import (")
    print("      MiniGridHomeostasisEnv, MG_TASK_SPECS,")
    print("      MG_TASK_SEQUENCE, MG_ALPHA_SEQUENCE)")