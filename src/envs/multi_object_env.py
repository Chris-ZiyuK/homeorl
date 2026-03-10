"""
Multi-Object Survival Environment for Homeostatic Grounding
============================================================
3 object types with hidden effects. Agent must learn which are
beneficial/harmful through interaction. Supports 3 observation modes:
  A: terminal-only (no energy visible)
  B: energy-aware (energy in obs)  
  C: energy + QUERY (energy + hint in obs)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MultiObjectEnv(gym.Env):
    """
    7x7 grid with 3 object types + exit.
    
    Objects (positions randomized each episode):
      Type 0 ('a', red):   harmful,   E -= 40
      Type 1 ('b', blue):  beneficial, E += 30
      Type 2 ('c', green): neutral,   E += 0

    Agent must reach exit (+1 reward) without dying (E <= 0).
    Energy starts at 100, costs 5/step. Must eat beneficial to survive.
    
    obs_mode controls what agent can see:
      'terminal':  positions + object types only
      'energy':    + normalized energy
      'full':      + energy + hint slots
    """

    def __init__(self, grid_size=7, max_steps=40,
                 E_init=100, c_step=5, c_query=8,
                 harmful_cost=40, beneficial_gain=30,
                 obs_mode='energy', reward_type='terminal'):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.E_init = E_init
        self.c_step = c_step
        self.c_query = c_query
        self.harmful_cost = harmful_cost
        self.beneficial_gain = beneficial_gain
        self.obs_mode = obs_mode  # 'terminal', 'energy', 'full'
        self.reward_type = reward_type
        self.setpoint = float(E_init)

        # Actions: 0=up 1=down 2=left 3=right, 4=QUERY (only in 'full')
        n_act = 5 if obs_mode == 'full' else 4
        self.action_space = spaces.Discrete(n_act)

        # Obs dimensions vary by mode
        # base: agent(2) + 3 objects(6) + exit(2) + obj_types(3) = 13
        # energy: base + energy(1) = 14
        # full: base + energy(1) + hint(3) = 17
        if obs_mode == 'terminal':
            self.obs_dim = 13
        elif obs_mode == 'energy':
            self.obs_dim = 14
        else:  # full
            self.obs_dim = 17

        self.observation_space = spaces.Box(
            low=-0.2, high=1.5, shape=(self.obs_dim,), dtype=np.float32)

        self._dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Object effects: type_id -> energy change
        self.effects = {0: -harmful_cost, 1: +beneficial_gain, 2: 0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        g = self.grid_size

        self.exit_pos = (g - 1, g - 1)
        used = [self.exit_pos]

        # Agent at (0, 0)
        self.agent_pos = (0, 0)
        used.append(self.agent_pos)

        # Place 3 objects (one of each type) in interior
        self.objects = []  # [(row, col, type_id, exists)]
        for type_id in range(3):
            pos = self._rand_pos(used)
            used.append(pos)
            self.objects.append([pos[0], pos[1], type_id, True])

        self.energy = float(self.E_init)
        self.hint = np.zeros(3, dtype=np.float32)  # one-hot per type
        self.has_queried = False
        self.steps = 0

        # Stats
        self.stats = {
            "ate_harmful": 0, "ate_beneficial": 0, "ate_neutral": 0,
            "reached_exit": False, "energy_depleted": False,
            "queries": 0,
            "approach_decisions": {0: [0, 0], 1: [0, 0], 2: [0, 0]},
            # [approach_count, avoid_count] for each type
        }

        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        
        old_drive = abs(self.setpoint - self.energy)

        if action == 4 and self.obs_mode == 'full':
            # QUERY: reveal all object effects
            self.energy -= self.c_query
            self.stats["queries"] += 1
            if not self.has_queried:
                self.has_queried = True
                # hint encodes: [harmful_flag, beneficial_flag, neutral_flag]
                # type 0=harmful(-1), type 1=beneficial(+1), type 2=neutral(0)
                self.hint = np.array([-1.0, 1.0, 0.0], dtype=np.float32)
        elif action < 4:
            dr, dc = self._dirs[action]
            nr = int(np.clip(self.agent_pos[0] + dr, 0, self.grid_size - 1))
            nc = int(np.clip(self.agent_pos[1] + dc, 0, self.grid_size - 1))
            self.agent_pos = (nr, nc)

        # Per-step energy cost
        self.energy -= self.c_step

        # Check object interactions
        for obj in self.objects:
            if obj[3] and (self.agent_pos[0] == obj[0] and
                           self.agent_pos[1] == obj[1]):
                obj[3] = False  # consumed
                effect = self.effects[obj[2]]
                self.energy += effect

                if obj[2] == 0:
                    self.stats["ate_harmful"] += 1
                elif obj[2] == 1:
                    self.stats["ate_beneficial"] += 1
                else:
                    self.stats["ate_neutral"] += 1

        self.energy = np.clip(self.energy, -999, self.E_init + 50)

        # Reached exit?
        if self.agent_pos == self.exit_pos:
            if self.reward_type != 'hrrl':
                reward = 1.0
            else:
                reward = 1.0  # Keep external exit reward +1 for hybrid
            terminated = True
            self.stats["reached_exit"] = True

        # Energy depleted?
        if self.energy <= 0:
            if self.reward_type != 'hrrl':
                reward = -1.0
            terminated = True
            self.stats["energy_depleted"] = True
            
        if self.reward_type == 'hrrl':
            new_drive = abs(self.setpoint - self.energy)
            r_internal = old_drive - new_drive
            reward += r_internal

        truncated = (self.steps >= self.max_steps) and not terminated

        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        g = self.grid_size - 1
        parts = [
            self.agent_pos[0] / g, self.agent_pos[1] / g,  # agent pos
        ]
        # 3 objects: pos + type one-hot-ish
        for obj in self.objects:
            if obj[3]:  # exists
                parts.extend([obj[0] / g, obj[1] / g])
            else:
                parts.extend([-0.1, -0.1])  # consumed
        # Exit
        parts.extend([self.exit_pos[0] / g, self.exit_pos[1] / g])
        # Object type identifiers (always visible, but meaning unknown)
        for obj in self.objects:
            parts.append((obj[2] + 1) / 3.0)  # 0.33, 0.67, 1.0

        if self.obs_mode in ('energy', 'full'):
            parts.append(self.energy / self.E_init)

        if self.obs_mode == 'full':
            parts.extend(self.hint.tolist())

        return np.array(parts, dtype=np.float32)

    def _rand_pos(self, exclude):
        while True:
            pos = (int(self.np_random.integers(1, self.grid_size - 1)),
                   int(self.np_random.integers(1, self.grid_size - 1)))
            if pos not in exclude:
                return pos
