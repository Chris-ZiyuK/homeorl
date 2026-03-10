"""
Poison Food Environment v2
=============================================
Core improvement: agent must eat food to survive and reach the exit.
- Grid 7x7, agent starts at top-left (0,0), exit at bottom-right (6,6)
- Shortest path takes 12 steps, each step costs c_step=8 energy -> at least 96 energy consumed
- Initial energy E_init=80 -> will definitely die without eating food (80 < 96)
- Safe food gives +50 energy, poison food -> instant death
- Therefore, agent must eat safe food to reach the exit
- QUERY costs c_query=5 energy to get a hint (which one is safe)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PoisonFoodEnv(gym.Env):
    """
    Observation (11 dimensions):
      [agent_r, agent_c, food1_r, food1_c, food2_r, food2_c,
       exit_r, exit_c, energy, hint_0, hint_1]

    Actions (5): Up/Down/Left/Right/QUERY
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size=7, max_steps=40,
                 E_init=80, c_step=8, c_query=5,
                 safe_gain=50, reward_type='terminal'):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.E_init = E_init
        self.c_step = c_step
        self.c_query = c_query
        self.safe_gain = safe_gain
        self.reward_type = reward_type
        self.setpoint = float(E_init)

        self.action_space = spaces.Discrete(5)   # Up, Down, Left, Right + QUERY
        self.observation_space = spaces.Box(
            low=-0.2, high=1.2, shape=(11,), dtype=np.float32)

        self._directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = (0, 0)
        self.exit_pos = (self.grid_size - 1, self.grid_size - 1)

        # Place two foods near the path from agent to exit
        # Food is placed in the middle area, a required path for the agent
        mid = self.grid_size // 2
        possible_food_cells = []
        for r in range(1, self.grid_size - 1):
            for c in range(1, self.grid_size - 1):
                possible_food_cells.append((r, c))

        chosen = self.np_random.choice(len(possible_food_cells), 2,
                                       replace=False)
        self.food1_pos = possible_food_cells[chosen[0]]
        self.food2_pos = possible_food_cells[chosen[1]]

        # Randomly choose which one is safe
        self.safe_food = self.np_random.choice([1, 2])

        self.food1_exists = True
        self.food2_exists = True
        self.energy = float(self.E_init)
        self.hint = np.array([0.0, 0.0], dtype=np.float32)
        self.has_queried = False
        self.steps = 0

        self.stats = {
            "queries": 0,
            "ate_safe": False,
            "ate_poison": False,
            "reached_exit": False,
            "energy_depleted": False,
        }

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        
        # Keep track of old drive for HRRL
        old_drive = abs(self.setpoint - self.energy)

        if action == 4:  # QUERY
            self.energy -= self.c_query
            self.stats["queries"] += 1
            if not self.has_queried:
                self.has_queried = True
                if self.safe_food == 1:
                    self.hint = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    self.hint = np.array([0.0, 1.0], dtype=np.float32)
        else:
            # Move
            dr, dc = self._directions[action]
            new_r = np.clip(self.agent_pos[0] + dr, 0, self.grid_size - 1)
            new_c = np.clip(self.agent_pos[1] + dc, 0, self.grid_size - 1)
            self.agent_pos = (int(new_r), int(new_c))

        # Consume energy per step
        self.energy -= self.c_step

        # Step on food -> automatically eat
        if self.food1_exists and self.agent_pos == self.food1_pos:
            self.food1_exists = False
            if self.safe_food == 1:
                self.energy += self.safe_gain
                self.stats["ate_safe"] = True
                if self.reward_type != 'hrrl':
                    reward += 0.3  # Small reward for eating safe food
            else:
                # Poison food -> instant death!
                self.stats["ate_poison"] = True
                self.energy = 0

        if self.food2_exists and self.agent_pos == self.food2_pos:
            self.food2_exists = False
            if self.safe_food == 2:
                self.energy += self.safe_gain
                self.stats["ate_safe"] = True
                if self.reward_type != 'hrrl':
                    reward += 0.3
            else:
                self.stats["ate_poison"] = True
                self.energy = 0

        self.energy = min(self.energy, float(self.E_init + self.safe_gain))

        # External/terminal rewards for reaching exit or dying
        if self.agent_pos == self.exit_pos:
            if self.reward_type != 'hrrl':
                reward += 1.0
            else:
                reward += 1.0 # Keep external exit reward +1 for hybrid
            terminated = True
            self.stats["reached_exit"] = True

        if self.energy <= 0:
            if self.reward_type != 'hrrl':
                reward = -1.0
            terminated = True
            self.stats["energy_depleted"] = True
            
        # HRRL internal reward calculation
        if self.reward_type == 'hrrl':
            new_drive = abs(self.setpoint - self.energy)
            r_internal = old_drive - new_drive
            reward += r_internal

        # Timeout
        truncated = (self.steps >= self.max_steps) and not terminated

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        g = self.grid_size - 1
        f1 = self.food1_pos if self.food1_exists else (-1, -1)
        f2 = self.food2_pos if self.food2_exists else (-1, -1)
        return np.array([
            self.agent_pos[0] / g, self.agent_pos[1] / g,
            f1[0] / g if self.food1_exists else -0.1,
            f1[1] / g if self.food1_exists else -0.1,
            f2[0] / g if self.food2_exists else -0.1,
            f2[1] / g if self.food2_exists else -0.1,
            self.exit_pos[0] / g, self.exit_pos[1] / g,
            self.energy / self.E_init,
            self.hint[0], self.hint[1],
        ], dtype=np.float32)

    def render(self):
        grid = [["." for _ in range(self.grid_size)]
                for _ in range(self.grid_size)]
        grid[self.exit_pos[0]][self.exit_pos[1]] = "E"
        if self.food1_exists:
            grid[self.food1_pos[0]][self.food1_pos[1]] = "%"
        if self.food2_exists:
            grid[self.food2_pos[0]][self.food2_pos[1]] = "%"
        grid[self.agent_pos[0]][self.agent_pos[1]] = "@"
        lines = [f"E={self.energy:.0f}  hint={self.hint}  step={self.steps}  safe=food{self.safe_food}"]
        for row in grid:
            lines.append("  " + " ".join(row))
        return "\n".join(lines)
