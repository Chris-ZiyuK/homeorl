"""
毒食物辨别环境 v2 (Poison Food Environment)
=============================================
核心改进：agent 必须吃食物才能活着到达出口。
- 网格 7×7，agent 左上角 (0,0)，出口右下角 (6,6)
- 最短路径需要 12 步，每步耗能 c_step=8 → 至少消耗 96 能量
- 初始能量 E₀=80 → 不吃食物一定死 (80 < 96)
- 安全食物 +50 能量，毒食物 → 直接死亡
- 因此 agent 必须吃到安全食物才可能到达出口
- QUERY 花费 c_query=5 能量换取 hint (哪个安全)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PoisonFoodEnv(gym.Env):
    """
    观测 (11 维):
      [agent_r, agent_c, food1_r, food1_c, food2_r, food2_c,
       exit_r, exit_c, energy, hint_0, hint_1]

    动作 (5 个): 上/下/左/右/QUERY
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size=7, max_steps=40,
                 E_init=80, c_step=8, c_query=5,
                 safe_gain=50):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.E_init = E_init
        self.c_step = c_step
        self.c_query = c_query
        self.safe_gain = safe_gain

        self.action_space = spaces.Discrete(5)   # 上下左右 + QUERY
        self.observation_space = spaces.Box(
            low=-0.2, high=1.2, shape=(11,), dtype=np.float32)

        self._directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = (0, 0)
        self.exit_pos = (self.grid_size - 1, self.grid_size - 1)

        # 两个食物放在从 agent 到 exit 的路径附近
        # 食物放在中间区域，agent 必经之路
        mid = self.grid_size // 2
        possible_food_cells = []
        for r in range(1, self.grid_size - 1):
            for c in range(1, self.grid_size - 1):
                possible_food_cells.append((r, c))

        chosen = self.np_random.choice(len(possible_food_cells), 2,
                                       replace=False)
        self.food1_pos = possible_food_cells[chosen[0]]
        self.food2_pos = possible_food_cells[chosen[1]]

        # 随机哪个安全
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
            # 移动
            dr, dc = self._directions[action]
            new_r = np.clip(self.agent_pos[0] + dr, 0, self.grid_size - 1)
            new_c = np.clip(self.agent_pos[1] + dc, 0, self.grid_size - 1)
            self.agent_pos = (int(new_r), int(new_c))

        # 每步消耗能量
        self.energy -= self.c_step

        # 踩到食物 → 自动吃
        if self.food1_exists and self.agent_pos == self.food1_pos:
            self.food1_exists = False
            if self.safe_food == 1:
                self.energy += self.safe_gain
                self.stats["ate_safe"] = True
                reward += 0.3  # 吃到安全食物的小奖励
            else:
                # 毒食物 → 直接死亡！
                self.stats["ate_poison"] = True
                self.energy = 0

        if self.food2_exists and self.agent_pos == self.food2_pos:
            self.food2_exists = False
            if self.safe_food == 2:
                self.energy += self.safe_gain
                self.stats["ate_safe"] = True
                reward += 0.3
            else:
                self.stats["ate_poison"] = True
                self.energy = 0

        self.energy = min(self.energy, float(self.E_init + self.safe_gain))

        # 到达出口
        if self.agent_pos == self.exit_pos:
            reward += 1.0
            terminated = True
            self.stats["reached_exit"] = True

        # 能量耗尽
        if self.energy <= 0:
            reward = -1.0
            terminated = True
            self.stats["energy_depleted"] = True

        # 超时
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
