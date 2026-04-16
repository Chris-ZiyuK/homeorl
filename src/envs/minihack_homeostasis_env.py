"""
MiniHack Sequential Homeostasis Environment
=============================================
A MiniHack-backed environment that satisfies the exact same interface as
SequentialHomeostasisEnv so it can be dropped into the existing runner,
agent suite, and analysis pipeline without any changes.

TASK SEQUENCE (structural analogue of the gridworld α sequence)
---------------------------------------------------------------
  mh_reach         : empty lit room, navigate to stair down
  mh_recharge      : apple on floor, must eat to survive the longer path
  mh_hazard_reach  : lava tiles block direct path, must navigate around
  mh_detour        : apple + lava, must eat AND avoid hazards
  mh_tight_detour  : larger room, apple further away, tighter energy margin

INTERFACE CONTRACT (matches SequentialHomeostasisEnv exactly)
-------------------------------------------------------------
  Constructor : MiniHackHomeostasisEnv(task_name, reward_mode,
                                        observation_mode, ...)
  Gymnasium   : reset(seed) → obs, {}
                step(action) → obs, reward, terminated, truncated, {}
  Attributes  : energy, energy_cap, agent_pos, food_available (list),
                _hazard_set (set of (r,c)), stats dict, obs_dim
  reward_mode : "task" | "homeostatic" | "mixed" | "eval"
  observation_mode: "full" | "masked"

ENERGY MODEL
------------
Energy is tracked by this wrapper, not NetHack's native hunger system,
so the dynamics are identical to the gridworld: -step_cost per step,
+food_gain on food pickup, -hazard_cost on hazard tile.
This ensures the carryover experiment is a fair comparison.

OBSERVATION VECTOR (obs_dim = 19, same as gridworld)
-----------------------------------------------------
  [agent_r, agent_c,          (2)  — normalised by grid size
   exit_r,  exit_c,           (2)
   food_r,  food_c,           (2)  — (-0.25, -0.25) if eaten/absent
   food_available_flag,       (1)
   energy_obs,                (1)  — energy/cap, or -1.0 if masked
   task_id,                   (1)  — task index / (n_tasks-1)
   hazard_r1, hazard_c1,      (2)  — first hazard, (-0.25,-0.25) if none
   hazard_r2, hazard_c2,      (2)  — second hazard
   hazard_r3, hazard_c3,      (2)  — third hazard (lava cells)
   chars_crop_food_signal,    (1)  — 1.0 if % visible in crop, else 0.0
   chars_crop_lava_signal,    (1)  — 1.0 if L visible in crop, else 0.0
  ]  → total 19 dims

INSTALLATION
------------
    pip install minihack
    python -c "import minihack; print('OK')"
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import minihack  # noqa — registers envs
    from minihack import LevelGenerator
    from nle import nethack
    _MINIHACK_AVAILABLE = True
except ImportError:
    _MINIHACK_AVAILABLE = False
    raise ImportError(
        "MiniHack not installed. Run: pip install minihack"
    )

# ── Actions ───────────────────────────────────────────────────────────────────

_MOVE_ACTIONS = (
    nethack.CompassDirection.N,
    nethack.CompassDirection.S,
    nethack.CompassDirection.W,
    nethack.CompassDirection.E,
)

# ── ASCII codes ───────────────────────────────────────────────────────────────

_CHAR_FOOD   = ord('%')   # comestible (apple)
_CHAR_STAIR  = ord('>')   # stair down = exit
_CHAR_LAVA   = ord('L')   # lava tile  = hazard

# ── Task specifications ───────────────────────────────────────────────────────
# Designed as structural analogues of the gridworld task sequence.
# Energy numbers are tuned so:
#   mh_reach     : agent has enough energy to reach exit with ~30% margin
#   mh_recharge  : agent MUST eat food or dies before reaching exit
#   mh_hazard_*  : lava forces a detour that costs extra energy

MH_TASK_SPECS = {
    "mh_reach": {
        "room_size":    9,
        "max_steps":    60,
        "initial_energy": 25.0,
        "energy_cap":   28.0,
        "step_cost":    1.0,
        "food_gain":    0.0,    # no food — not needed
        "hazard_cost":  0.0,
        "n_apples":     0,
        "n_lava":       0,
        "description":  "Empty room. Navigate to stair.",
    },
    "mh_recharge": {
        "room_size":    9,
        "max_steps":    70,
        "initial_energy": 12.0,
        "energy_cap":   20.0,
        "step_cost":    1.0,
        "food_gain":    12.0,
        "hazard_cost":  0.0,
        "n_apples":     1,
        "n_lava":       0,
        "description":  "Must find and eat apple to survive.",
    },
    "mh_hazard_reach": {
        "room_size":    9,
        "max_steps":    70,
        "initial_energy": 13.0,
        "energy_cap":   16.0,
        "step_cost":    1.0,
        "food_gain":    5.0,
        "hazard_cost":  4.0,
        "n_apples":     1,
        "n_lava":       3,
        "description":  "Lava blocks direct path. Optional food.",
    },
    "mh_detour": {
        "room_size":    9,
        "max_steps":    75,
        "initial_energy": 9.0,
        "energy_cap":   16.0,
        "step_cost":    1.0,
        "food_gain":    8.0,
        "hazard_cost":  4.0,
        "n_apples":     1,
        "n_lava":       3,
        "description":  "Must eat apple AND avoid lava.",
    },
    "mh_tight_detour": {
        "room_size":    11,
        "max_steps":    90,
        "initial_energy": 9.0,
        "energy_cap":   17.0,
        "step_cost":    1.0,
        "food_gain":    9.0,
        "hazard_cost":  5.0,
        "n_apples":     1,
        "n_lava":       4,
        "description":  "Larger room, tighter margins.",
    },
}

# Task index for observation encoding
MH_TASK_INDEX = {name: i for i, name in enumerate(MH_TASK_SPECS)}

# Sequence matching the gridworld α sequence structure
MH_TASK_SEQUENCE = list(MH_TASK_SPECS.keys())


# ── Level builder ─────────────────────────────────────────────────────────────

def _build_level(spec: dict) -> str:
    """Build a MiniHack des-file from a task spec.

    Lava tiles are placed at fixed positions that force a detour without
    making the task unsolvable. Positions block the direct path from
    top-left to bottom-right.
    """
    sz = spec["room_size"]
    lvl = LevelGenerator(w=sz, h=sz, lit=True)
    lvl.set_start_rect((0, 0), (2, 2))

    for _ in range(spec["n_apples"]):
        lvl.add_object("apple", "%")

    # Fixed lava positions scaled to room size — block midpoint
    mid = sz // 2
    lava_positions = [
        (mid,     mid),
        (mid,     mid + 1),
        (mid - 1, mid),
        (mid + 1, mid + 1),
    ]
    for i in range(min(spec["n_lava"], len(lava_positions))):
        x, y = lava_positions[i]
        lvl.add_terrain(coord=(x, y), flag="L")

    lvl.add_goal_pos((sz - 1, sz - 1))
    return lvl.get_des()


# ── Main environment ──────────────────────────────────────────────────────────

class MiniHackHomeostasisEnv(gym.Env):
    """
    MiniHack-backed sequential homeostasis environment.

    Drop-in replacement for SequentialHomeostasisEnv in the runner,
    agent suite, and analysis pipeline.
    """

    metadata = {"render_modes": ["ansi"]}

    # Fixed obs_dim matching the gridworld (19 dims)
    obs_dim = 19

    def __init__(
        self,
        task_name: str = "mh_reach",
        reward_mode: str = "homeostatic",
        observation_mode: str = "full",
        # Reward shaping coefficients (same defaults as gridworld)
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

        if task_name not in MH_TASK_SPECS:
            raise ValueError(
                f"Unknown task_name '{task_name}'. "
                f"Valid: {list(MH_TASK_SPECS.keys())}"
            )
        if reward_mode not in {"task", "homeostatic", "mixed", "eval"}:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")
        if observation_mode not in {"full", "masked"}:
            raise ValueError(f"Unknown observation_mode: {observation_mode}")

        self.task_name           = task_name
        self.reward_mode         = reward_mode
        self.observation_mode    = observation_mode
        self.exit_bonus          = exit_bonus
        self.death_penalty       = death_penalty
        self.progress_coef       = progress_coef
        self.food_bonus          = food_bonus
        self.hazard_penalty      = hazard_penalty
        self.internal_coef       = internal_coef
        self.task_reward_coef    = task_reward_coef
        self.initial_energy_override = initial_energy_override

        self._spec = MH_TASK_SPECS[task_name]
        self._load_spec()

        # Build underlying MiniHack env
        self._mh = gym.make(
            "MiniHack-Skill-Custom-v0",
            des_file=_build_level(self._spec),
            observation_keys=("chars_crop", "chars", "blstats"),
            actions=_MOVE_ACTIONS,
            max_episode_steps=self._spec["max_steps"],
            autopickup=True,
            options=("pickup_types:%",),
            pet=False,
            spawn_monsters=False,
        )

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.5,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Episode state — populated in reset()
        self.energy        = self.initial_energy
        self.energy_cap    = self._spec["energy_cap"]
        self.agent_pos     = (0, 0)          # (row, col) absolute
        self.food_available: list[tuple]= []  # list of (row,col) for any(food_available)
        self._hazard_set: set[tuple]    = set()  # set of (row,col) lava positions
        self._exit_pos     = (0, 0)
        self._inv_hash     = 0
        self._n_apples_eaten = 0
        self.stats         = {}

    # ── Spec loading ──────────────────────────────────────────────────────────

    def _load_spec(self):
        s = self._spec
        if self.initial_energy_override is not None:
            self.initial_energy = float(
                np.clip(self.initial_energy_override, 0.0, s["energy_cap"])
            )
        else:
            self.initial_energy = float(s["initial_energy"])
        self.energy_cap  = float(s["energy_cap"])
        self.setpoint    = self.energy_cap   # HRRL setpoint = cap
        self.max_steps   = s["max_steps"]

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_spec()

        mh_obs, _ = self._mh.reset()

        self.energy          = self.initial_energy
        self._inv_hash       = _inv_hash(mh_obs["inv_glyphs"]) \
            if "inv_glyphs" in mh_obs else 0
        self._n_apples_eaten = 0

        self.stats = {
            "success":         False,
            "energy_depleted": False,
            "food_collected":  False,
            "hazard_hits":     0,
            "energy_left":     self.energy,
            "start_energy":    self.energy,
        }

        bl = mh_obs["blstats"]
        self.agent_pos = (int(bl[1]), int(bl[0]))

        # Scan full map for objects
        self.food_available = self._scan(mh_obs["chars"], _CHAR_FOOD)
        self._hazard_set    = set(self._scan(mh_obs["chars"], _CHAR_LAVA))
        exit_pos            = self._first(mh_obs["chars"], _CHAR_STAIR)
        self._exit_pos      = exit_pos if exit_pos else (
            self._spec["room_size"] - 1,
            self._spec["room_size"] - 1,
        )

        return self._obs(mh_obs), {}

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: int):
        old_energy = self.energy
        old_drive  = abs(self.setpoint - self.energy)

        mh_obs, _, mh_term, mh_trunc, _ = self._mh.step(action)

        bl = mh_obs["blstats"]
        self.agent_pos = (int(bl[1]), int(bl[0]))

        # Per-step energy cost
        self.energy -= self._spec["step_cost"]

        # Pickup detection via inv_glyphs hash
        new_inv_hash = _inv_hash(mh_obs.get("inv_glyphs",
                                             np.zeros(55, dtype=np.int16)))
        if new_inv_hash != self._inv_hash:
            living = set(self._scan(mh_obs["chars"], _CHAR_FOOD))
            consumed = [p for p in self.food_available if p not in living]
            for _ in consumed:
                self.energy = float(np.clip(
                    self.energy + self._spec["food_gain"],
                    -999, self.energy_cap
                ))
                self.stats["food_collected"] = True
                self._n_apples_eaten += 1
            self.food_available = list(living)
            self._inv_hash = new_inv_hash

        # Hazard detection — agent stepped on lava
        if self.agent_pos in self._hazard_set:
            self.energy -= self._spec["hazard_cost"]
            self.stats["hazard_hits"] += 1

        self.energy = float(np.clip(self.energy, -999.0, self.energy_cap))

        # Refresh hazard set (lava doesn't move but good practice)
        self._hazard_set = set(self._scan(mh_obs["chars"], _CHAR_LAVA))

        # Exit detection
        exit_pos = self._first(mh_obs["chars"], _CHAR_STAIR)
        if exit_pos:
            self._exit_pos = exit_pos
        if self.agent_pos == self._exit_pos or mh_term:
            self.stats["success"] = True

        if self.energy <= 0:
            self.stats["energy_depleted"] = True

        terminated = self.stats["success"] or self.stats["energy_depleted"]
        truncated  = mh_trunc and not terminated
        self.stats["energy_left"] = max(self.energy, 0.0)

        # ── Reward ────────────────────────────────────────────────────────────
        reward = 0.0
        new_drive = abs(self.setpoint - self.energy)

        if self.reward_mode in {"task", "mixed"}:
            # Progress toward current target
            target = self.food_available[0] if self.food_available else self._exit_pos
            old_dist = _manhattan(self.agent_pos, target)
            reward  += self.task_reward_coef * self.progress_coef * (
                old_dist - _manhattan(self.agent_pos, target)
            )
            if self.stats["food_collected"] and self._n_apples_eaten == 1:
                reward += self.task_reward_coef * self.food_bonus
            if self.agent_pos in self._hazard_set:
                reward -= self.task_reward_coef * self.hazard_penalty

        if self.reward_mode in {"homeostatic", "mixed"}:
            reward += self.internal_coef * (old_drive - new_drive)

        if self.stats["success"]:
            reward += self.exit_bonus
        if self.stats["energy_depleted"]:
            reward -= self.death_penalty

        return self._obs(mh_obs), reward, terminated, truncated, {}

    # ── Observation builder ───────────────────────────────────────────────────

    def _obs(self, mh_obs) -> np.ndarray:
        sz   = float(self._spec["room_size"] - 1)
        ar, ac = self.agent_pos
        er, ec = self._exit_pos

        # Food position (first alive food, or sentinel)
        if self.food_available:
            fr, fc = self.food_available[0]
            food_flag = 1.0
        else:
            fr, fc = -1, -1
            food_flag = 0.0

        # Up to 3 hazard positions
        hazards = list(self._hazard_set)[:3]
        while len(hazards) < 3:
            hazards.append((-1, -1))

        # Crop signals — fast boolean from 9×9 chars_crop
        crop = mh_obs["chars_crop"]
        has_food_in_crop = float(np.any(crop == _CHAR_FOOD))
        has_lava_in_crop = float(np.any(crop == _CHAR_LAVA))

        energy_obs = (self.energy / self.energy_cap) \
            if self.observation_mode == "full" else -1.0

        obs = np.array([
            ar / sz, ac / sz,
            er / sz, ec / sz,
            fr / sz if fr >= 0 else -0.25,
            fc / sz if fc >= 0 else -0.25,
            food_flag,
            float(np.clip(energy_obs, -1.0, 1.5)),
            MH_TASK_INDEX[self.task_name] / max(len(MH_TASK_INDEX) - 1, 1),
            hazards[0][0] / sz if hazards[0][0] >= 0 else -0.25,
            hazards[0][1] / sz if hazards[0][1] >= 0 else -0.25,
            hazards[1][0] / sz if hazards[1][0] >= 0 else -0.25,
            hazards[1][1] / sz if hazards[1][1] >= 0 else -0.25,
            hazards[2][0] / sz if hazards[2][0] >= 0 else -0.25,
            hazards[2][1] / sz if hazards[2][1] >= 0 else -0.25,
            has_food_in_crop,
            has_lava_in_crop,
            float(self.stats["food_collected"]),
            float(len(self.food_available)),   # 0/1 apples remaining
        ], dtype=np.float32)

        assert len(obs) == self.obs_dim, \
            f"obs length {len(obs)} != obs_dim {self.obs_dim}"
        return obs

    # ── Map scanning utilities ────────────────────────────────────────────────

    @staticmethod
    def _scan(full_chars: np.ndarray, char_code: int) -> list[tuple[int, int]]:
        """Return all (row, col) positions of char_code in full map."""
        rows, cols = np.where(full_chars == char_code)
        return list(zip(rows.tolist(), cols.tolist()))

    @staticmethod
    def _first(full_chars: np.ndarray,
               char_code: int) -> tuple[int, int] | None:
        """Return first (row, col) occurrence of char_code, or None."""
        rows, cols = np.where(full_chars == char_code)
        if len(rows) == 0:
            return None
        return (int(rows[0]), int(cols[0]))

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        ar, ac = self.agent_pos
        lines = [
            f"task={self.task_name}  reward={self.reward_mode}  "
            f"E={self.energy:.1f}/{self.energy_cap:.1f}  "
            f"step={self.stats.get('hazard_hits', 0)}"
        ]
        lines.append(f"  agent@({ar},{ac})  exit@{self._exit_pos}")
        lines.append(f"  food={self.food_available}  lava={self._hazard_set}")
        return "\n".join(lines)

    def close(self):
        self._mh.close()
        super().close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inv_hash(inv_glyphs: np.ndarray) -> int:
    return hash(np.asarray(inv_glyphs).tobytes())


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("MiniHackHomeostasisEnv — smoke test")
    print("=" * 55)

    for task_name in MH_TASK_SPECS:
        for reward_mode in ("task", "homeostatic", "eval"):
            for obs_mode in ("full", "masked"):
                env = MiniHackHomeostasisEnv(
                    task_name=task_name,
                    reward_mode=reward_mode,
                    observation_mode=obs_mode,
                )
                obs, _ = env.reset(seed=42)
                assert obs.shape == (env.obs_dim,), \
                    f"Shape mismatch: {obs.shape}"
                assert hasattr(env, "energy")
                assert hasattr(env, "energy_cap")
                assert hasattr(env, "agent_pos")
                assert hasattr(env, "food_available") and \
                    isinstance(env.food_available, list)
                assert hasattr(env, "_hazard_set") and \
                    isinstance(env._hazard_set, set)

                total_r = 0.0
                for _ in range(30):
                    a = env.action_space.sample()
                    obs, r, term, trunc, _ = env.step(a)
                    total_r += r
                    if term or trunc:
                        break

                print(f"  {task_name:<18} {reward_mode:<12} {obs_mode:<6} "
                      f"obs={obs.shape}  E={env.energy:.1f}  r={total_r:+.2f}  "
                      f"stats={env.stats}")
                env.close()

    print("\n✅ All checks passed.")
    print("   Plug in with: from src.envs.minihack_homeostasis_env import "
          "MiniHackHomeostasisEnv")