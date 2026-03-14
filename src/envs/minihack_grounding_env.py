"""
MiniHack Grounding Environment — v5
=====================================
Drop-in replacement for MultiObjectEnv backed by MiniHack.

FIXES FROM DEBUG
----------------
- inv=55 at start: NetHack character starts with equipment filling inventory.
  Fix: track inv_glyphs HASH change, not count. Any change = new item picked up.
- Autopickup not working for food: NetHack's default pickup_types excludes '%'.
  Fix: pass options=("pickup_types:%",) to force food autopickup.
- Agent reaches apple tile but nothing happens: confirmed by step 3 agent=(9,38)
  same as apple=(9,38). Fix: both above together solve this.

HOW TO USE
----------
    from minihack_grounding_env import MiniHackGroundingEnv as MultiObjectEnv
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import minihack  # noqa
    from minihack import LevelGenerator
    from nle import nethack
except ImportError:
    raise ImportError("Run: pip install minihack")

_MOVE_ACTIONS = (
    nethack.CompassDirection.N,
    nethack.CompassDirection.S,
    nethack.CompassDirection.W,
    nethack.CompassDirection.E,
)

_CHAR_COMESTIBLE = ord('%')
_CHAR_STAIR      = ord('>')
_BL_X = 0
_BL_Y = 1

# Force NetHack to autopickup food ('%') only
_MH_OPTIONS = ("pickup_types:%",)


def _build_des_file(room_size: int = 9) -> str:
    lvl = LevelGenerator(w=room_size, h=room_size, lit=True)
    lvl.set_start_rect((0, 0), (2, 2))
    lvl.add_object("apple", "%")
    lvl.add_object("apple", "%")
    lvl.add_object("apple", "%")
    lvl.add_goal_pos((room_size - 1, room_size - 1))
    return lvl.get_des()


def _inv_hash(inv_glyphs: np.ndarray) -> int:
    """Hash of inv_glyphs — changes whenever inventory contents change."""
    return hash(inv_glyphs.tobytes())


class MiniHackGroundingEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size: int = 9,
        max_steps: int = 100,
        E_init: float = 100.0,
        c_step: float = 2.0,
        c_query: float = 8.0,
        harmful_cost: float = 40.0,
        beneficial_gain: float = 30.0,
        obs_mode: str = "energy",
        reward_type: str = "terminal",
    ):
        super().__init__()

        self.grid_size      = grid_size
        self.max_steps      = max_steps
        self.E_init         = float(E_init)
        self.c_step         = float(c_step)
        self.c_query        = float(c_query)
        self.obs_mode       = obs_mode
        self.reward_type    = reward_type
        self.setpoint       = float(E_init)

        self._effects = {
            0: -float(harmful_cost),
            1: +float(beneficial_gain),
            2:  0.0,
        }

        self._mh = gym.make(
            "MiniHack-Skill-Custom-v0",
            des_file=_build_des_file(grid_size),
            observation_keys=("chars", "blstats", "inv_glyphs"),
            actions=_MOVE_ACTIONS,
            max_episode_steps=max_steps,
            autopickup=True,
            options=_MH_OPTIONS,   # force food pickup
            pet=False,
            spawn_monsters=False,
        )

        self.action_space = spaces.Discrete(5 if obs_mode == "full" else 4)

        self.obs_dim = {"terminal": 13, "energy": 14, "full": 17}[obs_mode]

        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(self.obs_dim,), dtype=np.float32
        )

        self._agent_abs  = (0, 0)
        self._obj        = {}
        self._exit_abs   = (0, 0)
        self._inv_hash   = 0        # track hash, not count
        self._n_apples_eaten = 0    # how many apples consumed this episode
        self.energy      = self.E_init
        self.hint        = np.zeros(3, dtype=np.float32)
        self.has_queried = False
        self.steps       = 0
        self.stats       = {}

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mh_obs, _ = self._mh.reset()

        self.energy          = self.E_init
        self.hint            = np.zeros(3, dtype=np.float32)
        self.has_queried     = False
        self.steps           = 0
        self._n_apples_eaten = 0
        self.stats = {
            "ate_harmful": 0, "ate_beneficial": 0, "ate_neutral": 0,
            "reached_exit": False, "energy_depleted": False,
            "queries": 0,
            "approach_decisions": {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        }

        bl = mh_obs["blstats"]
        self._agent_abs = (int(bl[_BL_Y]), int(bl[_BL_X]))
        self._inv_hash  = _inv_hash(mh_obs["inv_glyphs"])

        # Register objects by spawn order
        self._obj = {}
        for idx, pos in enumerate(
                self._scan_char(mh_obs["chars"], _CHAR_COMESTIBLE)[:3]):
            self._obj[idx] = {"abs": pos, "alive": True}
        for tid in range(3):
            if tid not in self._obj:
                self._obj[tid] = {"abs": (-1, -1), "alive": False}

        stair = self._first_char(mh_obs["chars"], _CHAR_STAIR)
        self._exit_abs = stair if stair else (-1, -1)

        return self._build_obs(), {}

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        self.steps += 1
        reward     = 0.0
        terminated = False
        old_drive  = abs(self.setpoint - self.energy)
        mh_term    = False

        if action == 4 and self.obs_mode == "full":
            self.energy -= self.c_query
            self.stats["queries"] += 1
            if not self.has_queried:
                self.has_queried = True
                self.hint = np.array([-1.0, 1.0, 0.0], dtype=np.float32)
        else:
            mh_obs, _, mh_term, _, _ = self._mh.step(action)
            self.energy -= self.c_step
            self._process_obs(mh_obs)

        if self.stats["reached_exit"] or mh_term:
            self.stats["reached_exit"] = True
            reward     = 1.0 if self.reward_type != "hrrl" else reward + 1.0
            terminated = True

        if self.energy <= 0 and not terminated:
            self.stats["energy_depleted"] = True
            reward     = -1.0 if self.reward_type != "hrrl" else reward
            terminated = True

        if self.reward_type == "hrrl":
            new_drive  = abs(self.setpoint - self.energy)
            reward    += old_drive - new_drive

        truncated = (self.steps >= self.max_steps) and not terminated
        return self._build_obs(), reward, terminated, truncated, {}

    # ── internal ──────────────────────────────────────────────────────────────

    def _process_obs(self, mh_obs):
        bl = mh_obs["blstats"]
        self._agent_abs = (int(bl[_BL_Y]), int(bl[_BL_X]))

        # Detect which apples disappeared from the chars map this step.
        # Objects don't move in NetHack — only disappear when picked up.
        # This is more reliable than inv_hash because inv can change for
        # other reasons (NetHack auto-eats food when hungry, etc.)
        living_on_map = set(self._scan_char(mh_obs["chars"], _CHAR_COMESTIBLE))
        for tid, obj in self._obj.items():
            if not obj["alive"]:
                continue
            if tuple(obj["abs"]) not in living_on_map:
                # Apple gone from map → consumed
                obj["alive"] = False
                self._n_apples_eaten += 1
                self.energy = float(np.clip(
                    self.energy + self._effects[tid], -999, self.E_init + 50
                ))
                if tid == 0:
                    self.stats["ate_harmful"]    += 1
                elif tid == 1:
                    self.stats["ate_beneficial"] += 1
                else:
                    self.stats["ate_neutral"]    += 1

        # Keep inv_hash in sync (don't use it for detection, just housekeeping)
        self._inv_hash = _inv_hash(mh_obs["inv_glyphs"])

        # NOTE: do NOT update object positions — they are fixed at spawn.
        # The position-refresh logic caused drift when chars map showed
        # nearby '%' that got greedily matched to wrong type_ids.

        stair = self._first_char(mh_obs["chars"], _CHAR_STAIR)
        if stair:
            self._exit_abs = stair
        if self._agent_abs == self._exit_abs:
            self.stats["reached_exit"] = True

        # Approach decisions for FGS metric
        ar, ac = self._agent_abs
        for tid, obj in self._obj.items():
            if not obj["alive"]:
                continue
            if abs(ar - obj["abs"][0]) + abs(ac - obj["abs"][1]) <= 1:
                self.stats["approach_decisions"][tid][0] += 1

    @staticmethod
    def _scan_char(full_chars, char_code):
        return [(r, c)
                for r in range(full_chars.shape[0])
                for c in range(full_chars.shape[1])
                if int(full_chars[r, c]) == char_code]

    @staticmethod
    def _first_char(full_chars, char_code):
        for r in range(full_chars.shape[0]):
            for c in range(full_chars.shape[1]):
                if int(full_chars[r, c]) == char_code:
                    return (r, c)
        return None

    def _build_obs(self):
        ar, ac = self._agent_abs
        half   = 4.0
        parts  = [0.0, 0.0]

        for tid in range(3):
            obj = self._obj[tid]
            if obj["alive"]:
                parts.extend([
                    float(np.clip((obj["abs"][0] - ar) / half, -1.5, 1.5)),
                    float(np.clip((obj["abs"][1] - ac) / half, -1.5, 1.5)),
                ])
            else:
                parts.extend([-1.5, -1.5])

        er, ec = self._exit_abs
        parts.extend([
            float(np.clip((er - ar) / half, -1.5, 1.5)),
            float(np.clip((ec - ac) / half, -1.5, 1.5)),
        ])
        for tid in range(3):
            parts.append((tid + 1) / 3.0)

        if self.obs_mode in ("energy", "full"):
            parts.append(self.energy / self.E_init)
        if self.obs_mode == "full":
            parts.extend(self.hint.tolist())

        return np.array(parts, dtype=np.float32)

    def render(self):
        ar, ac = self._agent_abs
        lines = [f"E={self.energy:.0f}/{self.E_init:.0f}  step={self.steps}"]
        for tid, obj in self._obj.items():
            lines.append(f"  type{tid}@{obj['abs']} [{'alive' if obj['alive'] else 'eaten'}]")
        lines.append(f"  agent@({ar},{ac})  exit@{self._exit_abs}")
        return "\n".join(lines)

    def close(self):
        self._mh.close()
        super().close()


# ── Debug: greedy walk toward apple ──────────────────────────────────────────

def _debug_pickup():
    print("\n" + "=" * 55)
    print("DEBUG: Greedy walk toward all apples — testing pickup")
    print("=" * 55)

    env = MiniHackGroundingEnv(obs_mode="energy", reward_type="terminal")
    obs, _ = env.reset()

    print(f"Reset: agent={env._agent_abs}")
    for tid, obj in env._obj.items():
        print(f"  type{tid} @ {obj['abs']}")
    print(f"  exit @ {env._exit_abs}")

    def _next_alive_target():
        """Return live (row,col) of the closest alive object."""
        ar, ac = env._agent_abs
        best = None
        best_d = float("inf")
        for obj in env._obj.values():
            if not obj["alive"]:
                continue
            d = abs(ar - obj["abs"][0]) + abs(ac - obj["abs"][1])
            if d < best_d:
                best_d = d
                best = obj["abs"]
        return best

    def _greedy_action(tr, tc):
        ar, ac = env._agent_abs
        # Move along the axis with larger gap first, resolves oscillation
        dr, dc = tr - ar, tc - ac
        if abs(dr) >= abs(dc):
            return 1 if dr > 0 else 0   # S or N
        else:
            return 3 if dc > 0 else 2   # E or W

    for step in range(60):
        target = _next_alive_target()
        if target is None:
            # All eaten — walk to exit
            target = env._exit_abs

        action = _greedy_action(*target)
        obs, r, term, trunc, _ = env.step(action)
        ar2, ac2 = env._agent_abs
        print(f"  step {step+1:2d}: agent=({ar2},{ac2})  target={target}  "
              f"ate={env.stats['ate_harmful']}/{env.stats['ate_beneficial']}/{env.stats['ate_neutral']}  "
              f"E={env.energy:.0f}  r={r:.2f}")
        if term or trunc:
            print(f"  Done (term={term} trunc={trunc})")
            break

    print(f"\nFinal stats: {env.stats}")
    env.close()


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--debug" in sys.argv:
        _debug_pickup()
        sys.exit(0)

    print("=" * 55)
    print("MiniHackGroundingEnv — smoke test")
    print("=" * 55)

    for mode in ("terminal", "energy", "full"):
        env = MiniHackGroundingEnv(obs_mode=mode, reward_type="terminal")
        obs, _ = env.reset(seed=42)
        print(f"\nobs_mode='{mode}'  obs_dim={env.obs_dim}  obs.shape={obs.shape}")
        assert obs.shape == (env.obs_dim,), "Shape mismatch!"

        total_r = 0.0
        for _ in range(100):
            a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            total_r += r
            if term or trunc:
                break

        print(f"  steps={env.steps}  energy={env.energy:.1f}  reward={total_r:.2f}")
        print(f"  stats={env.stats}")
        env.close()

    print("\n✅ Smoke test done.")
    print("   Run with --debug to walk agent toward apple and verify pickup.")