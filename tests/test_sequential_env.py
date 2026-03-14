import unittest

from src.envs.sequential_homeostasis_env import SequentialHomeostasisEnv


class SequentialEnvTest(unittest.TestCase):
    def test_reset_has_fixed_observation_size(self):
        env = SequentialHomeostasisEnv(task_name="reach", reward_mode="homeostatic")
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (13,))

    def test_recharge_task_collects_food(self):
        env = SequentialHomeostasisEnv(task_name="recharge", reward_mode="eval")
        env.reset(seed=0)
        for action in [1, 1, 3]:
            _, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(terminated or truncated)
        self.assertTrue(env.stats["food_collected"])
        self.assertGreater(env.energy, 0)

    def test_detour_task_hazard_cost_applies(self):
        env = SequentialHomeostasisEnv(task_name="detour", reward_mode="eval")
        env.reset(seed=0)
        start_energy = env.energy
        for action in [1, 1, 3, 3]:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        self.assertGreaterEqual(env.stats["hazard_hits"], 1)
        self.assertLess(env.energy, start_energy)

    def test_hazard_reach_has_no_food_but_has_hazards(self):
        env = SequentialHomeostasisEnv(task_name="hazard_reach", reward_mode="eval")
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (13,))
        self.assertFalse(env.food_available)
        self.assertGreater(len(env.hazards), 0)

    def test_tight_detour_uses_larger_grid(self):
        env = SequentialHomeostasisEnv(task_name="tight_detour", reward_mode="eval")
        env.reset(seed=0)
        self.assertEqual(env.grid_size, 6)
        self.assertTrue(env.food_available)
        self.assertEqual(len(env.hazards), 2)


if __name__ == "__main__":
    unittest.main()
