#!/usr/bin/env python3
"""
Benchmark: single env vs SubprocVecEnv for Crafter PPO training.
Tests 5000 steps with 1, 2, 4, 8 parallel envs.

Usage: python experiments/crafter/benchmark_vecenv.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def benchmark(n_envs: int, total_steps: int = 5000):
    """Benchmark PPO training with n parallel environments."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from src.envs.crafter_gymnasium_adapter import make_gymnasium_crafter

    def make_env(rank):
        def _init():
            return make_gymnasium_crafter(
                agent_type='vanilla', seed=rank, record=False
            )
        return _init

    if n_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    model = PPO(
        'CnnPolicy', env,
        n_steps=max(256 // n_envs, 32),  # Adjust per-env steps
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        verbose=0,
        seed=42,
    )

    start = time.time()
    model.learn(total_timesteps=total_steps)
    elapsed = time.time() - start

    fps = total_steps / elapsed
    env.close()

    return elapsed, fps


if __name__ == '__main__':
    total_steps = 5000
    print(f"Benchmarking PPO on Crafter ({total_steps} steps)")
    print(f"{'n_envs':>8} {'Time (s)':>10} {'FPS':>8} {'Speedup':>8}")
    print("-" * 40)

    baseline_fps = None
    for n_envs in [1, 2, 4]:
        try:
            elapsed, fps = benchmark(n_envs, total_steps)
            if baseline_fps is None:
                baseline_fps = fps
            speedup = fps / baseline_fps
            print(f"{n_envs:>8} {elapsed:>10.1f} {fps:>8.1f} {speedup:>7.2f}x")
        except Exception as e:
            print(f"{n_envs:>8} FAILED: {e}")

    print("\nDone!")
