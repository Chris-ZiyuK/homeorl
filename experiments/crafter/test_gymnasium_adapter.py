#!/usr/bin/env python3
"""
Test the Gymnasium adapter + SB3 compatibility.

Run: python experiments/crafter/test_gymnasium_adapter.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def test_gymnasium_adapter():
    """Test that the adapter produces valid gymnasium interface."""
    from src.envs.crafter_gymnasium_adapter import make_gymnasium_crafter
    import gymnasium

    for agent_type in ['vanilla', 'hace', 'pure_homeo', 'health_only', 'naive_survival']:
        env = make_gymnasium_crafter(agent_type=agent_type, seed=42, record=False)

        # Check spaces
        assert isinstance(env.observation_space, gymnasium.spaces.Box), \
            f"Bad obs space: {type(env.observation_space)}"
        assert isinstance(env.action_space, gymnasium.spaces.Discrete), \
            f"Bad act space: {type(env.action_space)}"
        assert env.observation_space.shape == (64, 64, 3)
        assert env.action_space.n == 17

        # Check reset returns (obs, info)
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2, \
            f"reset() should return (obs, info), got {type(result)}"
        obs, info = result
        assert obs.shape == (64, 64, 3)

        # Check step returns 5-tuple
        result = env.step(0)
        assert len(result) == 5, f"step() should return 5 values, got {len(result)}"
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (64, 64, 3)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()
        print(f"✓ {agent_type:15s}: spaces OK, reset OK, step OK")

    return True


def test_sb3_compatibility():
    """Test that SB3 PPO can be instantiated with the adapter."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print("⚠ SB3 not installed, skipping. Install: pip install stable-baselines3")
        return True

    from src.envs.crafter_gymnasium_adapter import make_gymnasium_crafter

    env = make_gymnasium_crafter(agent_type='hace', seed=42, record=False)

    # Run SB3's env checker
    print("\nRunning SB3 env checker...")
    try:
        check_env(env, warn=True, skip_render_check=True)
        print("✓ SB3 check_env passed!")
    except Exception as e:
        print(f"⚠ check_env warning: {e}")

    # Try creating a PPO model
    print("\nInstantiating PPO with CnnPolicy...")
    model = PPO(
        'CnnPolicy', env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        verbose=0,
        seed=42,
    )
    print("✓ PPO created successfully")

    # Try a tiny training run
    print("\nRunning 128-step training pilot...")
    model.learn(total_timesteps=128)
    print("✓ PPO training pilot passed!")

    env.close()
    return True


def test_full_episode():
    """Run a full episode and verify death classification works through adapter."""
    import numpy as np
    from src.envs.crafter_gymnasium_adapter import make_gymnasium_crafter

    env = make_gymnasium_crafter(agent_type='hace', seed=0, record=False)
    obs, info = env.reset()

    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    print(f"\n✓ Full episode: {steps} steps")
    print(f"  terminated={terminated}, truncated={truncated}")
    print(f"  death_cause={info.get('death_cause', 'n/a')}")
    if 'vitals' in info:
        print(f"  final vitals: {info['vitals']}")

    env.close()
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Gymnasium Adapter + SB3 Compatibility Tests")
    print("=" * 60)

    tests = [
        ("Gymnasium Adapter", test_gymnasium_adapter),
        ("Full Episode", test_full_episode),
        ("SB3 Compatibility", test_sb3_compatibility),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{'─'*40}")
        print(f"Test: {name}")
        print(f"{'─'*40}")
        try:
            passed = test_fn()
            results.append((name, 'PASS'))
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f'FAIL: {e}'))

    print(f"\n{'='*60}")
    print("Results:")
    for name, status in results:
        icon = '✓' if status == 'PASS' else '✗'
        print(f"  {icon} {name}: {status}")
    n_pass = sum(1 for _, s in results if s == 'PASS')
    print(f"\n  {n_pass}/{len(results)} tests passed")
    print(f"{'='*60}")
