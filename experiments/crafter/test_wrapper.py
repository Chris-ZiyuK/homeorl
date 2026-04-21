#!/usr/bin/env python3
"""
Quick sanity test for the Crafter HACE wrappers.

Run: python experiments/crafter/test_wrapper.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def test_crafter_install():
    """Verify Crafter is installed and runs."""
    import crafter
    env = crafter.Env(seed=42)
    obs = env.reset()
    print(f"✓ Crafter installed. Obs shape: {obs.shape}")

    # Run a few random steps (use np.random instead of action_space.sample)
    import numpy as np
    n_actions = env.action_space.n
    for i in range(50):
        action = np.random.randint(n_actions)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    print(f"  Inventory keys: {sorted(info.get('inventory', {}).keys())}")
    inv = info['inventory']
    print(f"  health={inv['health']}, food={inv['food']}, "
          f"drink={inv['drink']}, energy={inv['energy']}")
    return True


def test_hace_wrapper():
    """Test CrafterHACEWrapper produces correct reward signals."""
    import crafter
    import numpy as np
    from src.envs.crafter_hace_wrapper import CrafterHACEWrapper

    env = crafter.Env(seed=42, reward=True)
    env = CrafterHACEWrapper(env, alpha=1.0, beta=1.0)
    obs = env.reset()
    n_actions = env.action_space.n
    print(f"\n✓ HACE wrapper created. Obs shape: {obs.shape}")

    total_homeo = 0
    total_orig = 0
    steps = 0

    for episode in range(3):
        obs = env.reset()
        done = False
        ep_len = 0
        while not done:
            action = np.random.randint(n_actions)
            obs, reward, done, info = env.step(action)
            ep_len += 1
            total_homeo += info.get('homeo_reward', 0)
            total_orig += info.get('original_reward', 0)
            steps += 1

        death_cause = info.get('death_cause', '?')
        vitals = info.get('vitals', {})
        print(f"  Ep {episode}: len={ep_len}, death={death_cause}, "
              f"vitals={{food:{vitals.get('food',0):.0f}, "
              f"drink:{vitals.get('drink',0):.0f}, "
              f"energy:{vitals.get('energy',0):.0f}}}")

    print(f"\n  Total steps: {steps}")
    print(f"  Avg homeo reward/step: {total_homeo/steps:.4f}")
    print(f"  Avg orig reward/step:  {total_orig/steps:.4f}")
    return True


def test_health_only_wrapper():
    """Test CrafterHealthOnlyWrapper."""
    import crafter
    import numpy as np
    from src.envs.crafter_hace_wrapper import CrafterHealthOnlyWrapper

    env = crafter.Env(seed=42, reward=True)
    env = CrafterHealthOnlyWrapper(env, alpha=1.0, beta=1.0)
    obs = env.reset()
    n_actions = env.action_space.n
    print(f"\n✓ Health-Only wrapper created.")

    done = False
    steps = 0
    while not done and steps < 500:
        action = np.random.randint(n_actions)
        obs, reward, done, info = env.step(action)
        steps += 1

    print(f"  Ran {steps} steps. Done={done}")
    print(f"  Death cause: {info.get('death_cause', 'n/a')}")
    print(f"  Homeo reward last step: {info.get('homeo_reward', 'n/a'):.4f}")
    return True


def test_naive_survival_wrapper():
    """Test CrafterNaiveSurvivalWrapper."""
    import crafter
    import numpy as np
    from src.envs.crafter_hace_wrapper import CrafterNaiveSurvivalWrapper

    env = crafter.Env(seed=42, reward=True)
    env = CrafterNaiveSurvivalWrapper(env, alpha=1.0, bonus_coef=0.1)
    obs = env.reset()
    n_actions = env.action_space.n
    print(f"\n✓ Naive Survival wrapper created.")

    done = False
    steps = 0
    while not done and steps < 500:
        action = np.random.randint(n_actions)
        obs, reward, done, info = env.step(action)
        steps += 1

    print(f"  Ran {steps} steps. Done={done}")
    print(f"  Naive reward last step: {info.get('naive_reward', 'n/a'):.4f}")
    print(f"  Death cause: {info.get('death_cause', 'n/a')}")
    return True


def test_factory():
    """Test the make_crafter_env factory function."""
    import numpy as np
    from src.envs.crafter_hace_wrapper import make_crafter_env

    for agent_type in ['vanilla', 'hace', 'pure_homeo', 'health_only', 'naive_survival']:
        env = make_crafter_env(agent_type=agent_type, seed=42, record=False)
        obs = env.reset()
        obs, r, done, info = env.step(0)
        print(f"✓ Factory: {agent_type:15s} → reward={r:.4f}")

    return True


def test_death_classification():
    """Run enough episodes to see different death causes."""
    import crafter
    import numpy as np
    from src.envs.crafter_hace_wrapper import CrafterHACEWrapper
    from collections import Counter

    env = crafter.Env(seed=0, reward=True)
    env = CrafterHACEWrapper(env, alpha=1.0, beta=1.0)
    n_actions = env.action_space.n

    death_causes = []
    for ep in range(20):
        obs = env.reset()
        done = False
        while not done:
            action = np.random.randint(n_actions)
            obs, reward, done, info = env.step(action)
        death_causes.append(info.get('death_cause', 'unknown'))

    counts = Counter(death_causes)
    print(f"\n✓ Death cause distribution (20 random episodes):")
    for cause, count in counts.most_common():
        print(f"  {cause:20s}: {count:3d} ({count/20*100:.0f}%)")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Crafter HACE Wrapper — Sanity Tests")
    print("=" * 60)

    tests = [
        ("Crafter Install", test_crafter_install),
        ("HACE Wrapper", test_hace_wrapper),
        ("Health-Only Wrapper", test_health_only_wrapper),
        ("Naive Survival Wrapper", test_naive_survival_wrapper),
        ("Factory Function", test_factory),
        ("Death Classification", test_death_classification),
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
