#!/usr/bin/env python3
"""
Train PPO agents on Crafter with various reward configurations.

This script trains a PPO agent (via Stable-Baselines3) on the Crafter
benchmark using one of five reward configurations:

  1. vanilla       — Original Crafter reward only
  2. hace          — Crafter reward + multi-dim homeostatic drive reduction
  3. pure_homeo    — Multi-dim HACE reward only (no Crafter reward)
  4. health_only   — Crafter reward + health-only HACE (ablation)
  5. naive_survival — Crafter reward + raw vital bonus (sham control)

Usage:
    python train_crafter.py --agent hace --seed 0 --steps 1000000
    python train_crafter.py --agent vanilla --seed 0 --steps 500000 --pilot

For Oscar HPC:
    See run_crafter_oscar.sh for SLURM job submission.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO on Crafter with HACE')
    parser.add_argument('--agent', type=str, default='vanilla',
                        choices=['vanilla', 'hace', 'pure_homeo',
                                 'health_only', 'naive_survival'],
                        help='Agent/reward configuration')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--steps', type=int, default=1_000_000,
                        help='Total training timesteps')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for Crafter reward (ignored for pure_homeo)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for HACE reward')
    parser.add_argument('--pilot', action='store_true',
                        help='Pilot mode: 100K steps, less logging')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory (default: experiments/crafter/results/<agent>_s<seed>)')
    parser.add_argument('--eval-freq', type=int, default=50_000,
                        help='Evaluate every N steps')
    parser.add_argument('--eval-episodes', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--no-record', action='store_true',
                        help='Disable Crafter recording')
    return parser.parse_args()


def make_env(agent_type: str, seed: int, logdir: str,
             alpha: float, beta: float, record: bool):
    """Create gymnasium-compatible Crafter env with appropriate wrapper."""
    from src.envs.crafter_gymnasium_adapter import make_gymnasium_crafter
    return make_gymnasium_crafter(
        agent_type=agent_type,
        alpha=alpha,
        beta=beta,
        seed=seed,
        logdir=logdir if record else None,
        record=record,
    )


class VitalTracker:
    """Callback-compatible tracker for episode-level survival metrics."""

    def __init__(self):
        self.episode_data = []
        self._current_episode = {
            'vitals': [],
            'rewards': [],
        }

    def on_step(self, info: dict):
        """Called every step to accumulate vital data."""
        if 'vitals' in info:
            self._current_episode['vitals'].append(info['vitals'].copy())
        if 'original_reward' in info:
            self._current_episode['rewards'].append(info['original_reward'])

    def on_episode_end(self, info: dict):
        """Called at episode end to store summary."""
        episode_summary = {
            'death_cause': info.get('death_cause', 'unknown'),
            'episode_length': info.get('episode_length', 0),
            'vital_averages': info.get('vital_averages', {}),
            'achievements': info.get('achievements', {}),
        }

        # Compute survival stats from accumulated vitals
        vitals_list = self._current_episode['vitals']
        if vitals_list:
            for v in ('food', 'drink', 'energy'):
                values = [vt.get(v, 0) for vt in vitals_list]
                episode_summary[f'avg_{v}'] = float(np.mean(values))
                episode_summary[f'min_{v}'] = float(np.min(values))
                episode_summary[f'time_at_zero_{v}'] = sum(
                    1 for x in values if x <= 0
                ) / max(len(values), 1)

        self.episode_data.append(episode_summary)

        # Reset for next episode
        self._current_episode = {'vitals': [], 'rewards': []}

    def save(self, filepath: str):
        """Save episode data to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.episode_data, f, indent=2, default=str)
        print(f"Saved {len(self.episode_data)} episodes to {filepath}")


def train_with_manual_loop(args):
    """Train using a manual loop (for detailed vital tracking).

    We use a manual training loop rather than SB3's model.learn()
    to have full control over per-step vital tracking and death
    cause analysis.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("ERROR: stable-baselines3 is required.")
        print("Install with: pip install stable-baselines3[extra]")
        sys.exit(1)

    # ── Setup ───────────────────────────────────────────────
    if args.pilot:
        args.steps = min(args.steps, 100_000)
        args.eval_freq = 10_000
        args.eval_episodes = 5

    outdir = args.outdir or str(
        PROJECT_ROOT / 'experiments' / 'crafter' / 'results' /
        f'{args.agent}_s{args.seed}'
    )
    os.makedirs(outdir, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config['outdir'] = outdir
    config['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(outdir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"{'='*60}")
    print(f"Crafter HACE Experiment")
    print(f"  Agent:  {args.agent}")
    print(f"  Seed:   {args.seed}")
    print(f"  Steps:  {args.steps:,}")
    print(f"  Alpha:  {args.alpha}")
    print(f"  Beta:   {args.beta}")
    print(f"  Output: {outdir}")
    print(f"{'='*60}")

    # ── Create environment ──────────────────────────────────
    crafter_logdir = os.path.join(outdir, 'crafter_logs')

    # For SB3, we need the env wrapped in a function
    def env_fn():
        return make_env(
            agent_type=args.agent,
            seed=args.seed,
            logdir=crafter_logdir,
            alpha=args.alpha,
            beta=args.beta,
            record=not args.no_record,
        )

    env = env_fn()

    # ── Create PPO agent ────────────────────────────────────
    # Using standard PPO hyperparameters for Crafter
    # (following community baselines)
    model = PPO(
        policy='CnnPolicy',
        env=env,
        verbose=1,
        seed=args.seed,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        tensorboard_log=os.path.join(outdir, 'tb_logs'),
    )

    # ── Training with vital tracking ────────────────────────
    tracker = VitalTracker()
    eval_results = []

    print(f"\nStarting training for {args.steps:,} steps...")
    start_time = time.time()

    obs = env.reset()
    total_steps = 0
    episode_count = 0
    episode_reward = 0
    episode_steps = 0

    while total_steps < args.steps:
        # Use the model to select action
        action, _ = model.predict(obs, deterministic=False)

        result = env.step(action)
        if len(result) == 4:
            next_obs, reward, done, info = result
        else:
            next_obs, reward, done, _, info = result

        total_steps += 1
        episode_steps += 1
        episode_reward += reward

        # Track vitals (for HACE-wrapped envs)
        tracker.on_step(info)

        if done:
            episode_count += 1
            tracker.on_episode_end(info)

            if episode_count % 50 == 0:
                elapsed = time.time() - start_time
                fps = total_steps / max(elapsed, 1)
                death = info.get('death_cause', '?')
                print(f"  Ep {episode_count:4d} | "
                      f"Steps {total_steps:>8,}/{args.steps:,} | "
                      f"Len {episode_steps:5d} | "
                      f"R {episode_reward:7.2f} | "
                      f"Death: {death:15s} | "
                      f"FPS {fps:.0f}")

            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
        else:
            obs = next_obs

        # ── Periodic model updates ──────────────────────────
        # Note: PPO collects rollouts internally via model.learn()
        # For the manual loop, we just call model.learn() in chunks
        if total_steps % args.eval_freq == 0:
            # Save checkpoint
            model.save(os.path.join(outdir, f'model_{total_steps}'))

    # Actually, the manual loop approach has issues with PPO's internal
    # buffer management. Let's use model.learn() with a custom callback.
    print("\nNote: Switching to SB3's native training loop with callback...")
    print("(Manual loop above is for demonstration; see train_with_sb3 below)")


def train_with_sb3(args):
    """Train using SB3's native loop with a custom callback for tracking."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    except ImportError:
        print("ERROR: stable-baselines3 is required.")
        print("Install: pip install 'stable-baselines3[extra]'")
        sys.exit(1)

    if args.pilot:
        args.steps = min(args.steps, 100_000)
        args.eval_freq = 10_000
        args.eval_episodes = 5

    outdir = args.outdir or str(
        PROJECT_ROOT / 'experiments' / 'crafter' / 'results' /
        f'{args.agent}_s{args.seed}'
    )
    os.makedirs(outdir, exist_ok=True)

    config = vars(args).copy()
    config['outdir'] = outdir
    config['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(outdir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"{'='*60}")
    print(f"Crafter HACE Experiment (SB3 native loop)")
    print(f"  Agent:  {args.agent}")
    print(f"  Seed:   {args.seed}")
    print(f"  Steps:  {args.steps:,}")
    print(f"  Alpha:  {args.alpha}")
    print(f"  Beta:   {args.beta}")
    print(f"  Output: {outdir}")
    print(f"{'='*60}")

    crafter_logdir = os.path.join(outdir, 'crafter_logs')

    env = make_env(
        agent_type=args.agent,
        seed=args.seed,
        logdir=crafter_logdir,
        alpha=args.alpha,
        beta=args.beta,
        record=not args.no_record,
    )

    # ── Custom callback for vital tracking ──────────────────

    class VitalCallback(BaseCallback):
        """Track vitals and death causes during training."""

        def __init__(self, save_path: str, verbose=0):
            super().__init__(verbose)
            self.save_path = save_path
            self.episodes = []
            self._ep_vitals = []
            self._ep_len = 0

        def _on_step(self) -> bool:
            infos = self.locals.get('infos', [{}])
            for info in infos if isinstance(infos, list) else [infos]:
                self._ep_len += 1
                if 'vitals' in info:
                    self._ep_vitals.append(info['vitals'].copy())

                # Check for episode end
                if info.get('death_cause') or self.locals.get('dones', [False])[0]:
                    ep_data = {
                        'step': self.num_timesteps,
                        'episode': len(self.episodes),
                        'length': info.get('episode_length', self._ep_len),
                        'death_cause': info.get('death_cause', 'unknown'),
                    }

                    # Vital averages
                    if self._ep_vitals:
                        for v in ('food', 'drink', 'energy'):
                            vals = [vt.get(v, 0) for vt in self._ep_vitals]
                            ep_data[f'avg_{v}'] = float(np.mean(vals))
                            ep_data[f'min_{v}'] = float(np.min(vals))

                    # Achievement count
                    achievements = info.get('achievements', {})
                    ep_data['n_achievements'] = sum(
                        1 for v in achievements.values() if v > 0
                    )

                    self.episodes.append(ep_data)
                    self._ep_vitals = []
                    self._ep_len = 0

                    # Periodic save
                    if len(self.episodes) % 100 == 0:
                        self._save()
                        if self.verbose:
                            print(f"  [{len(self.episodes)} episodes tracked]")

            return True

        def _save(self):
            with open(self.save_path, 'w') as f:
                json.dump(self.episodes, f, indent=2)

        def _on_training_end(self):
            self._save()
            print(f"Saved {len(self.episodes)} episode records to {self.save_path}")

    # ── Create PPO agent ────────────────────────────────────
    model = PPO(
        policy='CnnPolicy',
        env=env,
        verbose=1,
        seed=args.seed,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        tensorboard_log=os.path.join(outdir, 'tb_logs'),
    )

    # ── Train ───────────────────────────────────────────────
    vital_cb = VitalCallback(
        save_path=os.path.join(outdir, 'episode_data.json'),
        verbose=1,
    )

    start = time.time()
    model.learn(
        total_timesteps=args.steps,
        callback=vital_cb,
        progress_bar=True,
    )
    elapsed = time.time() - start

    # ── Save final model and summary ────────────────────────
    model.save(os.path.join(outdir, 'model_final'))

    summary = {
        'agent': args.agent,
        'seed': args.seed,
        'total_steps': args.steps,
        'training_time_s': elapsed,
        'n_episodes': len(vital_cb.episodes),
        'alpha': args.alpha,
        'beta': args.beta,
    }

    # Compute aggregate stats
    if vital_cb.episodes:
        lengths = [e['length'] for e in vital_cb.episodes]
        summary['avg_episode_length'] = float(np.mean(lengths))
        summary['median_episode_length'] = float(np.median(lengths))

        # Death cause distribution
        causes = [e['death_cause'] for e in vital_cb.episodes]
        cause_counts = {}
        for c in causes:
            cause_counts[c] = cause_counts.get(c, 0) + 1
        summary['death_causes'] = cause_counts

    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Time:     {elapsed/3600:.1f} hours")
    print(f"  Episodes: {len(vital_cb.episodes)}")
    if vital_cb.episodes:
        print(f"  Avg len:  {summary['avg_episode_length']:.0f}")
        print(f"  Deaths:   {summary.get('death_causes', {})}")
    print(f"  Output:   {outdir}")
    print(f"{'='*60}")

    env.close()


if __name__ == '__main__':
    args = parse_args()
    train_with_sb3(args)
