#!/usr/bin/env python3
"""
Analyze Crafter HACE experiment results.

Reads episode_data.json files from each run and produces:
  1. Survival metrics comparison (episode length, vital averages)
  2. Death cause distribution (stacked bar chart)
  3. Crafter-native metrics (from crafter stats.json)
  4. LaTeX-ready summary table

Usage:
    python analyze_crafter.py --results-dir experiments/crafter/results
    python analyze_crafter.py --results-dir experiments/crafter/results --latex
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not found. Plotting disabled.")


# ── Agent display configuration ────────────────────────────────────────────

AGENT_DISPLAY = {
    'vanilla':        {'label': 'Vanilla',       'color': '#EF4444', 'short': 'Van'},
    'hace':           {'label': 'HACE (ours)',    'color': '#3B82F6', 'short': 'HACE'},
    'pure_homeo':     {'label': 'Pure Homeo',     'color': '#8B5CF6', 'short': 'PH'},
    'health_only':    {'label': 'Health-Only',    'color': '#F59E0B', 'short': 'HO'},
    'naive_survival': {'label': 'Naive Survival', 'color': '#6B7280', 'short': 'NS'},
}

DEATH_COLORS = {
    'starved': '#EF4444',
    'dehydrated': '#3B82F6',
    'exhausted': '#F59E0B',
    'multiple_depletion': '#8B5CF6',
    'combat_death': '#10B981',
    'timeout': '#6B7280',
    'unknown': '#D1D5DB',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str,
                        default='experiments/crafter/results',
                        help='Root directory containing run folders')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Where to save plots (default: results-dir/analysis)')
    parser.add_argument('--latex', action='store_true',
                        help='Print LaTeX table')
    parser.add_argument('--window', type=int, default=50,
                        help='Smoothing window for learning curves')
    return parser.parse_args()


def load_results(results_dir: str) -> dict:
    """Load all episode data from the results directory.

    Returns:
        Dict[agent_type, List[Dict[seed, episodes]]]
    """
    results = defaultdict(list)
    results_path = Path(results_dir)

    for run_dir in sorted(results_path.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / 'config.json'
        data_path = run_dir / 'episode_data.json'

        if not config_path.exists() or not data_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)
        with open(data_path) as f:
            episodes = json.load(f)

        agent = config.get('agent', 'unknown')
        seed = config.get('seed', -1)

        results[agent].append({
            'seed': seed,
            'config': config,
            'episodes': episodes,
            'run_dir': str(run_dir),
        })

    return dict(results)


# ── Metric Computation ──────────────────────────────────────────────────────

def compute_survival_stats(episodes: list) -> dict:
    """Compute survival-related statistics from episode data."""
    if not episodes:
        return {}

    lengths = [e.get('length', 0) for e in episodes]
    deaths = [e.get('death_cause', 'unknown') for e in episodes]

    stats = {
        'n_episodes': len(episodes),
        'avg_length': float(np.mean(lengths)),
        'median_length': float(np.median(lengths)),
        'std_length': float(np.std(lengths)),
        'max_length': float(np.max(lengths)),
    }

    # Death cause distribution
    cause_counts = defaultdict(int)
    for d in deaths:
        cause_counts[d] += 1
    stats['death_distribution'] = dict(cause_counts)
    stats['death_rates'] = {
        k: v / len(episodes) for k, v in cause_counts.items()
    }

    # Vital averages (if available)
    for v in ('food', 'drink', 'energy'):
        avg_key = f'avg_{v}'
        vals = [e[avg_key] for e in episodes if avg_key in e]
        if vals:
            stats[f'mean_avg_{v}'] = float(np.mean(vals))
            stats[f'std_avg_{v}'] = float(np.std(vals))

    # Achievement count
    achievements = [e.get('n_achievements', 0) for e in episodes]
    if achievements:
        stats['avg_achievements'] = float(np.mean(achievements))

    return stats


def compute_learning_curves(runs: list, window: int = 50) -> dict:
    """Compute smoothed learning curves across seeds.

    Returns dict with 'mean' and 'std' arrays for episode_length.
    """
    all_lengths = []
    for run in runs:
        lengths = [e.get('length', 0) for e in run['episodes']]
        all_lengths.append(lengths)

    if not all_lengths:
        return {}

    # Pad to same length
    max_len = max(len(l) for l in all_lengths)
    padded = np.full((len(all_lengths), max_len), np.nan)
    for i, lengths in enumerate(all_lengths):
        padded[i, :len(lengths)] = lengths

    # Smooth each seed
    smoothed = np.full_like(padded, np.nan)
    for i in range(padded.shape[0]):
        valid = ~np.isnan(padded[i])
        if valid.sum() > window:
            cumsum = np.nancumsum(padded[i])
            smoothed[i, window-1:] = (
                cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])
            ) / window

    mean = np.nanmean(smoothed, axis=0)
    std = np.nanstd(smoothed, axis=0)

    return {'mean': mean, 'std': std, 'n_episodes': max_len}


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_episode_length_comparison(results: dict, output_path: str, window: int = 50):
    """Plot smoothed episode length learning curves for all agents."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for agent, runs in results.items():
        display = AGENT_DISPLAY.get(agent, {'label': agent, 'color': '#333'})
        curves = compute_learning_curves(runs, window)
        if not curves:
            continue

        x = np.arange(curves['n_episodes'])
        mean = curves['mean']
        std = curves['std']

        valid = ~np.isnan(mean)
        ax.plot(x[valid], mean[valid], label=display['label'],
                color=display['color'], linewidth=2)
        ax.fill_between(x[valid], (mean - std)[valid], (mean + std)[valid],
                        alpha=0.15, color=display['color'])

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Length (smoothed)', fontsize=12)
    ax.set_title('Survival Duration: HACE vs Baselines on Crafter', fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_death_cause_distribution(results: dict, output_path: str):
    """Stacked bar chart of death causes per agent."""
    if not HAS_MPL:
        return

    agents = list(results.keys())
    cause_types = ['starved', 'dehydrated', 'exhausted',
                   'multiple_depletion', 'combat_death', 'timeout']

    # Aggregate death rates across seeds
    agent_rates = {}
    for agent, runs in results.items():
        all_episodes = []
        for run in runs:
            all_episodes.extend(run['episodes'])
        stats = compute_survival_stats(all_episodes)
        agent_rates[agent] = stats.get('death_rates', {})

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(agents))
    width = 0.6
    bottom = np.zeros(len(agents))

    for cause in cause_types:
        values = [agent_rates.get(a, {}).get(cause, 0) for a in agents]
        ax.bar(x, values, width, bottom=bottom,
               label=cause.replace('_', ' ').title(),
               color=DEATH_COLORS.get(cause, '#999'))
        bottom += values

    labels = [AGENT_DISPLAY.get(a, {}).get('label', a) for a in agents]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Fraction of Episodes', fontsize=12)
    ax.set_title('Death Cause Distribution by Agent', fontsize=14)
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.25, 1))
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_vital_comparison(results: dict, output_path: str):
    """Bar chart comparing average vital levels across agents."""
    if not HAS_MPL:
        return

    agents = list(results.keys())
    vitals = ['food', 'drink', 'energy']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, vital in enumerate(vitals):
        ax = axes[idx]
        means = []
        stds = []
        colors = []

        for agent in agents:
            all_episodes = []
            for run in results[agent]:
                all_episodes.extend(run['episodes'])
            stats = compute_survival_stats(all_episodes)
            means.append(stats.get(f'mean_avg_{vital}', 0))
            stds.append(stats.get(f'std_avg_{vital}', 0))
            colors.append(AGENT_DISPLAY.get(agent, {}).get('color', '#333'))

        x = np.arange(len(agents))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, edgecolor='white')

        labels = [AGENT_DISPLAY.get(a, {}).get('short', a) for a in agents]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(f'Average {vital.capitalize()}', fontsize=12)
        ax.set_ylim(0, 9.5)
        ax.axhline(y=9, color='gray', linestyle='--', alpha=0.5, label='Setpoint')
        ax.grid(True, alpha=0.15, axis='y')

    axes[0].set_ylabel('Average Level (0-9)', fontsize=12)
    fig.suptitle('Vital Level Maintenance: HACE vs Baselines', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ── Summary Table ───────────────────────────────────────────────────────────

def print_summary_table(results: dict, latex: bool = False):
    """Print or generate latex for the summary comparison table."""
    print("\n" + "="*80)
    print("CRAFTER HACE EXPERIMENT — SUMMARY")
    print("="*80)

    header = f"{'Agent':<18} {'Ep.Len':>8} {'Avg Food':>9} {'Avg Drink':>10} " \
             f"{'Avg Energy':>11} {'Achieve':>8} {'Starved%':>9} {'Combat%':>9}"
    print(header)
    print("-"*80)

    rows = []
    for agent, runs in results.items():
        all_episodes = []
        for run in runs:
            all_episodes.extend(run['episodes'])
        stats = compute_survival_stats(all_episodes)

        label = AGENT_DISPLAY.get(agent, {}).get('label', agent)
        row = {
            'agent': label,
            'ep_len': stats.get('avg_length', 0),
            'food': stats.get('mean_avg_food', 0),
            'drink': stats.get('mean_avg_drink', 0),
            'energy': stats.get('mean_avg_energy', 0),
            'achievements': stats.get('avg_achievements', 0),
            'starved': stats.get('death_rates', {}).get('starved', 0) * 100,
            'combat': stats.get('death_rates', {}).get('combat_death', 0) * 100,
        }
        rows.append(row)

        print(f"{label:<18} {row['ep_len']:>8.0f} {row['food']:>9.2f} "
              f"{row['drink']:>10.2f} {row['energy']:>11.2f} "
              f"{row['achievements']:>8.1f} {row['starved']:>8.1f}% "
              f"{row['combat']:>8.1f}%")

    if latex:
        print("\n% LaTeX table:")
        print("\\begin{table}[t]")
        print("\\centering\\small")
        print("\\caption{Generalization: HACE on Crafter benchmark.}")
        print("\\begin{tabular}{@{}lcccccc@{}}")
        print("\\toprule")
        print("\\textbf{Agent} & \\textbf{Ep. Len} & \\textbf{Avg Food} & "
              "\\textbf{Avg Drink} & \\textbf{Avg Energy} & "
              "\\textbf{Starved\\%} & \\textbf{Combat\\%} \\\\")
        print("\\midrule")
        for r in rows:
            boldstart = "\\textbf{" if 'HACE' in r['agent'] else ""
            boldend = "}" if 'HACE' in r['agent'] else ""
            print(f"{boldstart}{r['agent']}{boldend} & "
                  f"{r['ep_len']:.0f} & {r['food']:.2f} & "
                  f"{r['drink']:.2f} & {r['energy']:.2f} & "
                  f"{r['starved']:.1f} & {r['combat']:.1f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or os.path.join(results_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)

    if not results:
        print("No results found! Check the results directory.")
        sys.exit(1)

    print(f"Found {len(results)} agent types:")
    for agent, runs in results.items():
        total_eps = sum(len(r['episodes']) for r in runs)
        print(f"  {agent}: {len(runs)} seeds, {total_eps} total episodes")

    # Generate all outputs
    print_summary_table(results, latex=args.latex)

    if HAS_MPL:
        plot_episode_length_comparison(
            results,
            os.path.join(output_dir, 'episode_length.png'),
            window=args.window,
        )
        plot_death_cause_distribution(
            results,
            os.path.join(output_dir, 'death_causes.png'),
        )
        plot_vital_comparison(
            results,
            os.path.join(output_dir, 'vital_levels.png'),
        )

    # Save aggregated stats
    agg_stats = {}
    for agent, runs in results.items():
        all_episodes = []
        for run in runs:
            all_episodes.extend(run['episodes'])
        agg_stats[agent] = compute_survival_stats(all_episodes)

    with open(os.path.join(output_dir, 'aggregate_stats.json'), 'w') as f:
        json.dump(agg_stats, f, indent=2)
    print(f"\nSaved aggregate stats to {output_dir}/aggregate_stats.json")


if __name__ == '__main__':
    main()
