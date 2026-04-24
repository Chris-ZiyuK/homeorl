#!/usr/bin/env python3
"""
Full Run Analysis: Crafter HACE Experiment (1M steps, 5 agents × 5 seeds)
Generates publication-quality figures for the NeurIPS paper.

Usage: python3 experiments/crafter/full_analysis.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────
RESULTS_DIR = Path('experiments/crafter/results')
OUTPUT_DIR = RESULTS_DIR / 'analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

AGENTS_ORDER = ['vanilla', 'hace', 'health_only', 'naive_survival', 'pure_homeo']

AGENT_CONFIG = {
    'vanilla':        {'label': 'Vanilla PPO',    'color': '#EF4444', 'order': 0},
    'hace':           {'label': 'HACE (ours)',     'color': '#3B82F6', 'order': 1},
    'health_only':    {'label': 'Health-Only',     'color': '#F59E0B', 'order': 2},
    'naive_survival': {'label': 'Naïve Survival',  'color': '#6B7280', 'order': 3},
    'pure_homeo':     {'label': 'Pure Homeo',      'color': '#8B5CF6', 'order': 4},
}

DEATH_CATS = ['combat_death', 'dehydrated', 'starved', 'multiple_depletion']
DEATH_LABELS = ['Combat', 'Dehydrated', 'Starved', 'Multi-Vital']
DEATH_COLORS = ['#10B981', '#3B82F6', '#EF4444', '#8B5CF6']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
})


# ── Data Loading (only full_* directories) ─────────────────────
def load_full_results():
    """Load only full_* run data, returning per-agent per-seed stats."""
    data = defaultdict(list)
    
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith('full_'):
            continue
        
        summary_path = run_dir / 'summary.json'
        episode_path = run_dir / 'episode_data.json'
        
        if not summary_path.exists():
            continue
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        episodes = []
        if episode_path.exists():
            with open(episode_path) as f:
                episodes = json.load(f)
        
        agent = summary['agent']
        seed = summary['seed']
        
        # Compute per-episode stats
        lengths = [e.get('length', 0) for e in episodes]
        deaths = defaultdict(int)
        for ep in episodes:
            cause = ep.get('death_cause', 'unknown')
            deaths[cause] += 1
        
        # Vital averages (may be absent for some agents)
        vitals = {}
        for v in ['food', 'drink', 'energy']:
            vals = [e.get(f'avg_{v}', None) for e in episodes]
            vals = [x for x in vals if x is not None]
            if vals:
                vitals[v] = np.mean(vals)
        
        data[agent].append({
            'seed': seed,
            'summary': summary,
            'n_episodes': len(episodes),
            'lengths': lengths,
            'deaths': dict(deaths),
            'vitals': vitals,
            'episodes': episodes,
        })
    
    return dict(data)


def compute_agent_stats(runs):
    """Aggregate statistics across seeds for one agent."""
    per_seed_avg_lens = [np.mean(r['lengths']) for r in runs]
    per_seed_med_lens = [np.median(r['lengths']) for r in runs]
    
    all_deaths = defaultdict(int)
    total_eps = 0
    for r in runs:
        total_eps += r['n_episodes']
        for cause, count in r['deaths'].items():
            all_deaths[cause] += count
    
    death_rates = {k: v / total_eps for k, v in all_deaths.items()}
    viability_rate = sum(v for k, v in death_rates.items() if k != 'combat_death')
    
    # Vitals (mean across seeds)
    vitals_means = {}
    for v in ['food', 'drink', 'energy']:
        vals = [r['vitals'].get(v, None) for r in runs]
        vals = [x for x in vals if x is not None]
        if vals:
            vitals_means[v] = (np.mean(vals), np.std(vals))
    
    return {
        'n_seeds': len(runs),
        'total_episodes': total_eps,
        'avg_len_mean': np.mean(per_seed_avg_lens),
        'avg_len_std': np.std(per_seed_avg_lens),
        'avg_len_seeds': per_seed_avg_lens,
        'med_len_mean': np.mean(per_seed_med_lens),
        'death_rates': death_rates,
        'viability_rate': viability_rate,
        'vitals': vitals_means,
    }


# ── Plotting ───────────────────────────────────────────────────

def plot_figure1(data, stats):
    """Main comparison figure: 4 panels."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    agents = AGENTS_ORDER
    x = np.arange(len(agents))
    colors = [AGENT_CONFIG[a]['color'] for a in agents]
    labels = [AGENT_CONFIG[a]['label'] for a in agents]
    
    # ── Panel A: Episode Length ──
    ax = axes[0]
    means = [stats[a]['avg_len_mean'] for a in agents]
    stds = [stats[a]['avg_len_std'] for a in agents]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5, width=0.6)
    for i, agent in enumerate(agents):
        seeds = stats[agent]['avg_len_seeds']
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(seeds))
        ax.scatter(x[i] + jitter, seeds, color='black', s=20, zorder=5, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('Avg Episode Length')
    ax.set_title('(A) Survival Duration', fontweight='bold')
    ax.set_ylim(160, 195)
    ax.grid(True, alpha=0.15, axis='y')
    
    # ── Panel B: Death Cause Distribution ──
    ax = axes[1]
    bottom = np.zeros(len(agents))
    for j, (cat, cat_label, cat_color) in enumerate(zip(DEATH_CATS, DEATH_LABELS, DEATH_COLORS)):
        values = [stats[a]['death_rates'].get(cat, 0) for a in agents]
        ax.bar(x, values, bottom=bottom, color=cat_color, label=cat_label,
               width=0.6, edgecolor='white', linewidth=0.5)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('Fraction of Episodes')
    ax.set_title('(B) Death Causes', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.15, axis='y')
    
    # ── Panel C: Viability Failure Rate ──
    ax = axes[2]
    viab = [stats[a]['viability_rate'] * 100 for a in agents]
    bars = ax.bar(x, viab, color=colors, alpha=0.85, edgecolor='white',
                  linewidth=1.5, width=0.6)
    for bar, val in zip(bars, viab):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('Viability Failure Rate (%)')
    ax.set_title('(C) Non-Combat Deaths', fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y')
    ax.set_ylim(0, max(viab) * 1.35)
    
    # ── Panel D: Vital Levels ──
    ax = axes[3]
    vital_names = ['food', 'drink', 'energy']
    vital_labels_short = ['Food', 'Drink', 'Energy']
    # Only show agents that have vital data
    agents_with_vitals = [a for a in agents if stats[a]['vitals']]
    n_vitals = len(vital_names)
    n_agents_v = len(agents_with_vitals)
    width = 0.15
    xv = np.arange(n_vitals)
    
    for i, agent in enumerate(agents_with_vitals):
        vals = [stats[agent]['vitals'].get(v, (0, 0))[0] for v in vital_names]
        errs = [stats[agent]['vitals'].get(v, (0, 0))[1] for v in vital_names]
        offset = (i - n_agents_v / 2 + 0.5) * width
        ax.bar(xv + offset, vals, width, yerr=errs, capsize=3,
               color=AGENT_CONFIG[agent]['color'], alpha=0.85,
               label=AGENT_CONFIG[agent]['label'], edgecolor='white')
    
    ax.axhline(y=9, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(2.5, 9.1, 'Setpoint', fontsize=7, color='gray', ha='right')
    ax.set_xticks(xv)
    ax.set_xticklabels(vital_labels_short, fontsize=9)
    ax.set_ylabel('Average Level (0-9)')
    ax.set_title('(D) Vital Maintenance', fontweight='bold')
    ax.legend(fontsize=7, loc='lower left', framealpha=0.9)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.15, axis='y')
    
    plt.suptitle('Crafter HACE Full Experiment — 1M Steps, 5 Agents × 5 Seeds',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / 'full_comparison.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_learning_curves(data, stats):
    """Smoothed learning curves across training."""
    fig, ax = plt.subplots(figsize=(12, 6))
    window = 100
    
    for agent in AGENTS_ORDER:
        cfg = AGENT_CONFIG[agent]
        runs = data[agent]
        
        # Collect all episode lengths
        all_lens = []
        for run in runs:
            all_lens.append(run['lengths'])
        
        # Pad to same length and smooth
        max_ep = max(len(l) for l in all_lens)
        padded = np.full((len(all_lens), max_ep), np.nan)
        for i, lens in enumerate(all_lens):
            padded[i, :len(lens)] = lens
        
        # Running mean per seed
        smoothed = np.full_like(padded, np.nan)
        for i in range(padded.shape[0]):
            valid = ~np.isnan(padded[i])
            valid_len = valid.sum()
            if valid_len > window:
                cumsum = np.nancumsum(padded[i])
                smoothed[i, window-1:valid_len] = (
                    cumsum[window-1:valid_len] - 
                    np.concatenate([[0], cumsum[:valid_len-window]])
                ) / window
        
        # Mean and std across seeds (only where >= 3 seeds have data)
        seed_count = np.sum(~np.isnan(smoothed), axis=0)
        valid_x = seed_count >= 3
        
        mean = np.nanmean(smoothed[:, valid_x], axis=0)
        std = np.nanstd(smoothed[:, valid_x], axis=0)
        x = np.where(valid_x)[0]
        
        ax.plot(x, mean, label=cfg['label'], color=cfg['color'], linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=cfg['color'])
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Length (smoothed, window=100)', fontsize=12)
    ax.set_title('Learning Curves: Survival Duration Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(50, 250)
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'learning_curves.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_dehydration_focus(stats):
    """Focused comparison of viability failure modes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    agents = AGENTS_ORDER
    x = np.arange(len(agents))
    colors = [AGENT_CONFIG[a]['color'] for a in agents]
    labels = [AGENT_CONFIG[a]['label'] for a in agents]
    
    # Panel A: Dehydrated vs Starved
    ax = axes[0]
    width = 0.3
    dehy = [stats[a]['death_rates'].get('dehydrated', 0) * 100 for a in agents]
    starv = [stats[a]['death_rates'].get('starved', 0) * 100 for a in agents]
    multi = [stats[a]['death_rates'].get('multiple_depletion', 0) * 100 for a in agents]
    
    b1 = ax.bar(x - width, dehy, width, label='Dehydrated', color='#3B82F6', alpha=0.85)
    b2 = ax.bar(x, starv, width, label='Starved', color='#EF4444', alpha=0.85)
    b3 = ax.bar(x + width, multi, width, label='Multi-Vital', color='#8B5CF6', alpha=0.85)
    
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Death Rate (%)')
    ax.set_title('(A) Viability Failure Breakdown', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15, axis='y')
    
    # Panel B: Drink level comparison (key HACE metric)
    ax = axes[1]
    agents_with_drink = [a for a in agents if stats[a]['vitals'].get('drink')]
    xd = np.arange(len(agents_with_drink))
    drink_means = [stats[a]['vitals']['drink'][0] for a in agents_with_drink]
    drink_stds = [stats[a]['vitals']['drink'][1] for a in agents_with_drink]
    drink_colors = [AGENT_CONFIG[a]['color'] for a in agents_with_drink]
    drink_labels = [AGENT_CONFIG[a]['label'] for a in agents_with_drink]
    
    bars = ax.bar(xd, drink_means, yerr=drink_stds, capsize=6,
                  color=drink_colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.axhline(y=9, color='gray', linestyle='--', alpha=0.4)
    ax.text(len(agents_with_drink) - 0.5, 9.1, 'Setpoint', fontsize=8, color='gray')
    
    for bar, val in zip(bars, drink_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(xd)
    ax.set_xticklabels(drink_labels, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Average Drink Level')
    ax.set_title('(B) Hydration Maintenance', fontweight='bold')
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.15, axis='y')
    
    plt.suptitle('Viability Analysis: HACE Drive Reduction Effect',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / 'viability_analysis.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def print_table(stats):
    """Print summary table + LaTeX."""
    agents = AGENTS_ORDER
    
    print("\n" + "=" * 95)
    print("CRAFTER HACE FULL EXPERIMENT — 1M Steps, 5 Agents × 5 Seeds")
    print("=" * 95)
    print(f"{'Agent':<18} {'Seeds':>5} {'Ep.Len (μ±σ)':>15} {'Eps':>7} "
          f"{'Combat%':>8} {'Dehydr%':>8} {'Starve%':>8} {'Multi%':>7} {'Viab.F%':>8}")
    print("-" * 95)
    
    for agent in agents:
        s = stats[agent]
        label = AGENT_CONFIG[agent]['label']
        dr = s['death_rates']
        print(f"{label:<18} {s['n_seeds']:>5} {s['avg_len_mean']:>8.1f}±{s['avg_len_std']:<5.1f} "
              f"{s['total_episodes']:>7} "
              f"{dr.get('combat_death', 0)*100:>7.1f}% "
              f"{dr.get('dehydrated', 0)*100:>7.1f}% "
              f"{dr.get('starved', 0)*100:>7.1f}% "
              f"{dr.get('multiple_depletion', 0)*100:>6.1f}% "
              f"{s['viability_rate']*100:>7.1f}%")
    
    print("=" * 95)
    
    # Vital levels
    print("\nVital Levels (agents with tracking):")
    print(f"{'Agent':<18} {'Food':>12} {'Drink':>12} {'Energy':>12}")
    print("-" * 55)
    for agent in agents:
        s = stats[agent]
        if not s['vitals']:
            continue
        label = AGENT_CONFIG[agent]['label']
        food = s['vitals'].get('food', (0, 0))
        drink = s['vitals'].get('drink', (0, 0))
        energy = s['vitals'].get('energy', (0, 0))
        print(f"{label:<18} {food[0]:>6.2f}±{food[1]:<4.2f} "
              f"{drink[0]:>6.2f}±{drink[1]:<4.2f} "
              f"{energy[0]:>6.2f}±{energy[1]:<4.2f}")
    print()


# ── Main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading full_* results...")
    data = load_full_results()
    
    if not data:
        print("ERROR: No full_* results found!")
        exit(1)
    
    for agent in AGENTS_ORDER:
        runs = data.get(agent, [])
        total_eps = sum(r['n_episodes'] for r in runs)
        print(f"  {agent}: {len(runs)} seeds, {total_eps} episodes")
    
    stats = {agent: compute_agent_stats(data[agent]) for agent in AGENTS_ORDER}
    
    print_table(stats)
    plot_figure1(data, stats)
    plot_learning_curves(data, stats)
    plot_dehydration_focus(stats)
    
    print("\nAll plots saved to:", OUTPUT_DIR)
