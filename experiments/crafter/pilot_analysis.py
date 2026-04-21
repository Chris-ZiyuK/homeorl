#!/usr/bin/env python3
"""
Analyze pilot run results (100K steps, 5 agents × 3 seeds).
Generates publication-style comparison plots.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Raw data from Oscar pilot run ─────────────────────────────────
# Parsed from summary.json files (only 100K-step runs)

data = {
    'hace': [
        # s0 was only 10K steps, skip it
        {'seed': 1, 'steps': 100000, 'n_episodes': 567, 'avg_len': 176.39, 'median_len': 173.0,
         'deaths': {'combat_death': 488, 'dehydrated': 37, 'starved': 22, 'multiple_depletion': 20}},
        {'seed': 2, 'steps': 100000, 'n_episodes': 545, 'avg_len': 183.49, 'median_len': 177.0,
         'deaths': {'combat_death': 461, 'dehydrated': 39, 'starved': 27, 'multiple_depletion': 18}},
    ],
    'vanilla': [
        {'seed': 0, 'steps': 100000, 'n_episodes': 558, 'avg_len': 179.21, 'median_len': 171.0,
         'deaths': {'combat_death': 464, 'dehydrated': 56, 'starved': 16, 'multiple_depletion': 22}},
        {'seed': 1, 'steps': 100000, 'n_episodes': 559, 'avg_len': 179.03, 'median_len': 172.0,
         'deaths': {'combat_death': 485, 'dehydrated': 39, 'starved': 16, 'multiple_depletion': 19}},
        {'seed': 2, 'steps': 100000, 'n_episodes': 570, 'avg_len': 175.50, 'median_len': 171.5,
         'deaths': {'combat_death': 484, 'dehydrated': 49, 'starved': 19, 'multiple_depletion': 18}},
    ],
    'health_only': [
        {'seed': 0, 'steps': 100000, 'n_episodes': 564, 'avg_len': 177.47, 'median_len': 172.0,
         'deaths': {'combat_death': 476, 'dehydrated': 51, 'starved': 17, 'multiple_depletion': 20}},
        {'seed': 1, 'steps': 100000, 'n_episodes': 561, 'avg_len': 178.37, 'median_len': 173.0,
         'deaths': {'combat_death': 450, 'dehydrated': 79, 'starved': 14, 'multiple_depletion': 18}},
        {'seed': 2, 'steps': 100000, 'n_episodes': 582, 'avg_len': 171.77, 'median_len': 172.0,
         'deaths': {'combat_death': 510, 'dehydrated': 47, 'starved': 9, 'multiple_depletion': 16}},
    ],
    'naive_survival': [
        {'seed': 0, 'steps': 100000, 'n_episodes': 576, 'avg_len': 173.39, 'median_len': 171.0,
         'deaths': {'combat_death': 496, 'dehydrated': 53, 'starved': 11, 'multiple_depletion': 16}},
        {'seed': 1, 'steps': 100000, 'n_episodes': 536, 'avg_len': 186.56, 'median_len': 179.0,
         'deaths': {'combat_death': 407, 'dehydrated': 72, 'starved': 22, 'multiple_depletion': 35}},
        {'seed': 2, 'steps': 100000, 'n_episodes': 542, 'avg_len': 184.56, 'median_len': 177.0,
         'deaths': {'combat_death': 446, 'dehydrated': 54, 'starved': 23, 'multiple_depletion': 19}},
    ],
    'pure_homeo': [
        # s0 was only 10K steps, skip it
        {'seed': 1, 'steps': 100000, 'n_episodes': 562, 'avg_len': 178.01, 'median_len': 173.0,
         'deaths': {'combat_death': 484, 'dehydrated': 38, 'starved': 24, 'multiple_depletion': 16}},
        {'seed': 2, 'steps': 100000, 'n_episodes': 553, 'avg_len': 180.66, 'median_len': 176.0,
         'deaths': {'combat_death': 463, 'dehydrated': 37, 'starved': 25, 'multiple_depletion': 28}},
    ],
}

# ── Display config ────────────────────────────────────────────────
AGENT_CONFIG = {
    'hace':           {'label': 'HACE (ours)',    'color': '#3B82F6', 'order': 0},
    'vanilla':        {'label': 'Vanilla',        'color': '#EF4444', 'order': 1},
    'health_only':    {'label': 'Health-Only',    'color': '#F59E0B', 'order': 2},
    'naive_survival': {'label': 'Naive Survival', 'color': '#6B7280', 'order': 3},
    'pure_homeo':     {'label': 'Pure Homeo',     'color': '#8B5CF6', 'order': 4},
}

DEATH_CATEGORIES = ['combat_death', 'dehydrated', 'starved', 'multiple_depletion']
DEATH_LABELS = ['Combat', 'Dehydrated', 'Starved', 'Multi-Depletion']
DEATH_COLORS = ['#10B981', '#3B82F6', '#EF4444', '#8B5CF6']

# Sort agents by order
agents_sorted = sorted(data.keys(), key=lambda a: AGENT_CONFIG[a]['order'])


def compute_stats(agent_data):
    """Compute per-agent aggregate statistics."""
    avg_lens = [d['avg_len'] for d in agent_data]
    total_episodes = sum(d['n_episodes'] for d in agent_data)
    
    # Aggregate death causes
    death_totals = {}
    for d in agent_data:
        for cause, count in d['deaths'].items():
            death_totals[cause] = death_totals.get(cause, 0) + count
    death_rates = {k: v / total_episodes for k, v in death_totals.items()}
    
    # Viability failure rate = everything except combat_death
    viability_failures = sum(v for k, v in death_totals.items() if k != 'combat_death')
    viability_rate = viability_failures / total_episodes
    
    return {
        'avg_len_mean': np.mean(avg_lens),
        'avg_len_std': np.std(avg_lens),
        'avg_len_seeds': avg_lens,
        'total_episodes': total_episodes,
        'death_rates': death_rates,
        'viability_rate': viability_rate,
        'n_seeds': len(agent_data),
    }


stats = {agent: compute_stats(data[agent]) for agent in agents_sorted}

# ── Print summary table ───────────────────────────────────────────
print("=" * 85)
print("CRAFTER PILOT (100K steps) — SUMMARY")
print("=" * 85)
print(f"{'Agent':<18} {'Seeds':>5} {'Ep.Len (μ±σ)':>15} {'Episodes':>9} {'Combat%':>8} {'Dehydr%':>8} {'Starve%':>8} {'Viab.Fail%':>10}")
print("-" * 85)
for agent in agents_sorted:
    s = stats[agent]
    label = AGENT_CONFIG[agent]['label']
    print(f"{label:<18} {s['n_seeds']:>5} {s['avg_len_mean']:>8.1f}±{s['avg_len_std']:<5.1f} "
          f"{s['total_episodes']:>9} "
          f"{s['death_rates'].get('combat_death', 0)*100:>7.1f}% "
          f"{s['death_rates'].get('dehydrated', 0)*100:>7.1f}% "
          f"{s['death_rates'].get('starved', 0)*100:>7.1f}% "
          f"{s['viability_rate']*100:>9.1f}%")
print("=" * 85)


# ── Figure 1: Episode Length Comparison ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Average Episode Length (bar + individual seeds)
ax = axes[0]
x = np.arange(len(agents_sorted))
means = [stats[a]['avg_len_mean'] for a in agents_sorted]
stds = [stats[a]['avg_len_std'] for a in agents_sorted]
colors = [AGENT_CONFIG[a]['color'] for a in agents_sorted]
labels = [AGENT_CONFIG[a]['label'] for a in agents_sorted]

bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85,
              edgecolor='white', linewidth=1.5, width=0.65)

# Overlay individual seed points
for i, agent in enumerate(agents_sorted):
    seed_vals = stats[agent]['avg_len_seeds']
    jitter = np.random.uniform(-0.12, 0.12, len(seed_vals))
    ax.scatter(x[i] + jitter, seed_vals, color='black', s=30, zorder=5,
               alpha=0.7, edgecolors='white', linewidths=0.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
ax.set_ylabel('Average Episode Length', fontsize=11)
ax.set_title('(A) Survival Duration', fontsize=12, fontweight='bold')
ax.set_ylim(165, 195)
ax.grid(True, alpha=0.15, axis='y')
ax.axhline(y=np.mean(means), color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

# Panel B: Death Cause Distribution (stacked bar)
ax = axes[1]
bottom = np.zeros(len(agents_sorted))
for j, (cat, cat_label, cat_color) in enumerate(zip(DEATH_CATEGORIES, DEATH_LABELS, DEATH_COLORS)):
    values = [stats[a]['death_rates'].get(cat, 0) for a in agents_sorted]
    ax.bar(x, values, bottom=bottom, color=cat_color, label=cat_label,
           width=0.65, edgecolor='white', linewidth=0.5)
    bottom += values

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
ax.set_ylabel('Fraction of Episodes', fontsize=11)
ax.set_title('(B) Death Cause Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
ax.set_ylim(0, 1.08)
ax.grid(True, alpha=0.15, axis='y')

# Panel C: Viability Failure Rate (focused bar)
ax = axes[2]
viab_rates = [stats[a]['viability_rate'] * 100 for a in agents_sorted]
bars = ax.bar(x, viab_rates, color=colors, alpha=0.85, edgecolor='white',
              linewidth=1.5, width=0.65)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, viab_rates)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
ax.set_ylabel('Viability Failure Rate (%)', fontsize=11)
ax.set_title('(C) Non-Combat Deaths', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.15, axis='y')
ax.set_ylim(0, max(viab_rates) * 1.3)

plt.suptitle('Crafter HACE Pilot — 100K Steps, 5 Agents',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('experiments/crafter/results/pilot_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved: experiments/crafter/results/pilot_analysis.png")

# ── Figure 2: Dehydration focus ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

dehydration_rates = [stats[a]['death_rates'].get('dehydrated', 0) * 100 for a in agents_sorted]
starvation_rates = [stats[a]['death_rates'].get('starved', 0) * 100 for a in agents_sorted]

width = 0.35
ax.bar(x - width/2, dehydration_rates, width, label='Dehydrated', color='#3B82F6', alpha=0.85)
ax.bar(x + width/2, starvation_rates, width, label='Starved', color='#EF4444', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Death Rate (%)', fontsize=11)
ax.set_title('Viability Failure Breakdown: Dehydration vs Starvation', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15, axis='y')

# Add value labels
for i in range(len(agents_sorted)):
    ax.text(x[i] - width/2, dehydration_rates[i] + 0.2, f'{dehydration_rates[i]:.1f}%',
            ha='center', va='bottom', fontsize=8)
    ax.text(x[i] + width/2, starvation_rates[i] + 0.2, f'{starvation_rates[i]:.1f}%',
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('experiments/crafter/results/pilot_viability_breakdown.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: experiments/crafter/results/pilot_viability_breakdown.png")

print("\nDone!")
