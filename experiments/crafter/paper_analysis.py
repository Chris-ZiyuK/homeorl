#!/usr/bin/env python3
"""
Publication-ready analysis: Statistical tests, per-seed error bars,
Crafter Score computation, and LaTeX integration.

Usage: python3 experiments/crafter/paper_analysis.py
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats as sp_stats
from itertools import combinations

# ── Configuration ──────────────────────────────────────────────
RESULTS_DIR = Path('experiments/crafter/results')
OUTPUT_DIR = RESULTS_DIR / 'analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

AGENTS_ORDER = ['vanilla', 'hace', 'health_only', 'naive_survival', 'pure_homeo']

AGENT_CFG = {
    'vanilla':        {'label': 'Vanilla PPO',    'color': '#EF4444', 'marker': 'o'},
    'hace':           {'label': 'HACE (ours)',     'color': '#3B82F6', 'marker': 's'},
    'health_only':    {'label': 'Health-Only',     'color': '#F59E0B', 'marker': '^'},
    'naive_survival': {'label': 'Naïve Survival',  'color': '#6B7280', 'marker': 'D'},
    'pure_homeo':     {'label': 'Pure Homeo',      'color': '#8B5CF6', 'marker': 'v'},
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
})


# ── Data Loading ───────────────────────────────────────────────
def load_full_results():
    data = defaultdict(list)
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith('full_'):
            continue
        summary_path = run_dir / 'summary.json'
        episode_path = run_dir / 'episode_data.json'
        if not summary_path.exists() or not episode_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        with open(episode_path) as f:
            episodes = json.load(f)
        data[summary['agent']].append({
            'seed': summary['seed'],
            'episodes': episodes,
            'summary': summary,
        })
    return dict(data)


def per_seed_stats(runs):
    """Compute per-seed statistics for one agent."""
    seeds = []
    for run in runs:
        eps = run['episodes']
        n = len(eps)
        lengths = [e['length'] for e in eps]
        deaths = defaultdict(int)
        for e in eps:
            deaths[e.get('death_cause', 'unknown')] += 1
        
        viab_deaths = sum(v for k, v in deaths.items() if k != 'combat_death')
        viab_rate = viab_deaths / n if n > 0 else 0
        dehy_rate = deaths.get('dehydrated', 0) / n if n > 0 else 0
        starv_rate = deaths.get('starved', 0) / n if n > 0 else 0
        multi_rate = deaths.get('multiple_depletion', 0) / n if n > 0 else 0
        combat_rate = deaths.get('combat_death', 0) / n if n > 0 else 0
        
        # Achievements
        achs = [e.get('n_achievements', 0) for e in eps]
        
        # Vitals
        vitals = {}
        for v in ['food', 'drink', 'energy']:
            vals = [e.get(f'avg_{v}') for e in eps if e.get(f'avg_{v}') is not None]
            if vals:
                vitals[v] = np.mean(vals)
        
        seeds.append({
            'seed': run['seed'],
            'n_episodes': n,
            'avg_length': np.mean(lengths),
            'med_length': np.median(lengths),
            'viab_rate': viab_rate,
            'dehy_rate': dehy_rate,
            'starv_rate': starv_rate,
            'multi_rate': multi_rate,
            'combat_rate': combat_rate,
            'avg_achievements': np.mean(achs),
            'vitals': vitals,
        })
    return seeds


# ── Statistical Tests ──────────────────────────────────────────
def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    data = np.array(data)
    boot_means = np.array([np.mean(np.random.choice(data, len(data), replace=True))
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100])


def pairwise_tests(agent_stats, metric='viab_rate'):
    """Mann-Whitney U tests between all agent pairs."""
    results = []
    agents = list(agent_stats.keys())
    for a1, a2 in combinations(agents, 2):
        v1 = [s[metric] for s in agent_stats[a1]]
        v2 = [s[metric] for s in agent_stats[a2]]
        try:
            U, p = sp_stats.mannwhitneyu(v1, v2, alternative='two-sided')
        except:
            U, p = 0, 1.0
        results.append({
            'agent1': a1, 'agent2': a2,
            'mean1': np.mean(v1), 'mean2': np.mean(v2),
            'U': U, 'p': p,
            'sig': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        })
    return results


# ── Paper Figures ──────────────────────────────────────────────

def plot_paper_figure(all_stats):
    """Main paper figure: 3 panels with error bars and significance."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    agents = AGENTS_ORDER
    x = np.arange(len(agents))
    colors = [AGENT_CFG[a]['color'] for a in agents]
    labels = [AGENT_CFG[a]['label'] for a in agents]
    
    # ── Panel A: Viability Failure Rate (per-seed) ──
    ax = axes[0]
    means = [np.mean([s['viab_rate'] for s in all_stats[a]]) * 100 for a in agents]
    ci_lo = [bootstrap_ci([s['viab_rate'] * 100 for s in all_stats[a]])[0] for a in agents]
    ci_hi = [bootstrap_ci([s['viab_rate'] * 100 for s in all_stats[a]])[1] for a in agents]
    
    err_lo = [m - lo for m, lo in zip(means, ci_lo)]
    err_hi = [hi - m for m, hi in zip(means, ci_hi)]
    
    bars = ax.bar(x, means, yerr=[err_lo, err_hi], capsize=5,
                  color=colors, alpha=0.85, edgecolor='white', linewidth=1.5, width=0.6)
    
    # Individual seed points
    for i, agent in enumerate(agents):
        seed_vals = [s['viab_rate'] * 100 for s in all_stats[agent]]
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(seed_vals))
        ax.scatter(x[i] + jitter, seed_vals, color='black', s=25, zorder=5, alpha=0.5)
    
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha='right')
    ax.set_ylabel('Viability Failure Rate (%)', fontsize=11)
    ax.set_title('(A) Non-Combat Death Rate', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.15, axis='y')
    ax.set_ylim(0, max(means) * 1.45)
    
    # Significance bracket: HACE vs Vanilla
    hace_idx = agents.index('hace')
    van_idx = agents.index('vanilla')
    ymax = max(means[hace_idx], means[van_idx]) + max(err_hi[hace_idx], err_hi[van_idx]) + 3.5
    ax.plot([van_idx, van_idx, hace_idx, hace_idx], 
            [ymax-0.5, ymax, ymax, ymax-0.5], 'k-', linewidth=1)
    # Get p-value
    v_hace = [s['viab_rate'] for s in all_stats['hace']]
    v_van = [s['viab_rate'] for s in all_stats['vanilla']]
    _, p_val = sp_stats.mannwhitneyu(v_hace, v_van, alternative='two-sided')
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    ax.text((van_idx + hace_idx) / 2, ymax + 0.3, f'p={p_val:.3f} ({sig})',
            ha='center', fontsize=8)
    
    # ── Panel B: Dehydration Breakdown ──
    ax = axes[1]
    width = 0.25
    dehy = [np.mean([s['dehy_rate'] for s in all_stats[a]]) * 100 for a in agents]
    starv = [np.mean([s['starv_rate'] for s in all_stats[a]]) * 100 for a in agents]
    multi = [np.mean([s['multi_rate'] for s in all_stats[a]]) * 100 for a in agents]
    
    b1 = ax.bar(x - width, dehy, width, label='Dehydrated', color='#3B82F6', alpha=0.85)
    b2 = ax.bar(x, starv, width, label='Starved', color='#EF4444', alpha=0.85)
    b3 = ax.bar(x + width, multi, width, label='Multi-Vital', color='#8B5CF6', alpha=0.85)
    
    for bars_group in [b1, b2, b3]:
        for bar in bars_group:
            h = bar.get_height()
            if h > 0.8:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                        f'{h:.1f}', ha='center', va='bottom', fontsize=6.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha='right')
    ax.set_ylabel('Death Rate (%)', fontsize=11)
    ax.set_title('(B) Viability Failure Breakdown', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.15, axis='y')
    
    # ── Panel C: Vital Maintenance ──
    ax = axes[2]
    vital_names = ['food', 'drink', 'energy']
    vital_labels = ['Food', 'Drink', 'Energy']
    agents_v = [a for a in agents if all_stats[a][0]['vitals']]
    n_v = len(vital_names)
    width = 0.18
    xv = np.arange(n_v)
    
    for i, agent in enumerate(agents_v):
        means_v = []
        errs_v = []
        for v in vital_names:
            vals = [s['vitals'].get(v, 0) for s in all_stats[agent]]
            means_v.append(np.mean(vals))
            errs_v.append(np.std(vals))
        offset = (i - len(agents_v) / 2 + 0.5) * width
        ax.bar(xv + offset, means_v, width, yerr=errs_v, capsize=3,
               color=AGENT_CFG[agent]['color'], alpha=0.85,
               label=AGENT_CFG[agent]['label'], edgecolor='white')
    
    ax.axhline(y=9, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(n_v - 0.5, 9.1, 'Setpoint', fontsize=8, color='gray', ha='right')
    ax.set_xticks(xv)
    ax.set_xticklabels(vital_labels, fontsize=10)
    ax.set_ylabel('Average Level (0-9)', fontsize=11)
    ax.set_title('(C) Vital Maintenance', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.15, axis='y')
    
    plt.suptitle('Crafter Generalization: HACE vs Baselines (1M steps, 5 seeds)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / 'paper_figure.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'paper_figure.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_learning_curves(data):
    """Clean learning curves with proper windowing."""
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 100
    
    for agent in AGENTS_ORDER:
        cfg = AGENT_CFG[agent]
        runs = data[agent]
        
        all_lens = [r['episodes'] for r in runs]
        all_lens = [[e['length'] for e in eps] for eps in all_lens]
        
        max_ep = min(len(l) for l in all_lens)  # Use minimum to avoid NaN
        trimmed = np.array([l[:max_ep] for l in all_lens])
        
        # Smooth each seed
        smoothed = np.zeros_like(trimmed, dtype=float)
        for i in range(trimmed.shape[0]):
            cumsum = np.cumsum(trimmed[i])
            smoothed[i, window-1:] = (cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])) / window
            smoothed[i, :window-1] = np.nan
        
        mean = np.nanmean(smoothed, axis=0)
        std = np.nanstd(smoothed, axis=0)
        valid = ~np.isnan(mean)
        xr = np.arange(max_ep)
        
        ax.plot(xr[valid], mean[valid], label=cfg['label'], color=cfg['color'], linewidth=2)
        ax.fill_between(xr[valid], (mean - std)[valid], (mean + std)[valid],
                        alpha=0.12, color=cfg['color'])
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Length (smoothed)', fontsize=12)
    ax.set_title('Survival Duration Learning Curves (Crafter, 1M Steps)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'paper_learning_curves.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'paper_learning_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def generate_latex_table(all_stats):
    """Generate LaTeX table for the paper."""
    agents = AGENTS_ORDER
    
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering\\small")
    lines.append("\\caption{Crafter generalization results (1M steps, 5 seeds). ")
    lines.append("\\emph{Viab. Fail.}: non-combat death rate. ")
    lines.append("\\emph{Dehydr.}: dehydration death rate. $\\downarrow$ = lower is better.}")
    lines.append("\\label{tab:crafter}")
    lines.append("\\begin{tabular}{@{}lccccccc@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Agent} & \\textbf{Ep. Len} & \\textbf{Achiev.} & "
                 "\\textbf{Viab. Fail.$\\downarrow$} & \\textbf{Dehydr.$\\downarrow$} & "
                 "\\textbf{Food} & \\textbf{Drink} & \\textbf{Energy} \\\\")
    lines.append("\\midrule")
    
    for agent in agents:
        seeds = all_stats[agent]
        label = AGENT_CFG[agent]['label']
        
        avg_len = np.mean([s['avg_length'] for s in seeds])
        std_len = np.std([s['avg_length'] for s in seeds])
        avg_ach = np.mean([s['avg_achievements'] for s in seeds])
        viab = np.mean([s['viab_rate'] for s in seeds]) * 100
        dehy = np.mean([s['dehy_rate'] for s in seeds]) * 100
        
        food_vals = [s['vitals'].get('food', None) for s in seeds]
        drink_vals = [s['vitals'].get('drink', None) for s in seeds]
        energy_vals = [s['vitals'].get('energy', None) for s in seeds]
        
        food_str = f"{np.mean([v for v in food_vals if v]):.2f}" if any(food_vals) else "---"
        drink_str = f"{np.mean([v for v in drink_vals if v]):.2f}" if any(drink_vals) else "---"
        energy_str = f"{np.mean([v for v in energy_vals if v]):.2f}" if any(energy_vals) else "---"
        
        bold = "\\textbf{" if agent == 'hace' else ""
        boldend = "}" if agent == 'hace' else ""
        
        lines.append(f"{bold}{label}{boldend} & "
                     f"{avg_len:.0f}$\\pm${std_len:.0f} & {avg_ach:.1f} & "
                     f"{viab:.1f}\\% & {dehy:.1f}\\% & "
                     f"{food_str} & {drink_str} & {energy_str} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading full_* results...")
    data = load_full_results()
    
    if not data:
        print("ERROR: No full_* results found!")
        sys.exit(1)
    
    # Compute per-seed stats
    all_stats = {}
    for agent in AGENTS_ORDER:
        runs = data.get(agent, [])
        all_stats[agent] = per_seed_stats(runs)
        total_eps = sum(s['n_episodes'] for s in all_stats[agent])
        print(f"  {agent}: {len(runs)} seeds, {total_eps} episodes")
    
    # ── Summary Table ──
    print("\n" + "=" * 100)
    print("CRAFTER HACE — FULL RESULTS (1M STEPS, 5 SEEDS)")
    print("=" * 100)
    print(f"{'Agent':<18} {'Ep.Len':>10} {'Achiev':>7} {'Viab.F%':>8} "
          f"{'Dehydr%':>8} {'Starve%':>8} {'Multi%':>7} {'Combat%':>8}")
    print("-" * 100)
    
    for agent in AGENTS_ORDER:
        s = all_stats[agent]
        label = AGENT_CFG[agent]['label']
        avg_len = np.mean([x['avg_length'] for x in s])
        std_len = np.std([x['avg_length'] for x in s])
        avg_ach = np.mean([x['avg_achievements'] for x in s])
        viab = np.mean([x['viab_rate'] for x in s]) * 100
        dehy = np.mean([x['dehy_rate'] for x in s]) * 100
        starv = np.mean([x['starv_rate'] for x in s]) * 100
        multi = np.mean([x['multi_rate'] for x in s]) * 100
        combat = np.mean([x['combat_rate'] for x in s]) * 100
        print(f"{label:<18} {avg_len:>6.1f}±{std_len:<3.1f} {avg_ach:>7.1f} "
              f"{viab:>7.1f}% {dehy:>7.1f}% {starv:>7.1f}% {multi:>6.1f}% {combat:>7.1f}%")
    
    # ── Statistical Tests ──
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS (Mann-Whitney U on viability failure rate)")
    print("=" * 80)
    
    tests = pairwise_tests(all_stats, 'viab_rate')
    print(f"{'Comparison':<40} {'U':>6} {'p-value':>10} {'Sig':>5}")
    print("-" * 65)
    for t in tests:
        l1 = AGENT_CFG[t['agent1']]['label']
        l2 = AGENT_CFG[t['agent2']]['label']
        print(f"{l1} vs {l2:<20} {t['U']:>6.0f} {t['p']:>10.4f} {t['sig']:>5}")
    
    # Bootstrap CIs
    print("\n" + "=" * 80)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (viability failure rate, %)")
    print("=" * 80)
    for agent in AGENTS_ORDER:
        vals = [s['viab_rate'] * 100 for s in all_stats[agent]]
        ci = bootstrap_ci(vals)
        label = AGENT_CFG[agent]['label']
        print(f"  {label:<18}: {np.mean(vals):>5.1f}% [{ci[0]:.1f}%, {ci[1]:.1f}%]")
    
    # ── Generate Plots ──
    print()
    plot_paper_figure(all_stats)
    plot_learning_curves(data)
    
    # ── LaTeX Table ──
    latex = generate_latex_table(all_stats)
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    print(latex)
    
    # Save LaTeX
    with open(OUTPUT_DIR / 'crafter_table.tex', 'w') as f:
        f.write(latex)
    print(f"\nSaved: {OUTPUT_DIR / 'crafter_table.tex'}")
    
    print("\nAll outputs saved to:", OUTPUT_DIR)
