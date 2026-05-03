"""
Generate publication-quality figures for the HACE course paper.

Reads raw JSON experiment data and produces individual PDF figures
with a cohesive premium color palette.

Usage:
    python generate_figures.py
"""

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

E2_RESET   = ROOT / "experiments/sequential_results/e2_reset/e2_reset.json"
E2_CARRY   = ROOT / "experiments/sequential_results/e2_carryover/e2_carryover.json"
E3_ALPHA   = ROOT / "experiments/sequential_results/e3_alpha_carryover/e3_alpha.json"

# ── Color Palette ────────────────────────────────────────────
COLORS = {
    "A_task_only":        "#F18F01",   # amber
    "B_energy_aware":     "#F4A942",   # light amber
    "C_hace":             "#2E86AB",   # teal
    "C_task_homeostatic": "#2E86AB",   # teal (E2 naming)
    "D_pure_homeostatic": "#7EB3C9",   # soft teal
    "E_task_oracle":      "#9E9E9E",   # grey
    "F_ewc":              "#A23B72",   # mauve
    "G_l2":               "#D08CBF",   # light mauve
    "H_er":               "#8B6FAE",   # lavender
    "I_hace_ewc":         "#C73E1D",   # vermillion
}

LABELS = {
    "A_task_only":        "Task-Only",
    "B_energy_aware":     "Energy-Aware",
    "C_hace":             "HACE (ours)",
    "C_task_homeostatic": "HACE (ours)",
    "D_pure_homeostatic": "Pure Homeo.",
    "E_task_oracle":      "Oracle",
    "F_ewc":              "EWC",
    "G_l2":               "L2",
    "H_er":               "Exp. Replay",
    "I_hace_ewc":         "HACE+EWC",
}

TASK_LABELS = {
    "reach": "Reach", "recharge": "Recharge", "dual_food": "Dual Food",
    "conservation": "Conserv.", "endurance": "Endurance",
    "collect_exit": "Coll. Exit", "detour": "Detour",
    "wall_maze": "Wall Maze", "sprint": "Sprint",
    "gauntlet_refuel": "Gauntlet",
    "hazard_cross": "Haz. Cross", "hazard_reach": "Haz. Reach",
    "tight_detour": "Tight Det.",
}

# ── Style Setup ──────────────────────────────────────────────
def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

# ── Data Loading ─────────────────────────────────────────────
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def validate_and_print(name, data):
    """Print summary for data validation."""
    agg = data["aggregated"]
    agents = list(agg["current"].keys())
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Agents: {agents}")
    pe = agg["phase_end"]
    for agent in agents:
        phases = pe[agent]
        n = len(phases)
        bsolv_vals = []
        for p in phases:
            b = p.get("boundary_solvability_mean")
            if b is not None and not math.isnan(b):
                bsolv_vals.append(b)
        avg_bs = np.mean(bsolv_vals) if bsolv_vals else float("nan")
        # final phase success on all tasks
        final = phases[-1]
        task_names = list(final["task_metrics"].keys())
        successes = {t: final["task_metrics"][t]["success_mean"] for t in task_names}
        avg_success = np.mean(list(successes.values()))
        print(f"  {agent:25s} | phases={n} | avg_bsolv={avg_bs:.3f} | avg_final_success={avg_success:.3f}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Learning curves (E3, focused agents)
# ═══════════════════════════════════════════════════════════════
def fig_learning_curves(e3):
    agg = e3["aggregated"]["current"]
    tasks = e3["config"]["tasks"]
    ep_per_task = e3["config"]["episodes_per_task"]

    # Focus agents for clarity
    focus = ["A_task_only", "C_hace", "F_ewc", "I_hace_ewc"]
    bg    = ["B_energy_aware", "D_pure_homeostatic", "E_task_oracle", "G_l2", "H_er"]

    fig, ax = plt.subplots(figsize=(7, 2.8))

    # Background agents: thin, low alpha
    for agent in bg:
        if agent not in agg:
            continue
        series = agg[agent]
        x = [s["global_episode"] for s in series]
        y = [s["success_mean"] for s in series]
        ax.plot(x, y, color=COLORS.get(agent, "#ccc"), alpha=0.25,
                linewidth=0.9, label=LABELS.get(agent, agent))

    # Focus agents: bold
    for agent in focus:
        if agent not in agg:
            continue
        series = agg[agent]
        x = [s["global_episode"] for s in series]
        y = [s["success_mean"] for s in series]
        std = [s["success_std"] for s in series]
        ax.plot(x, y, color=COLORS[agent], linewidth=2.2,
                label=LABELS[agent], zorder=5)
        ax.fill_between(x,
                        np.clip(np.array(y) - np.array(std), 0, 1),
                        np.clip(np.array(y) + np.array(std), 0, 1),
                        color=COLORS[agent], alpha=0.10, zorder=4)

    # Task boundary lines with labels
    for i in range(1, len(tasks)):
        bx = i * ep_per_task
        ax.axvline(bx, color="#d0d0d0", ls="--", lw=0.7, zorder=1)

    # Task labels at top
    for i, t in enumerate(tasks):
        cx = (i + 0.5) * ep_per_task
        ax.text(cx, 1.06, TASK_LABELS.get(t, t), ha="center", va="bottom",
                fontsize=5.5, color="#666", rotation=30)

    ax.set_xlim(0, len(tasks) * ep_per_task)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Global Episode")
    ax.set_ylabel("Success Rate")
    ax.legend(loc="lower left", ncol=3, framealpha=0.85,
              edgecolor="#ddd", fancybox=False)
    ax.grid(True, axis="y")

    fig.savefig(FIG_DIR / "fig_learning_curves.pdf")
    fig.savefig(FIG_DIR / "fig_learning_curves.png")
    plt.close(fig)
    print("  ✓ fig_learning_curves")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Viability collapse — reset vs carryover comparison
# ═══════════════════════════════════════════════════════════════
def fig_viability_collapse(e2_reset, e2_carry):
    """Bar chart: final task success under reset vs carryover for 3 reward regimes."""
    pe_reset = e2_reset["aggregated"]["phase_end"]
    pe_carry = e2_carry["aggregated"]["phase_end"]
    tasks = list(pe_reset[list(pe_reset.keys())[0]][-1]["task_metrics"].keys())

    # Map E2 agent names
    agents_e2 = ["A_task_only", "C_task_homeostatic", "D_pure_homeostatic"]
    labels_e2 = ["Task-Only", "HACE", "Pure Homeo."]
    colors_e2 = [COLORS["A_task_only"], COLORS["C_task_homeostatic"], COLORS["D_pure_homeostatic"]]

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)

    for ax_idx, (pe, mode_label) in enumerate([(pe_reset, "Reset"), (pe_carry, "Carryover")]):
        ax = axes[ax_idx]
        x = np.arange(len(tasks))
        n_agents = len(agents_e2)
        width = 0.78 / n_agents
        offsets = np.linspace(-0.39 + width/2, 0.39 - width/2, n_agents)

        for i, (agent, label, color) in enumerate(zip(agents_e2, labels_e2, colors_e2)):
            if agent not in pe:
                continue
            final = pe[agent][-1]
            means = [final["task_metrics"][t]["success_mean"] for t in tasks]
            stds  = [final["task_metrics"][t]["success_std"] for t in tasks]
            ax.bar(x + offsets[i], means, width=width, yerr=stds,
                   color=color, alpha=0.85, label=label if ax_idx == 0 else None,
                   edgecolor="white", linewidth=0.3,
                   error_kw={"linewidth": 0.7, "capsize": 2})

        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks],
                           rotation=40, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.1)
        ax.set_title(mode_label, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y")

    axes[0].set_ylabel("Success Rate")
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.savefig(FIG_DIR / "fig_viability_collapse.pdf")
    fig.savefig(FIG_DIR / "fig_viability_collapse.png")
    plt.close(fig)
    print("  ✓ fig_viability_collapse")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Boundary solvability + terminal energy (combined)
# ═══════════════════════════════════════════════════════════════
def fig_boundary_energy(e3):
    """Two-panel: boundary solvability bar + terminal energy bar per agent."""
    pe = e3["aggregated"]["phase_end"]
    tasks = e3["config"]["tasks"]

    agents_show = ["A_task_only", "C_hace", "F_ewc", "H_er", "I_hace_ewc"]

    # Compute avg boundary solvability and avg terminal energy per agent
    agent_bsolv = {}
    agent_energy = {}
    for agent in agents_show:
        if agent not in pe:
            continue
        bsolv_vals = []
        energy_vals = []
        for p in pe[agent]:
            b = p.get("boundary_solvability_mean")
            if b is not None and not math.isnan(b):
                bsolv_vals.append(b)
            e = p.get("policy_boundary_energy_mean")
            if e is not None and not math.isnan(e):
                energy_vals.append(e)
        agent_bsolv[agent] = np.mean(bsolv_vals) if bsolv_vals else 0
        agent_energy[agent] = np.mean(energy_vals) if energy_vals else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6))

    # Panel A: Boundary solvability
    x = np.arange(len(agents_show))
    bsolv_vals = [agent_bsolv.get(a, 0) for a in agents_show]
    bar_colors = [COLORS.get(a, "#999") for a in agents_show]
    bar_labels = [LABELS.get(a, a) for a in agents_show]

    bars = ax1.bar(x, bsolv_vals, color=bar_colors, edgecolor="white",
                   linewidth=0.5, alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels, rotation=35, ha="right", fontsize=7)
    ax1.set_ylabel("Avg. Boundary Solvability")
    ax1.set_title("(a) Boundary Solvability", fontsize=10)
    ax1.grid(True, axis="y")
    # Add value labels
    for bar, val in zip(bars, bsolv_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    # Panel B: Average terminal energy
    energy_vals = [agent_energy.get(a, 0) for a in agents_show]
    bars2 = ax2.bar(x, energy_vals, color=bar_colors, edgecolor="white",
                    linewidth=0.5, alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, rotation=35, ha="right", fontsize=7)
    ax2.set_ylabel("Avg. Terminal Energy")
    ax2.set_title("(b) Energy at Task Boundaries", fontsize=10)
    ax2.grid(True, axis="y")
    for bar, val in zip(bars2, energy_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=6.5)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIG_DIR / "fig_boundary_energy.pdf")
    fig.savefig(FIG_DIR / "fig_boundary_energy.png")
    plt.close(fig)
    print("  ✓ fig_boundary_energy")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Final performance heatmap (agents × tasks)
# ═══════════════════════════════════════════════════════════════
def fig_performance_heatmap(e3):
    """Heatmap of end-of-sequence success rates: agents × tasks."""
    pe = e3["aggregated"]["phase_end"]
    tasks = e3["config"]["tasks"]

    agents_show = ["A_task_only", "B_energy_aware", "C_hace",
                   "D_pure_homeostatic", "E_task_oracle",
                   "F_ewc", "G_l2", "H_er", "I_hace_ewc"]

    matrix = []
    ylabels = []
    for agent in agents_show:
        if agent not in pe:
            continue
        final = pe[agent][-1]
        row = [final["task_metrics"][t]["success_mean"] for t in tasks]
        matrix.append(row)
        ylabels.append(LABELS.get(agent, agent))

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7.5)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5.5, color=color, fontweight="bold" if val > 0.8 else "normal")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Success Rate", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("End-of-Sequence Success Rate (10-Task α, Carryover)", fontsize=10)

    fig.savefig(FIG_DIR / "fig_performance_heatmap.pdf")
    fig.savefig(FIG_DIR / "fig_performance_heatmap.png")
    plt.close(fig)
    print("  ✓ fig_performance_heatmap")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Death rate comparison across agents
# ═══════════════════════════════════════════════════════════════
def fig_death_rate(e3):
    """Stacked bar of death rate across tasks for key agents."""
    pe = e3["aggregated"]["phase_end"]
    tasks = e3["config"]["tasks"]

    agents_show = ["A_task_only", "C_hace", "F_ewc", "I_hace_ewc"]

    fig, ax = plt.subplots(figsize=(7, 2.6))

    x = np.arange(len(tasks))
    n = len(agents_show)
    width = 0.82 / n
    offsets = np.linspace(-0.41 + width/2, 0.41 - width/2, n)

    for i, agent in enumerate(agents_show):
        if agent not in pe:
            continue
        final = pe[agent][-1]
        deaths = [final["task_metrics"][t]["death_mean"] for t in tasks]
        ax.bar(x + offsets[i], deaths, width=width,
               color=COLORS.get(agent, "#999"), alpha=0.85,
               label=LABELS.get(agent, agent),
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks],
                       rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel("Death Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Agent Death Rate per Task (Carryover)", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
    ax.grid(True, axis="y")

    fig.savefig(FIG_DIR / "fig_death_rate.pdf")
    fig.savefig(FIG_DIR / "fig_death_rate.png")
    plt.close(fig)
    print("  ✓ fig_death_rate")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    setup_style()

    print("Loading data...")
    e2_reset = load_json(E2_RESET)
    e2_carry = load_json(E2_CARRY)
    e3_alpha = load_json(E3_ALPHA)

    print("\n── Data Validation ──")
    validate_and_print("E2 Reset", e2_reset)
    validate_and_print("E2 Carryover", e2_carry)
    validate_and_print("E3 Alpha Carryover", e3_alpha)

    print("\n── Generating Figures ──")
    fig_learning_curves(e3_alpha)
    fig_viability_collapse(e2_reset, e2_carry)
    fig_boundary_energy(e3_alpha)
    fig_performance_heatmap(e3_alpha)
    fig_death_rate(e3_alpha)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
