"""
Mechanistic analysis runner for HomeORL experiments.

After training completes and checkpoints are saved, this script runs:
  1. Linear probe analysis (what info is encoded in hidden layers)
  2. CKA stability analysis (do representations persist across tasks)
  3. Energy correlation analysis (why homeostatic agents transfer better)

Usage:
    python experiments/run_analysis.py \\
        --checkpoint_dir experiments/sequential_results/e3_alpha/checkpoints \\
        --task_sequence alpha \\
        --agents C_hace A_task_only F_ewc \\
        --output_dir experiments/analysis_results/e3_alpha
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.sequential_homeostasis_env import (
    SequentialHomeostasisEnv,
    TASK_SEQUENCES,
)
from src.agents.dqn_agent import DQNAgent
from src.agents.ewc_agent import EWCAgent
from src.analysis.linear_probe import collect_representations, train_linear_probe
from src.analysis.cka import linear_cka
from src.analysis.energy_correlation import analyze_energy_encoding


# ── NeurIPS-style plot settings ─────────────────────────────

def setup_plot_style():
    """Configure matplotlib/seaborn for NeurIPS-quality figures."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.8,
    })


# ── Probe Analysis ──────────────────────────────────────────

def run_probe_analysis(
    checkpoint_dir: Path,
    agents: list[str],
    task_sequence: list[str],
    output_dir: Path,
    n_episodes: int = 150,
):
    """Run linear probes on all agent checkpoints."""
    print("\n=== Linear Probe Analysis ===")
    env_kwargs = {"observation_mode": "full"}
    probe_labels = ["energy_level", "energy_low", "food_relevant", "hazard_nearby"]

    all_results = {}

    for agent_name in agents:
        print(f"\n  Agent: {agent_name}")
        agent_ckpt_dirs = sorted(checkpoint_dir.glob(f"{agent_name}_seed*"))

        if not agent_ckpt_dirs:
            print(f"    No checkpoints found, skipping.")
            continue

        # Use first seed for probe analysis
        ckpt_dir = agent_ckpt_dirs[0]
        agent_results = {}

        for ckpt_file in sorted(ckpt_dir.glob("*.pt")):
            phase_name = ckpt_file.stem  # e.g., "after_reach"
            agent = DQNAgent(obs_dim=19, n_actions=4)
            state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            agent.q_net.load_state_dict(state)

            phase_results = {}
            for task_name in task_sequence[:5]:  # probe on first 5 tasks
                hiddens, labels = collect_representations(
                    agent, SequentialHomeostasisEnv, task_name,
                    env_kwargs, n_episodes=n_episodes,
                )

                task_scores = {}
                for label_name in probe_labels:
                    y = labels[label_name]
                    unique = np.unique(y)
                    ptype = "classification" if len(unique) <= 5 else "regression"
                    result = train_linear_probe(hiddens, y, task=ptype)
                    task_scores[label_name] = result["test_score"]

                phase_results[task_name] = task_scores

            agent_results[phase_name] = phase_results
            print(f"    {phase_name}: energy_R²={phase_results[task_sequence[0]]['energy_level']:.3f}")

        all_results[agent_name] = agent_results

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot probe comparison
    _plot_probe_comparison(all_results, task_sequence, output_dir)

    return all_results


def _plot_probe_comparison(results, task_sequence, output_dir):
    """Plot energy encoding R² across agents and training phases."""
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Energy R² by agent across phases
    colors = sns.color_palette("Set2", n_colors=len(results))
    for idx, (agent_name, phases) in enumerate(results.items()):
        phase_names = list(phases.keys())
        r2_values = [
            phases[p].get(task_sequence[0], {}).get("energy_level", 0)
            for p in phase_names
        ]
        axes[0].plot(range(len(phase_names)), r2_values,
                     marker="o", label=agent_name, color=colors[idx])

    axes[0].set_xlabel("Training Phase")
    axes[0].set_ylabel("Energy R² (Linear Probe)")
    axes[0].set_title(f"Energy Encoding on '{task_sequence[0]}'")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-0.1, 1.1)

    # Right: Grouped bar chart of all probe labels for final checkpoint
    labels = ["energy_level", "energy_low", "food_relevant", "hazard_nearby"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(results), 1)

    for idx, (agent_name, phases) in enumerate(results.items()):
        last_phase = list(phases.keys())[-1] if phases else None
        if last_phase is None:
            continue
        # Average across tasks
        scores = []
        for label in labels:
            task_scores = [
                phases[last_phase].get(t, {}).get(label, 0)
                for t in task_sequence[:5]
            ]
            scores.append(np.mean(task_scores))

        offset = (idx - len(results) / 2 + 0.5) * width
        axes[1].bar(x + offset, scores, width=width * 0.9,
                    label=agent_name, color=colors[idx])

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
    axes[1].set_ylabel("Probe Score")
    axes[1].set_title("Final Checkpoint — All Probes")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "probe_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "probe_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved probe comparison to {output_dir}/probe_comparison.pdf")


# ── CKA Stability Analysis ─────────────────────────────────

def run_cka_analysis(
    checkpoint_dir: Path,
    agents: list[str],
    task_sequence: list[str],
    output_dir: Path,
    n_episodes: int = 100,
):
    """Compute CKA stability across training phases."""
    print("\n=== CKA Stability Analysis ===")
    env_kwargs = {"observation_mode": "full"}
    reference_task = task_sequence[0]

    all_cka = {}

    for agent_name in agents:
        agent_ckpt_dirs = sorted(checkpoint_dir.glob(f"{agent_name}_seed*"))
        if not agent_ckpt_dirs:
            continue

        ckpt_dir = agent_ckpt_dirs[0]
        ckpt_files = sorted(ckpt_dir.glob("*.pt"))

        # Collect representations on reference task at each checkpoint
        reps = {}
        for ckpt_file in ckpt_files:
            phase_name = ckpt_file.stem
            agent = DQNAgent(obs_dim=19, n_actions=4)
            state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            agent.q_net.load_state_dict(state)

            hiddens, _ = collect_representations(
                agent, SequentialHomeostasisEnv, reference_task,
                env_kwargs, n_episodes=n_episodes,
            )
            reps[phase_name] = hiddens

        # Compute pairwise CKA
        names = list(reps.keys())
        cka_matrix = np.zeros((len(names), len(names)))
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                cka_matrix[i, j] = linear_cka(reps[n1], reps[n2])

        all_cka[agent_name] = {"names": names, "matrix": cka_matrix.tolist()}
        print(f"  {agent_name}: {len(names)} checkpoints, "
              f"first-last CKA={cka_matrix[0, -1]:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot CKA heatmaps
    _plot_cka_heatmaps(all_cka, reference_task, output_dir)

    return all_cka


def _plot_cka_heatmaps(all_cka, reference_task, output_dir):
    """Plot CKA stability heatmaps for each agent."""
    setup_plot_style()
    n_agents = len(all_cka)
    if n_agents == 0:
        return

    fig, axes = plt.subplots(1, n_agents, figsize=(5 * n_agents, 4))
    if n_agents == 1:
        axes = [axes]

    for idx, (agent_name, data) in enumerate(all_cka.items()):
        matrix = np.array(data["matrix"])
        short_names = [n.replace("after_", "") for n in data["names"]]
        sns.heatmap(
            matrix, ax=axes[idx], annot=True, fmt=".2f",
            xticklabels=short_names, yticklabels=short_names,
            vmin=0, vmax=1, cmap="YlOrRd",
            annot_kws={"fontsize": 7},
        )
        axes[idx].set_title(f"{agent_name}\nCKA on '{reference_task}'", fontsize=10)
        axes[idx].tick_params(axis="both", labelsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "cka_stability.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "cka_stability.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved CKA heatmaps to {output_dir}/cka_stability.pdf")


# ── Energy Encoding Comparison ──────────────────────────────

def run_energy_analysis(
    checkpoint_dir: Path,
    agents: list[str],
    task_sequence: list[str],
    output_dir: Path,
    n_episodes: int = 150,
):
    """Compare energy encoding strength across agents."""
    print("\n=== Energy Encoding Analysis ===")
    env_kwargs = {"observation_mode": "full"}

    all_energy = {}

    for agent_name in agents:
        agent_ckpt_dirs = sorted(checkpoint_dir.glob(f"{agent_name}_seed*"))
        if not agent_ckpt_dirs:
            continue

        # Use final checkpoint
        ckpt_dir = agent_ckpt_dirs[0]
        ckpt_files = sorted(ckpt_dir.glob("*.pt"))
        if not ckpt_files:
            continue

        final_ckpt = ckpt_files[-1]
        agent = DQNAgent(obs_dim=19, n_actions=4)
        state = torch.load(final_ckpt, map_location="cpu", weights_only=True)
        agent.q_net.load_state_dict(state)

        analysis = analyze_energy_encoding(
            agent, SequentialHomeostasisEnv, task_sequence[:5],
            env_kwargs, n_episodes=n_episodes,
        )
        all_energy[agent_name] = analysis
        avg_r2 = np.mean([v["r2"] for v in analysis.values()])
        print(f"  {agent_name}: avg_energy_R²={avg_r2:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_energy_comparison(all_energy, task_sequence, output_dir)

    return all_energy


def _plot_energy_comparison(all_energy, task_sequence, output_dir):
    """Bar chart comparing energy R² across agents and tasks."""
    setup_plot_style()

    tasks = task_sequence[:5]
    agents = list(all_energy.keys())
    if not agents:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(tasks))
    width = 0.8 / max(len(agents), 1)
    colors = sns.color_palette("Set2", n_colors=len(agents))

    for idx, agent_name in enumerate(agents):
        r2_values = [all_energy[agent_name].get(t, {}).get("r2", 0.0) for t in tasks]
        offset = (idx - len(agents) / 2 + 0.5) * width
        ax.bar(x + offset, r2_values, width=width * 0.9,
               label=agent_name, color=colors[idx])

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Energy R² (Linear Probe)", fontsize=10)
    ax.set_title("Energy Encoding Strength by Agent", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.1)

    fig.tight_layout()
    fig.savefig(output_dir / "energy_encoding.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "energy_encoding.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved energy encoding comparison to {output_dir}/energy_encoding.pdf")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HomeORL mechanistic analysis")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task_sequence", type=str, default="alpha",
                        choices=list(TASK_SEQUENCES.keys()))
    parser.add_argument("--agents", type=str, nargs="+",
                        default=["A_task_only", "C_hace", "F_ewc"])
    parser.add_argument("--output_dir", type=str,
                        default="experiments/analysis_results")
    parser.add_argument("--n_episodes", type=int, default=150)
    parser.add_argument("--skip_probe", action="store_true")
    parser.add_argument("--skip_cka", action="store_true")
    parser.add_argument("--skip_energy", action="store_true")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    tasks = TASK_SEQUENCES[args.task_sequence]

    if not args.skip_probe:
        run_probe_analysis(ckpt_dir, args.agents, tasks, out_dir, args.n_episodes)

    if not args.skip_cka:
        run_cka_analysis(ckpt_dir, args.agents, tasks, out_dir, args.n_episodes)

    if not args.skip_energy:
        run_energy_analysis(ckpt_dir, args.agents, tasks, out_dir, args.n_episodes)

    print(f"\n✅ All analysis complete. Output: {out_dir}/")


if __name__ == "__main__":
    main()
