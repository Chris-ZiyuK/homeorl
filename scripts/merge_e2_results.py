#!/usr/bin/env python3
"""
Merge per-seed E2 results into a single aggregated JSON + publication-ready plots.

Usage:
    python scripts/merge_e2_results.py

This script:
1. Loads all per-seed JSON files from e2_reset/ and e2_carryover/
2. Aggregates into the same format as the multi-seed runner
3. Computes derived metrics: forgetting, forward transfer, boundary solvability
4. Produces publication-ready plots
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_sequential_tasks import aggregate_runs, plot_results


TASKS = ["reach", "recharge", "hazard_reach", "detour", "tight_detour"]
AGENTS = ["A_task_only", "B_energy_aware", "C_task_homeostatic", "D_pure_homeostatic", "E_task_oracle"]


def load_per_seed_runs(results_dir: Path) -> list[dict]:
    """Load all per-seed JSON files and extract raw_runs."""
    all_runs = []
    json_files = sorted(results_dir.glob("*_seed*.json"))
    if not json_files:
        # Try loading single aggregated file
        for f in results_dir.glob("*.json"):
            with open(f, "r") as fh:
                data = json.load(fh)
            if "raw_runs" in data:
                all_runs.extend(data["raw_runs"])
        return all_runs

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
        if "raw_runs" in data:
            all_runs.extend(data["raw_runs"])

    return all_runs


def compute_derived_metrics(aggregated: dict, tasks: list[str]) -> dict:
    """Compute forgetting, forward transfer, and boundary solvability summary."""
    derived = {}

    for agent, phases in aggregated["phase_end"].items():
        agent_metrics = {
            "forgetting": {},
            "boundary_solvability": [],
        }

        # --- Forgetting: peak success during training - final success ---
        for task_name in tasks:
            peak_success = 0.0
            final_success = 0.0
            for phase in phases:
                task_success = phase["task_metrics"][task_name]["success_mean"]
                if task_success > peak_success:
                    peak_success = task_success
                final_success = task_success
            agent_metrics["forgetting"][task_name] = round(peak_success - final_success, 4)

        # --- Boundary solvability across phases ---
        for phase in phases:
            bs = phase.get("boundary_solvability_mean")
            if bs is not None and not np.isnan(bs):
                agent_metrics["boundary_solvability"].append({
                    "phase": phase["phase_task"],
                    "solvability": round(bs, 4),
                    "solvability_std": round(phase.get("boundary_solvability_std", 0), 4),
                })

        derived[agent] = agent_metrics

    return derived


def plot_factorial_comparison(reset_agg: dict, carryover_agg: dict, output_dir: Path):
    """Generate the main E2 factorial comparison figure (Fig. 2 in the paper plan)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("E2 Factorial: Reward Regime × Boundary Mode", fontsize=16, fontweight="bold")

    core_agents = ["A_task_only", "C_task_homeostatic", "D_pure_homeostatic"]
    agent_labels = {
        "A_task_only": "Task-only",
        "C_task_homeostatic": "Task + Homeostatic",
        "D_pure_homeostatic": "Pure Homeostatic",
    }
    colors = {
        "A_task_only": "#4285f4",
        "C_task_homeostatic": "#34a853",
        "D_pure_homeostatic": "#ea4335",
    }

    for col_idx, (mode_name, agg) in enumerate([("Reset", reset_agg), ("Carryover", carryover_agg)]):
        # Row 0: Success rate learning curves
        ax_success = axes[0, col_idx]
        for agent in core_agents:
            if agent not in agg["current"]:
                continue
            series = agg["current"][agent]
            x = [row["global_episode"] for row in series]
            y = [row["success_mean"] for row in series]
            std = [row["success_std"] for row in series]
            ax_success.plot(x, y, label=agent_labels[agent], color=colors[agent], linewidth=2)
            ax_success.fill_between(
                x,
                np.maximum(0, np.array(y) - np.array(std)),
                np.minimum(1, np.array(y) + np.array(std)),
                color=colors[agent], alpha=0.12,
            )
        for i in range(1, len(TASKS)):
            ax_success.axvline(i * 600, color="#9ca3af", linestyle="--", lw=0.8)
        ax_success.set_title(f"Success Rate ({mode_name})", fontweight="bold")
        ax_success.set_ylim(0, 1.05)
        ax_success.set_xlabel("Global Episode")
        ax_success.set_ylabel("Success Rate")
        ax_success.legend(fontsize=8)
        ax_success.grid(True, alpha=0.3)

        # Row 1: Final evaluation bar chart
        ax_bar = axes[1, col_idx]
        x_pos = np.arange(len(TASKS))
        width = 0.25
        offsets = [-width, 0, width]
        for offset, agent in zip(offsets, core_agents):
            if agent not in agg["phase_end"]:
                continue
            means = []
            stds = []
            for task_idx, t in enumerate(TASKS):
                phase = agg["phase_end"][agent][task_idx]
                means.append(phase["task_metrics"][t]["success_mean"])
                stds.append(phase["task_metrics"][t]["success_std"])
            ax_bar.bar(
                x_pos + offset, means, width=width, yerr=stds,
                label=agent_labels[agent], color=colors[agent], alpha=0.85,
            )
        ax_bar.set_title(f"Final Evaluation ({mode_name})", fontweight="bold")
        ax_bar.set_xticks(x_pos, TASKS, rotation=30, ha="right")
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_ylabel("Success Rate")
        ax_bar.grid(True, axis="y", alpha=0.3)
        if col_idx == 0:
            ax_bar.legend(fontsize=8)

    # Col 2: Boundary solvability comparison
    ax_bs = axes[0, 2]
    for agent in core_agents:
        reset_bs = []
        carryover_bs = []
        for agg, lst in [(reset_agg, reset_bs), (carryover_agg, carryover_bs)]:
            if agent in agg["phase_end"]:
                for phase in agg["phase_end"][agent]:
                    bs = phase.get("boundary_solvability_mean")
                    if bs is not None and not np.isnan(bs):
                        lst.append(bs)
        mode_means = [np.mean(reset_bs) if reset_bs else 0, np.mean(carryover_bs) if carryover_bs else 0]
        x_modes = [0, 1]
        ax_bs.bar([x + core_agents.index(agent) * 0.2 - 0.2 for x in x_modes],
                  mode_means, width=0.18, label=agent_labels[agent], color=colors[agent], alpha=0.85)
    ax_bs.set_title("Boundary Solvability", fontweight="bold")
    ax_bs.set_xticks([0, 1], ["Reset", "Carryover"])
    ax_bs.set_ylim(0, 1.05)
    ax_bs.set_ylabel("Solvability")
    ax_bs.legend(fontsize=8)
    ax_bs.grid(True, axis="y", alpha=0.3)

    # Col 2, Row 1: Energy at boundary
    ax_energy = axes[1, 2]
    for agent in core_agents:
        for mode_name_inner, agg, x_offset in [("Reset", reset_agg, 0), ("Carryover", carryover_agg, 1)]:
            if agent in agg["phase_end"]:
                energies = []
                for phase in agg["phase_end"][agent]:
                    e = phase.get("policy_boundary_energy_mean")
                    if e is not None and not np.isnan(e):
                        energies.append(e)
                if energies:
                    ax_energy.bar(
                        x_offset + core_agents.index(agent) * 0.2 - 0.2,
                        np.mean(energies), width=0.18,
                        label=f"{agent_labels[agent]} ({mode_name_inner})" if x_offset == 0 else None,
                        color=colors[agent], alpha=0.5 + 0.35 * x_offset,
                    )
    ax_energy.set_title("Avg Terminal Energy", fontweight="bold")
    ax_energy.set_xticks([0, 1], ["Reset", "Carryover"])
    ax_energy.set_ylabel("Energy")
    ax_energy.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "e2_factorial_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'e2_factorial_comparison.png'}")


def main():
    results_base = ROOT / "experiments" / "sequential_results"
    reset_dir = results_base / "e2_reset"
    carryover_dir = results_base / "e2_carryover"

    print("Loading E2 reset results...")
    reset_runs = load_per_seed_runs(reset_dir)
    print(f"  Found {len(reset_runs)} runs")

    print("Loading E2 carryover results...")
    carryover_runs = load_per_seed_runs(carryover_dir)
    print(f"  Found {len(carryover_runs)} runs")

    if not reset_runs or not carryover_runs:
        print("ERROR: No results found. Run the experiments first.")
        sys.exit(1)

    # Aggregate
    reset_agg = aggregate_runs(reset_runs, TASKS)
    carryover_agg = aggregate_runs(carryover_runs, TASKS)

    # Save merged aggregated results
    for mode, runs, agg in [("reset", reset_runs, reset_agg), ("carryover", carryover_runs, carryover_agg)]:
        out_dir = results_base / f"e2_{mode}"
        merged = {"config": {"boundary_mode": mode, "num_seeds": len(runs) // len(AGENTS)},
                  "aggregated": agg, "raw_runs": runs}
        with open(out_dir / f"e2_{mode}_merged.json", "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Saved merged: {out_dir / f'e2_{mode}_merged.json'}")

    # Compute derived metrics
    for mode, agg in [("reset", reset_agg), ("carryover", carryover_agg)]:
        derived = compute_derived_metrics(agg, TASKS)
        out_dir = results_base / f"e2_{mode}"
        with open(out_dir / f"e2_{mode}_derived.json", "w") as f:
            json.dump(derived, f, indent=2)
        print(f"\n--- {mode.upper()} Derived Metrics ---")
        for agent, metrics in derived.items():
            print(f"  {agent}:")
            print(f"    Forgetting: {metrics['forgetting']}")
            print(f"    Boundary Solvability: {metrics['boundary_solvability']}")

    # Generate publication plots
    plot_factorial_comparison(reset_agg, carryover_agg, results_base)

    # Also generate individual mode plots
    for mode, agg in [("reset", reset_agg), ("carryover", carryover_agg)]:
        cfg = {
            "tasks": TASKS,
            "episodes_per_task": 600,
            "boundary_mode": mode,
            "logging": {
                "save_dir": str(results_base / f"e2_{mode}"),
                "plot_name": f"e2_{mode}.png",
                "json_name": f"e2_{mode}.json",
            },
        }
        plot_results(agg, cfg)
        print(f"Saved: {results_base / f'e2_{mode}' / f'e2_{mode}.png'}")

    print("\n✅ Merge complete!")


if __name__ == "__main__":
    main()
