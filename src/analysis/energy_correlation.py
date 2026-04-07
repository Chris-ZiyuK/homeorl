"""Energy-encoding analysis for understanding WHY homeostatic agents transfer.

Core hypothesis: Homeostatic agents develop representations where energy
state is deeply and linearly encoded in hidden activations, because their
loss function directly depends on energy changes. This creates a
"free" representation that transfers to any new task sharing the same
energy dynamics — without needing to re-learn energy management.

This module provides:
  1. Energy R² comparison across agent types
  2. Neuron-level energy correlation analysis
  3. Energy encoding stability across task switches
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def analyze_energy_encoding(
    agent,
    env_class,
    tasks: list[str],
    env_kwargs: dict,
    n_episodes: int = 200,
    seed_base: int = 42,
) -> dict[str, Any]:
    """Analyze how strongly energy state is encoded in hidden representations.

    Returns per-task R² of linear regression from hidden → energy, plus
    neuron-level Pearson correlations with energy.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from src.analysis.linear_probe import collect_representations

    results: dict[str, Any] = {}

    for task_name in tasks:
        hiddens, labels = collect_representations(
            agent, env_class, task_name, env_kwargs, n_episodes, seed_base
        )
        energy_levels = labels["energy_level"]

        if len(hiddens) < 50:
            results[task_name] = {"r2": 0.0, "top_neurons": [], "n_samples": len(hiddens)}
            continue

        # Overall R²
        X_train, X_test, y_train, y_test = train_test_split(
            hiddens, energy_levels, test_size=0.2, random_state=seed_base
        )
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)
        r2 = float(reg.score(X_test, y_test))

        # Per-neuron Pearson correlation with energy
        n_neurons = hiddens.shape[1]
        correlations = np.zeros(n_neurons)
        for j in range(n_neurons):
            if np.std(hiddens[:, j]) > 1e-8:
                correlations[j] = float(
                    np.corrcoef(hiddens[:, j], energy_levels)[0, 1]
                )

        # Top-5 energy-correlated neurons
        top_indices = np.argsort(np.abs(correlations))[::-1][:5]
        top_neurons = [
            {"neuron": int(idx), "correlation": float(correlations[idx])}
            for idx in top_indices
        ]

        results[task_name] = {
            "r2": r2,
            "mean_abs_correlation": float(np.mean(np.abs(correlations))),
            "max_abs_correlation": float(np.max(np.abs(correlations))),
            "top_neurons": top_neurons,
            "n_samples": len(hiddens),
        }

    return results


def compare_energy_encoding(
    agents: dict[str, Any],
    env_class,
    tasks: list[str],
    env_kwargs: dict,
    n_episodes: int = 200,
) -> dict[str, dict[str, float]]:
    """Compare energy encoding R² across multiple agents.

    Args:
        agents: {agent_name: agent_instance}

    Returns:
        {agent_name: {task_name: r2_score}}
    """
    results = {}
    for name, agent in agents.items():
        analysis = analyze_energy_encoding(agent, env_class, tasks, env_kwargs, n_episodes)
        results[name] = {task: data["r2"] for task, data in analysis.items()}
    return results


def encoding_stability_across_tasks(
    checkpoint_dir: str | Path,
    agent_class,
    agent_kwargs: dict,
    env_class,
    env_kwargs: dict,
    reference_task: str,
    task_sequence: list[str],
    n_episodes: int = 100,
) -> dict[str, float]:
    """Measure how energy encoding R² on a reference task changes as the
    agent trains on subsequent tasks.

    If R² stays high → the agent retains energy representation (anti-forgetting).
    If R² drops → energy representation is being overwritten.

    Returns:
        {checkpoint_name: r2_on_reference_task}
    """
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    results = {}

    for ckpt_file in sorted(checkpoint_dir.glob("*.pt")):
        ckpt_name = ckpt_file.stem
        agent = agent_class(**agent_kwargs)
        state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        agent.q_net.load_state_dict(state)

        analysis = analyze_energy_encoding(
            agent, env_class, [reference_task], env_kwargs, n_episodes
        )
        results[ckpt_name] = analysis[reference_task]["r2"]

    return results
