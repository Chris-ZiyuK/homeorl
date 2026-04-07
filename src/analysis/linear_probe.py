"""Linear probe analysis for understanding agent representations.

Core idea: freeze the agent's hidden layers and train a simple linear
classifier/regressor on top. If the probe achieves high accuracy for a
particular property (e.g., "should I eat this food?"), it means that
property is linearly decodable from the agent's representation —
i.e., the agent has "learned" to encode that information.

Key questions this module answers:
  1. Does the homeostatic agent encode energy state better than task-only?
  2. Does this encoding persist across task switches (anti-forgetting)?
  3. Are "functional" representations (should-eat, should-avoid) more
     stable than positional ones?
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def collect_representations(
    agent,
    env_class,
    task_name: str,
    env_kwargs: dict,
    n_episodes: int = 200,
    seed_base: int = 42,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Rollout the agent and collect hidden activations + ground-truth labels.

    Returns:
        hidden_acts: (N, hidden_dim) array of second-layer activations.
        labels: dict mapping label names to (N,) arrays.
    """
    from src.envs.sequential_homeostasis_env import TASK_SPECS

    spec = TASK_SPECS[task_name]
    all_hiddens = []
    all_labels: dict[str, list] = {
        "energy_level": [],
        "energy_low": [],        # binary: energy < 0.4 * cap
        "food_relevant": [],     # binary: any food available
        "hazard_nearby": [],     # binary: manhattan dist to nearest hazard <= 1
        "task_id": [],
    }

    for ep in range(n_episodes):
        env = env_class(task_name=task_name, reward_mode="eval", **env_kwargs)
        obs, _ = env.reset(seed=seed_base + ep)
        done = False

        while not done:
            # Get hidden activation
            h = agent.get_hidden_activations(obs)
            all_hiddens.append(h.squeeze(0))

            # Ground-truth labels
            all_labels["energy_level"].append(env.energy / env.energy_cap)
            all_labels["energy_low"].append(
                1.0 if env.energy < 0.4 * env.energy_cap else 0.0
            )
            all_labels["food_relevant"].append(
                1.0 if any(env.food_available) else 0.0
            )

            # Hazard proximity
            min_hazard_dist = float("inf")
            for h_pos in env._hazard_set:
                d = abs(env.agent_pos[0] - h_pos[0]) + abs(env.agent_pos[1] - h_pos[1])
                min_hazard_dist = min(min_hazard_dist, d)
            all_labels["hazard_nearby"].append(
                1.0 if min_hazard_dist <= 1 else 0.0
            )

            all_labels["task_id"].append(float(TASK_SPECS.get(task_name, 0) is not None))

            # Step
            action = agent.select_action(obs, eps=0.05)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    hidden_acts = np.array(all_hiddens)
    labels = {k: np.array(v) for k, v in all_labels.items()}
    return hidden_acts, labels


def train_linear_probe(
    hidden_acts: np.ndarray,
    labels: np.ndarray,
    task: str = "classification",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, float]:
    """Train a linear probe on frozen hidden activations.

    Args:
        hidden_acts: (N, hidden_dim) representation matrix.
        labels: (N,) target labels.
        task: "classification" or "regression".
        test_ratio: fraction held out for evaluation.
        seed: random seed for the train/test split.

    Returns:
        dict with "train_score", "test_score", and "n_samples".
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        hidden_acts, labels, test_size=test_ratio, random_state=seed
    )

    if task == "classification":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=500, random_state=seed)
        clf.fit(X_train, y_train)
        return {
            "train_score": float(clf.score(X_train, y_train)),
            "test_score": float(clf.score(X_test, y_test)),
            "n_samples": len(hidden_acts),
        }
    else:
        from sklearn.linear_model import Ridge

        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)
        return {
            "train_score": float(reg.score(X_train, y_train)),
            "test_score": float(reg.score(X_test, y_test)),
            "n_samples": len(hidden_acts),
        }


def probe_across_training(
    checkpoint_dir: str | Path,
    agent_class,
    agent_kwargs: dict,
    env_class,
    env_kwargs: dict,
    task_sequence: list[str],
    probe_labels: list[str] | None = None,
    n_episodes: int = 100,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run probes on checkpoints saved at each task boundary.

    Returns:
        Nested dict: {checkpoint_name: {task: {label: test_score}}}
    """
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    if probe_labels is None:
        probe_labels = ["energy_level", "energy_low", "food_relevant", "hazard_nearby"]

    results: dict[str, dict[str, dict[str, float]]] = {}

    for ckpt_file in sorted(checkpoint_dir.glob("*.pt")):
        ckpt_name = ckpt_file.stem  # e.g., "agent_C_seed0_after_recharge"
        agent = agent_class(**agent_kwargs)
        state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        agent.q_net.load_state_dict(state)

        results[ckpt_name] = {}

        for task_name in task_sequence:
            hiddens, labels = collect_representations(
                agent, env_class, task_name, env_kwargs, n_episodes
            )

            task_results = {}
            for label_name in probe_labels:
                y = labels[label_name]
                # Determine classification vs regression
                unique_vals = np.unique(y)
                probe_task = "classification" if len(unique_vals) <= 5 else "regression"
                score = train_linear_probe(hiddens, y, task=probe_task)
                task_results[label_name] = score["test_score"]

            results[ckpt_name][task_name] = task_results

    return results
