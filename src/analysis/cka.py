"""Centered Kernel Alignment (CKA) for comparing representations.

CKA measures similarity between two representation matrices. We use it to
answer: "Does the homeostatic agent's representation stay more stable
across task switches than the task-only agent's?"

A high CKA between representations on Task1 after training on Task1 vs
after training on Task5 means the agent retained its encoding structure.

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix."""
    n = K.shape[0]
    unit = np.ones((n, n)) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion (unbiased estimator)."""
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    return float(np.sum(Kc * Lc) / ((K.shape[0] - 1) ** 2))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear CKA between two representation matrices.

    Args:
        X: (n, d1) representation matrix from model/condition A.
        Y: (n, d2) representation matrix from model/condition B.

    Returns:
        CKA similarity in [0, 1]. 1.0 = identical representations.
    """
    assert X.shape[0] == Y.shape[0], "Same number of samples required."

    # Linear kernel
    K = X @ X.T
    L = Y @ Y.T

    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return float(hsic_kl / denom)


def compute_cka_matrix(
    checkpoint_dir: str | Path,
    agent_class,
    agent_kwargs: dict,
    env_class,
    env_kwargs: dict,
    task_sequence: list[str],
    reference_task: str | None = None,
    n_episodes: int = 100,
    seed_base: int = 42,
) -> dict:
    """Compute CKA similarity matrix across checkpoints and tasks.

    Produces two types of analysis:

    1. **Stability matrix**: For a fixed task (reference_task), how similar
       are the representations at different training stages?
       → High CKA = representation stability = less forgetting.

    2. **Cross-task matrix**: At a fixed checkpoint, how similar are
       representations across different tasks?
       → High CKA = task-agnostic representation = better transfer.

    Returns:
        dict with:
          "stability": {(ckpt_i, ckpt_j): cka_value}
          "cross_task": {ckpt: {(task_i, task_j): cka_value}}
          "checkpoint_names": list of checkpoint names
    """
    import torch
    from src.analysis.linear_probe import collect_representations

    checkpoint_dir = Path(checkpoint_dir)
    ckpt_files = sorted(checkpoint_dir.glob("*.pt"))

    if reference_task is None:
        reference_task = task_sequence[0]

    # Collect representations for each checkpoint on the reference task
    ckpt_reps: dict[str, np.ndarray] = {}
    ckpt_cross_reps: dict[str, dict[str, np.ndarray]] = {}

    for ckpt_file in ckpt_files:
        ckpt_name = ckpt_file.stem
        agent = agent_class(**agent_kwargs)
        state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        agent.q_net.load_state_dict(state)

        # Reference task representation
        hiddens, _ = collect_representations(
            agent, env_class, reference_task, env_kwargs, n_episodes, seed_base
        )
        ckpt_reps[ckpt_name] = hiddens

        # Cross-task representations
        ckpt_cross_reps[ckpt_name] = {}
        for task in task_sequence:
            h, _ = collect_representations(
                agent, env_class, task, env_kwargs, n_episodes, seed_base
            )
            ckpt_cross_reps[ckpt_name][task] = h

    # 1. Stability matrix
    names = list(ckpt_reps.keys())
    stability = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j >= i:
                cka = linear_cka(ckpt_reps[n1], ckpt_reps[n2])
                stability[(n1, n2)] = cka
                stability[(n2, n1)] = cka

    # 2. Cross-task matrix
    cross_task = {}
    for ckpt_name in names:
        cross_task[ckpt_name] = {}
        tasks = list(ckpt_cross_reps[ckpt_name].keys())
        for i, t1 in enumerate(tasks):
            for j, t2 in enumerate(tasks):
                if j >= i:
                    cka = linear_cka(
                        ckpt_cross_reps[ckpt_name][t1],
                        ckpt_cross_reps[ckpt_name][t2],
                    )
                    cross_task[ckpt_name][(t1, t2)] = cka
                    cross_task[ckpt_name][(t2, t1)] = cka

    return {
        "stability": stability,
        "cross_task": cross_task,
        "checkpoint_names": names,
    }
