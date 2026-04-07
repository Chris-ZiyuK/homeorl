"""Analysis tools for mechanistic explanation of homeostatic RL."""

from src.analysis.linear_probe import (
    collect_representations,
    train_linear_probe,
    probe_across_training,
)
from src.analysis.cka import linear_cka, compute_cka_matrix
from src.analysis.energy_correlation import analyze_energy_encoding

__all__ = [
    "collect_representations",
    "train_linear_probe",
    "probe_across_training",
    "linear_cka",
    "compute_cka_matrix",
    "analyze_energy_encoding",
]
