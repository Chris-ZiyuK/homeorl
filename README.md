# Homeostatic Grounding

> **Do Internal Energy States Help RL Agents Learn the Meaning of Symbols?**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project investigates whether analogous internal energy states (homeostatic signals) can accelerate **functional symbol grounding** in reinforcement learning agents operating under partial observability.

We compare three agent conditions in a controlled survival environment:
- **Group A (Terminal-Only):** Baseline with only sparse terminal reward
- **Group B (Energy-Aware):** Agent observes internal energy, providing immediate feedback
- **Group C (Energy + Query):** Energy observation plus queryable external knowledge

**Central Hypothesis:** Energy-aware agents develop correct object–effect associations significantly faster than terminal-reward-only agents.

**Course:** CS 2951X — Reintegrating Artificial Intelligence (Brown University, Spring 2026)

## Project Structure

```
.
├── docs/               # Research proposal, survey reports, references
├── paper/              # LaTeX paper drafts and figures
├── src/                # Core source code (envs, agents, metrics, utils)
├── configs/            # YAML experiment configurations
├── experiments/        # Experiment entry-point scripts
├── notebooks/          # Jupyter notebooks for analysis
├── results/            # Output: figures, logs, checkpoints
├── tests/              # Unit tests
└── scripts/            # Helper & tutorial scripts
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url> && cd homeostatic-grounding

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run an Experiment

```bash
# Run main experiment with default config
python experiments/run_experiment.py --config configs/default.yaml

# Run multi-seed experiment
python experiments/run_multiseed.py --config configs/default.yaml --seeds 10
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| **FGS** | Functional Grounding Score: P(approach beneficial) − P(approach harmful) |
| **Grounding Speed** | Episodes until FGS > 0.5 |
| **Survival Rate** | Fraction of episodes without agent death |
| **Exit Rate** | Fraction of episodes reaching the exit |

## Citation

```bibtex
@misc{kong2026homeostatic,
  title={Homeostatic Grounding: Do Internal Energy States Help RL Agents Learn the Meaning of Symbols?},
  author={Kong, Ziyu},
  year={2026},
  note={CS 2951X Course Project, Brown University}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
