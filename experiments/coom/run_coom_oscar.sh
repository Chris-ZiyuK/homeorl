#!/bin/bash
#SBATCH --job-name=coom_hace
#SBATCH --output=experiments/coom/logs/%x_%A_%a.out
#SBATCH --error=experiments/coom/logs/%x_%A_%a.err
#SBATCH --mem=32G
# If you get QOSMaxCpuPerUserLimit: norm-gpu QoS is often cpu=12 *per user* (see `myaccount`).
# Concurrent GPU jobs share that cap—e.g. two jobs cannot each use 12 CPUs at once.
#SBATCH --cpus-per-task=6
# ViZDoom: keep ppo_config.n_envs below ~cpus-per-task (see coom_experiment.yaml).
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-0
# One task per seed_index. Default CONFIG runs CO8 (continual sequence). Use AGENT=... for one agent.
# Edit array to num_seeds-1 for multi-seed sweeps.
#
# CPU-only / many cores (normal QoS, no GPU): experiments/coom/run_coom_oscar_batch.sh
#
# COOM (once on login node, venv on): bash scripts/install_coom_editable.sh /path/to/COOM
#   (creates <clone>/COOM/__init__.py if missing, then pip install -e)

set -eo pipefail

# Oscar homes are often /oscar/home/$USER — not /users/$USER. Submit from repo root
# (cd .../homeorl && sbatch ...) so SLURM_SUBMIT_DIR is correct, or: export PROJECT_DIR=/oscar/home/$USER/homeorl
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}"
# Default continual-learning config (DQN + CO8). Override for single-task runs, e.g.:
#   CONFIG=configs/coom_experiment.yaml sbatch experiments/coom/run_coom_oscar.sh
CONFIG="${CONFIG:-configs/coom_experiment_co8_oscar.yaml}"
SEED_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
AGENT="${AGENT:-}"

cd "$PROJECT_DIR" || {
  echo "ERROR: cannot cd to PROJECT_DIR=$PROJECT_DIR" >&2
  exit 1
}

# Oscar Lmod names change over time; failed loads must not abort (set -e).
# If everything 404s, use CCV system Python + your .venv (see docs.ccv.brown.edu/oscar/software/python-on-oscar).
# Pick exact names with: module spider python   and   module spider cuda
if command -v module >/dev/null 2>&1; then
  module load python/3.11.11-5e66 2>/dev/null || module load python/3.11.0 2>/dev/null || module load python/3.10.12 2>/dev/null || true
  module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true
fi

if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  source venv/bin/activate
fi
mkdir -p experiments/coom/logs

echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-$SEED_INDEX} | config=${CONFIG} seed_index=${SEED_INDEX}"

EXTRA=(--config "${CONFIG}" --seed-index "${SEED_INDEX}")
[ "${PILOT:-0}" = "1" ] || [ "${PILOT:-}" = "true" ] && EXTRA+=(--pilot)
[ -n "${OUTPUT_DIR:-}" ] && EXTRA+=(--output-dir "${OUTPUT_DIR}")
[ -n "${AGENT}" ] && EXTRA+=(--agent "${AGENT}")

python experiments/coom/train_coom.py "${EXTRA[@]}"
echo "Job complete seed_index ${SEED_INDEX}"
