#!/bin/bash
#SBATCH --job-name=coom_hace
#SBATCH --output=experiments/coom/logs/%x_%A_%a.out
#SBATCH --error=experiments/coom/logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-0
# One task per seed_index; each job runs all agents in CONFIG for that seed (edit array to num_seeds-1).

set -eo pipefail

PROJECT_DIR="/users/$USER/homeorl"  # Update to your Oscar clone path
CONFIG="${CONFIG:-configs/coom_experiment.yaml}"
SEED_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

cd "$PROJECT_DIR"

# Oscar Lmod names change over time; failed loads must not abort (set -e).
# If everything 404s, use CCV system Python + your .venv (see docs.ccv.brown.edu/oscar/software/python-on-oscar).
# Pick exact names with: module spider python   and   module spider cuda
if command -v module >/dev/null 2>&1; then
  module load python/3.11.0 2>/dev/null || module load python/3.10.12 2>/dev/null || true
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

python experiments/coom/train_coom.py "${EXTRA[@]}"
echo "Job complete seed_index ${SEED_INDEX}"
