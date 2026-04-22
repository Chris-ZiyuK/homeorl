#!/bin/bash
#SBATCH --job-name=coom_hace_cpu
#SBATCH --output=experiments/coom/logs/%x_%A_%a.out
#SBATCH --error=experiments/coom/logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=batch
# normal QoS: myaccount often shows cpu=64 for this partition — no GPU.
#SBATCH --cpus-per-task=64
# ViZDoom uses RAM; raise if OOM (many n_envs in coom_experiment_oscar_batch.yaml).
#SBATCH --mem=128G
#SBATCH --array=0-0

# CPU-only COOM training (ViZDoom-bound; GPU rarely helps). Uses configs/coom_experiment_oscar_batch.yaml.
#
#   sbatch experiments/coom/run_coom_oscar_batch.sh
#
# Override:
#   CONFIG=configs/coom_experiment.yaml sbatch experiments/coom/run_coom_oscar_batch.sh

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}"
CONFIG="${CONFIG:-configs/coom_experiment_oscar_batch.yaml}"
SEED_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

cd "$PROJECT_DIR" || {
  echo "ERROR: cannot cd to PROJECT_DIR=$PROJECT_DIR" >&2
  exit 1
}

if command -v module >/dev/null 2>&1; then
  module load python/3.11.0 2>/dev/null || module load python/3.10.12 2>/dev/null || true
fi

if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  source venv/bin/activate
fi
mkdir -p experiments/coom/logs

echo "CPU batch job | CUDA not required | config=${CONFIG} seed_index=${SEED_INDEX}"
python -c "import torch; print('torch.cuda.is_available:', torch.cuda.is_available())" || true

EXTRA=(--config "${CONFIG}" --seed-index "${SEED_INDEX}")
[ "${PILOT:-0}" = "1" ] || [ "${PILOT:-}" = "true" ] && EXTRA+=(--pilot)
[ -n "${OUTPUT_DIR:-}" ] && EXTRA+=(--output-dir "${OUTPUT_DIR}")

python experiments/coom/train_coom.py "${EXTRA[@]}"
echo "Job complete seed_index ${SEED_INDEX}"
