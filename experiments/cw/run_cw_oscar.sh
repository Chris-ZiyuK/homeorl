#!/bin/bash
#   Submit default config:
#     sbatch experiments/cw/run_cw_oscar.sh
#
#   Submit a specific YAML config:
#     CONFIG=configs/cw_baseline.yaml sbatch --array=0-0 experiments/cw/run_cw_oscar.sh
#
#   Run a quick pilot:
#     PILOT=1 CONFIG=configs/cw_baseline.yaml sbatch --array=0-0 experiments/cw/run_cw_oscar.sh
#
#   Run one agent from the YAML:
#     AGENT=vanilla CONFIG=configs/cw_baseline.yaml sbatch --array=0-0 experiments/cw/run_cw_oscar.sh
#
#   Outputs:
#     experiments/cw/results/<experiment>/<agent>/seed_<id>/
#     experiments/cw/logs/

#SBATCH --job-name=cw_hace
#SBATCH --output=experiments/cw/logs/%x_%A_%a.out
#SBATCH --error=experiments/cw/logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-2
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/homeorl}}"
CONFIG="${CONFIG:-configs/cw_experiment.yaml}"
SEED_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
AGENT="${AGENT:-}"

cd "$PROJECT_DIR" || {
  echo "ERROR: cannot cd to PROJECT_DIR=$PROJECT_DIR" >&2
  exit 1
}

# Load conda + MuJoCo/Mesa environment.
source "$PROJECT_DIR/scripts/load_homeorl_env.sh"

mkdir -p experiments/cw/logs

echo "Config: $CONFIG"
echo "Seed index: $SEED_INDEX"
echo "Agent: ${AGENT:-all}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"


EXTRA=(--config "${CONFIG}" --seed-index "${SEED_INDEX}")

if [ "${PILOT:-0}" = "1" ] || [ "${PILOT:-}" = "true" ]; then
  EXTRA+=(--pilot)
fi

if [ -n "${OUTPUT_DIR:-}" ]; then
  EXTRA+=(--output-dir "${OUTPUT_DIR}")
fi

if [ -n "${AGENT}" ]; then
  EXTRA+=(--agent "${AGENT}")
fi

python -u experiments/cw/train_cw.py "${EXTRA[@]}"

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
