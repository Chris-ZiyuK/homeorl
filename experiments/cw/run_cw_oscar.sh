#!/bin/bash
#SBATCH --job-name=cw_hace
#SBATCH --output=experiments/cw/logs/%x_%A_%a.out
#SBATCH --error=experiments/cw/logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2
# One task per seed index by default. Set --array=0-(num_seeds-1).

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}"
CONFIG="${CONFIG:-configs/cw_experiment.yaml}"
SEED_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
AGENT="${AGENT:-}"

cd "$PROJECT_DIR" || {
  echo "ERROR: cannot cd to PROJECT_DIR=$PROJECT_DIR" >&2
  exit 1
}

if command -v module >/dev/null 2>&1; then
  module load python/3.10.12 2>/dev/null || module load python/3.11.0 2>/dev/null || true
  module load gcc/11.3.0 2>/dev/null || module load gcc/10.2 2>/dev/null || true
fi

if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "ERROR: .venv missing. Run scripts/setup_cw_env_oscar.sh first." >&2
  exit 1
fi

mkdir -p experiments/cw/logs

python - <<'PY'
import importlib
mods = ("stable_baselines3", "metaworld", "mujoco_py")
for m in mods:
    importlib.import_module(m)
print("CW deps import check OK")
PY

EXTRA=(--config "${CONFIG}" --seed-index "${SEED_INDEX}")
[ "${PILOT:-0}" = "1" ] || [ "${PILOT:-}" = "true" ] && EXTRA+=(--pilot)
[ -n "${OUTPUT_DIR:-}" ] && EXTRA+=(--output-dir "${OUTPUT_DIR}")
[ -n "${AGENT}" ] && EXTRA+=(--agent "${AGENT}")

echo "CW job ${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-$SEED_INDEX} config=${CONFIG} seed_index=${SEED_INDEX} agent=${AGENT:-all}"
python experiments/cw/train_cw.py "${EXTRA[@]}"
