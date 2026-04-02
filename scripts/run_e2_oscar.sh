#!/bin/bash
#SBATCH --job-name=e2_homeorl
#SBATCH --array=0-19
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiments/sequential_results/e2_logs/slurm_%A_%a.out
#SBATCH --error=experiments/sequential_results/e2_logs/slurm_%A_%a.err

# ============================================================
# E2 Factorial Experiment: SLURM Job Array
#
# Each array task runs one seed for all 5 agents in one boundary mode.
# Usage:
#   sbatch scripts/run_e2_oscar.sh reset
#   sbatch scripts/run_e2_oscar.sh carryover
#
# After both jobs complete, merge results:
#   python scripts/merge_e2_results.py
# ============================================================

set -euo pipefail

BOUNDARY_MODE="${1:?Usage: sbatch scripts/run_e2_oscar.sh [reset|carryover]}"
SEED_INDEX=${SLURM_ARRAY_TASK_ID}

echo "=== E2 | mode=${BOUNDARY_MODE} | seed_index=${SEED_INDEX} | $(date) ==="

# Navigate to project root
cd "${SLURM_SUBMIT_DIR:-.}"

# Create log directory
mkdir -p experiments/sequential_results/e2_logs

# Activate environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Select config based on boundary mode
if [ "${BOUNDARY_MODE}" = "reset" ]; then
    CONFIG="configs/e2_reset.yaml"
    OUTPUT_DIR="experiments/sequential_results/e2_reset"
elif [ "${BOUNDARY_MODE}" = "carryover" ]; then
    CONFIG="configs/e2_carryover.yaml"
    OUTPUT_DIR="experiments/sequential_results/e2_carryover"
else
    echo "ERROR: boundary_mode must be 'reset' or 'carryover', got '${BOUNDARY_MODE}'"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python experiments/run_sequential_tasks.py \
    --config "${CONFIG}" \
    --seed_index "${SEED_INDEX}" \
    --boundary_mode "${BOUNDARY_MODE}" \
    --output_dir "${OUTPUT_DIR}"

echo "=== Done | mode=${BOUNDARY_MODE} | seed_index=${SEED_INDEX} | $(date) ==="
