#!/bin/bash
#SBATCH --job-name=crafter_hace
#SBATCH --output=experiments/crafter/logs/%x_%A_%a.out
#SBATCH --error=experiments/crafter/logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-49
# 5 agents × 10 seeds = 50 jobs
# NOTE: CPU-only — GPU utilization was <7%, workload is CPU-bound

# ============================================================
# Crafter HACE — Full Experiment
#
# Usage:
#   Full run (1M steps, 5 agents × 10 seeds = 50 jobs):
#     sbatch experiments/crafter/run_crafter_oscar.sh
#
#   Pilot run (100K steps, 5 agents × 3 seeds = 15 jobs):
#     sbatch --array=0-14 experiments/crafter/run_crafter_oscar.sh pilot
# ============================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
AGENTS=(vanilla hace pure_homeo health_only naive_survival)
PROJECT_DIR="/users/zkong10/codebase/homeorl"

# Pilot mode: fewer seeds, fewer steps
MODE="${1:-full}"
if [ "$MODE" = "pilot" ]; then
    NUM_SEEDS=3
    TOTAL_STEPS=100000
    PREFIX="pilot"
else
    NUM_SEEDS=10
    TOTAL_STEPS=1000000
    PREFIX="full"
fi

# ── Compute agent and seed from SLURM array task ID ────────────
AGENT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
AGENT=${AGENTS[$AGENT_IDX]}
OUTDIR="experiments/crafter/results/${PREFIX}_${AGENT}_s${SEED}"

echo "============================================"
echo "  Crafter HACE — ${MODE^^} RUN"
echo "  Job:   ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID})"
echo "  Agent: ${AGENT}"
echo "  Seed:  ${SEED}"
echo "  Steps: ${TOTAL_STEPS}"
echo "  Out:   ${OUTDIR}"
echo "  Time:  $(date)"
echo "============================================"

# ── Environment setup ──────────────────────────────────────────
cd "$PROJECT_DIR"
module load python/3.11.0 2>/dev/null || module load python/3.10.12 2>/dev/null || true

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found! Run setup_oscar_env.sh first."
    exit 1
fi

# Create directories
mkdir -p experiments/crafter/logs
mkdir -p "$OUTDIR"

# ── Pre-flight checks ─────────────────────────────────────────
python -c "import crafter, stable_baselines3, torch" || {
    echo "ERROR: Missing dependencies. Run setup_oscar_env.sh first."
    exit 1
}

# ── Run training ───────────────────────────────────────────────
python experiments/crafter/train_crafter.py \
    --agent "$AGENT" \
    --seed "$SEED" \
    --steps "$TOTAL_STEPS" \
    --outdir "$OUTDIR" \
    --no-record

echo ""
echo "Job complete: ${AGENT} seed ${SEED} — $(date)"
