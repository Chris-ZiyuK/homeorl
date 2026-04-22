#!/bin/bash
#SBATCH --job-name=crafter_hace
#SBATCH --output=experiments/crafter/logs/%x_%A_%a.out
#SBATCH --error=experiments/crafter/logs/%x_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=4

# ============================================================
# Crafter HACE — Optimized Full Experiment
#
# Packs 3 experiments per SLURM job (parallel via background
# processes). Each process is pinned to 1 thread to avoid
# CPU contention (PyTorch/BLAS default to multi-threaded).
#
# Full run: 5 agents × 5 seeds = 25 experiments
#   25 ÷ 3 = 9 jobs → sbatch --array=0-8
#
# Usage:
#   sbatch --array=0-8 experiments/crafter/run_crafter_packed.sh
#
# Monitor:
#   bash scripts/check_crafter_status.sh
# ============================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
AGENTS=(vanilla hace pure_homeo health_only naive_survival)
NUM_AGENTS=${#AGENTS[@]}
NUM_SEEDS=5
TOTAL_STEPS=1000000
RUNS_PER_JOB=3  # parallel experiments per SLURM job
PROJECT_DIR="/users/zkong10/codebase/homeorl"

TOTAL_RUNS=$((NUM_AGENTS * NUM_SEEDS))  # 25

# ── Determine which runs belong to this job ────────────────────
JOB_IDX=${SLURM_ARRAY_TASK_ID}
START_RUN=$((JOB_IDX * RUNS_PER_JOB))

echo "============================================"
echo "  Crafter HACE — PACKED RUN"
echo "  Job:       ${SLURM_JOB_ID} (array ${JOB_IDX})"
echo "  Runs:      ${START_RUN} to $((START_RUN + RUNS_PER_JOB - 1))"
echo "  Steps:     ${TOTAL_STEPS}"
echo "  Parallel:  ${RUNS_PER_JOB}"
echo "  Node:      $(hostname)"
echo "  CPUs:      ${SLURM_CPUS_PER_TASK:-1}"
echo "  Time:      $(date)"
echo "============================================"

# ── Environment setup ──────────────────────────────────────────
cd "$PROJECT_DIR"
module load python/3.11.0 2>/dev/null || module load python/3.10.12 2>/dev/null || true

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found! Run setup_oscar_env.sh first."
    exit 1
fi

mkdir -p experiments/crafter/logs

# Pre-flight
python -c "import crafter, stable_baselines3, torch" || {
    echo "ERROR: Missing dependencies."; exit 1
}

# ── Launch parallel experiments ────────────────────────────────
# CRITICAL: Pin each process to 1 thread to avoid CPU contention.
# Without this, PyTorch/BLAS spawns N threads per process,
# causing 12 threads on 4 CPUs → FPS drops from 130 to 3.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PIDS=()
FAILED=0

for OFFSET in $(seq 0 $((RUNS_PER_JOB - 1))); do
    RUN_IDX=$((START_RUN + OFFSET))

    # Skip if beyond total runs
    if [ "$RUN_IDX" -ge "$TOTAL_RUNS" ]; then
        echo "  [skip] Run ${RUN_IDX} — beyond total (${TOTAL_RUNS})"
        continue
    fi

    # Compute agent and seed
    AGENT_IDX=$((RUN_IDX / NUM_SEEDS))
    SEED=$((RUN_IDX % NUM_SEEDS))
    AGENT=${AGENTS[$AGENT_IDX]}
    OUTDIR="experiments/crafter/results/full_${AGENT}_s${SEED}"

    echo ""
    echo "  [${OFFSET}] Starting: ${AGENT} seed=${SEED} → ${OUTDIR}"

    mkdir -p "$OUTDIR"

    # Launch in background
    python experiments/crafter/train_crafter.py \
        --agent "$AGENT" \
        --seed "$SEED" \
        --steps "$TOTAL_STEPS" \
        --outdir "$OUTDIR" \
        --no-record \
        > "${OUTDIR}/train.log" 2>&1 &

    PIDS+=($!)
    echo "  [${OFFSET}] PID: ${PIDS[-1]}"
done

echo ""
echo "Waiting for ${#PIDS[@]} experiments to finish..."
echo "PIDs: ${PIDS[*]}"

# ── Wait for all and check exit codes ──────────────────────────
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    RUN_IDX=$((START_RUN + i))
    AGENT_IDX=$((RUN_IDX / NUM_SEEDS))
    SEED=$((RUN_IDX % NUM_SEEDS))
    AGENT=${AGENTS[$AGENT_IDX]}

    if wait "$PID"; then
        echo "  ✓ ${AGENT} seed=${SEED} (PID ${PID}) — SUCCESS"
    else
        echo "  ✗ ${AGENT} seed=${SEED} (PID ${PID}) — FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

# ── Summary ────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Job ${SLURM_JOB_ID} complete: $(date)"
echo "  Runs: $((${#PIDS[@]} - FAILED))/${#PIDS[@]} succeeded"
if [ "$FAILED" -gt 0 ]; then
    echo "  ⚠ ${FAILED} runs failed — check train.log in output dirs"
fi
echo "============================================"

# Exit with error if any run failed
[ "$FAILED" -eq 0 ] || exit 1
