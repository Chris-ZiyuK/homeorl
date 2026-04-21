#!/bin/bash
#SBATCH --job-name=crafter_hace
#SBATCH --output=experiments/crafter/logs/%x_%A_%a.out
#SBATCH --error=experiments/crafter/logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-49
# 5 agents × 10 seeds = 50 jobs

# ── Configuration ──────────────────────────────────────────────
AGENTS=(vanilla hace pure_homeo health_only naive_survival)
NUM_SEEDS=10
TOTAL_STEPS=1000000
PROJECT_DIR="/users/$USER/homeorl"  # Update this to your Oscar path

# ── Compute agent and seed from SLURM array task ID ────────────
AGENT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
AGENT=${AGENTS[$AGENT_IDX]}

echo "============================================"
echo "Job: ${SLURM_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Agent: ${AGENT}, Seed: ${SEED}"
echo "Steps: ${TOTAL_STEPS}"
echo "============================================"

# ── Environment setup ──────────────────────────────────────────
cd $PROJECT_DIR
module load python/3.10.12
module load cuda/12.1

# Activate virtual environment (update path as needed)
source .venv/bin/activate

# Create log directory
mkdir -p experiments/crafter/logs

# ── Run training ───────────────────────────────────────────────
python experiments/crafter/train_crafter.py \
    --agent $AGENT \
    --seed $SEED \
    --steps $TOTAL_STEPS \
    --outdir "experiments/crafter/results/${AGENT}_s${SEED}"

echo "Job complete: ${AGENT} seed ${SEED}"
