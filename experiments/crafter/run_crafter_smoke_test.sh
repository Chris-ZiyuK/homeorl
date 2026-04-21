#!/bin/bash
#SBATCH --job-name=crafter_smoke
#SBATCH --output=experiments/crafter/logs/smoke_%A_%a.out
#SBATCH --error=experiments/crafter/logs/smoke_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-1
# Smoke test: only 2 jobs (vanilla + hace), 1 seed each

# ============================================================
# Crafter HACE — Smoke Test (10K steps)
#
# 目的：验证 Oscar 环境可以正常跑完训练
# 预计时间：每个 job ~5 分钟
#
# 用法：
#   sbatch experiments/crafter/run_crafter_smoke_test.sh
#
# 完成后检查：
#   cat experiments/crafter/results/smoke_*/summary.json
# ============================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
AGENTS=(vanilla hace)
SEED=0
TOTAL_STEPS=10000
PROJECT_DIR="/users/zkong10/codebase/homeorl"

# ── Compute agent from array task ID ──────────────────────────
AGENT_IDX=${SLURM_ARRAY_TASK_ID}
AGENT=${AGENTS[$AGENT_IDX]}
OUTDIR="experiments/crafter/results/smoke_${AGENT}_s${SEED}"

echo "============================================"
echo "  Crafter HACE — SMOKE TEST"
echo "  Job:   ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID})"
echo "  Agent: ${AGENT}"
echo "  Seed:  ${SEED}"
echo "  Steps: ${TOTAL_STEPS}"
echo "  Out:   ${OUTDIR}"
echo "  Time:  $(date)"
echo "============================================"

# ── Environment setup ──────────────────────────────────────────
cd "$PROJECT_DIR"

# Load modules
module load python/3.11.0 2>/dev/null || module load python/3.10.12 2>/dev/null
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null

# Activate venv
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
echo ""
echo "Pre-flight checks:"
python -c "import crafter; print('  ✓ crafter installed')" || {
    echo "  ✗ crafter not installed"; exit 1
}
python -c "import stable_baselines3 as sb3; print(f'  ✓ SB3 {sb3.__version__}')" || {
    echo "  ✗ stable-baselines3 not installed"; exit 1
}
python -c "import torch; print(f'  ✓ torch {torch.__version__}, CUDA={torch.cuda.is_available()}')" || {
    echo "  ✗ torch not available"; exit 1
}
echo "  All checks passed."

# ── Run training ───────────────────────────────────────────────
echo ""
echo "Starting training..."
python experiments/crafter/train_crafter.py \
    --agent "$AGENT" \
    --seed "$SEED" \
    --steps "$TOTAL_STEPS" \
    --outdir "$OUTDIR" \
    --no-record \
    --eval-freq 5000 \
    --eval-episodes 3

# ── Post-run validation ───────────────────────────────────────
echo ""
echo "─── Post-run validation ───"

if [ -f "${OUTDIR}/summary.json" ]; then
    echo "✓ summary.json exists"
    cat "${OUTDIR}/summary.json"
else
    echo "✗ summary.json MISSING — training may have failed"
    exit 1
fi

if [ -f "${OUTDIR}/episode_data.json" ]; then
    N_EPISODES=$(python -c "import json; d=json.load(open('${OUTDIR}/episode_data.json')); print(len(d))")
    echo "✓ episode_data.json: ${N_EPISODES} episodes recorded"
else
    echo "✗ episode_data.json MISSING"
    exit 1
fi

echo ""
echo "============================================"
echo "  ✓ Smoke test PASSED: ${AGENT} seed ${SEED}"
echo "  Time: $(date)"
echo "============================================"
