#!/bin/bash
#SBATCH --job-name=crafter_bench
#SBATCH --output=experiments/crafter/logs/benchmark_%j.out
#SBATCH --error=experiments/crafter/logs/benchmark_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=6

# ============================================================
# Benchmark: SubprocVecEnv speedup on Crafter
# Tests 1, 2, 4 parallel envs with 5000 PPO steps each
# ~2 min total
#
# Usage: sbatch experiments/crafter/run_benchmark.sh
# ============================================================

set -euo pipefail

PROJECT_DIR="/users/zkong10/codebase/homeorl"
cd "$PROJECT_DIR"

module load python/3.11.0 2>/dev/null || module load python/3.10.12 2>/dev/null || true

source .venv/bin/activate
mkdir -p experiments/crafter/logs

echo "============================================"
echo "  Crafter VecEnv Benchmark"
echo "  Node: $(hostname)"
echo "  CPUs: ${SLURM_CPUS_PER_TASK:-1}"
echo "  Time: $(date)"
echo "============================================"

python experiments/crafter/benchmark_vecenv.py

echo ""
echo "Done: $(date)"
