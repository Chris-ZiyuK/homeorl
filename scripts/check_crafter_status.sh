#!/bin/bash
# ============================================================
# 查看 Crafter 实验运行状态
#
# 用法：bash scripts/check_crafter_status.sh
# ============================================================

PROJECT_DIR="/users/zkong10/codebase/homeorl"
RESULTS_DIR="${PROJECT_DIR}/experiments/crafter/results"

echo "============================================"
echo "  Crafter HACE — Experiment Status"
echo "  $(date)"
echo "============================================"

# ── 1. SLURM Job Status ───────────────────────────────────────
echo ""
echo "── SLURM Jobs ──"
RUNNING=$(squeue -u "$USER" -h -t RUNNING | wc -l)
PENDING=$(squeue -u "$USER" -h -t PENDING | wc -l)
echo "  Running: ${RUNNING}  |  Pending: ${PENDING}"
if [ "$RUNNING" -gt 0 ] || [ "$PENDING" -gt 0 ]; then
    squeue -u "$USER" -o "  %-12i %-15j %-8T %-10M %-6D %R"
fi

# ── 2. Completed Experiments ──────────────────────────────────
echo ""
echo "── Completed Experiments ──"

AGENTS=(vanilla hace pure_homeo health_only naive_survival)
TOTAL=0
DONE=0
FAILED=0

printf "  %-18s" "Agent"
for s in 0 1 2 3 4; do
    printf " s%-3s" "$s"
done
printf "  Done\n"
echo "  $(printf '%.0s─' {1..50})"

for agent in "${AGENTS[@]}"; do
    printf "  %-18s" "$agent"
    agent_done=0
    for seed in 0 1 2 3 4; do
        TOTAL=$((TOTAL + 1))
        dir="${RESULTS_DIR}/full_${agent}_s${seed}"
        summary="${dir}/summary.json"
        
        if [ -f "$summary" ]; then
            # Check if it completed all steps
            steps=$(python3 -c "import json; d=json.load(open('${summary}')); print(d.get('total_steps', 0))" 2>/dev/null || echo "0")
            if [ "$steps" -ge 1000000 ]; then
                printf " ✓   "
                DONE=$((DONE + 1))
                agent_done=$((agent_done + 1))
            else
                printf " ▶${steps:0:3} "
            fi
        elif [ -d "$dir" ]; then
            # Directory exists but no summary — still running or failed
            if [ -f "${dir}/train.log" ]; then
                # Check last line of train.log for progress
                last_line=$(tail -1 "${dir}/train.log" 2>/dev/null || echo "")
                if echo "$last_line" | grep -q "error\|Error\|ERROR\|Traceback" 2>/dev/null; then
                    printf " ✗   "
                    FAILED=$((FAILED + 1))
                else
                    printf " ▶   "
                fi
            else
                printf " ·   "
            fi
        else
            printf " ·   "
        fi
    done
    printf "  %d/5\n" "$agent_done"
done

echo ""
echo "  Total: ${DONE}/${TOTAL} completed, ${FAILED} failed"
echo "  Legend: ✓=done  ▶=running  ✗=failed  ·=not started"

# ── 3. Latest Activity ────────────────────────────────────────
echo ""
echo "── Latest Activity (most recent train.log updates) ──"
if ls ${RESULTS_DIR}/full_*/train.log 1>/dev/null 2>&1; then
    ls -lt ${RESULTS_DIR}/full_*/train.log 2>/dev/null | head -5 | while read line; do
        file=$(echo "$line" | awk '{print $NF}')
        dir=$(dirname "$file")
        name=$(basename "$dir")
        mod_time=$(echo "$line" | awk '{print $6, $7, $8}')
        last=$(tail -1 "$file" 2>/dev/null | head -c 80)
        echo "  ${name}: ${last}"
    done
else
    echo "  (no train.log files found yet)"
fi

# ── 4. Resource Usage (if any running) ────────────────────────
if [ "$RUNNING" -gt 0 ]; then
    echo ""
    echo "── Resource Usage (running jobs) ──"
    squeue -u "$USER" -h -t RUNNING -o "%i" | while read jobid; do
        echo "  Job ${jobid}: seff ${jobid} (run after completion)"
    done
fi

echo ""
echo "============================================"
