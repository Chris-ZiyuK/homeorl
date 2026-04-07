#!/usr/bin/env bash
# Quick monitor for running E3 experiments
# Usage: bash scripts/monitor_e3.sh

LOG="experiments/sequential_results/e3_alpha_carryover.log"

if [ ! -f "$LOG" ]; then
    echo "Log file not found: $LOG"
    exit 1
fi

TOTAL_RUNS=$((9 * 20))  # 9 agents × 20 seeds
DONE=$(grep -c "^Running " "$LOG" 2>/dev/null || echo 0)
PERCENT=$((DONE * 100 / TOTAL_RUNS))

echo "═══════════════════════════════════════"
echo "  E3 Alpha Training Progress"
echo "═══════════════════════════════════════"
echo "  Completed: $DONE / $TOTAL_RUNS ($PERCENT%)"
echo "  Last activity:"
tail -2 "$LOG"
echo ""

# Check if still running
if pgrep -f "run_sequential_tasks" > /dev/null; then
    echo "  Status: ✅ RUNNING"
    PID=$(pgrep -f "run_sequential_tasks.py" | head -1)
    CPU=$(ps -p $PID -o %cpu= 2>/dev/null)
    MEM=$(ps -p $PID -o %mem= 2>/dev/null)
    echo "  PID: $PID  CPU: ${CPU}%  MEM: ${MEM}%"
else
    echo "  Status: ⏹️  FINISHED (or crashed)"
    echo "  Check last lines for errors:"
    tail -5 "$LOG"
fi
echo "═══════════════════════════════════════"
