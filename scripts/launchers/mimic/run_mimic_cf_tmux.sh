#!/usr/bin/env bash
set -euo pipefail

# -------- Config --------
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="python"  # change to full path if needed
SCRIPT_PATH="$WORKDIR/scripts/analysis/mimic/mimic_circulatory_failure_analysis.py"
BASE_DIR="/home/gaga/data/physionet/mimic/circulatory_failure"
OUTPUT_DIR="$WORKDIR/results/mimic"

LOG_DIR="$WORKDIR/logs/outs"
TS="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="mimic_cf_${TS}"
LOG_FILE="$LOG_DIR/mimic_circulatory_failure_${TS}.log"

# Optional args
N_REPEATS=10
N_FOLDS=5
LOOKBACK_HOURS=12
SEED=920
# ------------------------

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKDIR"

CMD="$PYTHON_BIN $SCRIPT_PATH \
  --base-dir $BASE_DIR \
  --output-dir $OUTPUT_DIR \
  --lookback-hours $LOOKBACK_HOURS \
  --n-repeats $N_REPEATS \
  --n-folds $N_FOLDS \
  --seed $SEED"

# Start in detached tmux session and log output
tmux new-session -d -s "$SESSION_NAME" "cd '$WORKDIR' && $CMD > '$LOG_FILE' 2>&1"

echo "Started."
echo "tmux session : $SESSION_NAME"
echo "log file     : $LOG_FILE"
echo
echo "Monitor logs : tail -f '$LOG_FILE'"
echo "Attach tmux  : tmux attach -t '$SESSION_NAME'"
echo "List sessions: tmux ls"
echo "Stop job     : tmux kill-session -t '$SESSION_NAME'"