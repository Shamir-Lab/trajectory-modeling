#!/bin/bash
# Launch HiRID Circulatory Failure Analysis via nohup

# Usage:
#   bash scripts/launchers/hirid/run_hirid_circulatory_failure_nohup.sh \
#     [--base-dir PATH] [--output-dir PATH] [--lookback-hours N] \
#     [--eval-profile quick|full] [--n-jobs N] [--train-n-jobs N] \
#     [--train-backend threads|processes|threading]

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

# Parse arguments
BASE_DIR=${BASE_DIR:-${TRAJ_DATA_ROOT:-/home/gaga/data/physionet}/hirid/circulatory_failure}
OUTPUT_DIR=${OUTPUT_DIR:-${TRAJ_RESULTS_ROOT:-$REPO_ROOT/results}/hirid}
LOOKBACK_HOURS=${LOOKBACK_HOURS:-12}
EVAL_PROFILE=${EVAL_PROFILE:-full}
N_JOBS=${N_JOBS:-4}
TRAIN_N_JOBS=${TRAIN_N_JOBS:-1}
TRAIN_BACKEND=${TRAIN_BACKEND:-threads}

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --lookback-hours)
            LOOKBACK_HOURS="$2"
            shift 2
            ;;
        --train-n-jobs)
            TRAIN_N_JOBS="$2"
            shift 2
            ;;
        --n-jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --train-backend)
            TRAIN_BACKEND="$2"
            shift 2
            ;;
        --eval-profile)
            EVAL_PROFILE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Script expects threads|processes (normalize common alias)
if [[ "$TRAIN_BACKEND" == "threading" ]]; then
    TRAIN_BACKEND="threads"
fi

# Create logs directory if needed
mkdir -p logs/outs/hirid logs/errs/hirid

# Launch with nohup
nohup python scripts/analysis/hirid/hirid_circulatory_failure_analysis.py \
    --base-dir "$BASE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --lookback-hours "$LOOKBACK_HOURS" \
    --n-jobs "$N_JOBS" \
    --train-n-jobs "$TRAIN_N_JOBS" \
    --train-backend "$TRAIN_BACKEND" \
    --eval-profile "$EVAL_PROFILE" \
    > logs/outs/hirid/hirid_circulatory_failure_analysis.out \
    2> logs/errs/hirid/hirid_circulatory_failure_analysis.err &

PID=$!
echo "Process started with PID: $PID"
