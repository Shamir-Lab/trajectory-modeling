#!/bin/bash
# Launch eICU Circulatory Failure Analysis via nohup

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

# Parse arguments
BASE_DIR=${BASE_DIR:-/home/gaga/data/physionet/eicu/circulatory_failure}
OUTPUT_DIR=${OUTPUT_DIR:-results/eicu/circulatory_failure}
TRAIN_N_JOBS=${TRAIN_N_JOBS:-1}
TRAIN_BACKEND=${TRAIN_BACKEND:-threading}

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
        --train-n-jobs)
            TRAIN_N_JOBS="$2"
            shift 2
            ;;
        --train-backend)
            TRAIN_BACKEND="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Create logs directory if needed
mkdir -p logs/outs/eicu logs/errs/eicu

# Launch with nohup
nohup python scripts/analysis/eicu/eicu_circulatory_failure_analysis.py \
    --base-dir "$BASE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --train-n-jobs "$TRAIN_N_JOBS" \
    --train-backend "$TRAIN_BACKEND" \
    > logs/outs/eicu/eicu_circulatory_failure_analysis.out \
    2> logs/errs/eicu/eicu_circulatory_failure_analysis.err &

PID=$!
echo "Process started with PID: $PID"
