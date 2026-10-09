#!/bin/bash
# Launch MIMIC Circulatory Failure Analysis via nohup

# Usage:
#   bash scripts/launchers/mimic/run_mimic_circulatory_failure_nohup.sh \
#     [--base-dir PATH] [--output-dir PATH] [--lookback-hours N] \
#     [--n-repeats N] [--n-folds N] [--seed N] [--train-n-jobs N]

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

# Parse arguments
BASE_DIR=${BASE_DIR:-/home/gaga/data/physionet/mimic/circulatory_failure}
OUTPUT_DIR=${OUTPUT_DIR:-./results/mimic}
LOOKBACK_HOURS=${LOOKBACK_HOURS:-12}
N_REPEATS=${N_REPEATS:-10}
N_FOLDS=${N_FOLDS:-5}
SEED=${SEED:-920}
TRAIN_N_JOBS=${TRAIN_N_JOBS:-1}

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
        --n-repeats)
            N_REPEATS="$2"
            shift 2
            ;;
        --n-folds)
            N_FOLDS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --train-n-jobs)
            TRAIN_N_JOBS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Create logs directory if needed
mkdir -p logs/outs/mimic logs/errs/mimic

# Launch with nohup
nohup python scripts/analysis/mimic/mimic_circulatory_failure_analysis.py \
    --base-dir "$BASE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --lookback-hours "$LOOKBACK_HOURS" \
    --n-repeats "$N_REPEATS" \
    --n-folds "$N_FOLDS" \
    --seed "$SEED" \
    --train-n-jobs "$TRAIN_N_JOBS" \
    > logs/outs/mimic/mimic_circulatory_failure_analysis.out \
    2> logs/errs/mimic/mimic_circulatory_failure_analysis.err &

PID=$!
echo "Process started with PID: $PID"
