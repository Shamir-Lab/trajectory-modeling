#!/bin/bash
# Launch Mimic Sepsis Analysis via nohup
# Usage: ./scripts/launchers/mimic/run_mimic_sepsis_nohup.sh [--train-n-jobs N] [--train-backend threading|processes] [--parallel-axis repeat|feature-set] [--save-oof|--no-save-oof]

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

cd "$REPO_ROOT" || exit 1

TRAIN_N_JOBS=${TRAIN_N_JOBS:-1}
TRAIN_BACKEND=${TRAIN_BACKEND:-threading}
PARALLEL_AXIS=${PARALLEL_AXIS:-repeat}
OOF_SCOPE=${OOF_SCOPE:-representative-k}
OOF_K=${OOF_K:-10}
PLOT_K=${PLOT_K:-10}
SAVE_OOF=${SAVE_OOF:-0}
WITH_PROBS_SOURCE=${WITH_PROBS_SOURCE:-default}
RECOMPUTE_BOOTSTRAP_TRAJ=${RECOMPUTE_BOOTSTRAP_TRAJ:-0}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-n-jobs)
            TRAIN_N_JOBS="$2"
            shift 2
            ;;
        --train-backend)
            TRAIN_BACKEND="$2"
            shift 2
            ;;
        --parallel-axis)
            PARALLEL_AXIS="$2"
            shift 2
            ;;
        --oof-scope)
            OOF_SCOPE="$2"
            shift 2
            ;;
        --oof-k)
            OOF_K="$2"
            shift 2
            ;;
        --plot-k)
            PLOT_K="$2"
            shift 2
            ;;
        --with-probs-source)
            WITH_PROBS_SOURCE="$2"
            shift 2
            ;;
        --recompute-bootstrap-trajectories)
            RECOMPUTE_BOOTSTRAP_TRAJ=1
            shift
            ;;
        --save-oof)
            SAVE_OOF=1
            shift
            ;;
        --no-save-oof)
            SAVE_OOF=0
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting Mimic Sepsis Analysis (nohup)"
echo "  Train N-Jobs: $TRAIN_N_JOBS"
echo "  Train Backend: $TRAIN_BACKEND"
echo "  Parallel Axis: $PARALLEL_AXIS"
echo "  OOF Scope: $OOF_SCOPE"
echo "  OOF K: $OOF_K"
echo "  Plot K: $PLOT_K"
echo "  With Probs Source: $WITH_PROBS_SOURCE"
echo "  Recompute Bootstrap Trajectories: $RECOMPUTE_BOOTSTRAP_TRAJ"
echo "  Save OOF: $SAVE_OOF"
echo "  Start time: $(date)"
echo ""

mkdir -p logs/outs/mimic logs/errs/mimic

OOF_ARGS=(--oof-scope "$OOF_SCOPE" --oof-k "$OOF_K")
if [[ "$SAVE_OOF" == "1" ]]; then
    OOF_ARGS+=(--save-oof)
else
    OOF_ARGS+=(--no-save-oof)
fi

BOOTSTRAP_ARGS=()
if [[ "$RECOMPUTE_BOOTSTRAP_TRAJ" == "1" ]]; then
    BOOTSTRAP_ARGS+=(--recompute-bootstrap-trajectories)
fi

nohup python scripts/analysis/mimic/mimic_sepsis_analysis.py \
    --train-n-jobs "$TRAIN_N_JOBS" \
    --train-backend "$TRAIN_BACKEND" \
    --parallel-axis "$PARALLEL_AXIS" \
    --plot-k "$PLOT_K" \
    --with-probs-source "$WITH_PROBS_SOURCE" \
    "${BOOTSTRAP_ARGS[@]}" \
    "${OOF_ARGS[@]}" \
    > logs/outs/mimic/mimic_sepsis_analysis.out 2> logs/errs/mimic/mimic_sepsis_analysis.err &

PID=$!
echo "Process started with PID: $PID"
echo $PID > .mimic_sepsis_analysis.pid

echo "Output: logs/outs/mimic/mimic_sepsis_analysis.out"
echo "Errors: logs/errs/mimic/mimic_sepsis_analysis.err"
echo "To monitor: tail -f logs/outs/mimic/mimic_sepsis_analysis.out"
echo "To stop:    kill $PID"
