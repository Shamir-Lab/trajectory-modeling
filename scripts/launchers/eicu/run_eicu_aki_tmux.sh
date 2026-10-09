#!/bin/bash
# Launch Eicu Aki Analysis via tmux
# Usage: ./scripts/launchers/eicu/run_eicu_aki_tmux.sh [--train-n-jobs N] [--train-backend threading|processes] [--parallel-axis repeat|feature-set] [--save-oof|--no-save-oof]

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

SESSION_NAME="eicu_aki_analysis"
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

if [[ "$SAVE_OOF" == "1" ]]; then
    OOF_FLAG="--save-oof"
else
    OOF_FLAG="--no-save-oof"
fi

BOOTSTRAP_FLAG=""
if [[ "$RECOMPUTE_BOOTSTRAP_TRAJ" == "1" ]]; then
    BOOTSTRAP_FLAG="--recompute-bootstrap-trajectories"
fi

cd "$REPO_ROOT" || exit 1

mkdir -p logs/outs/eicu logs/errs/eicu

# Kill existing session if present
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create new session
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# Send command
tmux send-keys -t "$SESSION_NAME" \
    "cd $REPO_ROOT && python scripts/analysis/eicu/eicu_aki_analysis.py --train-n-jobs $TRAIN_N_JOBS --train-backend $TRAIN_BACKEND --parallel-axis $PARALLEL_AXIS --plot-k $PLOT_K --with-probs-source $WITH_PROBS_SOURCE $BOOTSTRAP_FLAG --oof-scope $OOF_SCOPE --oof-k $OOF_K $OOF_FLAG > logs/outs/eicu/eicu_aki_analysis.out 2> logs/errs/eicu/eicu_aki_analysis.err" \
    Enter

echo "✓ tmux session created: $SESSION_NAME"
echo "  Train N-Jobs: $TRAIN_N_JOBS"
echo "  Train Backend: $TRAIN_BACKEND"
echo "  Parallel Axis: $PARALLEL_AXIS"
echo "  OOF Scope: $OOF_SCOPE"
echo "  OOF K: $OOF_K"
echo "  Plot K: $PLOT_K"
echo "  With Probs Source: $WITH_PROBS_SOURCE"
echo "  Recompute Bootstrap Trajectories: $RECOMPUTE_BOOTSTRAP_TRAJ"
echo "  Save OOF: $SAVE_OOF"
echo "  Output log: logs/outs/eicu/eicu_aki_analysis.out"
echo "  Error log: logs/errs/eicu/eicu_aki_analysis.err"
echo ""
echo "Attach: tmux attach -t $SESSION_NAME"
echo "List sessions: tmux list-sessions"
echo "Kill session: tmux kill-session -t $SESSION_NAME"
