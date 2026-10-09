#!/usr/bin/env bash
# Generate launcher scripts for analysis tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Tasks configuration
declare -A TASK_SCRIPTS=(
    ["mimic_liver"]="mimic_liver_analysis.py"
    ["mimic_aki"]="mimic_aki_analysis.py"
    ["mimic_sepsis"]="mimic_sepsis_analysis.py"
    ["mimic_ventilator"]="mimic_ventilator_analysis.py"
    ["hirid_sepsis"]="hirid_sepsis_analysis.py"
    ["hirid_liver"]="hirid_liver_analysis.py"
    ["hirid_aki"]="hirid_aki_analysis.py"
    ["hirid_ventilator"]="hirid_ventilator_analysis.py"
    ["eicu_sepsis"]="eicu_sepsis_analysis.py"
    ["eicu_liver"]="eicu_liver_analysis.py"
    ["eicu_aki"]="eicu_aki_analysis.py"
    ["eicu_ventilator"]="eicu_ventilator_analysis.py"
)

# Function to create launcher scripts
create_launchers() {
    local task_key=$1
    local script_name=$2
    local task_name=$(echo $task_key | sed 's/_/ /g' | sed 's/^./\U&/g' | sed 's/ ./\U&/g')
    local dataset=$(echo "$task_key" | cut -d'_' -f1)
    local target_dir="$SCRIPT_DIR/$dataset"

    mkdir -p "$target_dir"
    
    # Create nohup launcher
    local nohup_file="$target_dir/run_${task_key}_nohup.sh"
    cat > "$nohup_file" << 'EOF_NOHUP'
#!/bin/bash
# Launch TASK_NAME Analysis via nohup
# Usage: ./scripts/launchers/TASK_DATASET/run_TASK_KEY_nohup.sh [--train-n-jobs N] [--train-backend threading|processes] [--parallel-axis repeat|feature-set] [--save-oof|--no-save-oof]

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

echo "Starting TASK_NAME Analysis (nohup)"
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

mkdir -p logs/outs/TASK_DATASET logs/errs/TASK_DATASET

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

nohup python scripts/analysis/TASK_DATASET/SCRIPT_NAME \
    --train-n-jobs "$TRAIN_N_JOBS" \
    --train-backend "$TRAIN_BACKEND" \
    --parallel-axis "$PARALLEL_AXIS" \
    --plot-k "$PLOT_K" \
    --with-probs-source "$WITH_PROBS_SOURCE" \
    "${BOOTSTRAP_ARGS[@]}" \
    "${OOF_ARGS[@]}" \
    > logs/outs/TASK_DATASET/TASK_KEY_analysis.out 2> logs/errs/TASK_DATASET/TASK_KEY_analysis.err &

PID=$!
echo "Process started with PID: $PID"
echo $PID > .TASK_KEY_analysis.pid

echo "Output: logs/outs/TASK_DATASET/TASK_KEY_analysis.out"
echo "Errors: logs/errs/TASK_DATASET/TASK_KEY_analysis.err"
echo "To monitor: tail -f logs/outs/TASK_DATASET/TASK_KEY_analysis.out"
echo "To stop:    kill $PID"
EOF_NOHUP
    
    # Replace placeholders
    sed -i "s|TASK_NAME|$task_name|g" "$nohup_file"
    sed -i "s|TASK_KEY|$task_key|g" "$nohup_file"
    sed -i "s|TASK_DATASET|$dataset|g" "$nohup_file"
    sed -i "s|SCRIPT_NAME|$script_name|g" "$nohup_file"
    chmod +x "$nohup_file"
    
    # Create tmux launcher
    local tmux_file="$target_dir/run_${task_key}_tmux.sh"
    cat > "$tmux_file" << 'EOF_TMUX'
#!/bin/bash
# Launch TASK_NAME Analysis via tmux
# Usage: ./scripts/launchers/TASK_DATASET/run_TASK_KEY_tmux.sh [--train-n-jobs N] [--train-backend threading|processes] [--parallel-axis repeat|feature-set] [--save-oof|--no-save-oof]

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

SESSION_NAME="TASK_KEY_analysis"
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

mkdir -p logs/outs/TASK_DATASET logs/errs/TASK_DATASET

# Kill existing session if present
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create new session
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# Send command
tmux send-keys -t "$SESSION_NAME" \
    "cd $REPO_ROOT && python scripts/analysis/TASK_DATASET/SCRIPT_NAME --train-n-jobs $TRAIN_N_JOBS --train-backend $TRAIN_BACKEND --parallel-axis $PARALLEL_AXIS --plot-k $PLOT_K --with-probs-source $WITH_PROBS_SOURCE $BOOTSTRAP_FLAG --oof-scope $OOF_SCOPE --oof-k $OOF_K $OOF_FLAG > logs/outs/TASK_DATASET/TASK_KEY_analysis.out 2> logs/errs/TASK_DATASET/TASK_KEY_analysis.err" \
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
echo "  Output log: logs/outs/TASK_DATASET/TASK_KEY_analysis.out"
echo "  Error log: logs/errs/TASK_DATASET/TASK_KEY_analysis.err"
echo ""
echo "Attach: tmux attach -t $SESSION_NAME"
echo "List sessions: tmux list-sessions"
echo "Kill session: tmux kill-session -t $SESSION_NAME"
EOF_TMUX
    
    # Replace placeholders
    sed -i "s|TASK_NAME|$task_name|g" "$tmux_file"
    sed -i "s|TASK_KEY|$task_key|g" "$tmux_file"
    sed -i "s|TASK_DATASET|$dataset|g" "$tmux_file"
    sed -i "s|SCRIPT_NAME|$script_name|g" "$tmux_file"
    chmod +x "$tmux_file"
    
    echo "✓ Created launchers for $task_key"
}

# Create all launchers
echo "Generating launcher scripts..."
for task_key in "${!TASK_SCRIPTS[@]}"; do
    create_launchers "$task_key" "${TASK_SCRIPTS[$task_key]}"
done

echo ""
echo "✓ All launcher scripts generated in dataset-specific folders under $SCRIPT_DIR"
echo ""
echo "Examples:"
echo "  nohup:  ./scripts/launchers/mimic/run_mimic_liver_nohup.sh --train-n-jobs 2"
echo "  tmux:   ./scripts/launchers/hirid/run_hirid_sepsis_tmux.sh --train-n-jobs 4 --train-backend threads"
