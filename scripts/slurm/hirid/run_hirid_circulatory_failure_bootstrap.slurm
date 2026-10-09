#!/usr/bin/env bash
#SBATCH --job-name=hirid_circ_bootstrap
#SBATCH --output=logs/outs/hirid_circ_bootstrap_%A_%a.out
#SBATCH --error=logs/errs/hirid_circ_bootstrap_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

#SBATCH --partition=cpu-killable
#SBATCH --array=0-7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs"
cd "$REPO_ROOT" || exit 1

# Activate conda environment
source "${CONDA_SH:-/home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh}"
conda activate pymc_env

echo "========================================================================"
echo "Starting HiRiD Circulatory Failure BOOTSTRAP trajectory run"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "========================================================================"

# Cohort splitting
COHORT_SPLITS=8
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

action_script="$REPO_ROOT/scripts/trajectory/hirid/circulatory_failure_trajs_bootstrap.py"

echo "Running: $action_script"
echo "Cohort: ${COHORT_INDEX}/${COHORT_SPLITS}"

OUTPUT_FORMAT="parquet"
PARQUET_COMPRESSION="zstd"

python $action_script \
    --data-dir "$DATA_ROOT/hirid/circulatory_failure" \
    --pred-dataset "$DATA_ROOT/hirid/circulatory_failure/circulatory_failure_prediction_dataset.csv" \
    --merged-output "$DATA_ROOT/hirid/circulatory_failure/circulatory_failure_prediction_dataset_with_bootstrap_probs" \
    --window-hours 12.0 \
    --n-bootstrap 1000 \
    --n-batches 8 \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX} \
    --output-format ${OUTPUT_FORMAT} \
    --parquet-compression ${PARQUET_COMPRESSION}

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "HiRiD Circulatory Failure BOOTSTRAP run completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================================"

exit $EXIT_CODE
