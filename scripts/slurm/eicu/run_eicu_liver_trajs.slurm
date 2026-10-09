#!/usr/bin/env bash
#SBATCH --job-name=eicu_liver_trajs
#SBATCH --output=logs/outs/eicu_liver_%A_%a.out
#SBATCH --error=logs/errs/eicu_liver_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

#SBATCH --partition=cpu-killable
#SBATCH --array=0-9


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
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

# Cohort splitting
COHORT_SPLITS=10
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

# Run eICU liver trajectory computation
echo "Starting eICU LIVER TRAJS run at $(date)"
echo "Cohort: 795 patients, 4,978 samples"
echo ""

python "$REPO_ROOT/scripts/trajectory/eicu/liver_trajs.py" \
    --input "$DATA_ROOT/eicu/liver/bilirubin_timeseries.csv" \
    --pred-dataset "$DATA_ROOT/eicu/liver/liver_prediction_dataset.csv" \
    --output "$DATA_ROOT/eicu/liver/liver_trajectory_probs.csv" \
    --probs-output "$DATA_ROOT/eicu/liver/liver_trajectory_probs_bayes.csv" \
    --window-days 5.0 \
    --flat-thr 0.5 \
    --decline-thr 1.0 \
    --nonlinear-gap 3.0 \
    --n-batches 5 \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "eICU LIVER TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Attempt to clean PyTensor cache for this job (best-effort)
if [ -f "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" ]; then
    bash "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" || true
fi

# Exit with the same code as the Python script
exit $EXIT_CODE
