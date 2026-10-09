#!/usr/bin/env bash
#SBATCH --job-name=eicu_sepsis_trajs
#SBATCH --output=logs/outs/eicu_sepsis_%A_%a.out
#SBATCH --error=logs/errs/eicu_sepsis_%A_%a.err
#SBATCH --time=36:00:00
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

echo "Starting eICU SEPSIS TRAJS run at $(date)"
echo ""

python "$REPO_ROOT/scripts/trajectory/eicu/sepsis_all_biomarkers.py" \
    --data-dir "$DATA_ROOT/eicu/sepsis" \
    --pred-dataset "$DATA_ROOT/eicu/sepsis/sepsis_prediction_dataset.csv" \
    --output "$DATA_ROOT/eicu/sepsis/sepsis_trajectory_probs.csv" \
    --window-days 3.0 \
    --n-batches 8 \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "eICU SEPSIS TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Attempt to clean PyTensor cache for this job (best-effort)
if [ -f "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" ]; then
    bash "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" || true
fi

# Exit with the same code as the Python script
exit $EXIT_CODE
