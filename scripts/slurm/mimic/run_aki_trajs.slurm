#!/usr/bin/env bash
#SBATCH --job-name=mimic_aki_trajs
#SBATCH --output=logs/outs/mimic_aki_%A_%a.out
#SBATCH --error=logs/errs/mimic_aki_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=cpu-killable
#SBATCH --array=0-44


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
	REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
export TRAJ_CACHE_ROOT="${TRAJ_CACHE_ROOT:-/home/gaga/tamarw1}"
export NUMBA_DISABLE_CACHING=1
export PYTHONPYCACHEPREFIX="${HOME}/.pycache_slurm/${SLURM_JOB_ID:-local}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs"
cd "$REPO_ROOT" || exit 1

# Activate conda environment
source "${CONDA_SH:-/home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh}"
conda activate pymc_env

# Cohort splitting
COHORT_SPLITS=45
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

# Run AKI creatinine trajectories
echo "Starting MIMIC AKI TRAJS run at $(date)"
echo ""

python "$REPO_ROOT/scripts/trajectory/mimic/aki_trajs.py" \
	--input "$DATA_ROOT/mimic/aki/creatinine_timeseries.csv" \
	--pred-dataset "$DATA_ROOT/mimic/aki/aki_prediction_dataset.csv" \
	--output "$DATA_ROOT/mimic/aki/aki_trajectory_probs_bayes.csv" \
	--merged-output "$DATA_ROOT/mimic/aki/aki_prediction_dataset_with_probs.csv" \
	--cohort-splits ${COHORT_SPLITS} \
	--cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "MIMIC AKI TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
if [ -f "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" ]; then
	bash "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" || true
fi

exit $EXIT_CODE