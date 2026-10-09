#!/usr/bin/env bash
#SBATCH --job-name=hirid_circulatory_trajs
#SBATCH --output=logs/outs/hirid_circulatory_%A_%a.out
#SBATCH --error=logs/errs/hirid_circulatory_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu-killable
#SBATCH --array=0-9

# HiRiD Circulatory Failure Trajectory Computation (multi-biomarker)

echo "=========================================="
echo "HiRiD Circulatory Failure Trajectory Computation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 128G"
echo "Start time: $(date)"
echo ""

# Cohort splitting
COHORT_SPLITS=10
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
export TRAJ_CACHE_ROOT="${TRAJ_CACHE_ROOT:-/home/gaga/tamarw1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs"

JOB_TMP="${TRAJ_CACHE_ROOT}/.job_tmp/hirid_cf/${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"
export NUMBA_DISABLE_CACHING=1
export PYTHONPYCACHEPREFIX="${HOME}/.pycache_slurm/${SLURM_JOB_ID:-local}"
mkdir -p "$PYTHONPYCACHEPREFIX"

PYTENSOR_CACHE_ROOT="${HOME}/.pytensor_cache/${SLURM_JOB_ID:-local}"
find "${HOME}/.pytensor_cache" -maxdepth 1 -mindepth 1 -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$PYTENSOR_CACHE_ROOT"
export PYTENSOR_FLAGS="base_compiledir=${PYTENSOR_CACHE_ROOT},optimizer=fast_compile,exception_verbosity=high"

trap 'bash "${REPO_ROOT}/scripts/slurm/cleanup_pytensor_cache.sh" 2>/dev/null || true; rm -rf "${JOB_TMP}" 2>/dev/null || true' EXIT

cd "$REPO_ROOT" || exit 1

# Run trajectory computation for all three biomarkers
python "$REPO_ROOT/scripts/trajectory/hirid/circulatory_failure_trajs.py" \
    --input-dir "$DATA_ROOT/hirid/circulatory_failure" \
    --output-dir "$DATA_ROOT/hirid/circulatory_failure" \
    --window-hours 12.0 \
    --n-batches 10 \
    --sampler nutpie \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

echo ""
echo "End time: $(date)"
echo "=========================================="
