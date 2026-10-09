#!/usr/bin/env bash
#SBATCH --job-name=hirid_circ_fail_bayes_traj
#SBATCH --output=logs/outs/%x_%A_%a.out
#SBATCH --error=logs/errs/%x_%A_%a.err
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu-killable
#SBATCH --exclude=rack-iscb-101,rack-iscb-33,rack-iscb-112

# HiRiD Circulatory Failure Bayesian Trajectory Generation
# Run: sbatch scripts/slurm/hirid/run_hirid_circulatory_failure_bayes_trajectories.slurm

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

# Cleanup pytensor cache and /dev/shm tmpdir on any exit (including SIGTERM from job timeout)
trap 'bash "${REPO_ROOT}/scripts/slurm/cleanup_pytensor_cache.sh" 2>/dev/null || true; rm -rf "${JOB_TMP}" 2>/dev/null || true; rm -rf "/dev/shm/tamarw1_${SLURM_JOB_ID:-local}" 2>/dev/null || true' EXIT

COHORT_INDEX=${SLURM_ARRAY_TASK_ID}
WINDOW_HOURS=12.0
N_BATCHES=8
DF_BASIS=${DF_BASIS:-4}
SAMPLER=nutpie

DATA_ROOT="${DATA_ROOT:-${TRAJ_DATA_ROOT:-/home/gaga/data/physionet}}"
export TRAJ_CACHE_ROOT="${TRAJ_CACHE_ROOT:-/home/gaga/tamarw1}"
export TRAJ_DATA_ROOT="$DATA_ROOT"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs"

# Route tmp*.py and other per-job temp files to /dev/shm (RAM, not NFS quota)
# so PyTensor/Numba intermediate files never land on gaga.
SHM_TMP="/dev/shm/tamarw1_${SLURM_JOB_ID:-local}"
find /dev/shm -maxdepth 1 -name "tamarw1_*" -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$SHM_TMP"
export TMPDIR="$SHM_TMP"
export TEMP="$SHM_TMP"
export TMP="$SHM_TMP"
export PYTHONPYCACHEPREFIX="$SHM_TMP/pycache"
# Small gaga JOB_TMP for matplotlib/arviz config only
JOB_TMP="${TRAJ_CACHE_ROOT}/.job_tmp/hirid_cf/${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_TMP/matplotlib" "$JOB_TMP/arviz" "$JOB_TMP/xdg_cache/arviz"
export NUMBA_DISABLE_CACHING=1
export MPLCONFIGDIR="$JOB_TMP/matplotlib"
export ARVIZ_DATA_HOME="$JOB_TMP/arviz"
export XDG_CACHE_HOME="$JOB_TMP/xdg_cache"
PYTENSOR_CACHE_ROOT="${TRAJ_CACHE_ROOT}/.pytensor_cache/${SLURM_JOB_ID:-local}"
find "${TRAJ_CACHE_ROOT}/.pytensor_cache" -maxdepth 1 -mindepth 1 -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$PYTENSOR_CACHE_ROOT"
export PYTENSOR_FLAGS="base_compiledir=${PYTENSOR_CACHE_ROOT},optimizer=fast_compile,exception_verbosity=high"

cd "$REPO_ROOT" || exit 1

# Activate conda environment
source "${CONDA_SH:-/home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh}"
conda activate pymc_env

echo "=========================================="
echo "HiRiD Bayesian Trajectory Generation"
echo "=========================================="
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Cohort Index: ${COHORT_INDEX}"
echo "N Patients: ${N_PATIENTS}"
echo "Seed: ${SEED}"
echo "Window: ${WINDOW_HOURS}h"
echo "Batches: ${N_BATCHES}"
echo "=========================================="

python scripts/trajectory/hirid/hirid_circulatory_failure_bayes_trajectories.py \
    --cohort-index ${COHORT_INDEX} \
    --cohort-splits 16 \
    --window-hours ${WINDOW_HOURS} \
    --n-batches ${N_BATCHES} \
    --df-basis ${DF_BASIS} \
    --sampler ${SAMPLER}

echo "✓ Trajectory generation complete for cohort ${COHORT_INDEX} with ${SAMPLER} sampler"
