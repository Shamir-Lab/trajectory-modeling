#!/usr/bin/env bash
#SBATCH --job-name=hirid_cf_trajs_container
#SBATCH --output=logs/outs/hirid_cf_container_%A_%a.out
#SBATCH --error=logs/errs/hirid_cf_container_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu-killable
#SBATCH --array=0-9

# HiRiD Circulatory Failure Trajectory Computation — UDocker container version
#
# Runs circulatory_failure_trajs.py with PyMC + g++ (via the trajectory-modeling
# UDocker container), giving C-compiled PyTensor ops unavailable on the server.
#
# Differences from run_hirid_circulatory_failure_trajs.slurm:
#   - Uses UDocker instead of conda; no conda activate needed
#   - Sampler: pymc (g++-accelerated) instead of nutpie (not in container)
#   - PyTensor cache is written to the host via a bind mount (/cache → TRAJ_CACHE_ROOT)
#
# Prerequisites:
#   udocker create --name=trajectory-modeling trajectory-modeling
#
# Run:
#   sbatch scripts/slurm/hirid/run_hirid_circulatory_failure_trajs_container.slurm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

COHORT_SPLITS=10
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}
CONTAINER_NAME=trajectory-modeling

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
TRAJ_CACHE_ROOT="${TRAJ_CACHE_ROOT:-/home/gaga/tamarw1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs"

# Per-job temp dir on cache filesystem (avoids NFS quota / contention)
JOB_TMP="${TRAJ_CACHE_ROOT}/.job_tmp/hirid_cf_container/${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_TMP"

# PyTensor cache dir — created here so the container can write to it via the
# /cache bind mount. The Python script constructs the same path from TRAJ_CACHE_ROOT
# + SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID, so directory must exist before launch.
PYTENSOR_CACHE="${TRAJ_CACHE_ROOT}/.pytensor_cache/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}"
find "${TRAJ_CACHE_ROOT}/.pytensor_cache" -maxdepth 1 -mindepth 1 -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$PYTENSOR_CACHE"

trap 'rm -rf "$PYTENSOR_CACHE" "$JOB_TMP" 2>/dev/null || true' EXIT

echo "=========================================="
echo "HiRiD Circulatory Failure (container/PyMC)"
echo "=========================================="
echo "Job ID    : ${SLURM_JOB_ID:-local}"
echo "Array task: ${COHORT_INDEX} / ${COHORT_SPLITS}"
echo "Node      : ${SLURM_NODELIST:-local}"
echo "CPUs      : ${SLURM_CPUS_PER_TASK:-64}"
echo "Container : ${CONTAINER_NAME}"
echo "Sampler   : pymc (g++-accelerated via PyTensor)"
echo "Cache     : ${PYTENSOR_CACHE}"
echo "Start     : $(date)"
echo "=========================================="

udocker run \
    -v "${REPO_ROOT}:/traj" \
    -v "${DATA_ROOT}:/data" \
    -v "${TRAJ_CACHE_ROOT}:/cache" \
    -e SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
    -e SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}" \
    -e SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}" \
    -e SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-64}" \
    -e TRAJ_CACHE_ROOT=/cache \
    -e TRAJ_DATA_ROOT=/data \
    -e TMPDIR=/cache/.job_tmp/hirid_cf_container/${SLURM_JOB_ID:-local} \
    -e NUMBA_DISABLE_CACHING=1 \
    -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    "${CONTAINER_NAME}" \
    python /traj/scripts/trajectory/hirid/circulatory_failure_trajs.py \
        --input-dir  /data/hirid/circulatory_failure \
        --output-dir /data/hirid/circulatory_failure \
        --window-hours 12.0 \
        --n-batches 10 \
        --sampler pymc \
        --cohort-splits ${COHORT_SPLITS} \
        --cohort-index  ${COHORT_INDEX}

echo ""
echo "End time: $(date)"
echo "=========================================="
