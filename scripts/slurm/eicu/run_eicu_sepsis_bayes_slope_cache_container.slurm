#!/usr/bin/env bash
#SBATCH --job-name=eicu_sepsis_bayes_slope_cache
#SBATCH --output=logs/outs/eicu_sepsis_bayes_slope_cache_%A_%a.out
#SBATCH --error=logs/errs/eicu_sepsis_bayes_slope_cache_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu-killable
#SBATCH --array=0-9

# eICU Sepsis Bayesian slope-sample cache — UDocker container version
#
# Builds the raw inter-point-slope cache consumed by
#   scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py --backend bayes
# using real PyMC posterior sampling (`_sample_post_trajs_scaled`), the same
# backend that produces {marker}_trajectory_probs_bayes.csv in production (see
# scripts/trajectory/eicu/sepsis_all_biomarkers.py). Runs PyMC + g++ (via the
# trajectory-modeling UDocker container) for C-compiled PyTensor ops unavailable
# on the bare server — see docs/CONTAINER_GUIDE.md. Without g++, PyTensor falls
# back to slow pure-Python compilation (10-50x slower).
#
# This ONLY builds the slope cache (--build-cache-only): it does not call
# prepare_features or run any CV, so 10 array tasks never race on writing the
# shared bootstrap-trajectory / summary-stats files that prepare_features
# produces. Each array task writes its own cohort shard
# ({marker}_bayes_slope_samples_cache_cohortNN.npz); nothing else touches disk.
#
# After ALL array tasks finish, merge the shards, then run the sensitivity
# script normally — the grid search + CV stage reuses the merged cache and
# needs no container:
#   bash scripts/slurm/eicu/merge_eicu_sepsis_bayes_slope_cache.sh lactate
#   python scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py --backend bayes --marker lactate
#
# Prerequisites:
#   udocker create --name=trajectory-modeling trajectory-modeling
#
# Usage (one marker per submission; MARKER defaults to lactate):
#   sbatch --export=MARKER=lactate  scripts/slurm/eicu/run_eicu_sepsis_bayes_slope_cache_container.slurm
#   sbatch --export=MARKER=wbc      scripts/slurm/eicu/run_eicu_sepsis_bayes_slope_cache_container.slurm
#   sbatch --export=MARKER=platelet scripts/slurm/eicu/run_eicu_sepsis_bayes_slope_cache_container.slurm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

MARKER="${MARKER:-lactate}"            # lactate | wbc | platelet
COHORT_SPLITS=10                        # matches sepsis_all_biomarkers.py / existing cohortNN files — keep in sync
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}
CONTAINER_NAME=trajectory-modeling

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
TRAJ_CACHE_ROOT="${TRAJ_CACHE_ROOT:-/home/gaga/tamarw1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs"

# Per-job temp dir on cache filesystem (avoids NFS quota / contention)
JOB_TMP="${TRAJ_CACHE_ROOT}/.job_tmp/eicu_sepsis_bayes_slope_cache/${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_TMP"

# PyTensor cache dir — created here so the container can write to it via the
# /cache bind mount. Unique per array task so 10 concurrent tasks never share
# a compiledir (each still benefits from the idempotent add_key /
# stale-key.pkl-tolerant patches in _apply_worker_process_patches for races
# *within* a task's own joblib workers).
PYTENSOR_CACHE="${TRAJ_CACHE_ROOT}/.pytensor_cache/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}"
find "${TRAJ_CACHE_ROOT}/.pytensor_cache" -maxdepth 1 -mindepth 1 -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$PYTENSOR_CACHE"

trap 'rm -rf "$PYTENSOR_CACHE" "$JOB_TMP" 2>/dev/null || true' EXIT

# Raise the open-files limit (best-effort) — some nodes default the soft
# ulimit far below the hard limit. This job spawns dozens of concurrent
# PyTensor-compiling joblib workers; too few fds causes intermittent
# "FileNotFoundError ... _SemLock._rebuild" in loky worker startup. See
# docker/run_eicu_sepsis_bayes_slope_cache.sh for the same fix and full context.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "ulimit -n: $(ulimit -Sn) (hard: $(ulimit -Hn))"

echo "=========================================="
echo "eICU Sepsis Bayesian slope cache (container/PyMC)"
echo "=========================================="
echo "Marker    : ${MARKER}"
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
    -e TMPDIR=/cache/.job_tmp/eicu_sepsis_bayes_slope_cache/${SLURM_JOB_ID:-local} \
    -e NUMBA_DISABLE_CACHING=1 \
    -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    "${CONTAINER_NAME}" \
    python /traj/scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py \
        --backend bayes \
        --marker "${MARKER}" \
        --base-dir /data/eicu/sepsis \
        --output-dir /traj/results/eicu/sepsis \
        --bayes-sampler pymc \
        --cache-n-jobs "${SLURM_CPUS_PER_TASK:-64}" \
        --build-cache-only \
        --cohort-splits ${COHORT_SPLITS} \
        --cohort-index  ${COHORT_INDEX}

echo ""
echo "End time: $(date)"
echo "=========================================="
