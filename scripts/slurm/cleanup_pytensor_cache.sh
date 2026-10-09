#!/usr/bin/env bash
# Remove PyTensor cache for this SLURM job and prune old cache dirs.
# PyTensor cache is routed to TRAJ_CACHE_ROOT (gaga home, ~150 GB quota)
# not $HOME (CS home, 4 GB quota).

set -euo pipefail

HOME_DIR=${HOME:-$(eval echo ~${USER:-$(whoami)})}
CACHE_BASE="${TRAJ_CACHE_ROOT:-$HOME_DIR}"
PYTENSOR_CACHE_DIR="${CACHE_BASE}/.pytensor_cache"

echo "[cleanup_pytensor_cache] running cleanup for job ${SLURM_JOB_ID:-unknown}"

if [ -d "$PYTENSOR_CACHE_DIR" ]; then
    # Clean by SLURM_JOB_ID (used by SLURM scripts)
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        TARGET="$PYTENSOR_CACHE_DIR/${SLURM_JOB_ID}"
        if [ -d "$TARGET" ]; then
            echo "[cleanup_pytensor_cache] removing $TARGET"
            rm -rf "$TARGET" || true
        fi
    fi
    # Clean by ARRAY_JOB_ID_TASK_ID (used by Python scripts)
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        ARRAY_TARGET="$PYTENSOR_CACHE_DIR/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        if [ -d "$ARRAY_TARGET" ]; then
            echo "[cleanup_pytensor_cache] removing $ARRAY_TARGET"
            rm -rf "$ARRAY_TARGET" || true
        fi
    fi
    echo "[cleanup_pytensor_cache] pruning cache dirs older than 1 day in $PYTENSOR_CACHE_DIR"
    find "$PYTENSOR_CACHE_DIR" -maxdepth 1 -mindepth 1 -type d -mtime +1 -print -exec rm -rf {} + || true
fi

# Also clean HiRID-style per-job pytensor dirs (CACHE_BASE/.pytensor_{job_id})
if [ -n "${SLURM_JOB_ID:-}" ]; then
    HIRID_TARGET="$CACHE_BASE/.pytensor_${SLURM_JOB_ID}"
    if [ -d "$HIRID_TARGET" ]; then
        echo "[cleanup_pytensor_cache] removing $HIRID_TARGET"
        rm -rf "$HIRID_TARGET" || true
    fi
fi
find "$CACHE_BASE" -maxdepth 1 -name ".pytensor_*" -type d -mtime +7 -print -exec rm -rf {} + 2>/dev/null || true

echo "[cleanup_pytensor_cache] done"

# Prune stranded .job_tmp dirs (left behind when SIGKILL bypasses the EXIT trap).
JOB_TMP_BASE="${CACHE_BASE}/.job_tmp"
if [ -d "$JOB_TMP_BASE" ]; then
    find "$JOB_TMP_BASE" -maxdepth 2 -mindepth 2 -type d -mtime +1 -print -exec rm -rf {} + 2>/dev/null || true
fi

# Remove orphaned tmp* compilation artifacts from the repo root.
# pytensor's compile_function_src creates NamedTemporaryFile(delete=False)
# which can land in CWD (repo root) when TMPDIR resolution falls back.
REPO_ROOT_CLEANUP="${SLURM_SUBMIT_DIR:-}"
if [ -z "$REPO_ROOT_CLEANUP" ] && [ -f "pyproject.toml" ]; then
    REPO_ROOT_CLEANUP="$PWD"
fi
if [ -n "$REPO_ROOT_CLEANUP" ] && [ -f "$REPO_ROOT_CLEANUP/pyproject.toml" ]; then
    n=$(find "$REPO_ROOT_CLEANUP" -maxdepth 1 -name 'tmp*' 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "[cleanup_pytensor_cache] removing $n tmp* artifacts from repo root"
        find "$REPO_ROOT_CLEANUP" -maxdepth 1 -name 'tmp*' -delete 2>/dev/null || true
    fi
fi
