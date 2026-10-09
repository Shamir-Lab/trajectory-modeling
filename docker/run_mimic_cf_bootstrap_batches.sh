#!/usr/bin/env bash
# run_mimic_cf_bootstrap_batches.sh — Sequential (non-SLURM) driver for MIMIC Circulatory
# Failure BOOTSTRAP trajectory probabilities, using the g++-accelerated `trajectory-modeling`
# UDocker container (see docs/CONTAINER_GUIDE.md).
#
# Standalone sibling of docker/run_cf_batches.sh (which drives the BAYESIAN CF trajectory
# scripts for hirid/mimic/eicu) — deliberately NOT folded into that script, to avoid any
# risk to its live Bayesian shards (run_cf_batches.sh is left completely untouched, including
# its lock files). This script:
#   - Calls scripts/trajectory/mimic/circulatory_failure_trajs_bootstrap.py (bootstrap
#     backend, ALL 3 biomarkers — lactate/heartrate/systolic — computed by one invocation),
#     not the bayes script.
#   - Processes cohorts ONE AT A TIME (bounded CPU/RAM on a shared, unscheduled machine).
#   - Skips a cohort's already-complete work (auto-detected from files already on disk),
#     so it's safe to interrupt and resume, and safe to re-run after a crash.
#   - Routes ALL scratch (TMPDIR, Numba, matplotlib) to /dev/shm (RAM-backed tmpfs), NOT
#     gaga home, deleting each cohort's scratch dir immediately after that cohort finishes.
#   - Re-checks gaga quota headroom before every cohort and aborts cleanly rather than
#     continuing into silent EDQUOT corruption.
#
# *** IMPORTANT — stale pre-existing bootstrap output ***
#   MIMIC cohort00-04 bootstrap output existed on disk BEFORE this script was written,
#   computed 2026-03/2026-05 by scripts/slurm/mimic/run_circulatory_failure_bootstrap.slurm
#   under COHORT_SPLITS=5 and --n-bootstrap 1000 — NOT this script's COHORT_SPLITS=24 /
#   --n-bootstrap 200 defaults. Cohort index N under a 5-way split is a DIFFERENT set of
#   patients than cohort index N under a 24-way split (same failure mode as the HiRID
#   "cohort partition drift" incident — see project memory feedback_slurm_bugs Bug 7).
#   Those old per-biomarker cohort00-04 files (and the merged
#   circulatory_failure_prediction_dataset_with_bootstrap_probs_cohort00-04.parquet files)
#   were LEFT IN PLACE as of 2026-09-01 (an attempt to archive them to
#   _stale_5way_bootstrap_backup/ was blocked by the operator's tooling permissions) —
#   under this script's default --cohort-splits 24, `--cohorts auto` would see those stale
#   files, wrongly conclude cohorts 0-4 are already complete, and SKIP them, silently
#   leaving 5-way-partitioned data mislabeled as 24-way cohorts. DO NOT use `--cohorts auto`
#   until either (a) those 9 stale files are moved/deleted, or (b) you've re-verified they
#   were regenerated under --cohort-splits 24. Until then, always pass an explicit --cohorts
#   list/range (which bypasses the skip-check entirely) — every cohort0-23 launch this
#   script's own header comment or a caller documents should do this.
#
# Logs (clear + easy to access):
#   logs/container_runs/mimic_cf_bootstrap/<RUN_ID>[_<suffix>]/
#     run.log             <- tail -f this for a live, single-file view of the whole run
#     status.tsv          <- one row per cohort: start/end/duration/exit code/status
#     cohort<NN>.out/.err <- full stdout/stderr for that cohort's container run
#
# Usage:
#   bash docker/run_mimic_cf_bootstrap_batches.sh --cohorts 5                # one cohort
#   bash docker/run_mimic_cf_bootstrap_batches.sh --cohorts 5-11 --lock-suffix a --cpus 16
#   nohup bash docker/run_mimic_cf_bootstrap_batches.sh --cohorts 12-17 --lock-suffix b --cpus 20 > /dev/null 2>&1 &
#   bash docker/run_mimic_cf_bootstrap_batches.sh --cohorts 5-23 --dry-run    # show plan, run nothing
#
# Multi-host parallelism (--lock-suffix):
#   By default this script takes a single lock, so only one instance can run anywhere at a
#   time. To run DIFFERENT, DISJOINT cohort ranges concurrently on multiple hosts, give each
#   concurrent invocation its own --lock-suffix and an explicit --cohorts range, e.g.:
#     Host A: --cohorts 5-9   --lock-suffix a --cpus 20
#     Host B: --cohorts 10-14 --lock-suffix b --cpus 20
#     Host C: --cohorts 15-19 --lock-suffix c --cpus 60
#   --lock-suffix requires an explicit --cohorts list (never "auto") for the same reason as
#   run_cf_batches.sh: two shards both left on "auto" would independently pick the same
#   incomplete cohorts and duplicate work. YOU are responsible for keeping shard --cohorts
#   ranges disjoint; the lock only prevents two processes from sharing one lock file.
#
# Lock namespace: .mimic_cf_bootstrap_batches[_suffix].lock — deliberately DISTINCT from
# run_cf_batches.sh's .mimic_cf_batches[_suffix].lock (bayes), even though the bayes MIMIC
# run is complete — kept namespace-separate on principle so a future bayes re-run and this
# bootstrap run can never collide on the same lock file.
#
# Options:
#   --cohorts LIST                Comma list / ranges of cohort indices (default: auto — every
#                                  cohort with incomplete output; see stale-file warning above)
#   --lock-suffix SUFFIX           Use a per-shard lock file (.mimic_cf_bootstrap_batches_{suffix}.lock)
#                                   instead of the default single-instance lock. Requires an
#                                   explicit --cohorts (not "auto"). Default: none.
#   --cohort-splits N               Total cohort splits (default: 24, matches the bayes MIMIC
#                                   run's convention — don't change once this dataset has
#                                   partial progress under a given split count).
#   --n-bootstrap N                  Bootstrap resamples per window (default: 200 — the
#                                   bootstrap script's own default; NOT the 1000 used by the
#                                   old, now-stale SLURM 5-way run).
#   --n-batches N                     Patient sub-batches per biomarker per cohort (default: 8,
#                                   the bootstrap script's own default).
#   --window-hours H                   Lookback window hours (default: 12.0)
#   --cpus N                            CPUs for this cohort — sets SLURM_CPUS_PER_TASK (which
#                                   circulatory_failure_trajs_bootstrap.py reads first) and
#                                   TRAJ_HEAVY_MAX_JOBS (its fallback) so the n_jobs fix takes
#                                   effect regardless of which var the script consults
#                                   (default: 32).
#   --output-format {csv,parquet,both} (default: parquet)
#   --container NAME                     UDocker container name (default: trajectory-modeling)
#   --udocker-dir DIR                     UDOCKER_DIR to export (default: /home/gaga/tamarw1/udocker_dir)
#   --data-root DIR                        Host data root (default: /home/gaga/data/physionet)
#   --cache-root DIR                        Scratch root, RAM-backed (default: /dev/shm/$USER/trajmodel_cache/mimic_bootstrap)
#   --min-quota-headroom-gb N              Abort if gaga headroom falls below this (default: 5)
#   --dry-run                                Print the plan and exit; run nothing
#   --help                                    Show this help

set -uo pipefail  # deliberately NOT -e: one failed cohort must not kill the run

# ── Locate repo root ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "ERROR: could not locate repo root" >&2; exit 1; }

# ── Defaults ───────────────────────────────────────────────────────────────────
COHORTS_ARG="auto"
LOCK_SUFFIX=""
COHORT_SPLITS=24
N_BOOTSTRAP=200
N_BATCHES=8
WINDOW_HOURS=12.0
CPUS=32
OUTPUT_FORMAT="parquet"
CONTAINER_NAME=trajectory-modeling
UDOCKER_DIR_OPT="/home/gaga/tamarw1/udocker_dir"
DATA_ROOT="/home/gaga/data/physionet"
CACHE_ROOT_OPT=""
MIN_QUOTA_HEADROOM_GB=5
DRY_RUN=0

# ── Arg parsing ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cohorts)                 COHORTS_ARG="$2"; shift 2 ;;
        --lock-suffix)             LOCK_SUFFIX="$2"; shift 2 ;;
        --cohort-splits)           COHORT_SPLITS="$2"; shift 2 ;;
        --n-bootstrap)             N_BOOTSTRAP="$2"; shift 2 ;;
        --n-batches)               N_BATCHES="$2"; shift 2 ;;
        --window-hours)            WINDOW_HOURS="$2"; shift 2 ;;
        --cpus)                    CPUS="$2"; shift 2 ;;
        --output-format)           OUTPUT_FORMAT="$2"; shift 2 ;;
        --container)               CONTAINER_NAME="$2"; shift 2 ;;
        --udocker-dir)             UDOCKER_DIR_OPT="$2"; shift 2 ;;
        --data-root)                DATA_ROOT="$2"; shift 2 ;;
        --cache-root)               CACHE_ROOT_OPT="$2"; shift 2 ;;
        --min-quota-headroom-gb)   MIN_QUOTA_HEADROOM_GB="$2"; shift 2 ;;
        --dry-run)                 DRY_RUN=1; shift ;;
        --help|-h)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$LOCK_SUFFIX" && "$COHORTS_ARG" == "auto" ]]; then
    echo "ERROR: --lock-suffix requires an explicit --cohorts range (not \"auto\") —" >&2
    echo "  two shards both left on \"auto\" would independently pick the same" >&2
    echo "  incomplete cohorts and duplicate work. See --help for the sharding pattern." >&2
    exit 1
fi

export UDOCKER_DIR="$UDOCKER_DIR_OPT"
CACHE_ROOT="${CACHE_ROOT_OPT:-/dev/shm/${USER}/trajmodel_cache/mimic_bootstrap}"

DATA_DIR="${DATA_ROOT}/mimic/circulatory_failure"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$${LOCK_SUFFIX:+_$LOCK_SUFFIX}"
LOG_ROOT="${REPO_ROOT}/logs/container_runs/mimic_cf_bootstrap/${RUN_ID}"
mkdir -p "$LOG_ROOT"
MASTER_LOG="${LOG_ROOT}/run.log"
STATUS_TSV="${LOG_ROOT}/status.tsv"
printf "cohort\tstart\tend\tduration_s\texit_code\tstatus\n" > "$STATUS_TSV"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=========================================="
echo "MIMIC Circulatory Failure BOOTSTRAP — sequential container batch run"
echo "=========================================="
echo "Run ID       : ${RUN_ID}"
echo "Host         : $(hostname)"
echo "Repo root    : ${REPO_ROOT}"
echo "Data dir     : ${DATA_DIR}"
echo "Container    : ${CONTAINER_NAME}"
echo "UDOCKER_DIR  : ${UDOCKER_DIR}"
echo "Cohort splits: ${COHORT_SPLITS}"
echo "Lock suffix  : ${LOCK_SUFFIX:-(none — default single-instance lock)}"
echo "N bootstrap  : ${N_BOOTSTRAP}"
echo "N batches    : ${N_BATCHES}"
echo "Window hours : ${WINDOW_HOURS}"
echo "CPUs         : ${CPUS}"
echo "Output format: ${OUTPUT_FORMAT}"
echo "Cache root   : ${CACHE_ROOT}  (RAM-backed tmpfs, cleaned per-cohort)"
echo "Logs         : ${LOG_ROOT}"
echo "=========================================="

# ── Single-instance lock (own namespace, distinct from run_cf_batches.sh's bayes locks) ──
if [[ -n "$LOCK_SUFFIX" ]]; then
    LOCK_FILE="${REPO_ROOT}/.mimic_cf_bootstrap_batches_${LOCK_SUFFIX}.lock"
else
    LOCK_FILE="${REPO_ROOT}/.mimic_cf_bootstrap_batches.lock"
fi
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "ERROR: another run_mimic_cf_bootstrap_batches.sh (lock-suffix '${LOCK_SUFFIX}') is already running (lock: ${LOCK_FILE})." >&2
    exit 1
fi

# ── Preflight checks ───────────────────────────────────────────────────────────
command -v udocker >/dev/null 2>&1 || { echo "ERROR: udocker not found in PATH on $(hostname)." >&2; exit 1; }

if ! udocker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "ERROR: udocker container '${CONTAINER_NAME}' not found under UDOCKER_DIR=${UDOCKER_DIR}." >&2
    echo "       See docs/CONTAINER_GUIDE.md." >&2
    exit 1
fi

for f in lactate_timeseries.csv heartrate_timeseries.csv systolic_timeseries.csv circulatory_failure_prediction_dataset.csv; do
    [[ -f "${DATA_DIR}/${f}" ]] || { echo "ERROR: missing input ${DATA_DIR}/${f}" >&2; exit 1; }
done

check_quota_headroom() {
    local line used quota_limit used_b quota_b headroom_gb
    line="$(quota -s 2>/dev/null | grep -A1 'Netapp5_vol_1/gaga' | tail -1)"
    if [[ -z "$line" ]]; then
        echo "  (could not read gaga quota — skipping headroom check)"
        return 0
    fi
    used="$(awk '{print $1}' <<<"$line")"
    quota_limit="$(awk '{print $2}' <<<"$line")"
    used_b="$(numfmt --from=iec "$used" 2>/dev/null || echo 0)"
    quota_b="$(numfmt --from=iec "$quota_limit" 2>/dev/null || echo 0)"
    headroom_gb=$(( (quota_b - used_b) / 1024 / 1024 / 1024 ))
    echo "  gaga quota: used=${used} limit=${quota_limit} headroom=${headroom_gb}G"
    if (( headroom_gb < MIN_QUOTA_HEADROOM_GB )); then
        echo "ERROR: gaga quota headroom (${headroom_gb}G) below --min-quota-headroom-gb (${MIN_QUOTA_HEADROOM_GB}G)." >&2
        echo "       Aborting rather than risk silent write failures / corrupted output." >&2
        return 1
    fi
    return 0
}

echo "Preflight: checking gaga quota headroom..."
check_quota_headroom || exit 1

mkdir -p "$CACHE_ROOT"
avail_shm_kb="$(df -k --output=avail "$CACHE_ROOT" | tail -1)"
echo "Preflight: /dev/shm avail on $(hostname) = $(( avail_shm_kb / 1024 / 1024 ))G"

echo "Preflight: container health check ($(hostname))..."
HEALTHCHECK_DIR="${CACHE_ROOT}/_healthcheck"
mkdir -p "$HEALTHCHECK_DIR"
cat > "${HEALTHCHECK_DIR}/check.py" <<'PYEOF'
import numpy as np
import pandas as pd
_ = np.arange(8) @ np.arange(8)
print("CONTAINER_HEALTH_CHECK_OK")
PYEOF
health_out="$(udocker run -v "${HEALTHCHECK_DIR}:/healthcheck" "${CONTAINER_NAME}" python3 /healthcheck/check.py 2>&1)"
rm -rf "$HEALTHCHECK_DIR"
if [[ "$health_out" != *"CONTAINER_HEALTH_CHECK_OK"* ]]; then
    echo "ERROR: container smoke test failed on $(hostname) — numpy/pandas did not import inside the container." >&2
    echo "  Output: ${health_out}" >&2
    exit 1
fi
echo "  OK"

# ── Expected output paths / completeness check per cohort ───────────────────────
# NOTE: relies on cohortNN files reflecting the CURRENT --cohort-splits/--n-bootstrap.
# See the stale-file warning at the top of this file — do not trust this check for
# cohorts 0-4 until the pre-existing 5-way-split files are archived or regenerated.
cohort_needs_work() {
    local cohort="$1" cp ext
    printf -v cp "%02d" "$cohort"
    ext="parquet"
    [[ "$OUTPUT_FORMAT" == "csv" ]] && ext="csv"
    for b in lactate heartrate systolic; do
        local p="${DATA_DIR}/${b}_trajectory_probs_bootstrap_cohort${cp}.${ext}"
        [[ -f "$p" && -s "$p" ]] || return 0
    done
    return 1
}

# ── Resolve which cohort indices to run ──────────────────────────────────────
expand_cohort_list() {
    # "3,4,10-12" -> "3 4 10 11 12"
    local spec="$1" part lo hi out=()
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
            lo="${part%-*}"; hi="${part#*-}"
            for ((i=lo; i<=hi; i++)); do out+=("$i"); done
        else
            out+=("$part")
        fi
    done
    echo "${out[@]}"
}

declare -a COHORT_INDICES=()
if [[ "$COHORTS_ARG" == "auto" ]]; then
    echo "WARNING: --cohorts auto — relying on on-disk file state to decide what's done." >&2
    echo "         See the stale-5-way-split warning at the top of this script." >&2
    for ((i=0; i<COHORT_SPLITS; i++)); do
        if cohort_needs_work "$i"; then
            COHORT_INDICES+=("$i")
        fi
    done
else
    COHORT_INDICES=($(expand_cohort_list "$COHORTS_ARG"))
fi

echo ""
echo "Plan: ${#COHORT_INDICES[@]} / ${COHORT_SPLITS} cohorts to run: ${COHORT_INDICES[*]:-none}"
echo ""

if [[ "${#COHORT_INDICES[@]}" -eq 0 ]]; then
    echo "Nothing to do."
    exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "--dry-run: exiting without launching any container."
    exit 0
fi

# ── Cleanup on interrupt/exit ─────────────────────────────────────────────────
RUN_CACHE_ROOT="${CACHE_ROOT}/${RUN_ID}"
mkdir -p "$RUN_CACHE_ROOT"
CURRENT_PGID=""
kill_current_cohort() {
    if [[ -n "$CURRENT_PGID" ]]; then
        echo "Interrupted — killing current cohort's process group (pgid ${CURRENT_PGID})..." >&2
        kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
        sleep 2
        kill -KILL -"$CURRENT_PGID" 2>/dev/null || true
    fi
}
cleanup_run() {
    kill_current_cohort
    rm -rf "$RUN_CACHE_ROOT" 2>/dev/null || true
}
trap cleanup_run EXIT INT TERM

# ── Main sequential loop ──────────────────────────────────────────────────────
N_TOTAL=${#COHORT_INDICES[@]}
N_DONE=0
N_FAILED=0
RUN_START=$(date +%s)

for cohort in "${COHORT_INDICES[@]}"; do
    N_DONE_SO_FAR=$((N_DONE + N_FAILED))
    printf -v cohort_padded "%02d" "$cohort"

    echo ""
    echo "=========================================="
    echo "Cohort ${cohort_padded} (${N_DONE_SO_FAR}/${N_TOTAL} attempted so far)"
    echo "Time: $(date)"
    echo "=========================================="

    echo "Checking gaga quota headroom before this cohort..."
    if ! check_quota_headroom; then
        echo "Aborting remaining run — see quota error above."
        break
    fi

    COHORT_CACHE="${RUN_CACHE_ROOT}/cohort${cohort_padded}"
    mkdir -p "$COHORT_CACHE"/{tmp,matplotlib,arviz,xdg_cache}

    OUT_LOG="${LOG_ROOT}/cohort${cohort_padded}.out"
    ERR_LOG="${LOG_ROOT}/cohort${cohort_padded}.err"

    cohort_start=$(date +%s)

    setsid udocker run \
        -v "${REPO_ROOT}:/traj" \
        -v "${DATA_ROOT}:/data" \
        -v "${COHORT_CACHE}:/cache" \
        -e SLURM_JOB_ID="${RUN_ID}" \
        -e SLURM_ARRAY_TASK_ID="${cohort}" \
        -e SLURM_CPUS_PER_TASK="${CPUS}" \
        -e TRAJ_HEAVY_MAX_JOBS="${CPUS}" \
        -e TRAJ_CACHE_ROOT=/cache \
        -e TRAJ_DATA_ROOT=/data \
        -e TMPDIR=/cache/tmp \
        -e TEMP=/cache/tmp \
        -e TMP=/cache/tmp \
        -e NUMBA_DISABLE_CACHING=1 \
        -e MPLCONFIGDIR=/cache/matplotlib \
        -e ARVIZ_DATA_HOME=/cache/arviz \
        -e XDG_CACHE_HOME=/cache/xdg_cache \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 \
        "${CONTAINER_NAME}" \
        python /traj/scripts/trajectory/mimic/circulatory_failure_trajs_bootstrap.py \
            --data-dir /data/mimic/circulatory_failure \
            --pred-dataset /data/mimic/circulatory_failure/circulatory_failure_prediction_dataset.csv \
            --window-hours "${WINDOW_HOURS}" \
            --n-bootstrap "${N_BOOTSTRAP}" \
            --n-batches "${N_BATCHES}" \
            --cohort-splits "${COHORT_SPLITS}" \
            --cohort-index "${cohort}" \
            --output-format "${OUTPUT_FORMAT}" \
        > >(tee -a "$OUT_LOG") 2> >(tee -a "$ERR_LOG" >&2) &
    child_pid=$!
    CURRENT_PGID="$(ps -o pgid= -p "$child_pid" 2>/dev/null | tr -d ' ')"
    wait "$child_pid"
    exit_code=$?
    CURRENT_PGID=""

    cohort_end=$(date +%s)
    duration=$((cohort_end - cohort_start))

    if [[ $exit_code -eq 0 ]]; then
        status="OK"
        N_DONE=$((N_DONE + 1))
    else
        status="FAILED"
        N_FAILED=$((N_FAILED + 1))
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$cohort_padded" "$(date -d "@$cohort_start" +%FT%T)" "$(date -d "@$cohort_end" +%FT%T)" \
        "$duration" "$exit_code" "$status" >> "$STATUS_TSV"

    echo "Cohort ${cohort_padded}: ${status} (exit=${exit_code}, ${duration}s). Logs: ${OUT_LOG} / ${ERR_LOG}"

    if [[ "$status" == "OK" ]]; then
        rm -f "$ERR_LOG"
    fi

    rm -rf "$COHORT_CACHE"
done

RUN_END=$(date +%s)

echo ""
echo "=========================================="
echo "Run ${RUN_ID} complete (mimic bootstrap)"
echo "=========================================="
echo "Attempted : $((N_DONE + N_FAILED)) / ${N_TOTAL}"
echo "Succeeded : ${N_DONE}"
echo "Failed    : ${N_FAILED}"
echo "Elapsed   : $(( (RUN_END - RUN_START) / 60 )) min"
echo "Status table: ${STATUS_TSV}"
echo "Full log    : ${MASTER_LOG}"
if [[ "$N_FAILED" -gt 0 ]]; then
    echo ""
    echo "Some cohorts failed — check the .err logs above, then re-run with an explicit"
    echo "--cohorts list covering the failed indices."
fi
echo ""
echo "=========================================="

[[ "$N_FAILED" -eq 0 ]]
