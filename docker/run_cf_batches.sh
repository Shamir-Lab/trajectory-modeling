#!/usr/bin/env bash
# run_cf_batches.sh — Sequential (non-SLURM) driver for Circulatory Failure Bayesian
# trajectory probabilities, for HiRID, MIMIC, or eICU, using the g++-accelerated
# `trajectory-modeling` UDocker container (see docs/CONTAINER_GUIDE.md).
#
# Generalizes docker/run_hirid_cf_batches.sh to all 3 datasets (that script is left
# untouched since a HiRID run may depend on it) — same design, dispatched per --dataset:
#   - Processes cohorts ONE AT A TIME (bounded CPU/RAM on a shared, unscheduled machine).
#   - Skips a cohort's already-complete work (auto-detected from files already on disk),
#     so it's safe to interrupt and resume, and safe to re-run after a crash.
#   - Routes ALL scratch (PyTensor compile cache, TMPDIR, Numba, matplotlib, arviz) to
#     /dev/shm (RAM-backed tmpfs), NOT gaga home, deleting each cohort's scratch dir
#     immediately after that cohort finishes.
#   - Re-checks gaga quota headroom before every cohort and aborts cleanly rather than
#     continuing into silent EDQUOT corruption.
#   - Designed to run unmodified on any server that mounts the same gaga/data filesystems
#     at the same paths — see --help for the UDOCKER_DIR note below.
#
# Per-dataset dispatch:
#   hirid : scripts/trajectory/hirid/circulatory_failure_trajs.py
#           always computes lactate+heartrate+systolic in one invocation, with its own
#           per-biomarker skip-if-exists. cohort-splits default: 16.
#   mimic : scripts/trajectory/mimic/circulatory_failure_trajs.py --biomarker all
#           (NEVER use single-biomarker mode — it derives the cohort-patient partition
#           from a different source file than --biomarker all does, which silently
#           produces inconsistent per-biomarker cohorts; see feedback_slurm_bugs Bug 7
#           for the HiRID incident this exact failure mode caused). Has its own
#           per-biomarker skip-if-exists (patched in). cohort-splits default: 24.
#   eicu  : scripts/trajectory/eicu/circulatory_failure_trajs.py
#           always computes lactate+heartrate+systolic in one invocation, with its own
#           per-biomarker skip-if-exists (matches hirid/mimic — previously this dataset
#           held all 3 biomarkers in memory and only wrote a single merged file at the
#           very end, so a kill anywhere in a cohort lost 100% of that cohort's compute;
#           fixed 2026-08-15). cohort-splits default: 8.
#
# Logs (clear + easy to access):
#   logs/container_runs/<dataset>_cf/<RUN_ID>/
#     run.log             <- tail -f this for a live, single-file view of the whole run
#     status.tsv          <- one row per cohort: start/end/duration/exit code/status
#     cohort<NN>.out/.err <- full stdout/stderr for that cohort's container run
#
# Usage:
#   bash docker/run_cf_batches.sh --dataset mimic                     # sane defaults, foreground
#   nohup bash docker/run_cf_batches.sh --dataset mimic > /dev/null 2>&1 &   # background, multi-day run
#   bash docker/run_cf_batches.sh --dataset eicu --dry-run             # show plan, run nothing
#   bash docker/run_cf_batches.sh --dataset mimic --cohorts 3,4,10-12  # only these cohort indices
#
# Running on a DIFFERENT server than where this was set up:
#   The udocker container's files live under $UDOCKER_DIR (default below), which is on
#   the shared gaga filesystem — no rebuild needed on another server that mounts gaga at
#   the same path. This script exports UDOCKER_DIR itself (doesn't rely on shell profile
#   parity across servers). It still needs the `udocker` BINARY to be installed on that
#   server (a local, non-NFS system path) — the preflight check below fails clearly if not.
#
# Multi-host parallelism (--lock-suffix):
#   By default this script takes a single per-dataset lock, so only one instance of a
#   given --dataset can run anywhere at a time — this is what makes it safe to resume
#   blindly after a crash/reboot without double-processing a cohort. To instead run
#   DIFFERENT, DISJOINT cohort ranges of the SAME dataset concurrently on multiple hosts
#   (e.g. to cut MIMIC's ~24-cohort sequential runtime by ~Nx across N hosts), give each
#   concurrent invocation its own --lock-suffix and an explicit --cohorts range:
#     Host A: --dataset mimic --cohorts 0-7   --lock-suffix a
#     Host B: --dataset mimic --cohorts 8-15  --lock-suffix b
#     Host C: --dataset mimic --cohorts 16-23 --lock-suffix c
#   --lock-suffix requires an explicit --cohorts list (never "auto") — two shards both
#   set to "auto" would independently pick the same incomplete cohorts and duplicate
#   work. YOU are responsible for keeping shard --cohorts ranges disjoint; the lock only
#   prevents two processes from sharing one lock file, not from choosing overlapping
#   ranges under different suffixes. The cf_batch_watchdog.sh cron watchdog only knows
#   about the default (no-suffix) single-instance lock — if you shard a dataset manually,
#   don't also rely on the watchdog to babysit it; resume that dataset's shards yourself.
#
# Options:
#   --dataset {hirid,mimic,eicu} REQUIRED
#   --cohorts LIST               Comma list / ranges of cohort indices (default: auto —
#                                 every cohort with incomplete output)
#   --lock-suffix SUFFIX          Use a per-shard lock file (.{dataset}_cf_batches_{suffix}.lock)
#                                 instead of the default per-dataset lock, so multiple
#                                 disjoint cohort ranges can run concurrently — see above.
#                                 Requires an explicit --cohorts (not "auto"). Default: none.
#   --cohort-splits N            Total cohort splits (default: dataset-specific, see above —
#                                 changing it reshuffles which patients land in which cohort
#                                 index, so don't change once a dataset has partial progress)
#   --n-batches N                 Patient sub-batches per biomarker per cohort (default: 4 — kept
#                                 low so joblib has enough windows per batch to fill --heavy-max-jobs
#                                 workers; a higher n-batches under-fills the pool, see HiRID incident)
#   --window-hours H               Lookback window hours (default: 12.0)
#   --cpus N                       CPUs for the light (lactate) config (default: 32)
#   --heavy-max-jobs N             CPU cap for heavy biomarkers, heartrate/systolic (default: 32)
#   --container NAME                UDocker container name (default: trajectory-modeling)
#   --udocker-dir DIR                UDOCKER_DIR to export (default: /home/gaga/tamarw1/udocker_dir)
#   --data-root DIR                   Host data root (default: /home/gaga/data/physionet)
#   --cache-root DIR                   Scratch root, RAM-backed (default: /dev/shm/$USER/trajmodel_cache)
#   --min-quota-headroom-gb N        Abort if gaga headroom falls below this (default: 5)
#   --dry-run                         Print the plan and exit; run nothing
#   --help                             Show this help

set -uo pipefail  # deliberately NOT -e: one failed cohort must not kill the run

# ── Locate repo root ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "ERROR: could not locate repo root" >&2; exit 1; }

# ── Defaults ───────────────────────────────────────────────────────────────────
DATASET=""
COHORTS_ARG="auto"
LOCK_SUFFIX=""
COHORT_SPLITS=""   # dataset-specific default filled in after parsing --dataset
N_BATCHES=4
WINDOW_HOURS=12.0
CPUS=32
HEAVY_MAX_JOBS=32
CONTAINER_NAME=trajectory-modeling
UDOCKER_DIR_OPT="/home/gaga/tamarw1/udocker_dir"
DATA_ROOT="/home/gaga/data/physionet"
CACHE_ROOT_OPT=""  # filled in after --dataset is known, unless overridden
MIN_QUOTA_HEADROOM_GB=5
DRY_RUN=0

# ── Arg parsing ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)                 DATASET="$2"; shift 2 ;;
        --cohorts)                 COHORTS_ARG="$2"; shift 2 ;;
        --lock-suffix)             LOCK_SUFFIX="$2"; shift 2 ;;
        --cohort-splits)           COHORT_SPLITS="$2"; shift 2 ;;
        --n-batches)               N_BATCHES="$2"; shift 2 ;;
        --window-hours)            WINDOW_HOURS="$2"; shift 2 ;;
        --cpus)                    CPUS="$2"; shift 2 ;;
        --heavy-max-jobs)          HEAVY_MAX_JOBS="$2"; shift 2 ;;
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

case "$DATASET" in
    hirid) [[ -z "$COHORT_SPLITS" ]] && COHORT_SPLITS=16 ;;
    mimic) [[ -z "$COHORT_SPLITS" ]] && COHORT_SPLITS=24 ;;
    eicu)  [[ -z "$COHORT_SPLITS" ]] && COHORT_SPLITS=8 ;;
    *) echo "ERROR: --dataset is required and must be one of: hirid, mimic, eicu" >&2; exit 1 ;;
esac

if [[ -n "$LOCK_SUFFIX" && "$COHORTS_ARG" == "auto" ]]; then
    echo "ERROR: --lock-suffix requires an explicit --cohorts range (not \"auto\") —" >&2
    echo "  two shards both left on \"auto\" would independently pick the same" >&2
    echo "  incomplete cohorts and duplicate work. See --help for the sharding pattern." >&2
    exit 1
fi

export UDOCKER_DIR="$UDOCKER_DIR_OPT"
CACHE_ROOT="${CACHE_ROOT_OPT:-/dev/shm/${USER}/trajmodel_cache/${DATASET}}"

DATA_DIR="${DATA_ROOT}/${DATASET}/circulatory_failure"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${REPO_ROOT}/logs/container_runs/${DATASET}_cf/${RUN_ID}"
mkdir -p "$LOG_ROOT"
MASTER_LOG="${LOG_ROOT}/run.log"
STATUS_TSV="${LOG_ROOT}/status.tsv"
printf "cohort\tstart\tend\tduration_s\texit_code\tstatus\n" > "$STATUS_TSV"

# Mirror everything to run.log while still printing to the terminal live.
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=========================================="
echo "${DATASET} Circulatory Failure — sequential container batch run"
echo "=========================================="
echo "Run ID       : ${RUN_ID}"
echo "Host         : $(hostname)"
echo "Repo root    : ${REPO_ROOT}"
echo "Data dir     : ${DATA_DIR}"
echo "Container    : ${CONTAINER_NAME}"
echo "UDOCKER_DIR  : ${UDOCKER_DIR}"
echo "Cohort splits: ${COHORT_SPLITS}"
echo "Lock suffix  : ${LOCK_SUFFIX:-(none — default per-dataset lock)}"
echo "N batches    : ${N_BATCHES}"
echo "Window hours : ${WINDOW_HOURS}"
echo "CPUs (light) : ${CPUS}   CPUs (heavy cap): ${HEAVY_MAX_JOBS}"
echo "Cache root   : ${CACHE_ROOT}  (RAM-backed tmpfs, cleaned per-cohort)"
echo "Logs         : ${LOG_ROOT}"
echo "=========================================="

# ── Single-instance lock, per dataset (lock file lives on shared NFS, so this also
#    prevents accidentally double-running the SAME dataset from two different servers) ──
# With --lock-suffix, this becomes a per-SHARD lock instead — deliberately opt-in, see
# the --lock-suffix note near the top of this file for the safety tradeoffs.
if [[ -n "$LOCK_SUFFIX" ]]; then
    LOCK_FILE="${REPO_ROOT}/.${DATASET}_cf_batches_${LOCK_SUFFIX}.lock"
else
    LOCK_FILE="${REPO_ROOT}/.${DATASET}_cf_batches.lock"
fi
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "ERROR: another run_cf_batches.sh --dataset ${DATASET} (lock-suffix '${LOCK_SUFFIX}') is already running (lock: ${LOCK_FILE})." >&2
    exit 1
fi

# ── Preflight checks ───────────────────────────────────────────────────────────
command -v udocker >/dev/null 2>&1 || { echo "ERROR: udocker not found in PATH on $(hostname)." >&2; exit 1; }

if ! udocker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "ERROR: udocker container '${CONTAINER_NAME}' not found under UDOCKER_DIR=${UDOCKER_DIR}." >&2
    echo "       See docs/CONTAINER_GUIDE.md. If UDOCKER_DIR is on shared gaga storage this" >&2
    echo "       should just work on any server mounting it at the same path — double-check" >&2
    echo "       --udocker-dir / UDOCKER_DIR if it doesn't." >&2
    exit 1
fi

case "$DATASET" in
    hirid)
        for f in lactate_timeseries.csv heartrate_timeseries.csv systolic_timeseries.csv; do
            [[ -f "${DATA_DIR}/${f}" ]] || { echo "ERROR: missing input ${DATA_DIR}/${f}" >&2; exit 1; }
        done
        ;;
    mimic)
        for f in lactate_timeseries.csv heartrate_timeseries.csv systolic_timeseries.csv circulatory_failure_prediction_dataset.csv; do
            [[ -f "${DATA_DIR}/${f}" ]] || { echo "ERROR: missing input ${DATA_DIR}/${f}" >&2; exit 1; }
        done
        ;;
    eicu)
        for f in lactate_timeseries.csv heartrate_timeseries.csv systolic_timeseries.csv circulatory_failure_prediction_dataset.csv; do
            [[ -f "${DATA_DIR}/${f}" ]] || { echo "ERROR: missing input ${DATA_DIR}/${f}" >&2; exit 1; }
        done
        ;;
esac

# gaga home quota headroom (netapp2 gaga volume) — writes silently fail past quota,
# corrupting output (see project memory: feedback_slurm_bugs Bug 2, Bug 7).
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

# Container health check: actually run something inside it before committing to a whole
# cohort loop. Catches host-level incompatibilities (e.g. a CPU too old for the
# container's compiled binaries — hit this exact failure mode on an AMD Opteron 6140,
# which predates SSSE3/SSE4.x/AVX) in one clear failure instead of N identical opaque
# exit=255s. `python3 --version` alone is NOT enough — it short-circuits most of
# CPython's startup and doesn't exercise numpy/scipy/pymc's compiled, CPU-dispatched
# code (confirmed: --version passed on the Opteron while the real workload still
# SIGILL'd on import). Import the same packages the real scripts import instead.
# Written to a scratch file rather than passed via `-c` — udocker/proot has been
# observed to mis-split a quoted -c argument containing spaces into separate argv
# entries, which looks like a crash but is a harness quoting quirk, not a real one.
echo "Preflight: container health check ($(hostname))..."
HEALTHCHECK_DIR="${CACHE_ROOT}/_healthcheck"
mkdir -p "$HEALTHCHECK_DIR"
cat > "${HEALTHCHECK_DIR}/check.py" <<'PYEOF'
import numpy as np
import pandas as pd
import pymc as pm
_ = np.arange(8) @ np.arange(8)
print("CONTAINER_HEALTH_CHECK_OK")
PYEOF
health_out="$(udocker run -v "${HEALTHCHECK_DIR}:/healthcheck" "${CONTAINER_NAME}" python3 /healthcheck/check.py 2>&1)"
rm -rf "$HEALTHCHECK_DIR"
if [[ "$health_out" != *"CONTAINER_HEALTH_CHECK_OK"* ]]; then
    echo "ERROR: container smoke test failed on $(hostname) — numpy/pandas/pymc did not import inside the container." >&2
    echo "  Output: ${health_out}" >&2
    echo "  If proot reports 'terminated with signal 4' (SIGILL) this is a CPU too old for the" >&2
    echo "  container's binaries (needs at least SSE4.2/AVX2-era x86-64) — use a different server." >&2
    exit 1
fi
echo "  OK"

# ── Expected output paths / completeness check per cohort (dataset-specific) ────
cohort_needs_work() {
    local cohort="$1" cp
    printf -v cp "%02d" "$cohort"
    case "$DATASET" in
        hirid|mimic|eicu)
            for b in lactate heartrate systolic; do
                local p="${DATA_DIR}/${b}_trajectory_probs_bayes_cohort${cp}.csv"
                [[ -f "$p" && -s "$p" ]] || return 0
            done
            return 1
            ;;
    esac
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
    for ((i=0; i<COHORT_SPLITS; i++)); do
        if cohort_needs_work "$i"; then
            COHORT_INDICES+=("$i")
        fi
    done
else
    COHORT_INDICES=($(expand_cohort_list "$COHORTS_ARG"))
fi

echo ""
echo "Plan: ${#COHORT_INDICES[@]} / ${COHORT_SPLITS} cohorts need work: ${COHORT_INDICES[*]:-none}"
for ((i=0; i<COHORT_SPLITS; i++)); do
    if ! cohort_needs_work "$i"; then
        echo "  cohort $(printf '%02d' "$i"): already complete — will skip"
    fi
done
echo ""

if [[ "${#COHORT_INDICES[@]}" -eq 0 ]]; then
    echo "Nothing to do — all cohorts already complete."
    exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "--dry-run: exiting without launching any container."
    exit 0
fi

# ── Cleanup on interrupt/exit: remove this run's whole scratch tree, and make sure
#    an interrupted cohort's udocker/loky worker tree actually dies with it ──────
# Killing just the top-level udocker PID does NOT reap its joblib/loky worker
# children — they're reparented to init and keep running (against a cache dir this
# script is about to delete, causing crash-loop noise). Confirmed the hard way
# earlier this session. Fix: run each cohort's udocker command in its own process
# group (setsid) and kill the whole group, not just one PID, on interrupt.
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

# ── Per-dataset container command builder ────────────────────────────────────
build_container_cmd() {
    local cohort="$1"
    case "$DATASET" in
        hirid)
            echo python /traj/scripts/trajectory/hirid/circulatory_failure_trajs.py \
                --input-dir  /data/hirid/circulatory_failure \
                --output-dir /data/hirid/circulatory_failure \
                --window-hours "${WINDOW_HOURS}" \
                --n-batches "${N_BATCHES}" \
                --sampler pymc \
                --cohort-splits "${COHORT_SPLITS}" \
                --cohort-index "${cohort}"
            ;;
        mimic)
            echo python /traj/scripts/trajectory/mimic/circulatory_failure_trajs.py \
                --biomarker all \
                --data-dir /data/mimic/circulatory_failure \
                --pred-dataset /data/mimic/circulatory_failure/circulatory_failure_prediction_dataset.csv \
                --window-hours "${WINDOW_HOURS}" \
                --n-batches "${N_BATCHES}" \
                --sampler pymc \
                --cohort-splits "${COHORT_SPLITS}" \
                --cohort-index "${cohort}"
            ;;
        eicu)
            echo python /traj/scripts/trajectory/eicu/circulatory_failure_trajs.py \
                --data-dir /data/eicu/circulatory_failure \
                --pred-dataset /data/eicu/circulatory_failure/circulatory_failure_prediction_dataset.csv \
                --window-hours "${WINDOW_HOURS}" \
                --n-batches "${N_BATCHES}" \
                --sampler pymc \
                --cohort-splits "${COHORT_SPLITS}" \
                --cohort-index "${cohort}"
            ;;
    esac
}

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
    mkdir -p "$COHORT_CACHE"/{pytensor,tmp,matplotlib,arviz,xdg_cache}

    OUT_LOG="${LOG_ROOT}/cohort${cohort_padded}.out"
    ERR_LOG="${LOG_ROOT}/cohort${cohort_padded}.err"

    cohort_start=$(date +%s)

    read -ra CONTAINER_CMD <<< "$(build_container_cmd "$cohort")"

    setsid udocker run \
        -v "${REPO_ROOT}:/traj" \
        -v "${DATA_ROOT}:/data" \
        -v "${COHORT_CACHE}:/cache" \
        -e SLURM_JOB_ID="${RUN_ID}" \
        -e SLURM_ARRAY_TASK_ID="${cohort}" \
        -e SLURM_CPUS_PER_TASK="${CPUS}" \
        -e TRAJ_HEAVY_MAX_JOBS="${HEAVY_MAX_JOBS}" \
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
        "${CONTAINER_CMD[@]}" \
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

    # Free the RAM-backed scratch for this cohort. The process group this cohort ran in
    # has already exited normally by this point (wait returned) — orphaned workers are
    # only a risk on interrupt, which kill_current_cohort (EXIT/INT/TERM trap) handles.
    rm -rf "$COHORT_CACHE"
done

RUN_END=$(date +%s)

echo ""
echo "=========================================="
echo "Run ${RUN_ID} complete (dataset=${DATASET})"
echo "=========================================="
echo "Attempted : $((N_DONE + N_FAILED)) / ${N_TOTAL}"
echo "Succeeded : ${N_DONE}"
echo "Failed    : ${N_FAILED}"
echo "Elapsed   : $(( (RUN_END - RUN_START) / 60 )) min"
echo "Status table: ${STATUS_TSV}"
echo "Full log    : ${MASTER_LOG}"
if [[ "$N_FAILED" -gt 0 ]]; then
    echo ""
    echo "Some cohorts failed — check the .err logs above, then re-run this script"
    echo "(it will only re-attempt cohorts that are still missing output)."
fi
echo ""
echo "Once all cohorts are complete, merge with:"
echo "  N_COHORTS=${COHORT_SPLITS} bash scripts/slurm/${DATASET}/merge_circulatory_failure_bayes.sh"
echo "=========================================="

[[ "$N_FAILED" -eq 0 ]]
