#!/usr/bin/env bash
# run_hirid_cf_batches.sh — Sequential (non-SLURM) driver for HiRiD Circulatory
# Failure Bayesian trajectory probabilities, using the g++-accelerated
# `trajectory-modeling` UDocker container (see docs/CONTAINER_GUIDE.md).
#
# Why this exists: SLURM array jobs (scripts/slurm/hirid/run_hirid_circulatory_failure_trajs_container.slurm)
# run all 16 cohorts in parallel, which is fine when SLURM is arbitrating shared
# nodes but is not appropriate to fire off directly on the login/interactive
# server. This script instead processes cohorts ONE AT A TIME, so:
#   - CPU/RAM use is bounded and predictable on a shared, unscheduled machine
#   - PyTensor/Numba compile caches never pile up across 16 concurrent jobs
#   - the gaga home quota (often nearly full — see docs/CONTAINER_GUIDE.md and
#     project memory) never gets hit by 16x simultaneous cache writes
#
# What it does per cohort (of --cohort-splits, default 16 — MUST stay 16 to
# stay aligned with already-computed cohortNN files on disk; changing it
# reshuffles which patients land in which cohort index):
#   - Skips the cohort entirely if lactate/heartrate/systolic outputs already
#     exist and are non-empty (auto-detected up front; the underlying Python
#     script also re-checks per-biomarker, so this is a fast-path, not the
#     only safety net).
#   - Runs circulatory_failure_trajs.py inside the container with --sampler pymc.
#   - Routes ALL scratch (PyTensor compile cache, TMPDIR, Numba, matplotlib,
#     arviz) to /dev/shm (RAM-backed tmpfs), NOT gaga home, and deletes that
#     cohort's scratch dir immediately after the cohort finishes — sequential
#     processing means only one cohort's cache exists on disk at any time.
#   - Re-checks gaga quota headroom before every cohort and aborts cleanly
#     (rather than continuing into silent EDQUOT corruption) if headroom
#     drops below --min-quota-headroom-gb.
#
# Logs (clear + easy to access):
#   logs/container_runs/hirid_cf/<RUN_ID>/
#     run.log            <- tail -f this for a live, single-file view of the whole run
#     status.tsv         <- one row per cohort: start/end/duration/exit code/status
#     cohort<NN>.out/.err <- full stdout/stderr for that cohort's container run
#
# Usage:
#   bash docker/run_hirid_cf_batches.sh                     # sane defaults, foreground
#   nohup bash docker/run_hirid_cf_batches.sh > /dev/null 2>&1 &   # background, multi-hour run
#   bash docker/run_hirid_cf_batches.sh --dry-run           # show plan, run nothing
#   bash docker/run_hirid_cf_batches.sh --cohorts 3,4,10-12 # only these cohort indices
#
# Options:
#   --cohorts LIST            Comma list / ranges of cohort indices to consider
#                              (default: auto — every cohort with >=1 missing biomarker)
#   --cohort-splits N          Total cohort splits (default: 16 — do not change; see above)
#   --n-batches N               Patient sub-batches per biomarker per cohort (default: 10)
#   --window-hours H            Lookback window hours (default: 12.0)
#   --cpus N                    CPUs for the light (lactate) config (default: 32)
#   --heavy-max-jobs N          CPU cap for heavy biomarkers, heartrate/systolic (default: 16)
#   --container NAME            UDocker container name (default: trajectory-modeling)
#   --data-root DIR              Host data root (default: /home/gaga/data/physionet)
#   --cache-root DIR              Scratch root, RAM-backed (default: /dev/shm/$USER/trajmodel_cache)
#   --min-quota-headroom-gb N   Abort if gaga headroom falls below this (default: 5)
#   --dry-run                    Print the plan and exit; run nothing
#   --help                        Show this help

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
COHORT_SPLITS=16
N_BATCHES=10
WINDOW_HOURS=12.0
CPUS=32
HEAVY_MAX_JOBS=16
CONTAINER_NAME=trajectory-modeling
DATA_ROOT="/home/gaga/data/physionet"
CACHE_ROOT="/dev/shm/${USER}/trajmodel_cache"
MIN_QUOTA_HEADROOM_GB=5
DRY_RUN=0

# ── Arg parsing ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cohorts)                COHORTS_ARG="$2"; shift 2 ;;
        --cohort-splits)          COHORT_SPLITS="$2"; shift 2 ;;
        --n-batches)              N_BATCHES="$2"; shift 2 ;;
        --window-hours)           WINDOW_HOURS="$2"; shift 2 ;;
        --cpus)                   CPUS="$2"; shift 2 ;;
        --heavy-max-jobs)         HEAVY_MAX_JOBS="$2"; shift 2 ;;
        --container)              CONTAINER_NAME="$2"; shift 2 ;;
        --data-root)              DATA_ROOT="$2"; shift 2 ;;
        --cache-root)             CACHE_ROOT="$2"; shift 2 ;;
        --min-quota-headroom-gb)  MIN_QUOTA_HEADROOM_GB="$2"; shift 2 ;;
        --dry-run)                DRY_RUN=1; shift ;;
        --help|-h)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ "$COHORT_SPLITS" -ne 16 ]]; then
    echo "WARNING: --cohort-splits=${COHORT_SPLITS} (not 16). This changes patient" >&2
    echo "         assignment for EVERY cohort index and will no longer align with" >&2
    echo "         already-computed cohortNN files on disk. Proceed only if intentional." >&2
fi

DATA_DIR="${DATA_ROOT}/hirid/circulatory_failure"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${REPO_ROOT}/logs/container_runs/hirid_cf/${RUN_ID}"
mkdir -p "$LOG_ROOT"
MASTER_LOG="${LOG_ROOT}/run.log"
STATUS_TSV="${LOG_ROOT}/status.tsv"
printf "cohort\tstart\tend\tduration_s\texit_code\tstatus\n" > "$STATUS_TSV"

# Mirror everything to run.log while still printing to the terminal live.
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=========================================="
echo "HiRiD Circulatory Failure — sequential container batch run"
echo "=========================================="
echo "Run ID       : ${RUN_ID}"
echo "Repo root    : ${REPO_ROOT}"
echo "Data dir     : ${DATA_DIR}"
echo "Container    : ${CONTAINER_NAME}"
echo "Cohort splits: ${COHORT_SPLITS}"
echo "N batches    : ${N_BATCHES}"
echo "Window hours : ${WINDOW_HOURS}"
echo "CPUs (light) : ${CPUS}   CPUs (heavy cap): ${HEAVY_MAX_JOBS}"
echo "Cache root   : ${CACHE_ROOT}  (RAM-backed tmpfs, cleaned per-cohort)"
echo "Logs         : ${LOG_ROOT}"
echo "=========================================="

# ── Single-instance lock (nothing here is SLURM-arbitrated) ──────────────────
LOCK_FILE="${REPO_ROOT}/.hirid_cf_batches.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "ERROR: another run_hirid_cf_batches.sh is already running (lock: ${LOCK_FILE})." >&2
    exit 1
fi

# ── Preflight checks ───────────────────────────────────────────────────────────
command -v udocker >/dev/null 2>&1 || { echo "ERROR: udocker not found in PATH." >&2; exit 1; }

if ! udocker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "ERROR: udocker container '${CONTAINER_NAME}' does not exist." >&2
    echo "       See docs/CONTAINER_GUIDE.md to build/load/create it." >&2
    exit 1
fi

for f in lactate_timeseries.csv heartrate_timeseries.csv systolic_timeseries.csv; do
    [[ -f "${DATA_DIR}/${f}" ]] || { echo "ERROR: missing input ${DATA_DIR}/${f}" >&2; exit 1; }
done

# gaga home quota headroom (netapp2 gaga volume) — writes silently fail past
# quota, corrupting output (see project memory: feedback_slurm_bugs Bug 2).
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
echo "Preflight: /dev/shm avail = $(( avail_shm_kb / 1024 / 1024 ))G"

# ── Expected output paths per (cohort, biomarker) ────────────────────────────
biomarker_out() {
    local biomarker="$1" cohort="$2"
    printf "%s/%s_trajectory_probs_bayes_cohort%02d.csv" "$DATA_DIR" "$biomarker" "$cohort"
}

is_done() {
    local path="$1"
    [[ -f "$path" && -s "$path" ]]
}

cohort_needs_work() {
    local cohort="$1"
    for b in lactate heartrate systolic; do
        is_done "$(biomarker_out "$b" "$cohort")" || return 0
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
        echo "  cohort $(printf '%02d' "$i"): already complete (lactate+heartrate+systolic present) — will skip"
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

# ── Cleanup on interrupt/exit: remove this run's whole scratch tree ─────────
RUN_CACHE_ROOT="${CACHE_ROOT}/${RUN_ID}"
mkdir -p "$RUN_CACHE_ROOT"
cleanup_run() {
    rm -rf "$RUN_CACHE_ROOT" 2>/dev/null || true
}
trap cleanup_run EXIT

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

    udocker run \
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
        python /traj/scripts/trajectory/hirid/circulatory_failure_trajs.py \
            --input-dir  /data/hirid/circulatory_failure \
            --output-dir /data/hirid/circulatory_failure \
            --window-hours "${WINDOW_HOURS}" \
            --n-batches "${N_BATCHES}" \
            --sampler pymc \
            --cohort-splits "${COHORT_SPLITS}" \
            --cohort-index "${cohort}" \
        > >(tee -a "$OUT_LOG") 2> >(tee -a "$ERR_LOG" >&2)
    exit_code=$?

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

    # Free the RAM-backed scratch for this cohort immediately — sequential
    # processing means we never need more than one cohort's cache at a time.
    rm -rf "$COHORT_CACHE"
done

RUN_END=$(date +%s)

echo ""
echo "=========================================="
echo "Run ${RUN_ID} complete"
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
echo "  bash scripts/slurm/hirid/merge_circulatory_failure_bayes.sh"
echo "=========================================="

[[ "$N_FAILED" -eq 0 ]]
