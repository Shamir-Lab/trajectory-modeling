#!/usr/bin/env bash
# run_eicu_sepsis_bayes_slope_cache.sh — Sequential (non-SLURM) driver for the
# eICU Sepsis Bayesian slope-sample cache, using the g++-accelerated
# `trajectory-modeling` UDocker container (see docs/CONTAINER_GUIDE.md).
#
# Builds the raw inter-point-slope cache consumed by
#   scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py --backend bayes
# via real PyMC posterior sampling (`_sample_post_trajs_scaled`) — the same
# backend that produces {marker}_trajectory_probs_bayes.csv in production
# (scripts/trajectory/eicu/sepsis_all_biomarkers.py). Without g++, PyTensor
# falls back to slow pure-Python compilation (10-50x slower).
#
# Why this exists (same rationale as docker/run_hirid_cf_batches.sh): this is
# for a plain server with no SLURM to arbitrate shared resources — cohorts are
# processed ONE AT A TIME so CPU/RAM stay bounded, PyTensor/Numba compile
# caches never pile up across concurrent jobs, and disk quota is never hit by
# simultaneous cache writes. If this server DOES have SLURM, prefer the array
# job instead: scripts/slurm/eicu/run_eicu_sepsis_bayes_slope_cache_container.slurm
#
# What it does per cohort (of --cohort-splits, default 10 — matches
# scripts/trajectory/eicu/sepsis_all_biomarkers.py / existing cohortNN files;
# changing it reshuffles which patients land in which cohort index):
#   - Skips the cohort entirely if lactate/wbc/platelet shard caches already
#     exist and are non-empty (auto-detected up front).
#   - Runs eicu_sepsis_sensitivity_analysis.py --backend bayes --marker all
#     --build-cache-only inside the container with --bayes-sampler pymc — ONE
#     container invocation covers all 3 markers per cohort.
#   - Routes ALL scratch (PyTensor compile cache, TMPDIR, Numba, matplotlib) to
#     RAM-backed tmpfs (--cache-root, NOT the data/repo filesystem), and
#     deletes that cohort's scratch dir immediately after the cohort finishes.
#   - Re-checks quota headroom before every cohort (best-effort; skipped if the
#     `quota` command / gaga volume aren't present on this server) and aborts
#     cleanly rather than continuing into silent write failures.
#
# This ONLY builds the slope cache shards — it does not run prepare_features or
# any CV. Once all cohorts are done, merge and run the sensitivity script
# normally (no container needed for either):
#   bash scripts/slurm/eicu/merge_eicu_sepsis_bayes_slope_cache.sh
#   python scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py --backend bayes --marker lactate
#
# Logs (clear + easy to access):
#   logs/container_runs/eicu_sepsis_bayes_slope_cache/<RUN_ID>/
#     run.log            <- tail -f this for a live, single-file view of the whole run
#     status.tsv         <- one row per cohort: start/end/duration/exit code/status
#     cohort<NN>.out/.err <- full stdout/stderr for that cohort's container run
#
# Usage:
#   bash docker/run_eicu_sepsis_bayes_slope_cache.sh                     # sane defaults, foreground
#   nohup bash docker/run_eicu_sepsis_bayes_slope_cache.sh > /dev/null 2>&1 &   # background, multi-hour run
#   bash docker/run_eicu_sepsis_bayes_slope_cache.sh --dry-run           # show plan, run nothing
#   bash docker/run_eicu_sepsis_bayes_slope_cache.sh --cohorts 3,4,10-12 # only these cohort indices
#   # Different server / different data mount:
#   bash docker/run_eicu_sepsis_bayes_slope_cache.sh --data-root /path/to/physionet --cache-root /dev/shm/$USER/trajmodel_cache
#
# Options:
#   --markers LIST              Comma list of markers to build (default: lactate,wbc,platelet)
#   --cohorts LIST               Comma list / ranges of cohort indices to consider
#                                 (default: auto — every cohort with >=1 missing marker shard)
#   --cohort-splits N             Total cohort splits (default: 10 — do not change once shards
#                                  exist; see above)
#   --window-days D                Lookback window in days (default: 3.0)
#   --n-samples N                    Posterior draws per window (default: 200)
#   --tune N                          Tuning steps (default: 300)
#   --chains N                         Chains per window (default: 4)
#   --cpus N                            CPUs for the container (default: 32)
#   --container NAME                     UDocker container name (default: trajectory-modeling)
#   --data-root DIR                       Host data root (default: /home/gaga/data/physionet)
#   --cache-root DIR                       Scratch root, RAM-backed (default: /dev/shm/$USER/trajmodel_cache)
#   --min-quota-headroom-gb N             Abort if headroom falls below this (default: 5; skipped
#                                          if quota can't be read on this server)
#   --dry-run                              Print the plan and exit; run nothing
#   --help                                  Show this help

set -uo pipefail  # deliberately NOT -e: one failed cohort must not kill the run

# ── Locate repo root ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "ERROR: could not locate repo root" >&2; exit 1; }

# ── Raise the open-files limit ────────────────────────────────────────────────
# Some hosts default the soft `ulimit -n` far below the hard limit (observed:
# 1024 soft vs 1048576 hard on one rack server, vs. already-1048576 on another).
# udocker/proot processes inherit the LAUNCHING shell's limit, not a container
# default, and this job spawns dozens of concurrent PyTensor-compiling joblib
# workers — each opens compiled .so files, cache locks, and a POSIX semaphore fd
# for its parent/child pipe. At 1024 fds this races: children intermittently
# fail with "FileNotFoundError ... _SemLock._rebuild" because the parent runs
# out of headroom to track everyone's semaphore/pipe fds. Best-effort: some
# hosts won't allow raising it (permission), which is fine — it just means this
# protection doesn't apply there.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# ── Defaults ───────────────────────────────────────────────────────────────────
MARKERS_ARG="lactate,wbc,platelet"
COHORTS_ARG="auto"
COHORT_SPLITS=10
WINDOW_DAYS=3.0
N_SAMPLES=200
TUNE=300
CHAINS=4
CPUS=32
CONTAINER_NAME=trajectory-modeling
DATA_ROOT="/home/gaga/data/physionet"
CACHE_ROOT="/dev/shm/${USER}/trajmodel_cache"
MIN_QUOTA_HEADROOM_GB=5
DRY_RUN=0

# ── Arg parsing ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --markers)                MARKERS_ARG="$2"; shift 2 ;;
        --cohorts)                COHORTS_ARG="$2"; shift 2 ;;
        --cohort-splits)          COHORT_SPLITS="$2"; shift 2 ;;
        --window-days)            WINDOW_DAYS="$2"; shift 2 ;;
        --n-samples)              N_SAMPLES="$2"; shift 2 ;;
        --tune)                   TUNE="$2"; shift 2 ;;
        --chains)                 CHAINS="$2"; shift 2 ;;
        --cpus)                   CPUS="$2"; shift 2 ;;
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

IFS=',' read -ra MARKERS <<< "$MARKERS_ARG"

if [[ "$COHORT_SPLITS" -ne 10 ]]; then
    echo "WARNING: --cohort-splits=${COHORT_SPLITS} (not 10). This changes patient" >&2
    echo "         assignment for EVERY cohort index and will no longer align with" >&2
    echo "         already-computed cohortNN shard files on disk. Proceed only if intentional." >&2
fi

DATA_DIR="${DATA_ROOT}/eicu/sepsis"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${REPO_ROOT}/logs/container_runs/eicu_sepsis_bayes_slope_cache/${RUN_ID}"
mkdir -p "$LOG_ROOT"
MASTER_LOG="${LOG_ROOT}/run.log"
STATUS_TSV="${LOG_ROOT}/status.tsv"
printf "cohort\tstart\tend\tduration_s\texit_code\tstatus\n" > "$STATUS_TSV"

# Mirror everything to run.log while still printing to the terminal live.
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=========================================="
echo "eICU Sepsis Bayesian slope cache — sequential container batch run"
echo "=========================================="
echo "Run ID       : ${RUN_ID}"
echo "Repo root    : ${REPO_ROOT}"
echo "Data dir     : ${DATA_DIR}"
echo "Container    : ${CONTAINER_NAME}"
echo "Markers      : ${MARKERS[*]}"
echo "Cohort splits: ${COHORT_SPLITS}"
echo "Window days  : ${WINDOW_DAYS}"
echo "n_samples/tune/chains: ${N_SAMPLES}/${TUNE}/${CHAINS}"
echo "CPUs         : ${CPUS}"
echo "Cache root   : ${CACHE_ROOT}  (RAM-backed tmpfs, cleaned per-cohort)"
echo "Logs         : ${LOG_ROOT}"
echo "ulimit -n    : $(ulimit -Sn) (hard: $(ulimit -Hn))"
echo "=========================================="

# ── Single-instance lock (nothing here is SLURM-arbitrated) ──────────────────
LOCK_FILE="${REPO_ROOT}/.eicu_sepsis_bayes_slope_cache.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "ERROR: another run_eicu_sepsis_bayes_slope_cache.sh is already running (lock: ${LOCK_FILE})." >&2
    exit 1
fi

# ── Preflight checks ───────────────────────────────────────────────────────────
command -v udocker >/dev/null 2>&1 || { echo "ERROR: udocker not found in PATH." >&2; exit 1; }

if ! udocker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "ERROR: udocker container '${CONTAINER_NAME}' does not exist on this server." >&2
    echo "       See docs/CONTAINER_GUIDE.md to export/load/create it:" >&2
    echo "         udocker load -i /path/to/trajectory-modeling.tar" >&2
    echo "         udocker create --name=trajectory-modeling trajectory-modeling" >&2
    exit 1
fi

for marker in "${MARKERS[@]}"; do
    case "$marker" in
        lactate) f=lactate_timeseries.csv ;;
        wbc)     f=wbc_timeseries.csv ;;
        platelet) f=platelets_timeseries.csv ;;
        *) echo "ERROR: unknown marker '${marker}' (expected lactate, wbc, or platelet)" >&2; exit 1 ;;
    esac
    [[ -f "${DATA_DIR}/${f}" ]] || { echo "ERROR: missing input ${DATA_DIR}/${f}" >&2; exit 1; }
done
[[ -f "${DATA_DIR}/sepsis_prediction_dataset.csv" ]] || {
    echo "ERROR: missing input ${DATA_DIR}/sepsis_prediction_dataset.csv" >&2; exit 1;
}

# Disk quota headroom — best-effort; some servers (e.g. a plain rack machine)
# won't have `quota` or the gaga volume at all, in which case this just skips.
check_quota_headroom() {
    local line used quota_limit used_b quota_b headroom_gb
    line="$(quota -s 2>/dev/null | grep -A1 'Netapp5_vol_1/gaga' | tail -1)"
    if [[ -z "$line" ]]; then
        echo "  (could not read gaga quota on this server — skipping headroom check)"
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

echo "Preflight: checking quota headroom..."
check_quota_headroom || exit 1

mkdir -p "$CACHE_ROOT"
avail_shm_kb="$(df -k --output=avail "$CACHE_ROOT" | tail -1)"
echo "Preflight: cache-root avail = $(( avail_shm_kb / 1024 / 1024 ))G"

# ── Expected output paths per (marker, cohort) ───────────────────────────────
marker_shard_out() {
    local marker="$1" cohort="$2"
    printf "%s/%s_bayes_slope_samples_cache_cohort%02d.npz" "$DATA_DIR" "$marker" "$cohort"
}

is_done() {
    local path="$1"
    [[ -f "$path" && -s "$path" ]]
}

cohort_needs_work() {
    local cohort="$1"
    for m in "${MARKERS[@]}"; do
        is_done "$(marker_shard_out "$m" "$cohort")" || return 0
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
        echo "  cohort $(printf '%02d' "$i"): already complete (${MARKERS[*]} shards present) — will skip"
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

    echo "Checking quota headroom before this cohort..."
    if ! check_quota_headroom; then
        echo "Aborting remaining run — see quota error above."
        break
    fi

    COHORT_CACHE="${RUN_CACHE_ROOT}/cohort${cohort_padded}"
    mkdir -p "$COHORT_CACHE"/{pytensor,tmp,matplotlib,xdg_cache}

    OUT_LOG="${LOG_ROOT}/cohort${cohort_padded}.out"
    ERR_LOG="${LOG_ROOT}/cohort${cohort_padded}.err"

    cohort_start=$(date +%s)

    udocker run \
        -v "${REPO_ROOT}:/traj" \
        -v "${DATA_ROOT}:/data" \
        -v "${COHORT_CACHE}:/cache" \
        -e TRAJ_CACHE_ROOT=/cache \
        -e TRAJ_DATA_ROOT=/data \
        -e TMPDIR=/cache/tmp \
        -e TEMP=/cache/tmp \
        -e TMP=/cache/tmp \
        -e NUMBA_DISABLE_CACHING=1 \
        -e MPLCONFIGDIR=/cache/matplotlib \
        -e XDG_CACHE_HOME=/cache/xdg_cache \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 \
        "${CONTAINER_NAME}" \
        python /traj/scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py \
            --backend bayes \
            --marker all \
            --base-dir /data/eicu/sepsis \
            --output-dir /traj/results/eicu/sepsis \
            --window "${WINDOW_DAYS}" \
            --bayes-sampler pymc \
            --bayes-n-samples "${N_SAMPLES}" \
            --bayes-tune "${TUNE}" \
            --bayes-chains "${CHAINS}" \
            --cache-n-jobs "${CPUS}" \
            --build-cache-only \
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
    echo "(it will only re-attempt cohorts that are still missing shard output)."
fi
echo ""
echo "Once all cohorts are complete, merge with:"
echo "  bash scripts/slurm/eicu/merge_eicu_sepsis_bayes_slope_cache.sh ${MARKERS[*]}"
echo "=========================================="

[[ "$N_FAILED" -eq 0 ]]
