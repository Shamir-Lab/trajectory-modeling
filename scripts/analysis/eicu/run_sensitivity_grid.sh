#!/usr/bin/env bash
# run_sensitivity_grid.sh — background driver for the eICU Sepsis trajectory-
# archetype sensitivity grid search (scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py).
#
# This stage needs NO container: by the time the slope cache is built
# (scripts/slurm/eicu/run_eicu_sepsis_bayes_slope_cache_container.slurm +
# merge_eicu_sepsis_bayes_slope_cache.sh), phi_sensitivity is pure numpy and
# run_cv is pure sklearn/XGBoost — no PyTensor/g++ involved. Runs directly in
# pymc_env.
#
# Why this exists: with the production defaults (CV_N_REPEATS=10,
# CV_N_SPLITS=5), a single grid point runs the full 27-feature-set x 4-model
# eICU Sepsis pipeline ~17x longer than the cv-repeats=1 smoke test (~15-17 min)
# — i.e. hours per grid point, ~2-3 days for one marker's full perturbation
# grid (baseline + 3 params x 4 factors = 13 points). This is a background,
# nohup-style job for exactly that reason, one process per marker so multiple
# markers can run concurrently (they're independent — each perturbs its own
# marker's thresholds against the same multi-marker downstream feature set).
#
# Usage:
#   bash scripts/analysis/eicu/run_sensitivity_grid.sh lactate
#   bash scripts/analysis/eicu/run_sensitivity_grid.sh lactate wbc platelet   # 3 concurrent background runs
#   nohup bash scripts/analysis/eicu/run_sensitivity_grid.sh lactate wbc platelet > /dev/null 2>&1 &
#
# Each marker's results land in results/eicu/sepsis/sensitivity_<marker>/
# (sensitivity_results_eicu_sepsis.csv + sensitivity_aupr_stability_eicu_sepsis.png)
# — separate directories so concurrent marker runs never collide on output.
#
# Extra args after the marker list are passed through to the python script, e.g.:
#   bash scripts/analysis/eicu/run_sensitivity_grid.sh lactate -- --backend bootstrap --cv-repeats 3
#
# Env overrides: CONDA_SH, TRAJ_DATA_ROOT

set -uo pipefail  # deliberately NOT -e: one failed marker must not kill the others

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "ERROR: could not locate repo root" >&2; exit 1; }
cd "$REPO_ROOT" || exit 1

MARKERS=()
EXTRA_ARGS=()
_in_extra=0
for arg in "$@"; do
    if [[ "$_in_extra" -eq 1 ]]; then
        EXTRA_ARGS+=("$arg")
    elif [[ "$arg" == "--" ]]; then
        _in_extra=1
    else
        MARKERS+=("$arg")
    fi
done

if [[ ${#MARKERS[@]} -eq 0 ]]; then
    echo "Usage: bash scripts/analysis/eicu/run_sensitivity_grid.sh MARKER [MARKER...] [-- EXTRA_PY_ARGS...]" >&2
    echo "  MARKER: lactate | wbc | platelet" >&2
    exit 1
fi

# Same fd-exhaustion protection as the cache-build scripts — sklearn/joblib
# repeat-level parallelism (and RandomForest's own internal n_jobs=-1) can hit
# the same low-default-soft-limit issue on some hosts.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# pymc_env's gcc_linux-64 activation hook references $SYS_SYSROOT without a
# default and is not nounset-safe, so relax -u just for the activation itself
# (see the same fix/explanation in merge_eicu_sepsis_bayes_slope_cache.sh).
set +u
source "${CONDA_SH:-/home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh}"
conda activate pymc_env
set -u

LOG_ROOT="${REPO_ROOT}/logs/sensitivity_grid"
mkdir -p "$LOG_ROOT"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "eICU Sepsis sensitivity grid — background run"
echo "=========================================="
echo "Run ID   : ${RUN_ID}"
echo "Repo root: ${REPO_ROOT}"
echo "Markers  : ${MARKERS[*]}"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
echo "ulimit -n: $(ulimit -Sn) (hard: $(ulimit -Hn))"
echo "=========================================="

declare -a PIDS=()
for marker in "${MARKERS[@]}"; do
    OUT_DIR="results/eicu/sepsis/sensitivity_${marker}"
    mkdir -p "$OUT_DIR"
    LOG_FILE="${LOG_ROOT}/${marker}_${RUN_ID}.log"

    echo "Launching marker=${marker} -> output-dir=${OUT_DIR}, log=${LOG_FILE}"
    # --backend bayes is the whole point of this run (reuse the merged
    # cache); listed before EXTRA_ARGS so --backend in EXTRA_ARGS can still
    # override it (argparse takes the last occurrence of a flag).
    nohup python scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py \
        --marker "${marker}" \
        --backend bayes \
        --output-dir "${OUT_DIR}" \
        "${EXTRA_ARGS[@]}" \
        > "${LOG_FILE}" 2>&1 &
    pid=$!
    disown
    PIDS+=("$pid")
    echo "  PID=${pid}"
done

echo ""
echo "All markers launched. PIDs: ${PIDS[*]}"
echo "Tail a marker's progress with:"
for marker in "${MARKERS[@]}"; do
    echo "  tail -f ${LOG_ROOT}/${marker}_${RUN_ID}.log"
done
