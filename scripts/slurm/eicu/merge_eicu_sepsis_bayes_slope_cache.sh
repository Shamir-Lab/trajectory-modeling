#!/usr/bin/env bash
# Merge eICU sepsis Bayesian slope-cache cohort shards — written by the parallel
# SLURM array job scripts/slurm/eicu/run_eicu_sepsis_bayes_slope_cache_container.slurm
# ({marker}_bayes_slope_samples_cache_cohortNN.npz) — into the final, unsuffixed
# cache consumed by:
#   python scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py --backend bayes
#
# Run this only after ALL array tasks for a marker have finished. No container
# needed — merging is pure numpy/pandas concatenation, not PyMC sampling.
#
# Usage:
#   bash scripts/slurm/eicu/merge_eicu_sepsis_bayes_slope_cache.sh                 # all 3 markers
#   bash scripts/slurm/eicu/merge_eicu_sepsis_bayes_slope_cache.sh lactate wbc      # subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
cd "$REPO_ROOT" || exit 1

# Activate conda environment (the sensitivity script imports sklearn/xgboost/joblib
# via analysis.analysis_template even in --merge-cohort-caches mode).
# pymc_env's gcc_linux-64 activation hook references $SYS_SYSROOT without a
# default and is not `nounset`-safe, so relax -u just for the activation itself.
set +u
source "${CONDA_SH:-/home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh}"
conda activate pymc_env
set -u

MARKERS=("$@")
if [[ ${#MARKERS[@]} -eq 0 ]]; then
    MARKERS=(lactate wbc platelet)
fi

echo "Merging eICU sepsis Bayesian slope-cache cohort shards for: ${MARKERS[*]}"
echo ""

for marker in "${MARKERS[@]}"; do
    echo "── ${marker} ──"
    python scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py \
        --backend bayes \
        --marker "${marker}" \
        --base-dir "${DATA_ROOT}/eicu/sepsis" \
        --merge-cohort-caches
    echo ""
done

echo "Merge complete! Run the sensitivity script normally now, e.g.:"
echo "  python scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py --backend bayes --marker lactate"
