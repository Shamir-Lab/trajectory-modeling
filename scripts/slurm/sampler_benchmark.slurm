#!/usr/bin/env bash
#SBATCH --job-name=sampler-bench
#SBATCH --output=logs/outs/bench_%j.out
#SBATCH --error=logs/errs/bench_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$SCRIPT_DIR"
    while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
        REPO_ROOT="$(dirname "$REPO_ROOT")"
    done
fi
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }
cd "$REPO_ROOT" || exit 1

BENCH="$REPO_ROOT/scripts/benchmarks/sampler_benchmark.py"

echo "==============================="
echo " Running nutpie (CPU) benchmark"
echo "==============================="
python "$BENCH" --sampler nutpie

echo ""
echo "================================="
echo " Running numpyro (GPU) benchmark"
echo "================================="
python "$BENCH" --sampler numpyro

echo ""
echo "=========="
echo " Comparison"
echo "=========="
python "$BENCH" --compare
