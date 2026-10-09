#!/usr/bin/env bash
# Merge all MIMIC AKI trajectory sub-cohort files into single datasets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
cd "$REPO_ROOT" || exit 1

echo "Merging MIMIC AKI trajectory probability sub-cohorts..."

echo ""
python3 << 'EOF'
import pandas as pd
from pathlib import Path
import os

result_dir = Path(os.environ.get("DATA_ROOT", "/home/gaga/data/physionet")) / "mimic" / "aki"

def merge_cohorts(pattern: str, output_name: str, id_col: str):
    cohort_files = sorted(result_dir.glob(pattern))
    if not cohort_files:
        print(f"ERROR: No cohort files found for {pattern}!")
        return

    print(f"Found {len(cohort_files)} cohort files for {output_name}:")
    for f in cohort_files:
        print(f"  - {f.name}")

    print("\nMerging...")
    cohorts = []
    for f in cohort_files:
        df = pd.read_csv(f)
        cohorts.append(df)
        print(f"  {f.name}: {len(df):,} rows, {df[id_col].nunique():,} patients")

    merged = pd.concat(cohorts, ignore_index=True)
    output_path = result_dir / output_name
    merged.to_csv(output_path, index=False)

    print(f"\n✓ Merged dataset saved: {output_path}")
    print(f"  Total rows: {len(merged):,}")
    print(f"  Total patients: {merged[id_col].nunique():,}")

merge_cohorts("aki_trajectory_probs_bayes_cohort*.csv", "aki_trajectory_probs_bayes.csv", "hadm_id")

# Build aki_prediction_dataset_with_probs.csv by merging the concatenated
# trajectory probs with the base prediction dataset. Per-cohort merged files
# are never written by the SLURM job, so we do it here from the full probs.
probs_path = result_dir / "aki_trajectory_probs_bayes.csv"
pred_path = result_dir / "aki_prediction_dataset.csv"
if not probs_path.exists():
    print("ERROR: aki_trajectory_probs_bayes.csv not found — run merge_cohorts first")
elif not pred_path.exists():
    print(f"ERROR: {pred_path} not found")
else:
    print(f"\nBuilding aki_prediction_dataset_with_probs.csv...")
    prob_cols = ["prob_stable", "prob_gradual_increase", "prob_rapid_increase"]
    probs = pd.read_csv(probs_path, usecols=["hadm_id", "time_day"] + prob_cols)
    pred = pd.read_csv(pred_path)
    merged = pred.merge(probs[["hadm_id", "time_day"] + prob_cols].drop_duplicates(["hadm_id", "time_day"]),
                        on=["hadm_id", "time_day"], how="left")
    missing = merged[prob_cols].isna().any(axis=1).sum()
    if missing > 0:
        print(f"  WARNING: {missing:,} rows ({100*missing/len(merged):.1f}%) have missing probs")
    out = result_dir / "aki_prediction_dataset_with_probs.csv"
    merged.to_csv(out, index=False)
    print(f"✓ Saved: {out}")
    print(f"  Rows: {len(merged):,}, patients: {merged['hadm_id'].nunique():,}")
EOF

echo ""
echo "Merge complete!"