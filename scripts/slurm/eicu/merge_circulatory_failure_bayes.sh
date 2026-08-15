#!/usr/bin/env bash
# Merge Bayesian trajectory cohort files for eICU circulatory failure.
# Run after all run_eicu_circulatory_failure_bayes_trajectories.slurm tasks complete.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
DATA_DIR="${DATA_ROOT}/eicu/circulatory_failure"
N_COHORTS="${N_COHORTS:-8}"

echo "Merging eICU Circulatory Failure Bayesian trajectory files (N_COHORTS=${N_COHORTS})..."

# Note: the per-cohort "circulatory_failure_prediction_dataset_with_probs_cohortNN" merge
# below only exists for cohorts where all 3 biomarkers were computed in the SAME invocation
# (circulatory_failure_trajs.py skips its own per-invocation merge if any biomarker was
# already done and skipped) — with resumable runs this will often be absent per-cohort;
# the per-biomarker merge above it is the reliable artifact to use downstream.
N_COHORTS="$N_COHORTS" DATA_ROOT="$DATA_ROOT" python - <<'PY'
from pathlib import Path
import pandas as pd
import os

data_dir = Path(os.environ.get("DATA_ROOT", "/home/gaga/data/physionet")) / "eicu" / "circulatory_failure"
n_cohorts = int(os.environ.get("N_COHORTS", "8"))

def merge_base(base_name: str) -> pd.DataFrame | None:
    parts = []
    for i in range(n_cohorts):
        p = data_dir / f"{base_name}_cohort{i:02d}.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
        else:
            print(f"  WARNING: Missing cohort file {p}")

    if not parts:
        print(f"  WARNING: No cohort files found for {base_name}")
        return None

    merged = pd.concat(parts, ignore_index=True)
    out = data_dir / f"{base_name}.csv"
    merged.to_csv(out, index=False)
    print(f"  ✓ {out} ({len(merged):,} rows)")
    return merged


for biomarker in ["lactate", "heartrate", "systolic"]:
    print(f"Merging {biomarker}...")
    merge_base(f"{biomarker}_trajectory_probs_bayes")

print("Merging prediction dataset with probs...")
merge_base("circulatory_failure_prediction_dataset_with_probs")

print("\n✓ Merge complete!")
PY
