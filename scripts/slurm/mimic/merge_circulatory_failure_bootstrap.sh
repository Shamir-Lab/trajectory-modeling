#!/usr/bin/env bash
# Merge bootstrap trajectory cohort files for MIMIC circulatory failure
# Run after all SLURM array jobs complete

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
DATA_DIR="${DATA_ROOT}/mimic/circulatory_failure"
# Must match the --cohort-splits used by docker/run_mimic_cf_bootstrap_batches.sh
# (default 24). The old default here was 5, left over from the original
# 2026-03/2026-05 5-way SLURM run — see that script's header for why mixing
# 5-way and 24-way cohort files under the same cohortNN index is a silent
# data-loss trap (project memory feedback_slurm_bugs Bug 7).
COHORT_SPLITS="${COHORT_SPLITS:-24}"

echo "Merging MIMIC Circulatory Failure Bootstrap trajectory files..."

python - <<'PY'
from pathlib import Path
import pandas as pd
import os

data_dir = Path(os.environ.get("DATA_ROOT", "/home/gaga/data/physionet")) / "mimic" / "circulatory_failure"
n_cohorts = int(os.environ.get("COHORT_SPLITS", "24"))

def merge_base(base_name: str):
    parts = []
    for i in range(n_cohorts):
        p_parquet = data_dir / f"{base_name}_cohort{i:02d}.parquet"
        p_csv = data_dir / f"{base_name}_cohort{i:02d}.csv"
        if p_parquet.exists():
            parts.append(pd.read_parquet(p_parquet))
        elif p_csv.exists():
            parts.append(pd.read_csv(p_csv))
        else:
            print(f"  WARNING: Missing cohort file for {base_name}, cohort={i:02d}")

    if not parts:
        print(f"  WARNING: No cohort files found for {base_name}")
        return

    merged = pd.concat(parts, ignore_index=True)
    out_parquet = data_dir / f"{base_name}.parquet"
    out_csv = data_dir / f"{base_name}.csv"
    merged.to_parquet(out_parquet, index=False, compression="zstd")
    merged.to_csv(out_csv, index=False)
    print(f"  ✓ Created {out_parquet} ({len(merged):,} rows)")
    print(f"  ✓ Created {out_csv} ({len(merged):,} rows)")


for biomarker in ["lactate", "heartrate", "systolic"]:
    print(f"Merging {biomarker}...")
    merge_base(f"{biomarker}_trajectory_probs_bootstrap")

print("Merging prediction dataset...")
merge_base("circulatory_failure_prediction_dataset_with_bootstrap_probs")

print("\n✓ Merge complete!")
PY
