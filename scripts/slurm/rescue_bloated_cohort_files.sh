#!/usr/bin/env bash
# Rescue cohort files generated with the left-join bug.
#
# The bug: probs_ts was built with how='left' on the unfiltered time series,
# so every cohort file contains ALL patients' rows with NaN probs for
# non-cohort patients. Filtering to non-NaN prob rows recovers exactly
# what an inner join would have produced.
#
# Run from the repo root. Rewrites files in place.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"

python3 << 'EOF'
import pandas as pd
from pathlib import Path
import os

data_root   = Path(os.environ.get("DATA_ROOT",    "/home/gaga/data/physionet"))
results_root = Path(os.environ.get("RESULTS_ROOT", "/home/gaga/tamarw1/trajectory-modeling/results"))

# (directory, glob pattern, prob column to use as NaN sentinel, id column)
TARGETS = [
    (data_root   / "mimic" / "aki",  "aki_trajectory_probs_bayes_cohort*.csv",    "prob_stable",  "hadm_id"),
    (results_root / "eicu"  / "aki",  "aki_trajectory_probs_bayes_cohort*.csv",    "prob_stable",  "stay_id"),
]

for directory, pattern, sentinel_col, id_col in TARGETS:
    cohort_files = sorted(directory.glob(pattern))
    if not cohort_files:
        print(f"No files matched: {directory}/{pattern}  — skipping")
        continue

    print(f"\n{'='*70}")
    print(f"Rescuing {len(cohort_files)} files in {directory}")
    print(f"{'='*70}")

    total_before = 0
    total_after  = 0

    for f in cohort_files:
        df = pd.read_csv(f)
        n_before = len(df)

        if sentinel_col not in df.columns:
            print(f"  SKIP {f.name}: column '{sentinel_col}' not found")
            continue

        n_patients_before = df[id_col].nunique()
        df_clean = df[df[sentinel_col].notna()].reset_index(drop=True)
        n_after  = len(df_clean)
        n_patients_after = df_clean[id_col].nunique()

        if n_before == n_after:
            print(f"  {f.name}: already clean ({n_before:,} rows) — skipping")
            continue

        df_clean.to_csv(f, index=False)
        freed_mb = (n_before - n_after) * df.memory_usage(deep=False).sum() / len(df) / 1e6
        print(f"  {f.name}: {n_before:,} → {n_after:,} rows  |  "
              f"{n_patients_before:,} → {n_patients_after:,} patients  |  "
              f"~{freed_mb:.0f} MB freed")
        total_before += n_before
        total_after  += n_after

    print(f"\n  Total: {total_before:,} → {total_after:,} rows "
          f"({total_before - total_after:,} NaN rows removed)")

print("\nDone.")
EOF
