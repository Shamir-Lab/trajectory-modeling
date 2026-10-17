# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/concat_hirid_boot.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
"""Combine the per-split HiRID bootstrap caches (24 sub-cohorts) into one cache file per marker."""
from pathlib import Path
import pandas as pd
D = Path('/tmp/tw_rerun/hirid_boot')
for m in ['lactate', 'heartrate', 'systolic']:
    parts = sorted(D.glob(f'{m}_trajectory_probs_bootstrap_cohort*.parquet'))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    dup = df.duplicated(['patientid', 'time_hour']).sum()
    df.to_parquet(D / f'{m}_trajectory_probs_bootstrap.parquet', index=False)
    print(f'{m}: {len(parts)} splits, {len(df):,} windows, {df.patientid.nunique():,} patients, duplicate keys {dup}')
