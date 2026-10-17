# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/append_newstays.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
"""Append trajectory features of the new eICU ACLF stays to the cached files (same columns)."""
from pathlib import Path
import pandas as pd
old = Path('/home/gaga/data/physionet/eicu/liver'); ns = Path('/tmp/tw_rerun/eicu_liver_newstays')
out = Path('/tmp/tw_rerun/data_root/eicu/liver')
for name in ['liver_trajectory_probs_bayes.csv', 'liver_trajectory_probs_bootstrap.csv', 'bilirubin_trajectory_probs_bootstrap.csv']:
    a = pd.read_csv(old / name); b = pd.read_csv(ns / f'new_{name}')
    assert set(b.columns) == set(a.columns), (name, set(a.columns) ^ set(b.columns))
    assert not set(b.stay_id) & set(a.stay_id), name
    pd.concat([a, b[a.columns]], ignore_index=True).to_csv(out / name, index=False)
    print(f'  {name}: {len(a):,} cached + {len(b):,} new rows ({b.stay_id.nunique()} stays)')
