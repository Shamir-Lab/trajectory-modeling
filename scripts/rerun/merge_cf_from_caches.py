#!/usr/bin/env python
# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/merge_cf_from_caches.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
"""Rebuild the circulatory-failure merged datasets from the per-marker trajectory caches.

Same merge as the pipeline (scripts/slurm/*/merge_circulatory_failure_bayes.sh and the
*_trajs_bootstrap.py merge step): new plain prediction dataset LEFT JOIN each marker's
cached trajectory probabilities on (id, time_hour), one row per key (first kept).
No trajectory feature is recomputed.

usage: merge_cf_from_caches.py <dataset> [--bootstrap-from-old-merged]      (mimic | eicu | hirid)
       --bootstrap-from-old-merged: take bootstrap values from the old merged dataset instead of the
       per-marker bootstrap caches (HiRID: those caches do not match the merged dataset used so far)
       --bootstrap-cache-dir <dir>: take the bootstrap caches from <dir> (recomputed trajectories)
reads  /home/gaga/data/physionet/<dataset>/circulatory_failure/*_trajectory_probs_{bayes,bootstrap}.*
       /tmp/tw_rerun/base/<dataset>/circulatory_failure/circulatory_failure_prediction_dataset.csv
writes .../circulatory_failure_prediction_dataset_with_probs.csv
       .../circulatory_failure_prediction_dataset_with_bootstrap_probs.parquet (+ .csv twin)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ds = sys.argv[1]
BOOT_FROM_OLD_MERGED = "--bootstrap-from-old-merged" in sys.argv[2:]
BOOT_DIR = Path(sys.argv[sys.argv.index("--bootstrap-cache-dir") + 1]) if "--bootstrap-cache-dir" in sys.argv else None
OLD = Path(f"/home/gaga/data/physionet/{ds}/circulatory_failure")
NEW = Path(f"/tmp/tw_rerun/base/{ds}/circulatory_failure")
STEM = "circulatory_failure_prediction_dataset"
MARKERS = ["lactate", "heartrate", "systolic"]
IDS = ["stay_id", "hadm_id", "patientid"]


def read(p, cols=None):
    return pd.read_parquet(p, columns=cols) if p.suffix == ".parquet" else pd.read_csv(p, usecols=cols, low_memory=False)


def header(p):
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_schema(p).names
    return pd.read_csv(p, nrows=0).columns.tolist()


def cache_path(marker, backend):
    if backend == "bootstrap" and BOOT_FROM_OLD_MERGED:
        return OLD / f"{STEM}_with_bootstrap_probs.parquet"
    src = BOOT_DIR if (backend == "bootstrap" and BOOT_DIR is not None) else OLD
    for suf in (".parquet", ".csv"):  # parquet first, like the analysis loaders
        p = src / f"{marker}_trajectory_probs_{backend}{suf}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no {backend} cache for {marker} in {src}")


plain = pd.read_csv(NEW / f"{STEM}.csv", low_memory=False)
idc = next(c for c in IDS if c in plain.columns)
key = [idc, "time_hour"]
plain[idc] = plain[idc].astype("int64")
plain["time_hour"] = pd.to_numeric(plain["time_hour"]).astype("int64")
print(f"{ds}: new plain dataset {len(plain):,} rows, {plain[idc].nunique():,} stays")

for backend in ("bayes", "bootstrap"):
    out = plain.copy()
    for m in MARKERS:
        p = cache_path(m, backend)
        cols = header(p)
        dst = [f"{m}_stable", f"{m}_gradual", f"{m}_rapid"]
        if all(c in cols for c in dst):
            src = dst
        else:  # bayes caches keep the model's generic names
            direction = "decline" if m == "systolic" else "increase"
            src = ["prob_stable", f"prob_gradual_{direction}", f"prob_rapid_{direction}"]
        c = read(p, key + src).rename(columns=dict(zip(src, dst)))
        c = c.dropna(subset=key)
        c[idc] = c[idc].astype("int64")
        c["time_hour"] = pd.to_numeric(c["time_hour"]).astype("int64")
        dup = c[c.duplicated(key, keep=False)]
        conflicts = int((dup.groupby(key)[dst].nunique(dropna=True) > 1).any(axis=1).sum()) if len(dup) else 0
        c = c.drop_duplicates(key, keep="first")
        n0 = len(out)
        out = out.merge(c, on=key, how="left")
        assert len(out) == n0
        print(f"  {backend:9s} {m:9s} <- {p.name}: coverage {out[dst[0]].notna().mean():.1%}"
              f" (duplicate keys with differing values: {conflicts})")

    # agreement with the old merged dataset on windows present in both
    old_p = OLD / (f"{STEM}_with_probs.csv" if backend == "bayes" else f"{STEM}_with_bootstrap_probs.parquet")
    tcols = [f"{m}_{t}" for m in MARKERS for t in ("stable", "gradual", "rapid")]
    old = read(old_p, key + tcols)
    old[idc] = old[idc].astype("int64")
    old["time_hour"] = pd.to_numeric(old["time_hour"]).astype("int64")
    old = old.drop_duplicates(key)
    both = out[key + tcols].merge(old, on=key, suffixes=("", "_old"))
    a = both[tcols].to_numpy(float)
    b = both[[f"{c}_old" for c in tcols]].to_numpy(float)
    same = np.isclose(a, b, rtol=0, atol=1e-9) | (np.isnan(a) & np.isnan(b))
    print(f"  {backend}: {len(both):,} windows also in the old merged file; identical values in {same.all(axis=1).mean():.4%} of them")

    if backend == "bayes":
        targets = [NEW / f"{STEM}_with_probs.csv"]
    else:
        targets = [NEW / f"{STEM}_with_bootstrap_probs.parquet", NEW / f"{STEM}_with_bootstrap_probs.csv"]
    for t in targets:
        assert not t.is_symlink(), f"refusing to write through symlink {t}"
        if t.suffix == ".parquet":
            out.to_parquet(t, index=False)
        else:
            out.to_csv(t, index=False)
        print(f"  wrote {t}")
