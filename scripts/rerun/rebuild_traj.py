# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/rebuild_traj.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
"""Rebuild trajectory-merged datasets for one task from the new preprocessed rows and the
cached trajectory features (no trajectory feature is recomputed).

usage: rebuild_traj.py <old_task_dir> <new_task_dir> [extra_source_dir]
Writes rebuilt files into <new_task_dir> and prints a coverage report."""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd

old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
extra_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
IDS, WINS = ['stay_id', 'hadm_id', 'patientid'], ['time_hour', 'time_day']

def is_traj(c):
    return (c.startswith(('prob_', 'trajtype_')) or '_boot' in c
            or c.endswith(('_stable', '_gradual', '_rapid', '_increase', '_decline', '_prob', '_worsening')))

def read(p, cols=None):
    if p.suffix == '.parquet':
        return pd.read_parquet(p, columns=cols)
    return pd.read_csv(p, usecols=cols, low_memory=False)

def backend(p):
    return 'bootstrap' if 'bootstrap' in p.name.lower() else 'bayes'


def header(p):
    if p.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_schema(p).names
    return pd.read_csv(p, nrows=0).columns.tolist()

plain_name = next(p.name for p in new_dir.glob('*_prediction_dataset.csv'))
plain = pd.read_csv(new_dir / plain_name, low_memory=False)
idc = next(c for c in IDS if c in plain.columns); wc = next(c for c in WINS if c in plain.columns)
key = [idc, wc]
plain[idc] = plain[idc].astype('int64'); plain[wc] = pd.to_numeric(plain[wc]).astype('int64')

pat = re.compile(r'(trajectory_probs|_prediction_dataset_with_.*probs)(?!.*cohort\d)')
sources = [p for p in sorted(old_dir.iterdir()) if p.suffix in ('.csv', '.parquet') and pat.search(p.name) and not re.search(r'cohort\d+', p.name)]
# Like the analysis loaders, prefer <stem>.parquet over <stem>.csv when both exist
stems_parquet = {p.stem for p in sources if p.suffix == '.parquet'}
sources = [p for p in sources if not (p.suffix == '.csv' and p.stem in stems_parquet)]
if extra_dir:
    sources += sorted(extra_dir.glob('new_*.csv'))
groups, members = {}, {}
for p in sources:
    cols = header(p)
    if not set(key) <= set(cols):
        print(f'  skip (no key): {p.name}'); continue
    tcols = [c for c in cols if is_traj(c)]
    if not tcols:
        continue
    df = read(p, key + tcols)
    df[idc] = df[idc].astype('int64'); df[wc] = pd.to_numeric(df[wc]).astype('int64')
    df = df.drop_duplicates(key)
    sig = (backend(p), tuple(sorted(tcols)))
    groups.setdefault(sig, []).append(df); members.setdefault(sig, []).append(p.name)
union = {}
for sig, dfs in groups.items():
    allr = pd.concat(dfs, ignore_index=True)
    dup = allr[allr.duplicated(key, keep=False)]
    conflicts = 0
    if len(dup):
        g = dup.groupby(key)[list(sig[1])].nunique(dropna=True)
        conflicts = int((g > 1).any(axis=1).sum())
    union[sig] = allr.drop_duplicates(key, keep='first')
    print(f'  {sig[0]} set {list(sig[1])[:3]}...: {len(union[sig]):,} windows from {members[sig]}; conflicting keys: {conflicts}')

for p in sources:
    if extra_dir and p.parent == extra_dir:
        continue
    cols = header(p)
    if not set(key) <= set(cols):
        continue
    tcols = [c for c in cols if is_traj(c)]
    if not tcols:
        continue
    u = union[(backend(p), tuple(sorted(tcols)))]
    feat = [c for c in cols if c not in tcols and c not in key]
    is_merged = any(c in plain.columns for c in feat if c not in ('labname', 'lab_time_hours', 'time_days', 'time_hours')) and len(feat) > 8
    old_cov = read(p, key + tcols[:1])[tcols[0]].notna().mean()
    if not is_merged:
        print(f'  cache kept unchanged: {p.name}')
        continue
    out = plain.merge(u, on=key, how='left')
    out = out[[c for c in plain.columns] + [c for c in tcols if c not in plain.columns]]
    if p.suffix == '.parquet':
        out.to_parquet(new_dir / p.name, index=False)
        if (old_dir / f'{p.stem}.csv').exists():  # keep the CSV twin consistent
            out.to_csv(new_dir / f'{p.stem}.csv', index=False)
    else:
        out.to_csv(new_dir / p.name, index=False)
    print(f'  rebuilt {p.name} (merged dataset): {len(out):,} rows; trajectory coverage {old_cov:.1%} -> {out[tcols[0]].notna().mean():.1%}')

cov = {sig: plain.merge(u[key + [sig[1][0]]], on=key, how='left')[sig[1][0]].notna().mean() for sig, u in union.items()}
print(f'  new plain dataset: {len(plain):,} rows, {plain[idc].nunique():,} stays; trajectory coverage per set: ' +
      ', '.join(f'{s[0]}:{s[1][0]}={v:.1%}' for s, v in cov.items()))
