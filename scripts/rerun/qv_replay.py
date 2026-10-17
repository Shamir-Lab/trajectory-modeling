#!/usr/bin/env python
# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/qv_replay.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
"""Regenerate the qualitative-validation figures with the windows of an earlier manifest.

Only the plotting code (labels) changes; the plotted (patient, window) pairs are taken
from the old selected_examples_manifest.csv instead of being re-selected.
usage: qv_replay.py <old_manifest.csv> <output_dir>
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/gaga/tamarw1/trajectory-modeling")
old = pd.read_csv(sys.argv[1])
out_dir = Path(sys.argv[2])

spec_ = importlib.util.spec_from_file_location("qv", REPO / "scripts/figures/qualitative_validation.py")
qv = importlib.util.module_from_spec(spec_)
sys.modules["qv"] = qv
spec_.loader.exec_module(qv)
qv.OUTPUT_DIR = out_dir


def wanted(spec):
    return old[(old.dataset == spec.dataset) & (old.task == spec.task) & (old.parameter == spec.marker)]


def shortlist(probs, spec, threshold, verbose=True):
    w = wanted(spec)
    out = {}
    for a in qv.ARCHETYPES:
        rows = []
        for _, r in w[w.archetype == a].sort_values("panel_slot").iterrows():
            m = probs[np.isclose(probs[spec.id_col].astype(float), float(r.patient_id), rtol=0, atol=1e-6)
                      & np.isclose(probs[spec.window_col].astype(float), float(r.window_end), rtol=0, atol=1e-6)]
            assert len(m) >= 1, f"{spec.key} {a}: window {r.patient_id}/{r.window_end} not in prob file"
            rows.append(m.iloc[[0]])
        out[a] = pd.concat(rows, ignore_index=True) if rows else probs.iloc[0:0]
    return out


def select(shortlists, ts_df, spec, n_per_class, rng=None, verbose=True, min_span=None, ts_index=None):
    res = {}
    for a in qv.ARCHETYPES:
        c = shortlists[a]
        res[a] = [(r[spec.id_col], r[spec.window_col], float(r[spec.prob_cols[a]])) for _, r in c.iterrows()]
        if verbose:
            print(f"      [{spec.key}] '{a}': {len(res[a])} example(s) (replayed)")
    return res


qv.shortlist_candidates = shortlist
qv.select_examples = select
sys.argv = ["qualitative_validation.py"]
qv.main()

new = pd.read_csv(out_dir / "selected_examples_manifest.csv")
k = ["dataset", "task", "parameter", "archetype", "panel_slot"]
m = old.merge(new, on=k, suffixes=("_old", "_new"), how="outer", indicator=True)
print("rows old/new/both:", len(old), len(new), int((m._merge == "both").sum()))
for c in ["patient_id", "window_end", "posterior_prob", "n_measurements", "time_span_covered", "value_min", "value_max"]:
    a, b = m[f"{c}_old"].astype(float), m[f"{c}_new"].astype(float)
    print(f"  {c:18s} identical: {bool(np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True))}")
