#!/usr/bin/env python
# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/cf_shard.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
"""Run scripts/analysis/mimic/mimic_circulatory_failure_analysis.py split across hosts.

The analysis evaluates (model, feature set) pairs independently: every pair uses
the same prepared dataset and derives its shuffles and folds from seed + repeat.
So pairs can run in separate processes and be merged without changing results.

  run   <shard> <n_shards> <rf_jobs> [<summary_procs>] -- <analysis CLI args>
        Runs the unchanged analysis main(), but only the pairs with
        index % n_shards == shard (others return no rows). Random Forest fits use
        rf_jobs threads (n_jobs does not change the fitted trees).
        Writes the usual outputs to --output-dir plus jobs.json (pair order).

  merge <output_dir> <shard_dir> [<shard_dir> ...]
        Concatenates the shard fold results in the original pair order and
        recomputes summary/delta tables with the analysis' own run_cv/save_results.
"""
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/gaga/tamarw1/trajectory-modeling")
SCRIPT = REPO / "scripts/analysis/mimic/mimic_circulatory_failure_analysis.py"
FOLD_FILE = "mimic_circulatory_failure_fold_results.parquet"

spec = importlib.util.spec_from_file_location("cfa", SCRIPT)
cfa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfa)
RealParallel = cfa.Parallel


def shard_parallel(shard, n_shards, out_dir, rf_jobs=None):
    """Parallel replacement that runs only this shard's pairs."""

    class ShardParallel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, tasks):
            tasks = list(tasks)
            order = [[t[1][0], t[1][2]] for t in tasks]  # (model, feature set)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "jobs.json").write_text(json.dumps(order, indent=1))
            mine = [i for i in range(len(tasks)) if i % n_shards == shard]
            cfa.logger.info(f"Shard {shard}/{n_shards}: {len(mine)} of {len(tasks)} pairs, rf_jobs={rf_jobs}")
            kwargs = dict(self.kwargs, n_jobs=max(1, len(mine)))
            done = RealParallel(**kwargs)(tasks[i] for i in mine)
            res = [[] for _ in tasks]
            for i, rows in zip(mine, done):
                res[i] = rows
            return res

    return ShardParallel


_CHUNKS = []


def _summary_chunk(args):
    i, value_col, id_col, windowing_col, time_col, lookback_hours = args
    try:
        return ORIG_SUMMARY(_CHUNKS[i], value_col, id_col, windowing_col, time_col, lookback_hours)
    except ValueError:  # every row of this block dropped as missing: nothing to concatenate
        return pd.DataFrame()


ORIG_SUMMARY = cfa.biomarker_summary_stats


def parallel_summary(n_procs):
    """biomarker_summary_stats on contiguous blocks of sorted patient ids, in order.

    The original loops over groupby(id) (sorted ids) and concatenates, so running it
    on consecutive id blocks and concatenating the blocks gives the same table."""
    import multiprocessing as mp

    def summary(ts_df, value_col, id_col, windowing_col, time_col, lookback_hours=12):
        ids = pd.Series(ts_df[id_col].dropna().unique()).sort_values().to_numpy()
        blocks = [b for b in np.array_split(ids, n_procs * 4) if len(b)]
        global _CHUNKS
        _CHUNKS = [ts_df[ts_df[id_col].isin(b)] for b in blocks]
        jobs = [(i, value_col, id_col, windowing_col, time_col, lookback_hours) for i in range(len(blocks))]
        with mp.get_context("fork").Pool(n_procs) as pool:
            parts = pool.map(_summary_chunk, jobs, chunksize=1)
        _CHUNKS = []
        return pd.concat(parts, ignore_index=True)

    return summary


def run(shard, n_shards, rf_jobs, cli_args, sum_procs=16):
    from sklearn.ensemble import RandomForestClassifier

    cfa.biomarker_summary_stats = parallel_summary(sum_procs)

    orig_fit = RandomForestClassifier.fit

    def fit(self, X, y, sample_weight=None):
        self.n_jobs = rf_jobs
        return orig_fit(self, X, y, sample_weight)

    RandomForestClassifier.fit = fit
    args = cfa_args(cli_args)
    cfa.Parallel = shard_parallel(shard, n_shards, Path(args.output_dir), rf_jobs)
    cfa.main(args)


def cfa_args(cli_args):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lookback-hours", type=int, default=12)
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=920)
    p.add_argument("--train-n-jobs", type=int, default=1)
    return p.parse_args(cli_args)


def merge(out_dir, shard_dirs):
    shard_dirs = [Path(d) for d in shard_dirs]
    order = json.loads((shard_dirs[0] / "jobs.json").read_text())
    for d in shard_dirs[1:]:
        assert json.loads((d / "jobs.json").read_text()) == order, f"pair order differs in {d}"
    parts = [pd.read_parquet(d / FOLD_FILE) for d in shard_dirs]
    allres = pd.concat(parts, ignore_index=True)
    key = ["Model", "Feature Set", "repeat", "fold"]
    assert not allres.duplicated(key).any(), "duplicate (model, feature set, repeat, fold) rows"
    n_per_pair = allres.groupby(["Model", "Feature Set"]).size()
    pairs_found = set(n_per_pair.index)
    missing = [p for p in order if tuple(p) not in pairs_found]
    assert not missing, f"missing pairs: {missing}"
    assert n_per_pair.nunique() == 1, f"unequal fold counts: {n_per_pair.value_counts().to_dict()}"

    by_pair = {k: g.to_dict("records") for k, g in allres.groupby(["Model", "Feature Set"], sort=False)}

    class MergeParallel:
        def __init__(self, **kwargs):
            pass

        def __call__(self, tasks):
            tasks = list(tasks)
            got = [[t[1][0], t[1][2]] for t in tasks]
            assert got == order, "pair order differs from the shards"
            return [by_pair[(m, fs)] for m, fs in got]

    cfa.Parallel = MergeParallel
    feature_sets = {}
    for _, fs in order:
        feature_sets.setdefault(fs, [])
    stub = pd.DataFrame({"y": [0], "id": [0]})
    results_df, summary_df, delta_df = cfa.run_cv(stub, "y", "id", feature_sets, train_n_jobs=1)
    cfa.save_results(results_df, summary_df, delta_df, out_dir)
    print(f"merged {len(results_df)} fold rows, {len(order)} pairs, {n_per_pair.iloc[0]} folds each -> {out_dir}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        sep = sys.argv.index("--")
        run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[sep + 1:],
            sum_procs=int(sys.argv[5]) if sep > 5 else 16)
    elif mode == "merge":
        merge(Path(sys.argv[2]), sys.argv[3:])
    else:
        sys.exit(__doc__)
