#!/usr/bin/env python3
# Benchmark behind thesis Section 3.3.3 (September 2026), copied unchanged from /tmp/bench_backends.py on rack-shamir4.
# Run from the repo root on the server; it reads the shared data root. Results: docs/thesis_results.md.
"""
Runtime and memory benchmark: Bayesian MCMC backend vs bootstrap backend.

Both backends are given the identical set of patient-windows and the production
configuration for the chosen task. Peak memory is sampled across the whole
process tree (joblib workers are child processes), so the number reported is the
real footprint of the run, not just the parent.

Usage:
  python bench_backends.py --backend bayes     --n-patients 8  --n-jobs 1
  python bench_backends.py --backend bootstrap --n-patients 8  --n-jobs 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, "src")

from traj_features.backends.bayes import BayesianTraj, BayesConfig
from traj_features.backends.bayes.classify import pos_flags_from_traj
from traj_features.backends.bootstrap import BootstrapTraj, BootstrapConfig

DATA = "/home/gaga/data/physionet"

TASKS = {
    # production settings, from scripts/trajectory/<ds>/<task>_trajs.py
    "sepsis_lactate": dict(
        path=f"{DATA}/eicu/sepsis/lactate_timeseries.csv",
        id_col="stay_id", value_col="lactate",
        time_col="time_days", window_col="time_day",
        window=3.0, flat_thr=0.1, decline_thr=0.3, nonlinear_gap=0.5,
        min_points=4, grid_freq=2, n_bootstrap=200,
    ),
    "circ_lactate": dict(
        path=f"{DATA}/eicu/circulatory_failure/lactate_timeseries.csv",
        id_col="stay_id", value_col="lactate",
        time_col="time_hours", window_col="time_hour",
        window=12.0, flat_thr=0.04, decline_thr=0.15, nonlinear_gap=0.2,
        min_points=4, grid_freq=2, n_bootstrap=200,
    ),
    "circ_systolic": dict(
        path=f"{DATA}/eicu/circulatory_failure/systolic_timeseries.csv",
        id_col="stay_id", value_col="systolic",
        time_col="time_hours", window_col="time_hour",
        window=12.0, flat_thr=-1.5, decline_thr=-6.0, nonlinear_gap=8.0,
        min_points=4, grid_freq=2, n_bootstrap=200,
    ),
}

TRAJ_TYPES = ("stable", "gradual_increase", "rapid_increase")
LABEL_MAP = {"nonprogression": "stable", "linear": "gradual_increase",
             "nonlinear": "rapid_increase"}


class TreeMemorySampler(threading.Thread):
    """Peak RSS of this process plus all descendants, sampled periodically."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mb = 0.0
        self.peak_procs = 0
        self._stop_evt = threading.Event()

    def run(self) -> None:
        me = psutil.Process()
        while not self._stop_evt.is_set():
            total, n = 0, 0
            try:
                procs = [me] + me.children(recursive=True)
                for p in procs:
                    try:
                        total += p.memory_info().rss
                        n += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except psutil.Error:
                pass
            mb = total / 1e6
            if mb > self.peak_mb:
                self.peak_mb = mb
                self.peak_procs = n
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()


def build_input(cfg: dict, n_patients: int) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(cfg["path"], usecols=[cfg["id_col"], cfg["value_col"],
                                            cfg["time_col"], cfg["window_col"]])
    raw[cfg["value_col"]] = pd.to_numeric(raw[cfg["value_col"]], errors="coerce")
    raw = raw.dropna(subset=[cfg["value_col"]])

    counts = raw.groupby(cfg["id_col"]).size().sort_values(ascending=False)
    pids = list(counts[counts >= 8].index[:n_patients])
    sub = raw[raw[cfg["id_col"]].isin(pids)]

    ti = sub.rename(columns={cfg["id_col"]: "patientid",
                             cfg["time_col"]: "time",
                             cfg["window_col"]: "time_window",
                             cfg["value_col"]: "lab_value"})
    ti = ti[["patientid", "time", "time_window", "lab_value"]].sort_values(
        ["patientid", "time"])

    # Count the windows both backends will build, for a per-window cost.
    n_windows, sizes = 0, []
    for pid, g in ti.groupby("patientid"):
        for wid in np.unique(np.floor(g["time_window"].to_numpy(dtype=float))):
            w = g[(g["time_window"] >= wid - cfg["window"]) & (g["time_window"] <= wid)]
            if len(w) >= cfg["min_points"]:
                n_windows += 1
                sizes.append(len(w))
    stats = dict(n_patients=len(pids), n_rows=int(len(ti)),
                 n_windows=n_windows,
                 median_window_size=int(np.median(sizes)) if sizes else 0)
    return ti, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["bayes", "bootstrap"], required=True)
    ap.add_argument("--task", choices=list(TASKS), default="sepsis_lactate")
    ap.add_argument("--n-patients", type=int, default=8)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--sampler", default="nutpie")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = TASKS[args.task]
    ti, stats = build_input(cfg, args.n_patients)

    if args.backend == "bayes":
        model = BayesianTraj(BayesConfig(
            window_years=cfg["window"], df_basis=5,
            n_samples=200, tune=300, chains=4, target_accept=0.99,
            min_points_per_window=cfg["min_points"], grid_freq=cfg["grid_freq"],
            flat_thr=cfg["flat_thr"], decline_thr=cfg["decline_thr"],
            nonlinear_gap=cfg["nonlinear_gap"],
            pids="patientid", values="lab_value",
            time_col="time", windowing_col="time_window",
            sampler=args.sampler, n_jobs=args.n_jobs, progressbar=False,
            traj_types=TRAJ_TYPES, class_func=pos_flags_from_traj,
            label_map=LABEL_MAP,
        ))
    else:
        model = BootstrapTraj(BootstrapConfig(
            window_years=cfg["window"], n_bootstrap=cfg["n_bootstrap"],
            smoothing=None,
            min_points_per_window=cfg["min_points"], grid_freq=cfg["grid_freq"],
            flat_thr=cfg["flat_thr"], decline_thr=cfg["decline_thr"],
            nonlinear_gap=cfg["nonlinear_gap"],
            pids="patientid", values="lab_value",
            time_col="time", windowing_col="time_window",
            n_jobs=args.n_jobs, progressbar=False,
            traj_types=TRAJ_TYPES, class_func=pos_flags_from_traj,
            label_map=LABEL_MAP,
        ))

    mem = TreeMemorySampler()
    baseline = psutil.Process().memory_info().rss / 1e6
    mem.start()
    t0 = time.perf_counter()
    out = model.embed(ti.copy())
    elapsed = time.perf_counter() - t0
    mem.stop()
    mem.join(timeout=2)

    rec = dict(
        backend=args.backend, task=args.task, sampler=args.sampler if args.backend == "bayes" else None,
        n_jobs=args.n_jobs, **stats,
        rows_returned=int(len(out)),
        elapsed_s=round(elapsed, 2),
        windows_per_s=round(stats["n_windows"] / elapsed, 3) if elapsed else None,
        ms_per_window=round(1000 * elapsed / stats["n_windows"], 1) if stats["n_windows"] else None,
        baseline_rss_mb=round(baseline, 1),
        peak_tree_rss_mb=round(mem.peak_mb, 1),
        peak_over_baseline_mb=round(mem.peak_mb - baseline, 1),
        procs_at_peak=mem.peak_procs,
    )
    print(json.dumps(rec))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
