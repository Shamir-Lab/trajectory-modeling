"""
Sampler benchmark: nutpie (CPU) vs numpyro (JAX/GPU).

Run one sampler per invocation — submit two SLURM jobs and compare the JSONs:

    python scripts/benchmarks/sampler_benchmark.py --sampler nutpie
    python scripts/benchmarks/sampler_benchmark.py --sampler numpyro
    python scripts/benchmarks/sampler_benchmark.py --compare
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = REPO_ROOT / "logs" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Benchmark config — realistic production-like settings
# ---------------------------------------------------------------------------
N_PATIENTS   = 30
N_TIMEPOINTS = 28    # 4 weeks of daily measurements
N_SAMPLES    = 500
TUNE         = 250
CHAINS       = 4
BATCH_SIZE   = 10
# window_years is compared directly against windowing_col values (time_day, integers).
# 14-day lookback gives up to 15 points per window — realistic for ICU lab trending.
WINDOW_YEARS = 14.0
DF_BASIS     = 4
GRID_FREQ    = 12


def make_cohort(n_patients: int = N_PATIENTS,
                n_timepoints: int = N_TIMEPOINTS,
                seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for pid in range(n_patients):
        times = np.linspace(0, n_timepoints - 1, n_timepoints)
        # Slight per-patient trend variation to make the model interesting
        trend = rng.uniform(-0.02, 0.04)
        values = 1.0 + trend * times + rng.randn(n_timepoints) * 0.12
        for t, v in zip(times, values):
            rows.append({
                "patientid": pid,
                "time_days":  round(float(t), 4),
                "time_day":   int(t),
                "lab_value":  max(0.2, float(v)),
            })
    return pd.DataFrame(rows)


def run_benchmark(sampler: str) -> dict:
    from traj_features.backends.bayes import BayesConfig, BayesianTraj
    from traj_features.backends.bayes.classify import pos_flags_from_traj

    use_gpu = sampler == "numpyro"

    cfg = BayesConfig(
        window_years=WINDOW_YEARS,
        df_basis=DF_BASIS,
        n_samples=N_SAMPLES,
        tune=TUNE,
        chains=CHAINS,
        cores=1,
        min_points_per_window=5,
        grid_freq=GRID_FREQ,
        pids="patientid",
        values="lab_value",
        time_col="time_days",
        windowing_col="time_day",
        sampler=sampler,
        use_gpu=use_gpu,
        batch_size=BATCH_SIZE,
        n_jobs=1,
        progressbar=False,
        flat_thr=0.1,
        decline_thr=0.3,
        nonlinear_gap=0.5,
        class_func=pos_flags_from_traj,
        traj_types=("stable", "gradual_increase", "rapid_increase"),
        label_map={
            "nonprogression": "stable",
            "linear": "gradual_increase",
            "nonlinear": "rapid_increase",
        },
    )

    model = BayesianTraj(cfg=cfg)
    # Respect explicit sampler choice — don't let auto-detection override it
    model.cfg.sampler = sampler

    df = make_cohort()
    n_windows = df.groupby(["patientid", "time_day"]).ngroups
    print(f"\n[{sampler}] {N_PATIENTS} patients, {n_windows} windows")
    print(f"         {N_SAMPLES} samples, {TUNE} tune, {CHAINS} chains")

    t0 = time.time()
    result = model.embed(df)
    elapsed = time.time() - t0

    n_rows = len(result)
    windows_per_sec = n_windows / elapsed if elapsed > 0 else 0

    print(f"[{sampler}] {elapsed:.1f}s total | {n_rows} output rows | "
          f"{windows_per_sec:.2f} windows/s")

    record = {
        "sampler":         sampler,
        "n_patients":      N_PATIENTS,
        "n_windows":       n_windows,
        "n_samples":       N_SAMPLES,
        "tune":            TUNE,
        "chains":          CHAINS,
        "elapsed_s":       round(elapsed, 2),
        "windows_per_sec": round(windows_per_sec, 3),
        "n_output_rows":   n_rows,
    }

    out_path = RESULTS_DIR / f"benchmark_{sampler}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"[{sampler}] results saved → {out_path}")
    return record


def compare() -> None:
    results = {}
    for sampler in ("nutpie", "numpyro"):
        p = RESULTS_DIR / f"benchmark_{sampler}.json"
        if p.exists():
            results[sampler] = json.loads(p.read_text())
        else:
            print(f"  missing: {p}")

    if len(results) < 2:
        print("Need both nutpie and numpyro results to compare.")
        return

    a, b = results["nutpie"], results["numpyro"]

    # Sanity check: configs should match
    mismatch = {k for k in ("n_patients", "n_windows", "n_samples", "tune", "chains")
                if a.get(k) != b.get(k)}
    if mismatch:
        print(f"  WARNING: configs differ in {mismatch} — speedup numbers may be misleading")

    speedup = a["elapsed_s"] / b["elapsed_s"] if b["elapsed_s"] > 0 else float("nan")

    print("\n" + "=" * 52)
    print(f"  {'Sampler':<12} {'Time (s)':>10} {'Win/s':>10}")
    print("-" * 52)
    for r in (a, b):
        print(f"  {r['sampler']:<12} {r['elapsed_s']:>10.1f} {r['windows_per_sec']:>10.3f}")
    print("-" * 52)
    print(f"  numpyro is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than nutpie")
    print("=" * 52)
    print(f"  config: {a['n_patients']} patients, {a['n_windows']} windows, "
          f"{a['n_samples']} samples, {a['tune']} tune, {a['chains']} chains")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sampler", choices=["nutpie", "numpyro", "pymc"],
                       help="Sampler to benchmark")
    group.add_argument("--compare", action="store_true",
                       help="Compare saved results from previous runs")
    args = parser.parse_args()

    if args.compare:
        compare()
    else:
        run_benchmark(args.sampler)
