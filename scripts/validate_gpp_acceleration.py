#!/usr/bin/env python3
"""
Validate and benchmark g++ acceleration for PyMC/PyTensor trajectory sampling.

Runs two identical MCMC benchmarks in separate subprocesses:
  1. WITH g++ (C-compiled PyTensor ops)  — normal container mode
  2. WITHOUT g++ (pure-Python PyTensor)  — fallback mode

Reports timing, speedup ratio, and verifies outputs are non-empty.

Usage:
    python scripts/validate_gpp_acceleration.py [--n-patients N] [--n-samples N]

Expected output:
    [WITH g++]  elapsed: X.Xs | rows: N | probs in [0,1]: YES
    [NO  g++]   elapsed: X.Xs | rows: N | probs in [0,1]: YES
    Speedup: X.Xx (with g++ vs without)
    g++ acceleration is ACTIVE (or NOT ACTIVE)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CODE = textwrap.dedent("""\
import sys, time, json, os, numpy as np, pandas as pd
sys.path.insert(0, {src!r})

from traj_features.backends.bayes import BayesConfig, BayesianTraj
from traj_features.backends.bayes.classify import pos_flags_from_traj

rng = np.random.RandomState(42)
rows = []
for pid in range({n_patients}):
    t = np.linspace(0, 7, 8)
    v = 1.0 + 0.05 * t + rng.randn(8) * 0.1
    for ti, vi in zip(t, v):
        rows.append(dict(patientid=pid, time_days=round(float(ti), 4),
                         time_day=int(ti), lab_value=max(0.2, float(vi))))
df = pd.DataFrame(rows)

cfg = BayesConfig(
    window_years=7.0, df_basis=4,
    n_samples={n_samples}, tune={n_samples},
    chains=1, cores=1,
    min_points_per_window=5, grid_freq=3,
    pids="patientid", values="lab_value",
    time_col="time_days", windowing_col="time_day",
    sampler="pymc", use_gpu=False,
    batch_size=50, n_jobs=1, progressbar=False,
    flat_thr=0.1, decline_thr=0.3, nonlinear_gap=0.5,
    class_func=pos_flags_from_traj,
    traj_types=("stable", "gradual_increase", "rapid_increase"),
    label_map={{"nonprogression": "stable",
                "linear": "gradual_increase",
                "nonlinear": "rapid_increase"}},
)
model = BayesianTraj(cfg=cfg)
model.cfg.sampler = "pymc"

# Pass 1: warmup — triggers C compilation if g++ is active.
# This is the one-time cost paid on first use; subsequent calls hit the cache.
t_warmup = time.time()
model.embed(df)
elapsed_warmup = time.time() - t_warmup

# Pass 2: timed benchmark with warm cache (compiled ops already on disk).
t0 = time.time()
result = model.embed(df)
elapsed = time.time() - t0

prob_cols = [c for c in result.columns if c.startswith("trajtype_")]
probs_valid = all(result[c].between(0.0, 1.0).all() for c in prob_cols)

print(json.dumps(dict(
    elapsed=round(elapsed, 3),
    elapsed_warmup=round(elapsed_warmup, 3),
    n_rows=len(result),
    n_prob_cols=len(prob_cols),
    probs_valid=probs_valid,
    empty=result.empty,
)))
""")


def _run_benchmark(n_patients: int, n_samples: int, with_gpp: bool) -> dict:
    """Run benchmark in a subprocess, optionally disabling g++."""
    code = BENCHMARK_CODE.format(
        src=str(REPO_ROOT / "src"),
        n_patients=n_patients,
        n_samples=n_samples,
    )

    env = os.environ.copy()
    if with_gpp:
        # Let PyTensor auto-detect g++ (normal mode)
        env["PYTENSOR_FLAGS"] = (
            "base_compiledir=/tmp/pytensor_validate_gpp,"
            "optimizer=fast_compile"
        )
    else:
        # Disable C compilation: cxx= (empty string) tells PyTensor no compiler
        env["PYTENSOR_FLAGS"] = (
            "base_compiledir=/tmp/pytensor_validate_nogpp,"
            "optimizer=fast_compile,cxx="
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="traj_bench_"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, env=env, timeout=600
        )
    finally:
        os.unlink(tmp_path)

    if result.returncode != 0:
        print(f"  [STDERR]\n{result.stderr[-2000:]}", file=sys.stderr)
        raise RuntimeError(
            f"Benchmark subprocess failed (returncode={result.returncode})"
        )

    # Parse last JSON line (subprocess may print PyMC progress above it)
    json_lines = [l for l in result.stdout.strip().splitlines() if l.startswith("{")]
    if not json_lines:
        print(f"  [STDOUT]\n{result.stdout[-2000:]}", file=sys.stderr)
        raise RuntimeError("No JSON output from benchmark subprocess")

    return json.loads(json_lines[-1])


def _check_gpp_available() -> bool:
    """Return True if g++ is in PATH."""
    import shutil
    return shutil.which("g++") is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patients", type=int, default=4,
                        help="Synthetic patients per benchmark (default: 4)")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="MCMC samples per run (default: 20; use 100+ for stable timing)")
    parser.add_argument("--skip-no-gpp", action="store_true",
                        help="Skip the without-g++ benchmark (saves time)")
    args = parser.parse_args()

    gpp_available = _check_gpp_available()
    print("=" * 60)
    print("PyTensor / g++ Acceleration Validation")
    print("=" * 60)
    print(f"g++ in PATH : {'YES — C compilation will be active' if gpp_available else 'NO  — will run in pure-Python mode'}")
    print(f"n_patients  : {args.n_patients}")
    print(f"n_samples   : {args.n_samples}")
    print()

    # ── Benchmark 1: WITH g++ ──────────────────────────────────────────────
    print("Running benchmark 1 of 2: WITH g++ ...")
    t_start = time.time()
    try:
        r_gpp = _run_benchmark(args.n_patients, args.n_samples, with_gpp=True)
        ok_gpp = True
    except Exception as e:
        print(f"  FAILED: {e}")
        ok_gpp = False
        r_gpp = {}
    if ok_gpp:
        print(f"  warmup   : {r_gpp['elapsed_warmup']:.2f}s  (one-time C compilation cost)")
        print(f"  elapsed  : {r_gpp['elapsed']:.2f}s  (steady-state, compiled ops cached)")
        print(f"  rows     : {r_gpp['n_rows']}")
        print(f"  prob cols: {r_gpp['n_prob_cols']}")
        print(f"  probs OK : {'YES' if r_gpp['probs_valid'] else 'NO'}")
        print(f"  empty    : {'YES ← ERROR' if r_gpp['empty'] else 'NO'}")

    print()

    # ── Benchmark 2: WITHOUT g++ ───────────────────────────────────────────
    if not args.skip_no_gpp:
        print("Running benchmark 2 of 2: WITHOUT g++ (pure Python) ...")
        t_start = time.time()
        try:
            r_nogpp = _run_benchmark(args.n_patients, args.n_samples, with_gpp=False)
            ok_nogpp = True
        except Exception as e:
            print(f"  FAILED: {e}")
            ok_nogpp = False
            r_nogpp = {}
        t_wall_nogpp = time.time() - t_start

        if ok_nogpp:
            print(f"  elapsed  : {r_nogpp['elapsed']:.2f}s  (wall: {t_wall_nogpp:.1f}s)")
            print(f"  rows     : {r_nogpp['n_rows']}")
            print(f"  prob cols: {r_nogpp['n_prob_cols']}")
            print(f"  probs OK : {'YES' if r_nogpp['probs_valid'] else 'NO'}")
            print(f"  empty    : {'YES ← ERROR' if r_nogpp['empty'] else 'NO'}")
    else:
        ok_nogpp = False

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    # ── Speedup report ─────────────────────────────────────────────────────
    if ok_gpp and ok_nogpp:
        if r_nogpp["elapsed"] > 0:
            speedup = r_nogpp["elapsed"] / r_gpp["elapsed"]
            print(f"Speedup (g++ vs no-g++) : {speedup:.1f}x")
            if speedup > 1.5:
                print("Result: g++ acceleration is ACTIVE and providing a speedup.")
            else:
                print("Result: speedup is low (<1.5x). Check that g++ is installed and")
                print("        that PYTENSOR_FLAGS does not disable C compilation.")
        else:
            print("Could not compute speedup (no-g++ elapsed time is 0).")
    elif ok_gpp and not ok_nogpp:
        print("(without-g++ benchmark was skipped or failed)")
    elif not ok_gpp:
        print("ERROR: the with-g++ benchmark failed. Check error output above.")
        sys.exit(1)

    # ── Output integrity ───────────────────────────────────────────────────
    if ok_gpp:
        if r_gpp["empty"]:
            print("ERROR: embed() returned an empty DataFrame — outputs are empty!")
            sys.exit(1)
        if not r_gpp["probs_valid"]:
            print("ERROR: probability values outside [0, 1] detected!")
            sys.exit(1)
        print("Output check: PASSED (non-empty, probabilities in [0, 1])")

    # ── g++ status ─────────────────────────────────────────────────────────
    print()
    if gpp_available:
        print("g++ acceleration: ACTIVE (g++ found in PATH, PyTensor will compile C ops)")
    else:
        print("g++ acceleration: NOT ACTIVE (g++ not in PATH)")
        print("  → Inside the container, g++ is installed and this will be ACTIVE.")
        print("  → Outside the container (university server), this is expected.")


if __name__ == "__main__":
    main()
