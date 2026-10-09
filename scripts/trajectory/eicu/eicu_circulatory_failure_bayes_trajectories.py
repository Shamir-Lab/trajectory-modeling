#!/usr/bin/env python3
"""eICU Circulatory Failure Bayesian trajectory generation (full dataset with nutpie)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASET = "eicu"
DEFAULT_COHORT_SPLITS = 4
DATA_ROOT = os.path.join(os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet"), "eicu", "circulatory_failure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="eICU Circulatory Failure Bayesian Trajectory Generation (full dataset)")
    parser.add_argument("--cohort-index", type=int, default=0, help="Cohort index (0-indexed)")
    parser.add_argument("--cohort-splits", type=int, default=DEFAULT_COHORT_SPLITS, help="Total cohort splits")
    parser.add_argument("--window-hours", type=float, default=12.0, help="Trajectory window in hours")
    parser.add_argument("--n-batches", type=int, default=8, help="Parallel batches")
    parser.add_argument("--df-basis", type=int, default=5, help="Spline basis complexity / degrees of freedom")
    parser.add_argument("--sampler", type=str, choices=["pymc", "numpyro", "nutpie"], default="nutpie", help="Sampler backend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    traj_script = root / "scripts" / "trajectory" / "eicu" / "circulatory_failure_trajs.py"

    cmd = [
        sys.executable,
        str(traj_script),
        "--data-dir",
        DATA_ROOT,
        "--pred-dataset",
        f"{DATA_ROOT}/circulatory_failure_prediction_dataset.csv",
        "--merged-output",
        f"{DATA_ROOT}/circulatory_failure_prediction_dataset_with_probs.csv",
        "--window-hours",
        str(args.window_hours),
        "--n-batches",
        str(args.n_batches),
        "--df-basis",
        str(args.df_basis),
        "--cohort-splits",
        str(args.cohort_splits),
        "--cohort-index",
        str(args.cohort_index),
        "--sampler",
        args.sampler,
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
