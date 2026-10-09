#!/usr/bin/env python3
"""
Run Bayesian circulatory-failure trajectories on a smaller patient subset.

This script:
1) Samples N patients from each dataset's circulatory-failure prediction table
2) Writes subset prediction + timeseries CSVs to a dedicated subset directory
3) Launches the existing Bayesian trajectory scripts on those subset files

Default: runs for mimic + eicu + hirid, with 1500 patients each.

Example:
    python scripts/trajectory/run_circulatory_failure_bayes_subset.py

Custom example:
    python scripts/trajectory/run_circulatory_failure_bayes_subset.py \
      --datasets mimic eicu hirid --n-patients 1500 --seed 42 --n-batches 8 --window-hours 12
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet"))


DATASET_CFG = {
    "mimic": {
        "base_dir": DATA_ROOT / "mimic" / "circulatory_failure",
        "id_col": "hadm_id",
        "ts_files": [
            "lactate_timeseries.csv",
            "heartrate_timeseries.csv",
            "systolic_timeseries.csv",
        ],
        "traj_script": WORKSPACE_ROOT / "scripts" / "trajectory" / "mimic" / "circulatory_failure_trajs.py",
    },
    "eicu": {
        "base_dir": DATA_ROOT / "eicu" / "circulatory_failure",
        "id_col": "stay_id",
        "ts_files": [
            "lactate_timeseries.csv",
            "heartrate_timeseries.csv",
            "systolic_timeseries.csv",
        ],
        "traj_script": WORKSPACE_ROOT / "scripts" / "trajectory" / "eicu" / "circulatory_failure_trajs.py",
    },
    "hirid": {
        "base_dir": DATA_ROOT / "hirid" / "circulatory_failure",
        "id_col": "patientid",
        "ts_files": [
            "lactate_timeseries.csv",
            "heartrate_timeseries.csv",
            "systolic_timeseries.csv",
        ],
        "traj_script": WORKSPACE_ROOT / "scripts" / "trajectory" / "hirid" / "circulatory_failure_trajs.py",
    },
}


def _read_table(path_stem: Path) -> pd.DataFrame:
    csv_path = path_stem.with_suffix(".csv")
    parquet_path = path_stem.with_suffix(".parquet")

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(f"Missing table: {csv_path} or {parquet_path}")


def _load_prediction_df(base_dir: Path) -> pd.DataFrame:
    candidates = [
        "circulatory_failure_prediction_dataset",
        "circulatory_failure_prediction_dataset_with_probs",
        "circulatory_failure_prediction_dataset_with_bootstrap_probs",
    ]
    for stem in candidates:
        try:
            return _read_table(base_dir / stem)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No prediction dataset found under {base_dir}")


def _subset_patients(df: pd.DataFrame, id_col: str, n_patients: int, seed: int) -> pd.Index:
    unique_ids = pd.Index(df[id_col].dropna().unique())
    if len(unique_ids) == 0:
        raise ValueError(f"No patient IDs found in column '{id_col}'")
    if len(unique_ids) <= n_patients:
        return unique_ids
    return pd.Series(unique_ids).sample(n=n_patients, random_state=seed, replace=False).sort_values().values


def _filter_and_save_csv(in_path: Path, out_path: Path, id_col: str, selected_ids: Iterable) -> tuple[int, int]:
    ts_df = pd.read_csv(in_path)
    before = len(ts_df)
    if id_col not in ts_df.columns:
        raise ValueError(f"{in_path.name} missing ID column '{id_col}'")
    ts_df = ts_df[ts_df[id_col].isin(selected_ids)].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts_df.to_csv(out_path, index=False)
    return before, len(ts_df)


def _run(cmd: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def prepare_subset(dataset: str, n_patients: int, seed: int, out_root: Path) -> tuple[Path, Path, str, int]:
    cfg = DATASET_CFG[dataset]
    base_dir: Path = cfg["base_dir"]
    id_col: str = cfg["id_col"]

    pred_df = _load_prediction_df(base_dir)
    if id_col not in pred_df.columns:
        raise ValueError(f"Prediction dataset for {dataset} missing ID column '{id_col}'")

    selected_ids = _subset_patients(pred_df, id_col=id_col, n_patients=n_patients, seed=seed)
    subset_pred = pred_df[pred_df[id_col].isin(selected_ids)].copy()

    subset_dir = out_root / dataset / f"patients_{len(selected_ids)}_seed_{seed}"
    subset_dir.mkdir(parents=True, exist_ok=True)

    subset_pred_path = subset_dir / "circulatory_failure_prediction_dataset.csv"
    subset_pred.to_csv(subset_pred_path, index=False)

    print(f"\n[{dataset}] Selected patients: {len(selected_ids):,}")
    print(f"[{dataset}] Prediction rows: {len(subset_pred):,}")
    print(f"[{dataset}] Subset dir: {subset_dir}")

    for ts_name in cfg["ts_files"]:
        src = base_dir / ts_name
        if not src.exists():
            raise FileNotFoundError(f"Missing timeseries file for {dataset}: {src}")
        dst = subset_dir / ts_name
        before, after = _filter_and_save_csv(src, dst, id_col=id_col, selected_ids=selected_ids)
        print(f"[{dataset}] {ts_name}: {after:,}/{before:,} rows")

    return subset_dir, subset_pred_path, id_col, len(selected_ids)


def run_bayesian_trajectories(
    dataset: str,
    subset_dir: Path,
    subset_pred_path: Path,
    window_hours: float,
    n_batches: int,
    cohort_splits: int = 1,
    cohort_index: int = 0,
    dry_run: bool = False,
) -> None:
    script_path = DATASET_CFG[dataset]["traj_script"]
    if not script_path.exists():
        raise FileNotFoundError(f"Trajectory script not found: {script_path}")

    py = sys.executable

    if dataset == "mimic":
        merged_out = subset_dir / "circulatory_failure_prediction_dataset_with_probs.csv"
        cmd = [
            py,
            str(script_path),
            "--biomarker",
            "all",
            "--data-dir",
            str(subset_dir),
            "--pred-dataset",
            str(subset_pred_path),
            "--merged-output",
            str(merged_out),
            "--window-hours",
            str(window_hours),
            "--n-batches",
            str(n_batches),
            "--cohort-splits",
            str(cohort_splits),
            "--cohort-index",
            str(cohort_index),
        ]
    elif dataset == "eicu":
        merged_out = subset_dir / "circulatory_failure_prediction_dataset_with_probs.csv"
        cmd = [
            py,
            str(script_path),
            "--data-dir",
            str(subset_dir),
            "--pred-dataset",
            str(subset_pred_path),
            "--merged-output",
            str(merged_out),
            "--window-hours",
            str(window_hours),
            "--n-batches",
            str(n_batches),
            "--cohort-splits",
            str(cohort_splits),
            "--cohort-index",
            str(cohort_index),
        ]
    else:  # hirid
        cmd = [
            py,
            str(script_path),
            "--input-dir",
            str(subset_dir),
            "--output-dir",
            str(subset_dir),
            "--window-hours",
            str(window_hours),
            "--n-batches",
            str(n_batches),
            "--cohort-splits",
            str(cohort_splits),
            "--cohort-index",
            str(cohort_index),
            "--sampler",
            "pymc",
        ]

    _run(cmd, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Bayesian circulatory failure trajectories on 1500-patient subsets")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["mimic", "eicu", "hirid"],
        choices=["mimic", "eicu", "hirid"],
        help="Datasets to run",
    )
    p.add_argument("--n-patients", type=int, default=1500, help="Patients to sample per dataset")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed")
    p.add_argument("--window-hours", type=float, default=12.0, help="Bayesian trajectory lookback window (hours)")
    p.add_argument("--n-batches", type=int, default=8, help="Batches for trajectory scripts")
    p.add_argument("--cohort-splits", type=int, default=1, help="Total number of cohort splits per dataset")
    p.add_argument("--cohort-index", type=int, default=0, help="Which cohort to process (0-indexed)")
    p.add_argument(
        "--subset-root",
        type=str,
        default=str(DATA_ROOT / "subsets" / "circulatory_failure_bayes"),
        help="Root directory where subset files/results are written",
    )
    p.add_argument("--dry-run", action="store_true", help="Only print commands, do not execute")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    subset_root = Path(args.subset_root)

    if args.cohort_index < 0 or args.cohort_index >= args.cohort_splits:
        print(f"ERROR: cohort-index must be between 0 and {args.cohort_splits - 1}")
        sys.exit(1)

    print("=" * 88)
    print("Bayesian Circulatory Failure Subset Runner")
    print("=" * 88)
    print(f"Datasets: {args.datasets}")
    print(f"Patients per dataset: {args.n_patients}")
    print(f"Seed: {args.seed}")
    print(f"Cohorts per dataset: {args.cohort_splits}")
    print(f"Processing cohort: {args.cohort_index} / {args.cohort_splits}")
    print(f"Subset root: {subset_root}")

    for ds in args.datasets:
        subset_dir, subset_pred_path, id_col, actual_n = prepare_subset(
            dataset=ds,
            n_patients=args.n_patients,
            seed=args.seed,
            out_root=subset_root,
        )
        print(f"[{ds}] Running Bayesian trajectories on {actual_n:,} patients (cohort {args.cohort_index}/{args.cohort_splits})...")
        run_bayesian_trajectories(
            dataset=ds,
            subset_dir=subset_dir,
            subset_pred_path=subset_pred_path,
            window_hours=args.window_hours,
            n_batches=args.n_batches,
            cohort_splits=args.cohort_splits,
            cohort_index=args.cohort_index,
            dry_run=args.dry_run,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
