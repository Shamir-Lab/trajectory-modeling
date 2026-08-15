#!/usr/bin/env python3
"""
HiRiD circulatory failure trajectory probabilities for three biomarkers:
  - Lactate
  - Heart Rate
  - Systolic BP
"""

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add code directory to path
code_dir = Path(__file__).parent.parent / 'code'
sys.path.insert(0, str(code_dir))

from traj_features.backends.bayes import BayesianTraj, BayesConfig
from traj_features.backends.bayes.classify import flags_from_traj, pos_flags_from_traj

# Use job-specific compile directory to avoid lock contention.
# Route to TRAJ_CACHE_ROOT (gaga home, ~150 GB quota) not Path.home()
# which resolves to the CS home (4 GB quota, fills up fast).
array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'local'))
array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
job_id = f"{array_job_id}_{array_task_id}"
_cache_root = Path(os.environ.get('TRAJ_CACHE_ROOT', str(Path.home())))
pytensor_cache = _cache_root / '.pytensor_cache' / job_id
pytensor_cache.mkdir(parents=True, exist_ok=True)
os.environ['PYTENSOR_FLAGS'] = (
    f"base_compiledir={pytensor_cache},"
    "optimizer=fast_compile,exception_verbosity=high"
)

# Limit threading to prevent oversubscription
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

_DATA_ROOT = os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet")

N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))
# Heartrate and systolic have 200× more rows than lactate.  Running all
# SLURM CPUs in parallel exhausts 96 GB RAM (SIGKILL on workers).  Cap
# parallelism for these biomarkers; leave lactate uncapped.
HEAVY_MAX_JOBS = int(os.environ.get('TRAJ_HEAVY_MAX_JOBS', '16'))

print(f"[Setup] PyTensor compile dir: {pytensor_cache}")
print(f"[Setup] Thread limits: OMP/MKL/OpenBLAS = 1")
print(f"[Setup] n_jobs: {N_JOBS} (SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'unset')})")
print(f"[Setup] heavy_max_jobs: {HEAVY_MAX_JOBS} (heartrate/systolic capped to avoid OOM)")


@dataclass
class BiomarkerSpec:
    name: str
    input_path: Path
    output_path: Path
    value_col: str
    value_alias: str
    baseline_candidates: List[str]
    config: BayesConfig
    column_map: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute HiRiD circulatory failure trajectory probabilities for lactate, heart rate, and systolic BP'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default=os.path.join(_DATA_ROOT, 'hirid', 'circulatory_failure'),
        help='Directory containing lactate_timeseries.csv, heartrate_timeseries.csv, systolic_timeseries.csv'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.join(_DATA_ROOT, 'hirid', 'circulatory_failure'),
        help='Directory to write trajectory probability CSVs'
    )

    parser.add_argument(
        '--window-hours',
        type=float,
        default=12.0,
        help='Lookback window in hours for all biomarkers (default: 12.0)'
    )

    parser.add_argument(
        '--n-batches',
        type=int,
        default=10,
        help='Number of patient batches (default: 10)'
    )

    parser.add_argument(
        '--cohort-splits',
        type=int,
        default=1,
        help='Total number of cohort splits (default: 1)'
    )

    parser.add_argument(
        '--cohort-index',
        type=int,
        default=0,
        help='Which cohort split to process (0-indexed)'
    )

    parser.add_argument(
        '--sampler',
        type=str,
        choices=['pymc', 'numpyro', 'nutpie'],
        default='nutpie',
        help='Sampler backend for BayesianTraj (default: nutpie; use pymc in container)'
    )

    parser.add_argument(
        '--df-basis',
        type=int,
        default=5,
        help='Spline basis degrees of freedom for BayesConfig (default: 5)'
    )

    return parser.parse_args()


def _coalesce_baseline(df: pd.DataFrame, candidates: List[str], target: str) -> Optional[str]:
    available = [c for c in candidates if c in df.columns]
    if not available:
        return None
    df[target] = df[available].bfill(axis=1).iloc[:, 0]
    return target


def _make_configs(args: argparse.Namespace) -> Dict[str, BiomarkerSpec]:
    base_in = Path(args.input_dir)
    base_out = Path(args.output_dir)
    window = args.window_hours
    sampler = args.sampler

    lactate_cfg = BayesConfig(
        window_years=window,
        df_basis=args.df_basis,
        n_samples=300,
        tune=300,
        min_points_per_window=4,
        grid_freq=3,
        flat_thr=0.03,
        decline_thr=0.12,
        nonlinear_gap=0.18,
        pids='patientid',
        values='lab_value',
        time_col='time_hours',
        windowing_col='time_hour',
        use_gpu=False,
        sampler=sampler,
        target_accept=0.995,
        chains=4,
        n_jobs=N_JOBS,
        class_func=pos_flags_from_traj,
        traj_types=('stable', 'gradual_increase', 'rapid_increase'),
        label_map={'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'}
    )

    heavy_jobs = min(N_JOBS, HEAVY_MAX_JOBS) if N_JOBS > 0 else HEAVY_MAX_JOBS

    hr_cfg = BayesConfig(
        window_years=window,
        df_basis=args.df_basis,
        n_samples=300,
        tune=300,
        min_points_per_window=4,
        grid_freq=3,
        flat_thr=1.2,
        decline_thr=4.5,
        nonlinear_gap=7.0,
        pids='patientid',
        values='lab_value',
        time_col='time_hours',
        windowing_col='time_hour',
        use_gpu=False,
        sampler=sampler,
        target_accept=0.995,
        chains=4,
        n_jobs=heavy_jobs,
        class_func=pos_flags_from_traj,
        traj_types=('stable', 'gradual_increase', 'rapid_increase'),
        label_map={'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'}
    )

    sbp_cfg = BayesConfig(
        window_years=window,
        df_basis=args.df_basis,
        n_samples=300,
        tune=300,
        min_points_per_window=4,
        grid_freq=3,
        flat_thr=-1.2,
        decline_thr=-4.5,
        nonlinear_gap=7.0,
        pids='patientid',
        values='lab_value',
        time_col='time_hours',
        windowing_col='time_hour',
        use_gpu=False,
        sampler=sampler,
        target_accept=0.995,
        chains=4,
        n_jobs=heavy_jobs,
        class_func=flags_from_traj,
        traj_types=('stable', 'gradual_decline', 'rapid_decline'),
        label_map={'nonprogression': 'stable', 'linear': 'gradual_decline', 'nonlinear': 'rapid_decline'}
    )

    return {
        'lactate': BiomarkerSpec(
            name='lactate',
            input_path=base_in / 'lactate_timeseries.csv',
            output_path=base_out / 'lactate_trajectory_probs_bayes.csv',
            value_col='lactate',
            value_alias='lactate',
            baseline_candidates=['baseline_lactate'],
            config=lactate_cfg,
            column_map={
                'trajtype_stable_prob': 'prob_stable',
                'trajtype_gradual_increase_prob': 'prob_gradual_increase',
                'trajtype_rapid_increase_prob': 'prob_rapid_increase'
            }
        ),
        'heartrate': BiomarkerSpec(
            name='heartrate',
            input_path=base_in / 'heartrate_timeseries.csv',
            output_path=base_out / 'heartrate_trajectory_probs_bayes.csv',
            value_col='heartrate',
            value_alias='heartrate',
            baseline_candidates=['baseline_heartrate'],
            config=hr_cfg,
            column_map={
                'trajtype_stable_prob': 'prob_stable',
                'trajtype_gradual_increase_prob': 'prob_gradual_increase',
                'trajtype_rapid_increase_prob': 'prob_rapid_increase'
            }
        ),
        'systolic': BiomarkerSpec(
            name='systolic',
            input_path=base_in / 'systolic_timeseries.csv',
            output_path=base_out / 'systolic_trajectory_probs_bayes.csv',
            value_col='systolic',
            value_alias='systolic',
            baseline_candidates=['baseline_systolic'],
            config=sbp_cfg,
            column_map={
                'trajtype_stable_prob': 'prob_stable',
                'trajtype_gradual_decline_prob': 'prob_gradual_decline',
                'trajtype_rapid_decline_prob': 'prob_rapid_decline'
            }
        ),
    }


def _compute_biomarker_probs(spec: BiomarkerSpec, n_batches: int, cohort_patients: Optional[List[int]] = None) -> pd.DataFrame:
    if not spec.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {spec.input_path}")

    print("=" * 80)
    print(f"Processing {spec.name.upper()} trajectories")
    print("=" * 80)
    print(f"Input:  {spec.input_path}")
    print(f"Output: {spec.output_path}")

    df = pd.read_csv(spec.input_path)

    if spec.value_col not in df.columns:
        raise KeyError(
            f"Expected value column '{spec.value_col}' not found in {spec.input_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep the original biomarker-specific column while also creating a canonical
    # value column expected by BayesConfig (`values='lab_value'`).
    if spec.value_alias != spec.value_col:
        df = df.rename(columns={spec.value_col: spec.value_alias})
    df['lab_value'] = df[spec.value_alias]

    baseline_name = f"baseline_{spec.value_alias}"
    _coalesce_baseline(df, spec.baseline_candidates, baseline_name)

    df = df.dropna(subset=[spec.value_alias])
    df = df.sort_values(by=['patientid', 'time_hours'])
    df = df.drop_duplicates(subset=['patientid', 'time_hour', spec.value_alias])
    if cohort_patients is not None:
        df = df[df['patientid'].isin(cohort_patients)]

    print(f"Patients: {df['patientid'].nunique():,}")
    print(f"Measurements: {len(df):,}")

    model = BayesianTraj(cfg=spec.config)

    patients = df['patientid'].unique()
    npts = patients.size
    batch_size = max(1, npts // n_batches)

    outputs = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
        subset_patients = patients[start_idx:end_idx]

        print(f"  Batch {i + 1}/{n_batches} ({len(subset_patients)} patients)...")

        batch_probs = model.embed(df[df['patientid'].isin(subset_patients)])
        outputs.append(batch_probs)

        gc.collect()

    trajectory_probs = pd.concat(outputs, ignore_index=True)

    if not set(spec.column_map).issubset(trajectory_probs.columns):
        # Every window for this biomarker/cohort failed to sample (e.g. no
        # patient cleared min_points_per_window) — embed() returns a frame
        # without the trajtype_* columns instead of an empty-but-typed one.
        # Treat it as "nothing to save" instead of crashing the whole cohort
        # (and losing whichever other biomarkers already saved successfully).
        print(f"⚠️  ERROR: No windows produced valid trajectory probabilities for {spec.name} "
              f"(expected columns {sorted(spec.column_map)} not in output — likely too few points per patient).")
        return None

    trajectory_probs = trajectory_probs.rename(columns=spec.column_map)

    prob_cols = list(spec.column_map.values())
    trajectory_probs = trajectory_probs.rename(columns={'patientid': 'patientid'})

    probs_ts = df.merge(
        trajectory_probs[['patientid', 'time_hour'] + prob_cols],
        on=['patientid', 'time_hour'],
        how='inner'
    )

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    probs_ts.to_csv(spec.output_path, index=False)
    print(f"✓ Saved: {spec.output_path}")

    return trajectory_probs


def main():
    args = parse_args()
    specs = _make_configs(args)

    if args.cohort_index < 0 or args.cohort_index >= args.cohort_splits:
        print(f"ERROR: cohort-index must be between 0 and {args.cohort_splits - 1}")
        sys.exit(1)

    cohort_patients = None
    if args.cohort_splits > 1:
        cohort_source = None
        for spec in specs.values():
            if spec.input_path.exists():
                cohort_source = spec.input_path
                break

        if cohort_source is not None:
            cohort_df = pd.read_csv(cohort_source, usecols=['patientid'])
            all_patients = pd.Series(cohort_df['patientid'].unique()).sort_values().to_numpy()
            total_patients = len(all_patients)
            cohort_size = total_patients // args.cohort_splits
            start_idx = args.cohort_index * cohort_size
            end_idx = total_patients if args.cohort_index == args.cohort_splits - 1 else start_idx + cohort_size
            cohort_patients = all_patients[start_idx:end_idx].tolist()

            print(f"\n📊 Sub-cohort {args.cohort_index + 1}/{args.cohort_splits}:")
            print(f"   Processing patients {start_idx:,} to {end_idx:,} (of {total_patients:,} total)")
            print(f"   Cohort size: {len(cohort_patients):,} patients")

    for spec in specs.values():
        if args.cohort_splits > 1:
            spec.output_path = spec.output_path.parent / f"{spec.output_path.stem}_cohort{args.cohort_index:02d}{spec.output_path.suffix}"
        if spec.output_path.exists() and spec.output_path.stat().st_size > 0:
            print(f"✓ Skipping {spec.name} — output already exists: {spec.output_path}")
            continue
        _compute_biomarker_probs(spec, args.n_batches, cohort_patients)


if __name__ == '__main__':
    main()
