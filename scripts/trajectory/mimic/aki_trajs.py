"""
Compute creatinine trajectory probabilities for MIMIC AKI cohort.

Usage:
    python aki_trajs.py --input results/mimic/aki/creatinine_timeseries.csv --pred-dataset results/mimic/aki/aki_prediction_dataset.csv --output results/mimic/aki/aki_trajectory_probs_bayes.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import gc
import shutil
from pathlib import Path
import sys

# Ensure compiled artifacts and matplotlib cache land in a writable location
ARRAY_JOB_ID = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
ARRAY_TASK_ID = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
JOB_ID = f"{ARRAY_JOB_ID}_{ARRAY_TASK_ID}"
CACHE_ROOT = Path(os.environ.get("TRAJ_CACHE_ROOT", str(Path.home())))
PYTENSOR_CACHE = CACHE_ROOT / '.pytensor_cache' / JOB_ID
PYTENSOR_CACHE.mkdir(parents=True, exist_ok=True)
os.environ['PYTENSOR_FLAGS'] = f"base_compiledir={PYTENSOR_CACHE},optimizer=fast_compile,exception_verbosity=high"
os.environ['FILELOCK_TIMEOUT'] = '30'

MPL_CACHE = CACHE_ROOT / '.matplotlib'
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPL_CACHE))

_DATA_ROOT = os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet")

# Limit threading
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.abspath('src'))

from traj_features.backends.bayes import BayesianTraj, BayesConfig
from traj_features.backends.bayes.classify import pos_flags_from_traj

import pymc as pm
from patsy import dmatrix


def precompile_pytensor_functions(config):
    """
    Pre-compile PyTensor functions with a dummy run to avoid lock contention.
    """
    print("\n[Precompilation] Compiling PyTensor functions...")

    dummy_data = pd.DataFrame({
        'patientid': [1] * 10,
        'time_days': np.linspace(0, 7, 10),
        'lab_value': np.random.randn(10) + 1.0
    })

    try:
        df_window = dummy_data.copy()
        tmin, tmax = df_window['time_days'].min(), df_window['time_days'].max()
        span = max(1e-8, tmax - tmin)

        t_scaled = (df_window['time_days'] - tmin) / span
        df_window['time_scaled'] = t_scaled

        y_raw = df_window['lab_value'].to_numpy()
        y_mu = float(np.mean(y_raw))
        y_sd = float(np.std(y_raw)) if np.std(y_raw) > 0 else 1.0
        y_std = (y_raw - y_mu) / y_sd

        X = dmatrix(f"bs(x, df={config.df_basis}, include_intercept=True)",
                   {"x": t_scaled}, return_type='dataframe').to_numpy()

        with pm.Model() as m:
            beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X.shape[1])
            sigma = pm.HalfNormal("sigma", 1.0)
            mu = pm.math.dot(X, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_std)

            print("  [Precompilation] Compiling logp function...")
            logp_fn = m.compile_logp()
            test_point = m.initial_point()
            _ = logp_fn(test_point)

        print("  [Precompilation] ✓ PyTensor functions compiled successfully")
        return True

    except Exception as e:
        print(f"  [Precompilation] ⚠️  Warning: Could not precompile: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Compute MIMIC AKI trajectory probabilities')
    parser.add_argument('--input', type=str,
                        default=os.path.join(_DATA_ROOT, 'mimic', 'aki', 'creatinine_timeseries.csv'),
                        help='Path to raw creatinine time series CSV')
    parser.add_argument('--pred-dataset', type=str,
                        default=os.path.join(_DATA_ROOT, 'mimic', 'aki', 'aki_prediction_dataset.csv'),
                        help='Path to prediction dataset for merging')
    parser.add_argument('--output', type=str,
                        default=os.path.join(_DATA_ROOT, 'mimic', 'aki', 'aki_trajectory_probs_bayes.csv'),
                        help='Path to save trajectory probabilities (time series + probs)')
    parser.add_argument('--merged-output', type=str,
                        default=None,
                        help='Optional path to save prediction dataset merged with probabilities')
    parser.add_argument('--window-days', type=float, default=7.0,
                        help='Lookback window in days (default: 7.0)')
    parser.add_argument('--flat-thr', type=float, default=0.1,
                        help='Stable threshold in mg/dL (default: 0.1)')
    parser.add_argument('--decline-thr', type=float, default=0.3,
                        help='Increase threshold in mg/dL/day (default: 0.3)')
    parser.add_argument('--nonlinear-gap', type=float, default=0.5,
                        help='Nonlinear gap threshold (default: 0.5)')
    parser.add_argument('--n-batches', type=int, default=25,
                        help='Number of batches for processing (default: 25)')
    parser.add_argument('--cohort-splits', type=int, default=1,
                        help='Total number of cohort splits (default: 1)')
    parser.add_argument('--cohort-index', type=int, default=0,
                        help='Which cohort split to process (0-indexed)')

    args = parser.parse_args()

    if args.cohort_index < 0 or args.cohort_index >= args.cohort_splits:
        print(f"ERROR: cohort-index must be between 0 and {args.cohort_splits - 1}")
        sys.exit(1)

    print("=" * 80)
    print("MIMIC AKI Trajectory Modeling")
    if args.cohort_splits > 1:
        print(f"Sub-cohort: {args.cohort_index + 1}/{args.cohort_splits}")
    print("=" * 80)

    print(f"\n📊 Loading raw creatinine time series from: {args.input}")
    creatinine_ts = pd.read_csv(args.input)

    n_patients = creatinine_ts['hadm_id'].nunique()
    n_rows = len(creatinine_ts)
    print(f"   Patients: {n_patients:,}")
    print(f"   Total measurements: {n_rows:,}")
    if n_patients == 0 or n_rows == 0:
        print("\nERROR: No measurements found in the provided time series.")
        sys.exit(1)
    print(f"   Measurements per patient: {n_rows / n_patients:.1f}")

    traj_input = creatinine_ts[['hadm_id', 'time_days', 'time_day', 'creatinine']].copy()
    traj_input = traj_input.rename(columns={
        'hadm_id': 'patientid',
        'creatinine': 'lab_value'
    })
    traj_input = traj_input.dropna(subset=['lab_value']).sort_values(by=['patientid', 'time_days'])

    all_patients = traj_input['patientid'].unique()
    total_patients = len(all_patients)

    if args.cohort_splits > 1:
        cohort_size = total_patients // args.cohort_splits
        start_idx = args.cohort_index * cohort_size
        end_idx = total_patients if args.cohort_index == args.cohort_splits - 1 else start_idx + cohort_size
        cohort_patients = all_patients[start_idx:end_idx]
        traj_input = traj_input[traj_input['patientid'].isin(cohort_patients)]

        print(f"\n📊 Sub-cohort {args.cohort_index + 1}/{args.cohort_splits}:")
        print(f"   Processing patients {start_idx:,} to {end_idx:,} (of {total_patients:,} total)")
        print(f"   Cohort size: {len(cohort_patients):,} patients")

    print(f"\n1. Input data:")
    print(f"   Rows: {len(traj_input):,}")
    print(f"   Patients: {traj_input['patientid'].nunique():,}")
    print(f"   Timepoints per patient: {traj_input.groupby('patientid')['time_days'].count().mean():.1f}")

    config = BayesConfig(
        window_years=args.window_days,
        df_basis=5,
        n_samples=200,
        tune=300,
        min_points_per_window=6,
        grid_freq=2,
        flat_thr=args.flat_thr,
        decline_thr=args.decline_thr,
        nonlinear_gap=args.nonlinear_gap,
        pids='patientid',
        values='lab_value',
        time_col='time_days',
        windowing_col='time_day',
        use_gpu=False,
        sampler='nutpie',
        target_accept=0.99,
        chains=4,
        n_jobs=-1,
        class_func=pos_flags_from_traj,
        traj_types=('prolonged_nonprogression', 'linear_decline', 'nonlinear'),
    )

    print(f"\n2. BayesianTraj Configuration:")
    print(f"   Window: {args.window_days} days")
    print(f"   Stable threshold: ±{args.flat_thr} mg/dL")
    print(f"   Increase threshold: >{args.decline_thr} mg/dL/day")

    precompile_pytensor_functions(config)

    traj_model = BayesianTraj(cfg=config)

    print(f"\n3. Computing trajectory probabilities...")

    patients = traj_input['patientid'].unique()
    npts = patients.size
    n_batches = args.n_batches
    batch_size = npts // n_batches

    trajectory_probs_list = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
        subset_patients = patients[start_idx:end_idx]

        print(f"\n   Batch {i+1}/{n_batches} ({len(subset_patients)} patients)...")

        batch_input = traj_input[traj_input['patientid'].isin(subset_patients)]
        batch_probs = traj_model.embed(batch_input)
        trajectory_probs_list.append(batch_probs)

        del batch_input
        del batch_probs

        gc.collect()
        if PYTENSOR_CACHE.exists():
            try:
                print("   Cleaning PyTensor cache for next batch...")
                for item in PYTENSOR_CACHE.glob('compiledir*'):
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
            except Exception as e:
                print(f"   Warning: Could not clean cache: {e}")

    trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)
    del trajectory_probs_list
    gc.collect()

    print(f"\n✓ Trajectory probabilities computed!")

    trajectory_probs = trajectory_probs.rename(columns={
        'trajtype_prolonged_nonprogression_prob': 'prob_stable',
        'trajtype_linear_decline_prob': 'prob_gradual_increase',
        'trajtype_nonlinear_prob': 'prob_rapid_increase',
        'patientid': 'hadm_id'
    })

    prob_cols = ['prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']

    probs_ts = creatinine_ts.merge(
        trajectory_probs[['hadm_id', 'time_day'] + prob_cols],
        on=['hadm_id', 'time_day'],
        how='inner'
    )
    del creatinine_ts
    gc.collect()

    output_path = Path(args.output)
    if args.cohort_splits > 1:
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_cohort{args.cohort_index:02d}{suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    probs_ts.to_csv(output_path, index=False)

    print(f"\n✓ Saved trajectory probabilities: {output_path}")

    if args.pred_dataset and Path(args.pred_dataset).exists():
        print(f"\n4. Loading prediction dataset: {args.pred_dataset}")
        prediction_dataset = pd.read_csv(args.pred_dataset)
        print(f"   Samples: {len(prediction_dataset):,}")

        dataset_with_probs = prediction_dataset.merge(
            trajectory_probs[['hadm_id', 'time_day'] + prob_cols],
            on=['hadm_id', 'time_day'],
            how='inner'
        )

        print(f"\n5. Merged trajectory probabilities:")
        print(f"   Rows: {len(dataset_with_probs):,}")

        missing = dataset_with_probs[prob_cols].isna().any(axis=1).sum()
        if missing > 0:
            print(f"   WARNING: Missing probabilities for {missing} rows.")

        out_of_range = ((dataset_with_probs[prob_cols] < 0) | (dataset_with_probs[prob_cols] > 1)).any(axis=1).sum()
        if out_of_range > 0:
            print(f"   ERROR: {out_of_range} rows have probabilities outside [0, 1].")
            sys.exit(2)

        sum_probs = dataset_with_probs[prob_cols].sum(axis=1)
        invalid_sum = (np.abs(sum_probs - 1.0) > 0.05).sum()
        if invalid_sum > 0:
            print(f"   WARNING: {invalid_sum} rows have probabilities not summing to ~1.")

        dataset_with_probs['dominant_traj'] = dataset_with_probs[
            prob_cols
        ].idxmax(axis=1).str.replace('prob_', '')

        print(f"\n6. Trajectory Distribution:")
        for traj in ['stable', 'gradual_increase', 'rapid_increase']:
            subset = dataset_with_probs[dataset_with_probs['dominant_traj'] == traj]
            if len(subset) > 0:
                print(f"   {traj.replace('_', ' ').title():20s}: {len(subset):5,} ({100*len(subset)/len(dataset_with_probs):5.1f}%)")

        merged_path = Path(args.merged_output) if args.merged_output else output_path.with_name('aki_prediction_dataset_with_probs.csv')
        if args.cohort_splits > 1:
            stem = merged_path.stem
            suffix = merged_path.suffix
            merged_path = merged_path.parent / f"{stem}_cohort{args.cohort_index:02d}{suffix}"
        dataset_with_probs.to_csv(merged_path, index=False)
        print(f"\n✓ Saved merged prediction dataset: {merged_path}")
    else:
        print("\nℹ️  Prediction dataset not found or not provided. Skipping merged output.")
    if args.cohort_splits > 1:
        print("\n💡 To merge all cohorts, run:")
        print(
            f"   python -c \"import pandas as pd; pd.concat([pd.read_csv('/home/gaga/data/physionet/mimic/aki/aki_trajectory_probs_cohort{{i:02d}}.csv') for i in range({args.cohort_splits})]).to_csv('/home/gaga/data/physionet/mimic/aki/aki_trajectory_probs.csv', index=False)\""
        )
    print("=" * 80)


if __name__ == '__main__':
    main()