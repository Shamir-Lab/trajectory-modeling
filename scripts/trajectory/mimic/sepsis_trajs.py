"""
Compute biomarker trajectory probabilities for the MIMIC sepsis cohort.

Usage (examples):
    python sepsis_trajs.py --biomarker lactate  --input results/mimic/sepsis/lactate_timeseries.csv  --pred-dataset results/mimic/sepsis/sepsis_prediction_dataset.csv --output results/mimic/sepsis/lactate_trajectory_probs_bayes.csv
    python sepsis_trajs.py --biomarker wbc      --input results/mimic/sepsis/wbc_timeseries.csv      --pred-dataset results/mimic/sepsis/sepsis_prediction_dataset.csv --output results/mimic/sepsis/wbc_trajectory_probs_bayes.csv
    python sepsis_trajs.py --biomarker platelet --input results/mimic/sepsis/platelet_timeseries.csv --pred-dataset results/mimic/sepsis/sepsis_prediction_dataset.csv --output results/mimic/sepsis/platelet_trajectory_probs_bayes.csv
    python sepsis_trajs.py --biomarker all --data-dir results/mimic/sepsis --pred-dataset results/mimic/sepsis/sepsis_prediction_dataset.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import gc
from pathlib import Path
import sys

# Ensure compiled artifacts and matplotlib cache land in a writable location
ARRAY_JOB_ID = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
ARRAY_TASK_ID = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
JOB_ID = f"{ARRAY_JOB_ID}_{ARRAY_TASK_ID}"
CACHE_ROOT = Path(os.environ.get("TRAJ_CACHE_ROOT", str(Path.home())))
PYTENSOR_CACHE = CACHE_ROOT / '.pytensor_cache' / JOB_ID
PYTENSOR_CACHE.mkdir(parents=True, exist_ok=True)
os.environ['PYTENSOR_FLAGS'] = f"compiledir={PYTENSOR_CACHE},base_compiledir={PYTENSOR_CACHE},optimizer=fast_compile,exception_verbosity=high"

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
from traj_features.backends.bayes.classify import pos_flags_from_traj, flags_from_traj

import pymc as pm
from patsy import dmatrix


def precompile_pytensor_functions(config):
    """
    Pre-compile PyTensor functions with a dummy run to avoid lock contention.
    """
    print("\n[Precompilation] Compiling PyTensor functions...")

    dummy_data = pd.DataFrame({
        'patientid': [1] * 10,
        'time_days': np.linspace(0, 3, 10),
        'lab_value': np.random.randn(10) + 2.0
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


BIOMARKER_CONFIG = {
    'lactate': {
        'value_col': 'lactate',
        'flat_thr': 0.1,
        'decline_thr': 0.3,
        'nonlinear_gap': 0.5,
        'class_func': pos_flags_from_traj,
        'traj_types': ('stable', 'gradual_increase', 'rapid_increase'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'},
    },
    'wbc': {
        'value_col': 'wbc',
        'flat_thr': 1.0,
        'decline_thr': 3.0,
        'nonlinear_gap': 2.0,
        'class_func': pos_flags_from_traj,
        'traj_types': ('stable', 'gradual_increase', 'rapid_increase'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'},
    },
    'platelet': {
        'value_col': 'platelet',
        'flat_thr': -20.0,
        'decline_thr': -50.0,
        'nonlinear_gap': 30.0,
        'class_func': flags_from_traj,
        'traj_types': ('stable', 'gradual_decline', 'rapid_decline'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_decline', 'nonlinear': 'rapid_decline'},
    },
}


def compute_biomarker(
    biomarker: str,
    input_path: Path,
    output_path: Path,
    window_days: float,
    n_batches: int,
    cohort_patients: np.ndarray | None = None,
    value_col_override: str | None = None,
    flat_thr_override: float | None = None,
    decline_thr_override: float | None = None,
    nonlinear_gap_override: float | None = None,
):
    cfg_defaults = BIOMARKER_CONFIG[biomarker]
    value_col = value_col_override or cfg_defaults['value_col']
    flat_thr = cfg_defaults['flat_thr'] if flat_thr_override is None else flat_thr_override
    decline_thr = cfg_defaults['decline_thr'] if decline_thr_override is None else decline_thr_override
    nonlinear_gap = cfg_defaults['nonlinear_gap'] if nonlinear_gap_override is None else nonlinear_gap_override

    print("=" * 80)
    print(f"MIMIC Sepsis Trajectory Modeling - {biomarker.upper()}")
    print("=" * 80)

    print(f"\n📊 Loading raw {biomarker} time series from: {input_path}")
    ts_df = pd.read_csv(input_path)
    n_patients = ts_df['hadm_id'].nunique()
    n_rows = len(ts_df)
    print(f"   Patients: {n_patients:,}")
    print(f"   Total measurements: {n_rows:,}")
    if n_patients == 0 or n_rows == 0:
        print("\nERROR: No measurements found in the provided time series.")
        return None, None, None
    print(f"   Measurements per patient: {n_rows / n_patients:.1f}")

    if value_col not in ts_df.columns:
        print(f"\nERROR: Expected value column '{value_col}' not found in {input_path}. Columns: {list(ts_df.columns)}")
        return None, None, None

    traj_input = ts_df[['hadm_id', 'time_days', 'time_day', value_col]].copy()
    traj_input = traj_input.rename(columns={
        'hadm_id': 'patientid',
        value_col: 'lab_value'
    })

    traj_input = traj_input.dropna(subset=['lab_value']).sort_values(by=['patientid', 'time_days'])

    if cohort_patients is not None:
        traj_input = traj_input[traj_input['patientid'].isin(cohort_patients)]

    print(f"\n1. Input data:")
    print(f"   Rows: {len(traj_input):,}")
    print(f"   Patients: {traj_input['patientid'].nunique():,}")
    print(f"   Timepoints per patient: {traj_input.groupby('patientid')['time_days'].count().mean():.1f}")

    config = BayesConfig(
        window_years=window_days,
        df_basis=5,
        n_samples=200,
        tune=300,
        min_points_per_window=4,
        grid_freq=2,
        flat_thr=flat_thr,
        decline_thr=decline_thr,
        nonlinear_gap=nonlinear_gap,
        pids='patientid',
        values='lab_value',
        time_col='time_days',
        windowing_col='time_day',
        use_gpu=False,
        sampler='nutpie',
        target_accept=0.99,
        chains=4,
        n_jobs=-1,
        class_func=cfg_defaults['class_func'],
        traj_types=cfg_defaults['traj_types'],
        label_map=cfg_defaults['label_map'],
    )

    print(f"\n2. BayesianTraj Configuration:")
    print(f"   Biomarker: {biomarker}")
    print(f"   Window: {window_days} days")
    print(f"   Stable threshold: ±{flat_thr}")
    print(f"   Change threshold: {decline_thr}")
    print(f"   Nonlinear gap: {nonlinear_gap}")

    precompile_pytensor_functions(config)

    traj_model = BayesianTraj(cfg=config)

    print(f"\n3. Computing trajectory probabilities...")

    patients = traj_input['patientid'].unique()
    npts = patients.size
    batch_size = npts // n_batches

    trajectory_probs_list = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
        subset_patients = patients[start_idx:end_idx]

        print(f"\n   Batch {i+1}/{n_batches} ({len(subset_patients)} patients)...")

        batch_probs = traj_model.embed(traj_input[traj_input['patientid'].isin(subset_patients)])
        trajectory_probs_list.append(batch_probs)

        gc.collect()

    trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)

    print(f"\n✓ Trajectory probabilities computed!")

    prob_map = {f'trajtype_{t}_prob': f'prob_{t}' for t in cfg_defaults['traj_types']}
    trajectory_probs = trajectory_probs.rename(columns=prob_map)
    trajectory_probs = trajectory_probs.rename(columns={'patientid': 'hadm_id'})
    prob_cols = [prob_map[k] for k in prob_map]

    probs_ts = ts_df.merge(
        trajectory_probs[['hadm_id', 'time_day'] + prob_cols],
        on=['hadm_id', 'time_day'],
        how='inner'
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    probs_ts.to_csv(output_path, index=False)

    print(f"\n✓ Saved trajectory probabilities: {output_path}")

    return trajectory_probs, prob_cols, cfg_defaults['traj_types']


def main():
    parser = argparse.ArgumentParser(description='Compute MIMIC sepsis trajectory probabilities')
    parser.add_argument('--biomarker', type=str, choices=list(BIOMARKER_CONFIG.keys()) + ['all'], default='lactate',
                        help='Biomarker to model (lactate, wbc, platelet) or all')
    parser.add_argument('--input', type=str,
                        default=os.path.join(_DATA_ROOT, 'mimic', 'sepsis', 'lactate_timeseries.csv'),
                        help='Path to raw biomarker time series CSV (single-biomarker mode)')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(_DATA_ROOT, 'mimic', 'sepsis'),
                        help='Directory containing biomarker time series CSVs (all-biomarker mode)')
    parser.add_argument('--pred-dataset', type=str,
                        default=os.path.join(_DATA_ROOT, 'mimic', 'sepsis', 'sepsis_prediction_dataset.csv'),
                        help='Path to prediction dataset for merging')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save trajectory probabilities (single-biomarker mode)')
    parser.add_argument('--merged-output', type=str, default=None,
                        help='Optional path to save prediction dataset merged with probabilities')
    parser.add_argument('--window-days', type=float, default=3.0,
                        help='Lookback window in days (default: 3.0)')
    parser.add_argument('--flat-thr', type=float, default=None,
                        help='Override stable threshold (single-biomarker mode)')
    parser.add_argument('--decline-thr', type=float, default=None,
                        help='Override change threshold (single-biomarker mode)')
    parser.add_argument('--nonlinear-gap', type=float, default=None,
                        help='Override nonlinear gap (single-biomarker mode)')
    parser.add_argument('--value-col', type=str, default=None,
                        help='Override value column name (single-biomarker mode)')
    parser.add_argument('--n-batches', type=int, default=8,
                        help='Number of batches for processing (default: 8)')
    parser.add_argument('--cohort-splits', type=int, default=1,
                        help='Total number of cohort splits (default: 1)')
    parser.add_argument('--cohort-index', type=int, default=0,
                        help='Which cohort split to process (0-indexed)')

    args = parser.parse_args()

    if args.cohort_index < 0 or args.cohort_index >= args.cohort_splits:
        print(f"ERROR: cohort-index must be between 0 and {args.cohort_splits - 1}")
        sys.exit(1)

    if args.biomarker == 'all':
        data_dir = Path(args.data_dir)
        biomarker_results = {}

        cohort_patients = None
        if args.cohort_splits > 1 and args.pred_dataset and Path(args.pred_dataset).exists():
            prediction_dataset = pd.read_csv(args.pred_dataset)
            all_patients = prediction_dataset['hadm_id'].unique()
            total_patients = len(all_patients)
            cohort_size = total_patients // args.cohort_splits
            start_idx = args.cohort_index * cohort_size
            end_idx = total_patients if args.cohort_index == args.cohort_splits - 1 else start_idx + cohort_size
            cohort_patients = all_patients[start_idx:end_idx]

            print(f"\n📊 Sub-cohort {args.cohort_index + 1}/{args.cohort_splits}:")
            print(f"   Processing patients {start_idx:,} to {end_idx:,} (of {total_patients:,} total)")
            print(f"   Cohort size: {len(cohort_patients):,} patients")

        for biomarker in BIOMARKER_CONFIG.keys():
            input_path = data_dir / f"{biomarker}_timeseries.csv"
            output_path = data_dir / f"{biomarker}_trajectory_probs_bayes.csv"
            if args.cohort_splits > 1:
                stem = output_path.stem
                suffix = output_path.suffix
                output_path = output_path.parent / f"{stem}_cohort{args.cohort_index:02d}{suffix}"

            traj_probs, prob_cols, traj_types = compute_biomarker(
                biomarker=biomarker,
                input_path=input_path,
                output_path=output_path,
                window_days=args.window_days,
                n_batches=args.n_batches,
                cohort_patients=cohort_patients,
            )

            if traj_probs is None:
                continue

            prefixed = traj_probs[['hadm_id', 'time_day'] + prob_cols].copy()
            rename_map = {
                prob_cols[0]: f"{biomarker}_stable",
                prob_cols[1]: f"{biomarker}_gradual",
                prob_cols[2]: f"{biomarker}_rapid",
            }
            prefixed = prefixed.rename(columns=rename_map)

            biomarker_results[biomarker] = prefixed

        if not biomarker_results:
            print("\n❌ ERROR: No biomarker trajectories were computed successfully")
            sys.exit(1)

        if args.pred_dataset and Path(args.pred_dataset).exists():
            print(f"\n4. Loading prediction dataset: {args.pred_dataset}")
            prediction_dataset = pd.read_csv(args.pred_dataset)
            if cohort_patients is not None:
                prediction_dataset = prediction_dataset[prediction_dataset['hadm_id'].isin(cohort_patients)]
            print(f"   Samples: {len(prediction_dataset):,}")

            merged = prediction_dataset.copy()
            for biomarker, traj_df in biomarker_results.items():
                print(f"\nMerging {biomarker} trajectories...")
                merged = merged.merge(traj_df, on=['hadm_id', 'time_day'], how='left')

                prob_cols = [f"{biomarker}_stable", f"{biomarker}_gradual", f"{biomarker}_rapid"]
                missing = merged[prob_cols].isna().any(axis=1).sum()
                print(f"   Rows with {biomarker} probs: {len(merged) - missing:,} / {len(merged):,}")

            merged_path = Path(args.merged_output) if args.merged_output else data_dir / "sepsis_prediction_dataset_with_probs.csv"
            if args.cohort_splits > 1:
                stem = merged_path.stem
                suffix = merged_path.suffix
                merged_path = merged_path.parent / f"{stem}_cohort{args.cohort_index:02d}{suffix}"
            merged.to_csv(merged_path, index=False)
            print(f"\n✓ Saved merged prediction dataset: {merged_path}")
        else:
            print("\nℹ️  Prediction dataset not found or not provided. Skipping merged output.")
        print("=" * 80)
        return

    bm = args.biomarker
    output_path = Path(args.output) if args.output else Path(_DATA_ROOT) / "mimic" / "sepsis" / f"{bm}_trajectory_probs_bayes.csv"
    input_path = Path(args.input)

    cohort_patients = None
    if args.cohort_splits > 1:
        ts_df = pd.read_csv(input_path)
        all_patients = ts_df['hadm_id'].unique()
        total_patients = len(all_patients)
        cohort_size = total_patients // args.cohort_splits
        start_idx = args.cohort_index * cohort_size
        end_idx = total_patients if args.cohort_index == args.cohort_splits - 1 else start_idx + cohort_size
        cohort_patients = all_patients[start_idx:end_idx]

        print(f"\n📊 Sub-cohort {args.cohort_index + 1}/{args.cohort_splits}:")
        print(f"   Processing patients {start_idx:,} to {end_idx:,} (of {total_patients:,} total)")
        print(f"   Cohort size: {len(cohort_patients):,} patients")
    if args.cohort_splits > 1:
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_cohort{args.cohort_index:02d}{suffix}"

    traj_probs, prob_cols, traj_types = compute_biomarker(
        biomarker=bm,
        input_path=input_path,
        output_path=output_path,
        window_days=args.window_days,
        n_batches=args.n_batches,
        cohort_patients=cohort_patients,
        value_col_override=args.value_col,
        flat_thr_override=args.flat_thr,
        decline_thr_override=args.decline_thr,
        nonlinear_gap_override=args.nonlinear_gap,
    )

    if traj_probs is None:
        sys.exit(1)

    if args.pred_dataset and Path(args.pred_dataset).exists():
        print(f"\n4. Loading prediction dataset: {args.pred_dataset}")
        prediction_dataset = pd.read_csv(args.pred_dataset)
        if cohort_patients is not None:
            prediction_dataset = prediction_dataset[prediction_dataset['hadm_id'].isin(cohort_patients)]
        print(f"   Samples: {len(prediction_dataset):,}")

        merge_cols = ['hadm_id', 'time_day'] + prob_cols
        dataset_with_probs = prediction_dataset.merge(
            traj_probs[merge_cols],
            on=['hadm_id', 'time_day'],
            how='left'
        )

        print(f"\n5. Merged trajectory probabilities:")
        print(f"   Rows: {len(dataset_with_probs):,}")

        missing_mask = dataset_with_probs[prob_cols].isna().any(axis=1)
        n_missing = missing_mask.sum()
        if n_missing > 0:
            pct_missing = 100 * n_missing / len(dataset_with_probs)
            print(f"   ⚠️  Missing probabilities: {n_missing:,} / {len(dataset_with_probs):,} ({pct_missing:.1f}%)")
        else:
            print("   ✓ All rows have complete probability assignments")

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
        for traj in traj_types:
            subset = dataset_with_probs[dataset_with_probs['dominant_traj'] == traj]
            if len(subset) > 0:
                print(f"   {traj.replace('_', ' ').title():20s}: {len(subset):5,} ({100*len(subset)/len(dataset_with_probs):5.1f}%)")

        merged_path = Path(args.merged_output) if args.merged_output else output_path.with_name(f"{bm}_prediction_dataset_with_probs.csv")
        if args.cohort_splits > 1:
            stem = merged_path.stem
            suffix = merged_path.suffix
            merged_path = merged_path.parent / f"{stem}_cohort{args.cohort_index:02d}{suffix}"
        dataset_with_probs.to_csv(merged_path, index=False)
        print(f"\n✓ Saved merged prediction dataset: {merged_path}")
    else:
        print("\nℹ️  Prediction dataset not found or not provided. Skipping merged output.")
    print("=" * 80)


if __name__ == '__main__':
    main()
