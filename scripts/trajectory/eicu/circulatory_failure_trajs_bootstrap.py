"""
Compute BOOTSTRAP trajectory probabilities for circulatory failure biomarkers (lactate, heartrate, systolic).

This uses the faster bootstrap method instead of Bayesian inference.

Usage:
    python circulatory_failure_trajs_bootstrap.py --data-dir /home/gaga/data/physionet/eicu/circulatory_failure
"""

import pandas as pd
import numpy as np
import argparse
import os
import gc
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Limit threading for cluster jobs
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

sys.path.insert(0, os.path.abspath('src'))

from traj_features.backends.bootstrap import BootstrapTraj, BootstrapConfig
from traj_features.backends.bayes.classify import pos_flags_from_traj, flags_from_traj



BIOMARKERS = {
    'lactate': {
        'file': 'lactate_timeseries.csv',
        'value_col': 'lactate',
        'flat_thr': 0.04,
        'decline_thr': 0.15,
        'nonlinear_gap': 0.2,
        'class_func': pos_flags_from_traj,
        'traj_types': ('stable', 'gradual_increase', 'rapid_increase'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'},
    },
    'heartrate': {
        'file': 'heartrate_timeseries.csv',
        'value_col': 'heartrate',
        'flat_thr': 1.5,
        'decline_thr': 6.0,
        'nonlinear_gap': 10.0,
        'class_func': pos_flags_from_traj,
        'traj_types': ('stable', 'gradual_increase', 'rapid_increase'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'},
    },
    'systolic': {
        'file': 'systolic_timeseries.csv',
        'value_col': 'systolic',
        'flat_thr': -1.5,
        'decline_thr': -6.0,
        'nonlinear_gap': 8.0,
        'class_func': flags_from_traj,
        'traj_types': ('stable', 'gradual_decline', 'rapid_decline'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_decline', 'nonlinear': 'rapid_decline'},
    },
}


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _save_table(df: pd.DataFrame, path: Path, parquet_compression: str = 'zstd') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.parquet':
        df.to_parquet(path, index=False, compression=parquet_compression)
    else:
        df.to_csv(path, index=False)


def _resolve_input_path(data_dir: Path, csv_name: str) -> Path:
    parquet_path = data_dir / csv_name.replace('.csv', '.parquet')
    csv_path = data_dir / csv_name
    if parquet_path.exists():
        return parquet_path
    return csv_path


def compute_biomarker_trajectories(biomarker_name, config_dict, data_dir, window_hours, n_bootstrap, n_batches, id_col, cohort_patients=None):
    print("\n" + "=" * 80)
    print(f"Computing Bootstrap Trajectories: {biomarker_name.upper()}")
    print("=" * 80)

    ts_path = _resolve_input_path(data_dir, config_dict['file'])
    print(f"\n📊 Loading {biomarker_name} time series from: {ts_path}")

    if not ts_path.exists():
        print(f"   ⚠️  WARNING: File not found, skipping {biomarker_name}")
        return None

    ts_df = _load_table(ts_path)
    
    # Auto-detect ID column if not found
    actual_id_col = id_col
    if id_col not in ts_df.columns:
        for candidate in ['stay_id', 'hadm_id', 'patientid']:
            if candidate in ts_df.columns:
                actual_id_col = candidate
                break
        else:
            print(f"   ⚠️  WARNING: No ID column found, skipping {biomarker_name}")
            return None
    
    if cohort_patients is not None:
        ts_df = ts_df[ts_df[actual_id_col].isin(cohort_patients)].copy()
        print(f"   Filtered to cohort: {ts_df[actual_id_col].nunique():,} patients")

    n_patients = ts_df[actual_id_col].nunique()
    n_rows = len(ts_df)

    if n_patients == 0 or n_rows == 0:
        print(f"   ⚠️  WARNING: No data found, skipping {biomarker_name}")
        return None

    print(f"   Patients: {n_patients:,}")
    print(f"   Measurements: {n_rows:,}")
    print(f"   Measurements/patient: {n_rows / n_patients:.1f}")

    value_col = config_dict['value_col']
    if value_col not in ts_df.columns:
        print(f"   ⚠️  WARNING: Column '{value_col}' not found, skipping {biomarker_name}")
        return None

    # Detect time columns
    if 'time_hours' in ts_df.columns and 'time_hour' in ts_df.columns:
        time_col, windowing_col = 'time_hours', 'time_hour'
    elif 'time_days' in ts_df.columns and 'time_day' in ts_df.columns:
        time_col, windowing_col = 'time_days', 'time_day'
    else:
        print(f"   ⚠️  WARNING: No time columns found, skipping {biomarker_name}")
        return None

    ts_df = ts_df.drop_duplicates(subset=[id_col, time_col])
    traj_input = ts_df[[actual_id_col, time_col, windowing_col, value_col]].copy()
    traj_input = traj_input.rename(columns={
        actual_id_col: 'patientid',
        value_col: 'lab_value'
    })
    traj_input['lab_value'] = pd.to_numeric(traj_input['lab_value'], errors='coerce').astype(np.float32)
    traj_input = traj_input.dropna(subset=['lab_value']).sort_values(by=['patientid', time_col])
    del ts_df
    gc.collect()

    config = BootstrapConfig(
        window_years=window_hours,
        n_bootstrap=n_bootstrap,
        smoothing=None,
        min_points_per_window=4,
        grid_freq=2,
        flat_thr=config_dict['flat_thr'],
        decline_thr=config_dict['decline_thr'],
        nonlinear_gap=config_dict['nonlinear_gap'],
        pids='patientid',
        values='lab_value',
        time_col=time_col,
        windowing_col=windowing_col,
        n_jobs=-1,
        traj_types=config_dict['traj_types'],
        class_func=config_dict['class_func'],
        label_map=config_dict['label_map'],
        progressbar=True,
    )

    print(f"\nConfiguration:")
    print(f"   Window: {window_hours} hours")
    print(f"   Bootstrap samples: {n_bootstrap}")
    print(f"   Stable threshold: ±{config_dict['flat_thr']}")
    print(f"   Change threshold: {config_dict['decline_thr']}")
    print(f"   Nonlinear gap: {config_dict['nonlinear_gap']}")

    traj_model = BootstrapTraj(cfg=config)

    patients = traj_input['patientid'].unique()
    npts = patients.size
    batch_size = max(1, npts // n_batches)

    trajectory_probs_list = []

    print(f"\nComputing probabilities ({n_batches} batches)...")
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
        subset_patients = patients[start_idx:end_idx]

        if len(subset_patients) == 0:
            continue

        print(f"   Batch {i + 1}/{n_batches} ({len(subset_patients)} patients)...", end=' ', flush=True)

        try:
            batch_input = traj_input[traj_input['patientid'].isin(subset_patients)]
            batch_probs = traj_model.embed(batch_input)
            trajectory_probs_list.append(batch_probs)
            del batch_input
            print("✓")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            continue

        gc.collect()

    if not trajectory_probs_list:
        print(f"\n   ⚠️  ERROR: All batches failed for {biomarker_name}")
        return None

    trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)

    traj_type_names = config_dict['traj_types']
    prob_map = {
        f'trajtype_{traj_type_names[0]}_prob': f'{biomarker_name}_stable',
        f'trajtype_{traj_type_names[1]}_prob': f'{biomarker_name}_gradual',
        f'trajtype_{traj_type_names[2]}_prob': f'{biomarker_name}_rapid',
    }

    trajectory_probs = trajectory_probs.rename(columns=prob_map)
    trajectory_probs = trajectory_probs.rename(columns={'patientid': actual_id_col})

    # Ensure windowing column is present for merging
    if windowing_col not in trajectory_probs.columns:
        # Avoid explosive many-to-many merge by id only; derive from time if possible
        if time_col in trajectory_probs.columns:
            trajectory_probs[windowing_col] = np.floor(pd.to_numeric(trajectory_probs[time_col], errors='coerce')).astype('Int64')
        else:
            print(f"   ⚠️  WARNING: Missing '{windowing_col}' and '{time_col}' in model output; skipping {biomarker_name} to avoid invalid merge")
            return None

    result_cols = [actual_id_col, windowing_col, f'{biomarker_name}_stable', f'{biomarker_name}_gradual', f'{biomarker_name}_rapid']
    result = trajectory_probs[[c for c in result_cols if c in trajectory_probs.columns]].copy()
    for c in [f'{biomarker_name}_stable', f'{biomarker_name}_gradual', f'{biomarker_name}_rapid']:
        if c in result.columns:
            result[c] = pd.to_numeric(result[c], errors='coerce').astype(np.float32)

    prob_cols = [f'{biomarker_name}_stable', f'{biomarker_name}_gradual', f'{biomarker_name}_rapid']
    key_cols = [actual_id_col, windowing_col]
    dup_n = result.duplicated(subset=key_cols).sum()
    if dup_n > 0:
        print(f"   WARNING: {dup_n:,} duplicate key rows found for {biomarker_name}; aggregating by mean on probabilities")
        result = result.groupby(key_cols, as_index=False)[prob_cols].mean()

    missing = result[prob_cols].isna().any(axis=1).sum()
    if missing > 0:
        print(f"   WARNING: Missing probabilities for {missing} rows in {biomarker_name} time series.")

    del trajectory_probs, trajectory_probs_list, traj_input, patients
    gc.collect()

    print(f"   ✓ Computed {len(result):,} trajectory windows for {biomarker_name}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Compute bootstrap circulatory failure trajectory probabilities')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing biomarker time series CSVs')
    parser.add_argument('--pred-dataset', type=str, default=None,
                        help='Path to prediction dataset for merging')
    parser.add_argument('--merged-output', type=str, default=None,
                        help='Optional path to save prediction dataset merged with probabilities')
    parser.add_argument('--window-hours', type=float, default=12.0,
                        help='Lookback window in hours (default: 12.0)')
    parser.add_argument('--n-bootstrap', type=int, default=200,
                        help='Number of bootstrap samples (default: 200)')
    parser.add_argument('--n-batches', type=int, default=8,
                        help='Number of batches for processing (default: 8)')
    parser.add_argument('--cohort-splits', type=int, default=1,
                        help='Total number of cohort splits (default: 1)')
    parser.add_argument('--cohort-index', type=int, default=0,
                        help='Which cohort split to process (0-indexed)')
    parser.add_argument('--id-col', type=str, default='stay_id',
                        help='ID column name (default: stay_id)')
    parser.add_argument('--biomarkers', type=str, nargs='+', default=None,
                        help='Specific biomarkers to process (default: all)')
    parser.add_argument('--output-format', type=str, default='parquet', choices=['csv', 'parquet', 'both'],
                        help='Output format for trajectory probabilities (default: parquet)')
    parser.add_argument('--parquet-compression', type=str, default='zstd',
                        help='Parquet compression codec (default: zstd)')

    args = parser.parse_args()

    if args.cohort_index < 0 or args.cohort_index >= args.cohort_splits:
        print(f"ERROR: cohort-index must be between 0 and {args.cohort_splits - 1}")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    biomarker_results = {}

    # Determine which biomarkers to process
    biomarkers_to_process = BIOMARKERS
    if args.biomarkers:
        biomarkers_to_process = {k: v for k, v in BIOMARKERS.items() if k in args.biomarkers}

    # Handle cohort splitting
    cohort_patients = None
    if args.cohort_splits > 1:
        # Get all patients from prediction dataset or first time series
        if args.pred_dataset and Path(args.pred_dataset).exists():
            prediction_dataset = pd.read_csv(args.pred_dataset)
            id_col = args.id_col
            if id_col not in prediction_dataset.columns:
                for candidate in ['stay_id', 'hadm_id', 'patientid']:
                    if candidate in prediction_dataset.columns:
                        id_col = candidate
                        break
            all_patients = prediction_dataset[id_col].unique()
        else:
            first_biomarker = next(iter(biomarkers_to_process.values()))
            ts_path = data_dir / first_biomarker['file']
            if ts_path.exists():
                ts_df = pd.read_csv(ts_path)
                id_col = args.id_col
                if id_col not in ts_df.columns:
                    for candidate in ['stay_id', 'hadm_id', 'patientid']:
                        if candidate in ts_df.columns:
                            id_col = candidate
                            break
                all_patients = ts_df[id_col].unique()
            else:
                all_patients = None

        if all_patients is not None:
            total_patients = len(all_patients)
            cohort_size = total_patients // args.cohort_splits
            start_idx = args.cohort_index * cohort_size
            end_idx = total_patients if args.cohort_index == args.cohort_splits - 1 else start_idx + cohort_size
            cohort_patients = all_patients[start_idx:end_idx]

            print(f"\n📊 Sub-cohort {args.cohort_index + 1}/{args.cohort_splits}:")
            print(f"   Processing patients {start_idx:,} to {end_idx:,} (of {total_patients:,} total)")
            print(f"   Cohort size: {len(cohort_patients):,} patients")

    # Process each biomarker
    for biomarker, cfg in biomarkers_to_process.items():
        traj_df = compute_biomarker_trajectories(
            biomarker_name=biomarker,
            config_dict=cfg,
            data_dir=data_dir,
            window_hours=args.window_hours,
            n_bootstrap=args.n_bootstrap,
            n_batches=args.n_batches,
            id_col=args.id_col,
            cohort_patients=cohort_patients,
        )
        if traj_df is not None:
            # Save individual biomarker results
            base_name = f"{biomarker}_trajectory_probs_bootstrap"
            if args.cohort_splits > 1:
                base_name = f"{base_name}_cohort{args.cohort_index:02d}"

            output_paths = []
            if args.output_format in ('parquet', 'both'):
                output_paths.append(data_dir / f"{base_name}.parquet")
            if args.output_format in ('csv', 'both'):
                output_paths.append(data_dir / f"{base_name}.csv")

            for output_path in output_paths:
                _save_table(traj_df, output_path, parquet_compression=args.parquet_compression)
                print(f"   ✓ Saved: {output_path}")

            biomarker_results[biomarker] = output_paths[0]
            del traj_df
            gc.collect()

    if not biomarker_results:
        print("\n❌ ERROR: No biomarker trajectories were computed successfully")
        sys.exit(1)

    # Merge with prediction dataset if provided
    if args.pred_dataset and Path(args.pred_dataset).exists():
        print(f"\nLoading prediction dataset: {args.pred_dataset}")
        prediction_dataset = _load_table(Path(args.pred_dataset))
        
        # Auto-detect ID and windowing columns
        id_col = args.id_col
        if id_col not in prediction_dataset.columns:
            for candidate in ['stay_id', 'hadm_id', 'patientid']:
                if candidate in prediction_dataset.columns:
                    id_col = candidate
                    break
        
        windowing_col = 'time_hour' if 'time_hour' in prediction_dataset.columns else 'time_day'
        
        if cohort_patients is not None:
            prediction_dataset = prediction_dataset[prediction_dataset[id_col].isin(cohort_patients)]
        print(f"   Samples: {len(prediction_dataset):,}")

        key_cols = [id_col, windowing_col]
        pred_dup_n = prediction_dataset.duplicated(subset=key_cols).sum()
        if pred_dup_n > 0:
            print(f"   WARNING: prediction dataset has {pred_dup_n:,} duplicate key rows; keeping first per key")
            prediction_dataset = prediction_dataset.drop_duplicates(subset=key_cols, keep='first')

        merged = prediction_dataset.copy()
        del prediction_dataset
        gc.collect()

        for biomarker, traj_path in biomarker_results.items():
            print(f"\nMerging {biomarker} trajectories...")
            traj_df = _load_table(traj_path)
            prob_cols = [f"{biomarker}_stable", f"{biomarker}_gradual", f"{biomarker}_rapid"]
            traj_dup_n = traj_df.duplicated(subset=key_cols).sum()
            if traj_dup_n > 0:
                print(f"   WARNING: {traj_dup_n:,} duplicate key rows in {biomarker} file; aggregating before merge")
                traj_df = traj_df.groupby(key_cols, as_index=False)[prob_cols].mean()

            merged = merged.merge(traj_df, on=key_cols, how='left', validate='one_to_one')
            del traj_df
            gc.collect()

            available_cols = [c for c in prob_cols if c in merged.columns]
            if available_cols:
                missing = merged[available_cols].isna().any(axis=1).sum()
                print(f"   Rows with {biomarker} probs: {len(merged) - missing:,} / {len(merged):,}")

        base_merged_path = Path(args.merged_output) if args.merged_output else data_dir / "circulatory_failure_prediction_dataset_with_bootstrap_probs"
        if base_merged_path.suffix in ('.csv', '.parquet'):
            base_merged_path = base_merged_path.with_suffix('')
        if args.cohort_splits > 1:
            base_merged_path = base_merged_path.parent / f"{base_merged_path.name}_cohort{args.cohort_index:02d}"

        merged_paths = []
        if args.output_format in ('parquet', 'both'):
            merged_paths.append(base_merged_path.with_suffix('.parquet'))
        if args.output_format in ('csv', 'both'):
            merged_paths.append(base_merged_path.with_suffix('.csv'))

        for merged_path in merged_paths:
            _save_table(merged, merged_path, parquet_compression=args.parquet_compression)
            print(f"\n✓ Saved merged prediction dataset: {merged_path}")

    print("\n" + "=" * 80)
    print("Bootstrap trajectory computation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
