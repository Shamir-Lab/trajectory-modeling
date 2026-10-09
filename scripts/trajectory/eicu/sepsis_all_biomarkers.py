"""
Compute trajectory probabilities for ALL sepsis biomarkers (lactate, WBC, platelets) in one unified run.

This replaces the separate biomarker scripts and automatically merges all trajectory features.

Usage:
    python sepsis_all_biomarkers.py
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

# Biomarker-specific configuration
BIOMARKERS = {
    'lactate': {
        'file': 'lactate_timeseries.csv',
        'value_col': 'lactate',
        'flat_thr': 0.1,
        'decline_thr': 0.3,
        'nonlinear_gap': 0.5,
        'class_func': pos_flags_from_traj,
        'traj_types': ('stable', 'gradual_increase', 'rapid_increase'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'},
    },
    'wbc': {
        'file': 'wbc_timeseries.csv',
        'value_col': 'wbc',
        'flat_thr': 1.0,
        'decline_thr': 3.0,
        'nonlinear_gap': 2.0,
        'class_func': pos_flags_from_traj,
        'traj_types': ('stable', 'gradual_increase', 'rapid_increase'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'},
    },
    'platelet': {
        'file': 'platelets_timeseries.csv',
        'value_col': 'platelet',
        'flat_thr': -20.0,
        'decline_thr': -50.0,
        'nonlinear_gap': 30.0,
        'class_func': flags_from_traj,
        'traj_types': ('stable', 'gradual_decline', 'rapid_decline'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_decline', 'nonlinear': 'rapid_decline'},
    },
}

def compute_biomarker_trajectories(biomarker_name, config_dict, data_dir, window_days, n_batches, probs_output_path, cohort_patients=None):
    """
    Compute trajectory probabilities for a single biomarker.
    
    Returns: DataFrame with columns ['stay_id', 'time_day', '{biomarker}_stable', '{biomarker}_gradual', '{biomarker}_rapid']
    """
    print("\n" + "="*80)
    print(f"Computing Trajectories: {biomarker_name.upper()}")
    print("="*80)
    
    # Load time series
    ts_path = data_dir / config_dict['file']
    print(f"\n📊 Loading {biomarker_name} time series from: {ts_path}")
    
    if not ts_path.exists():
        print(f"   ⚠️  WARNING: File not found, skipping {biomarker_name}")
        return None
    
    ts_df = pd.read_csv(ts_path)
    n_patients = ts_df['stay_id'].nunique()
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
    
    # Prepare input
    traj_input = ts_df[['stay_id', 'time_days', 'time_day', value_col]].copy()
    traj_input = traj_input.rename(columns={
        'stay_id': 'patientid',
        value_col: 'lab_value'
    })
    traj_input = traj_input.dropna(subset=['lab_value']).sort_values(by=['patientid', 'time_days'])
    if cohort_patients is not None:
        traj_input = traj_input[traj_input['patientid'].isin(cohort_patients)]
    
    # Configure model
    config = BayesConfig(
        window_years=window_days,
        df_basis=5,
        n_samples=200,
        tune=300,
        min_points_per_window=4,
        grid_freq=2,
        flat_thr=config_dict['flat_thr'],
        decline_thr=config_dict['decline_thr'],
        nonlinear_gap=config_dict['nonlinear_gap'],
        pids='patientid',
        values='lab_value',
        time_col='time_days',
        windowing_col='time_day',
        use_gpu=False,
        sampler='nutpie',
        target_accept=0.99,
        chains=4,
        n_jobs=-1,
        class_func=config_dict['class_func'],
        traj_types=config_dict['traj_types'],
        label_map=config_dict['label_map'],
    )
    
    print(f"\nConfiguration:")
    print(f"   Window: {window_days} days")
    print(f"   Stable threshold: ±{config_dict['flat_thr']}")
    print(f"   Change threshold: {config_dict['decline_thr']}")
    print(f"   Nonlinear gap: {config_dict['nonlinear_gap']}")
    
    # Initialize model
    traj_model = BayesianTraj(cfg=config)
    
    # Process in batches
    patients = traj_input['patientid'].unique()
    npts = patients.size
    batch_size = npts // n_batches
    
    trajectory_probs_list = []
    
    print(f"\nComputing probabilities ({n_batches} batches)...")
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
        subset_patients = patients[start_idx:end_idx]
        
        print(f"   Batch {i+1}/{n_batches} ({len(subset_patients)} patients)...", end=' ', flush=True)
        
        try:
            batch_probs = traj_model.embed(traj_input[traj_input['patientid'].isin(subset_patients)])
            trajectory_probs_list.append(batch_probs)
            print("✓")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            # Continue with next batch instead of failing completely
            continue
        
        gc.collect()
    
    if not trajectory_probs_list:
        print(f"\n   ⚠️  ERROR: All batches failed for {biomarker_name}")
        return None
    
    trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)
    
    # Rename columns with biomarker prefix
    traj_type_names = config_dict['traj_types']
    prob_map = {
        f'trajtype_{traj_type_names[0]}_prob': f'{biomarker_name}_stable',
        f'trajtype_{traj_type_names[1]}_prob': f'{biomarker_name}_gradual',
        f'trajtype_{traj_type_names[2]}_prob': f'{biomarker_name}_rapid',
    }
    
    trajectory_probs = trajectory_probs.rename(columns=prob_map)
    trajectory_probs = trajectory_probs.rename(columns={'patientid': 'stay_id'})
    
    # Select relevant columns
    result_cols = ['stay_id', 'time_day', f'{biomarker_name}_stable', f'{biomarker_name}_gradual', f'{biomarker_name}_rapid']
    result = trajectory_probs[result_cols].copy()

    # Validate probabilities
    prob_cols = [f'{biomarker_name}_stable', f'{biomarker_name}_gradual', f'{biomarker_name}_rapid']
    missing = result[prob_cols].isna().any(axis=1).sum()
    if missing > 0:
        print(f"   WARNING: Missing probabilities for {missing} rows in {biomarker_name} time series.")

    out_of_range = ((result[prob_cols] < 0) | (result[prob_cols] > 1)).any(axis=1).sum()
    if out_of_range > 0:
        print(f"   ERROR: {out_of_range} rows have probabilities outside [0, 1] for {biomarker_name}.")
        sys.exit(2)

    sum_probs = result[prob_cols].sum(axis=1)
    invalid_sum = (np.abs(sum_probs - 1.0) > 0.05).sum()
    if invalid_sum > 0:
        print(f"   WARNING: {invalid_sum} rows have probabilities not summing to ~1 for {biomarker_name}.")

    # Save time series probabilities
    probs_output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(probs_output_path, index=False)
    print(f"   ✓ Saved {biomarker_name} probabilities: {probs_output_path}")
    
    print(f"\n✓ Computed {len(result):,} trajectory timepoints for {biomarker_name}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Compute eICU sepsis trajectory probabilities for all biomarkers')
    parser.add_argument('--data-dir', type=str,
                       default=os.path.join(_DATA_ROOT, 'eicu', 'sepsis'),
                       help='Directory containing time series CSVs')
    parser.add_argument('--pred-dataset', type=str,
                       default=os.path.join(_DATA_ROOT, 'eicu', 'sepsis', 'sepsis_prediction_dataset.csv'),
                       help='Path to prediction dataset for merging')
    parser.add_argument('--output', type=str,
                       default=os.path.join(_DATA_ROOT, 'eicu', 'sepsis', 'sepsis_trajectory_probs.csv'),
                       help='Path to save merged prediction dataset with probabilities')
    parser.add_argument('--merged-output', type=str,
                       default=None,
                       help='Optional path to save merged prediction dataset (overrides --output)')
    parser.add_argument('--window-days', type=float, default=3.0,
                       help='Lookback window in days (default: 3.0)')
    parser.add_argument('--n-batches', type=int, default=8,
                       help='Number of batches per biomarker (default: 8)')
    parser.add_argument('--cohort-splits', type=int, default=1,
                       help='Total number of cohort splits (default: 1)')
    parser.add_argument('--cohort-index', type=int, default=0,
                       help='Which cohort split to process (0-indexed)')
    
    args = parser.parse_args()

    if args.cohort_index < 0 or args.cohort_index >= args.cohort_splits:
        print(f"ERROR: cohort-index must be between 0 and {args.cohort_splits - 1}")
        sys.exit(1)
    
    data_dir = Path(args.data_dir)
    
    print("="*80)
    print("eICU Sepsis Multi-Biomarker Trajectory Modeling")
    print("="*80)
    print(f"\nBiomarkers: lactate, WBC, platelet")
    print(f"Data directory: {data_dir}")
    print(f"Prediction dataset: {args.pred_dataset}")
    print(f"Window: {args.window_days} days")
    if args.cohort_splits > 1:
        print(f"Sub-cohort: {args.cohort_index + 1}/{args.cohort_splits}")
    
    # PRE-COMPILE PyTensor functions once (before all biomarkers)
    dummy_config = BayesConfig(
        window_years=args.window_days,
        df_basis=5,
        n_samples=200,
        tune=300,
        min_points_per_window=4,
    )
    precompile_pytensor_functions(dummy_config)
    
    # Determine cohort split using prediction dataset (preferred)
    cohort_patients = None
    if args.cohort_splits > 1 and Path(args.pred_dataset).exists():
        prediction_dataset = pd.read_csv(args.pred_dataset)
        all_patients = prediction_dataset['stay_id'].unique()
        total_patients = len(all_patients)
        cohort_size = total_patients // args.cohort_splits
        start_idx = args.cohort_index * cohort_size
        end_idx = total_patients if args.cohort_index == args.cohort_splits - 1 else start_idx + cohort_size
        cohort_patients = all_patients[start_idx:end_idx]

        print(f"\n📊 Sub-cohort {args.cohort_index + 1}/{args.cohort_splits}:")
        print(f"   Processing patients {start_idx:,} to {end_idx:,} (of {total_patients:,} total)")
        print(f"   Cohort size: {len(cohort_patients):,} patients")

    # Compute trajectories for each biomarker
    biomarker_results = {}
    
    for biomarker_name, config_dict in BIOMARKERS.items():
        probs_output_path = data_dir / f"{biomarker_name}_trajectory_probs_bayes.csv"
        if args.cohort_splits > 1:
            stem = probs_output_path.stem
            suffix = probs_output_path.suffix
            probs_output_path = probs_output_path.parent / f"{stem}_cohort{args.cohort_index:02d}{suffix}"
        result = compute_biomarker_trajectories(
            biomarker_name=biomarker_name,
            config_dict=config_dict,
            data_dir=data_dir,
            window_days=args.window_days,
            n_batches=args.n_batches,
            probs_output_path=probs_output_path,
            cohort_patients=cohort_patients,
        )
        
        if result is not None:
            biomarker_results[biomarker_name] = result
    
    if not biomarker_results:
        print("\n❌ ERROR: No biomarker trajectories were computed successfully")
        sys.exit(1)
    
    # Load prediction dataset
    print("\n" + "="*80)
    print("Merging with Prediction Dataset")
    print("="*80)
    
    print(f"\nLoading prediction dataset: {args.pred_dataset}")
    prediction_dataset = pd.read_csv(args.pred_dataset)
    if cohort_patients is not None:
        prediction_dataset = prediction_dataset[prediction_dataset['stay_id'].isin(cohort_patients)]
    print(f"   Samples: {len(prediction_dataset):,}")
    print(f"   Patients: {prediction_dataset['stay_id'].nunique():,}")
    
    # Merge all biomarker trajectories
    merged = prediction_dataset.copy()
    
    for biomarker_name, traj_df in biomarker_results.items():
        print(f"\nMerging {biomarker_name} trajectories...")
        merged = merged.merge(traj_df, on=['stay_id', 'time_day'], how='left')
        
        prob_cols = [f'{biomarker_name}_stable', f'{biomarker_name}_gradual', f'{biomarker_name}_rapid']
        missing = merged[prob_cols].isna().any(axis=1).sum()
        print(f"   Rows with {biomarker_name} probs: {len(merged) - missing:,} / {len(merged):,}")
        print(f"   Missing {biomarker_name} probs: {missing:,} ({100*missing/len(merged):.1f}%)")
    
    # Summary
    print(f"\n" + "="*80)
    print("Final Merged Dataset")
    print("="*80)
    print(f"\nRows: {len(merged):,}")
    print(f"Columns: {len(merged.columns)}")
    print(f"\nTrajectory probability columns:")
    
    all_traj_cols = [col for col in merged.columns if any(bm in col for bm in biomarker_results.keys()) 
                     and any(suffix in col for suffix in ['_stable', '_gradual', '_rapid'])]
    for col in sorted(all_traj_cols):
        non_missing = merged[col].notna().sum()
        print(f"   {col:30s}: {non_missing:7,} non-missing ({100*non_missing/len(merged):5.1f}%)")
    
    # Validate merged dataset
    for biomarker_name in biomarker_results.keys():
        prob_cols = [f'{biomarker_name}_stable', f'{biomarker_name}_gradual', f'{biomarker_name}_rapid']
        missing = merged[prob_cols].isna().any(axis=1).sum()
        if missing > 0:
            print(f"   WARNING: Missing probabilities for {missing} rows in merged {biomarker_name}.")

        out_of_range = ((merged[prob_cols] < 0) | (merged[prob_cols] > 1)).any(axis=1).sum()
        if out_of_range > 0:
            print(f"   ERROR: {out_of_range} rows have probabilities outside [0, 1] for merged {biomarker_name}.")
            sys.exit(2)

        sum_probs = merged[prob_cols].sum(axis=1)
        invalid_sum = (np.abs(sum_probs - 1.0) > 0.05).sum()
        if invalid_sum > 0:
            print(f"   WARNING: {invalid_sum} rows have probabilities not summing to ~1 for merged {biomarker_name}.")

    # Save merged dataset
    output_path = Path(args.merged_output) if args.merged_output else Path(args.output)
    if args.cohort_splits > 1:
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_cohort{args.cohort_index:02d}{suffix}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved merged dataset: {output_path}")
    print("="*80)

if __name__ == '__main__':
    main()
