"""
Compute biomarker trajectory probabilities for the eICU sepsis cohort.

Usage (examples):
    python sepsis_trajs.py --biomarker lactate   --input results/eicu/sepsis/lactate_timeseries.csv   --pred-dataset results/eicu/sepsis/sepsis_prediction_dataset.csv --output results/eicu/sepsis/lactate_trajectory_probs.csv
    python sepsis_trajs.py --biomarker wbc       --input results/eicu/sepsis/wbc_timeseries.csv       --pred-dataset results/eicu/sepsis/sepsis_prediction_dataset.csv --output results/eicu/sepsis/wbc_trajectory_probs.csv
    python sepsis_trajs.py --biomarker platelets --input results/eicu/sepsis/platelets_timeseries.csv --pred-dataset results/eicu/sepsis/sepsis_prediction_dataset.csv --output results/eicu/sepsis/platelets_trajectory_probs.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import gc
from pathlib import Path
import sys

# Ensure compiled artifacts and matplotlib cache land in a writable location
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
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

# Default biomarker-specific settings (mirrors MIMIC)
BIOMARKER_CONFIG = {
    'lactate': {
        'value_col': 'lactate',
        'flat_thr': 0.1,
        'decline_thr': 0.3,  # positive = worsening
        'nonlinear_gap': 0.5,
        'class_func': pos_flags_from_traj,
        'traj_types': ('stable', 'gradual_increase', 'rapid_increase'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'},
    },
    'wbc': {
        'value_col': 'wbc',
        'flat_thr': 1.0,
        'decline_thr': 3.0,  # positive = worsening (leukocytosis)
        'nonlinear_gap': 2.0,
        'class_func': pos_flags_from_traj,
        'traj_types': ('stable', 'gradual_increase', 'rapid_increase'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'},
    },
    'platelets': {
        'value_col': 'platelet',
        'flat_thr': -20.0,
        'decline_thr': -50.0,  # negative = worsening (thrombocytopenia)
        'nonlinear_gap': 30.0,
        'class_func': flags_from_traj,
        'traj_types': ('stable', 'gradual_decline', 'rapid_decline'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_decline', 'nonlinear': 'rapid_decline'},
    },
}

def main():
    parser = argparse.ArgumentParser(description='Compute eICU sepsis trajectory probabilities')
    parser.add_argument('--biomarker', type=str, choices=list(BIOMARKER_CONFIG.keys()), default='lactate',
                       help='Biomarker to model (lactate, wbc, platelets)')
    parser.add_argument('--input', type=str,
                       default=os.path.join(_DATA_ROOT, 'eicu', 'sepsis', 'lactate_timeseries.csv'),
                       help='Path to raw biomarker time series CSV')
    parser.add_argument('--pred-dataset', type=str,
                       default=os.path.join(_DATA_ROOT, 'eicu', 'sepsis', 'sepsis_prediction_dataset.csv'),
                       help='Path to prediction dataset for merging')
    parser.add_argument('--output', type=str,
                       default=os.path.join(_DATA_ROOT, 'eicu', 'sepsis', 'sepsis_trajectory_probs.csv'),
                       help='Path to save trajectory probabilities')
    parser.add_argument('--window-days', type=float, default=3.0,
                       help='Lookback window in days (default: 3.0)')
    parser.add_argument('--flat-thr', type=float, default=None,
                       help='Override stable threshold (per-day units). Defaults are biomarker-specific.')
    parser.add_argument('--decline-thr', type=float, default=None,
                       help='Override change threshold. Defaults are biomarker-specific.')
    parser.add_argument('--nonlinear-gap', type=float, default=None,
                       help='Override nonlinear gap. Defaults are biomarker-specific.')
    parser.add_argument('--value-col', type=str, default=None,
                       help='Override value column name in the time series CSV')
    parser.add_argument('--n-batches', type=int, default=8,
                       help='Number of batches for processing (default: 8)')
    
    args = parser.parse_args()
    
    bm = args.biomarker
    cfg_defaults = BIOMARKER_CONFIG[bm]
    value_col = args.value_col or cfg_defaults['value_col']
    flat_thr = cfg_defaults['flat_thr'] if args.flat_thr is None else args.flat_thr
    decline_thr = cfg_defaults['decline_thr'] if args.decline_thr is None else args.decline_thr
    nonlinear_gap = cfg_defaults['nonlinear_gap'] if args.nonlinear_gap is None else args.nonlinear_gap

    print("="*80)
    print(f"eICU Sepsis Trajectory Modeling - {bm.upper()}")
    print("="*80)
    
    # Load raw biomarker time series
    print(f"\n📊 Loading raw {bm} time series from: {args.input}")
    ts_df = pd.read_csv(args.input)
    n_patients = ts_df['stay_id'].nunique()
    n_rows = len(ts_df)
    print(f"   Patients: {n_patients:,}")
    print(f"   Total measurements: {n_rows:,}")
    if n_patients == 0 or n_rows == 0:
        print("\nERROR: No measurements found in the provided time series. "
              "Verify preprocessing produced a non-empty CSV and the value column name matches the biomarker.")
        sys.exit(1)
    print(f"   Measurements per patient: {n_rows / n_patients:.1f}")
    
    if value_col not in ts_df.columns:
        print(f"\nERROR: Expected value column '{value_col}' not found in {args.input}. Columns: {list(ts_df.columns)}")
        sys.exit(1)
    
    # Prepare trajectory input (rename for BayesianTraj)
    traj_input = ts_df[['stay_id', 'time_days', 'time_day', value_col]].copy()
    traj_input = traj_input.rename(columns={
        'stay_id': 'patientid',
        value_col: 'lab_value'
    })
    
    traj_input = traj_input.dropna(subset=['lab_value']).sort_values(by=['patientid', 'time_days'])
    
    print(f"\n1. Input data:")
    print(f"   Rows: {len(traj_input):,}")
    print(f"   Patients: {traj_input['patientid'].nunique():,}")
    print(f"   Timepoints per patient: {traj_input.groupby('patientid')['time_days'].count().mean():.1f}")
    
    # Configure Bayesian trajectory model
    config = BayesConfig(
        window_years=args.window_days,
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
    print(f"   Biomarker: {bm}")
    print(f"   Window: {args.window_days} days")
    print(f"   Stable threshold: ±{flat_thr}")
    print(f"   Change threshold: {decline_thr}")
    print(f"   Nonlinear gap: {nonlinear_gap}")
    
    # PRE-COMPILE PyTensor functions (avoids lock contention in parallel workers)
    precompile_pytensor_functions(config)
    
    # Initialize model
    traj_model = BayesianTraj(cfg=config)
    
    print(f"\n3. Computing trajectory probabilities...")
    
    # Process in batches
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
        
        batch_probs = traj_model.embed(traj_input[traj_input['patientid'].isin(subset_patients)])
        trajectory_probs_list.append(batch_probs)
        
        gc.collect()
    
    trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)
    
    print(f"\n✓ Trajectory probabilities computed!")
    
    # Rename columns
    # Rename probability columns using the biomarker-aware traj labels
    prob_map = {f'trajtype_{t}_prob': f'{bm}_{t}' for t in cfg_defaults['traj_types']}
    trajectory_probs = trajectory_probs.rename(columns=prob_map)
    trajectory_probs = trajectory_probs.rename(columns={'patientid': 'stay_id'})
    
    # Load prediction dataset and merge
    print(f"\n4. Loading prediction dataset: {args.pred_dataset}")
    prediction_dataset = pd.read_csv(args.pred_dataset)
    print(f"   Samples: {len(prediction_dataset):,}")
    
    merge_cols = ['stay_id', 'time_day'] + [prob_map[k] for k in prob_map]
    dataset_with_probs = prediction_dataset.merge(
        trajectory_probs[merge_cols],
        on=['stay_id', 'time_day'],
        how='left'
    )
    
    print(f"\n5. Merged trajectory probabilities:")
    print(f"   Rows: {len(dataset_with_probs):,}")
    
    # Analyze missing probabilities
    prob_cols = [prob_map[k] for k in prob_map]
    missing_mask = dataset_with_probs[prob_cols].isna().any(axis=1)
    n_missing = missing_mask.sum()
    
    if n_missing > 0:
        pct_missing = 100 * n_missing / len(dataset_with_probs)
        print(f"   ⚠️  Missing probabilities: {n_missing:,} / {len(dataset_with_probs):,} ({pct_missing:.1f}%)")
        
        # Diagnose reasons
        missing_samples = dataset_with_probs[missing_mask]
        missing_stay_ids = missing_samples['stay_id'].unique()
        traj_stay_ids = trajectory_probs['stay_id'].unique()
        
        no_traj_data = sum(1 for sid in missing_stay_ids if sid not in traj_stay_ids)
        if no_traj_data > 0:
            print(f"      - {no_traj_data} patients have no trajectory probabilities (insufficient observations)")
        
        missing_time_days = set(zip(missing_samples['stay_id'], missing_samples['time_day']))
        traj_time_days = set(zip(trajectory_probs['stay_id'], trajectory_probs['time_day']))
        missing_windows = len(missing_time_days - traj_time_days)
        if missing_windows > 0:
            print(f"      - {missing_windows} time windows have no probabilities (not enough data in lookback window)")
        
        print(f"   ✓ Proceeding with available probabilities (missing values preserved as NaN)")
    else:
        print(f"   ✓ All rows have complete probability assignments")

    out_of_range = ((dataset_with_probs[prob_cols] < 0) | (dataset_with_probs[prob_cols] > 1)).any(axis=1).sum()
    if out_of_range > 0:
        print(f"   ERROR: {out_of_range} rows have probabilities outside [0, 1].")
        sys.exit(2)

    sum_probs = dataset_with_probs[prob_cols].sum(axis=1)
    invalid_sum = (np.abs(sum_probs - 1.0) > 0.05).sum()
    if invalid_sum > 0:
        print(f"   WARNING: {invalid_sum} rows have probabilities not summing to ~1.")

    # Dominant trajectory
    dataset_with_probs['dominant_traj'] = dataset_with_probs[
        prob_cols
    ].idxmax(axis=1).str.replace(f"{bm}_", '')
    
    print(f"\n6. Trajectory Distribution:")
    for traj in cfg_defaults['traj_types']:
        subset = dataset_with_probs[dataset_with_probs['dominant_traj'] == traj]
        if len(subset) > 0:
            print(f"   {traj.replace('_', ' ').title():20s}: {len(subset):5,} ({100*len(subset)/len(dataset_with_probs):5.1f}%)")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_with_probs.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved: {output_path}")
    print("="*80)

if __name__ == '__main__':
    main()
