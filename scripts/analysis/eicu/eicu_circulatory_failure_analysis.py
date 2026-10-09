#!/usr/bin/env python
"""
eICU Circulatory Failure Prediction Analysis
Compares model and feature configurations across single vs multi-biomarker trajectories and summary statistics.

Run from workspace root:
    python scripts/analysis/eicu/eicu_circulatory_failure_analysis.py
    python scripts/analysis/eicu/eicu_circulatory_failure_analysis.py --train-n-jobs 4 --train-backend threading
"""
import os
import sys
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

SEED = 920


def _pick_id_col(df):
    """Auto-detect ID column."""
    for col in ['stay_id', 'patientunitstayid', 'hadm_id']:
        if col in df.columns:
            return col
    raise ValueError('No ID column found')


def _pick_time_cols(df):
    """Auto-detect time columns."""
    if 'time_hours' in df.columns and 'time_hour' in df.columns:
        return 'time_hours', 'time_hour'
    if 'time_days' in df.columns and 'time_day' in df.columns:
        return 'time_days', 'time_day'
    if 'time_hour' in df.columns:
        return 'time_hour', 'time_hour'
    if 'time_day' in df.columns:
        return 'time_day', 'time_day'
    raise ValueError('No time columns found')


def _read_table(path: Path) -> pd.DataFrame:
    """Read table (parquet or CSV)."""
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_named_table(base_dir: Path, stem_name: str) -> pd.DataFrame:
    """Load named table (parquet-first fallback to CSV)."""
    candidates = [base_dir / f'{stem_name}.parquet', base_dir / f'{stem_name}.csv']
    for p in candidates:
        if p.exists():
            logger.info(f'  Loading: {p}')
            return _read_table(p)
    raise FileNotFoundError(f'Missing table for {stem_name}. Tried: {candidates}')


def _safe_left_merge(base_df, add_df, key_cols, add_name):
    """Left-merge add_df onto base_df on key_cols, de-duplicating either side
    first so pandas' validate='one_to_one' doesn't raise. Mirrors
    scripts/analysis/hirid/hirid_circulatory_failure_analysis.py::_safe_left_merge.
    """
    base_dup = int(base_df.duplicated(subset=key_cols).sum())
    add_dup = int(add_df.duplicated(subset=key_cols).sum())

    if base_dup > 0:
        logger.warning(f'base dataset has {base_dup:,} duplicate keys; keeping first')
        base_df = base_df.drop_duplicates(subset=key_cols, keep='first')

    if add_dup > 0:
        prob_cols = [c for c in add_df.columns if c not in key_cols]
        logger.warning(f'{add_name} has {add_dup:,} duplicate keys; aggregating means')
        add_df = add_df.groupby(key_cols, as_index=False)[prob_cols].mean()

    return base_df.merge(add_df, on=key_cols, how='left', validate='one_to_one')


def load_prediction_dataset(base_dir: Path):
    """Load the eICU circulatory-failure prediction dataset, combining the
    bayes and bootstrap trajectory-probability backends when both are
    available.

    Mirrors the dual-backend merge in
    scripts/analysis/hirid/hirid_circulatory_failure_analysis.py::main():
    when both a pre-merged bayes prediction dataset
    (`circulatory_failure_prediction_dataset_with_probs`) and a pre-merged
    bootstrap prediction dataset
    (`circulatory_failure_prediction_dataset_with_bootstrap_probs`) exist
    with all 9 raw trajectory columns, they are combined into a single
    dataframe: bootstrap trajectory columns are renamed with a "_boot"
    suffix, bayes trajectory columns are left unsuffixed, and the two are
    left-joined on [id_col, windowing_col]. This keeps the two trajectory
    representations disjoint so build_feature_sets() can build independent
    "(Bootstrap)" feature sets alongside the bayes-backed ones, matching the
    poster pipeline's (Bootstrap)-suffix convention (see
    src/analysis/poster/feature_set_mapper.py DEFAULT_ALIASES). Falls back to
    whichever single backend is available, then to the raw (no trajectory
    probs) prediction dataset.
    """
    pred_stem_bootstrap = 'circulatory_failure_prediction_dataset_with_bootstrap_probs'
    pred_stem_bayes = 'circulatory_failure_prediction_dataset_with_probs'
    pred_stem_base = 'circulatory_failure_prediction_dataset'

    expected_traj_cols = [
        'lactate_stable', 'lactate_gradual', 'lactate_rapid',
        'heartrate_stable', 'heartrate_gradual', 'heartrate_rapid',
        'systolic_stable', 'systolic_gradual', 'systolic_rapid',
    ]

    def _try_load(stem):
        try:
            return _load_named_table(base_dir, stem)
        except FileNotFoundError:
            return None

    dataset_boot = _try_load(pred_stem_bootstrap)
    dataset_bayes = _try_load(pred_stem_bayes)

    has_boot_merged = dataset_boot is not None and all(c in dataset_boot.columns for c in expected_traj_cols)
    has_bayes_merged = dataset_bayes is not None and all(c in dataset_bayes.columns for c in expected_traj_cols)
    logger.info(f'Pre-merged bootstrap trajectory dataset present: {has_boot_merged}')
    logger.info(f'Pre-merged bayes trajectory dataset present: {has_bayes_merged}')

    if has_boot_merged and has_bayes_merged:
        logger.info('Both trajectory backends found — merging into one working dataframe.')
        id_col = _pick_id_col(dataset_boot)
        _, windowing_col = _pick_time_cols(dataset_boot)
        key_cols = [id_col, windowing_col]
        dataset = dataset_boot.rename(columns={c: f'{c}_boot' for c in expected_traj_cols})
        dataset = _safe_left_merge(dataset, dataset_bayes[key_cols + expected_traj_cols], key_cols, 'bayes_traj')
        pred_source_stem = f'{pred_stem_bootstrap} + {pred_stem_bayes}'
    elif has_boot_merged:
        dataset = dataset_boot
        pred_source_stem = pred_stem_bootstrap
    elif has_bayes_merged:
        dataset = dataset_bayes
        pred_source_stem = pred_stem_bayes
    else:
        dataset = _load_named_table(base_dir, pred_stem_base)
        pred_source_stem = pred_stem_base

    logger.info(f'Loaded prediction dataset ({pred_source_stem}): {len(dataset):,} samples')
    return dataset, has_boot_merged, has_bayes_merged


def load_data(base_dir: Path):
    """Load prediction dataset and biomarker timeseries."""
    logger.info(f'BASE_DIR: {base_dir}')

    dataset, has_boot_merged, has_bayes_merged = load_prediction_dataset(base_dir)
    # kept for backward-compat logging/return signature; True only when the
    # bootstrap backend's raw trajectory columns (now "_boot"-suffixed when
    # both backends are present) made it into the dataset.
    has_bootstrap_probs = has_boot_merged

    # Load biomarker timeseries
    lactate_ts = _load_named_table(base_dir, 'lactate_timeseries')
    heartrate_ts = _load_named_table(base_dir, 'heartrate_timeseries')
    systolic_ts = _load_named_table(base_dir, 'systolic_timeseries')

    logger.info(f'Lactate:   {len(lactate_ts):,} rows')
    logger.info(f'Heartrate: {len(heartrate_ts):,} rows')
    logger.info(f'Systolic:  {len(systolic_ts):,} rows')

    return dataset, lactate_ts, heartrate_ts, systolic_ts, has_bootstrap_probs


def biomarker_summary_stats(ts_df, value_col, id_col, windowing_col, time_col, lookback_hours=12):
    """Compute windowed summary statistics per biomarker."""
    ts_df = ts_df.copy()
    ts_df[time_col] = pd.to_numeric(ts_df[time_col], errors='coerce')
    ts_df[windowing_col] = pd.to_numeric(ts_df[windowing_col], errors='coerce')
    ts_df[value_col] = pd.to_numeric(ts_df[value_col], errors='coerce')
    ts_df = ts_df.dropna(subset=[time_col, windowing_col, value_col])

    trend_failures = 0

    summary_df_list = []
    for group, group_df in ts_df.groupby(id_col):
        group_df = group_df.sort_values(time_col)
        summary_list = []
        for current_time in group_df[windowing_col].unique():
            window_start = current_time - lookback_hours
            window_data = group_df[group_df[windowing_col].between(window_start, current_time, inclusive='both')]
            if len(window_data) > 0:
                value_mean = window_data[value_col].mean()
                value_max = window_data[value_col].max()
                value_min = window_data[value_col].min()
                value_change = window_data[value_col].iloc[-1] - window_data[value_col].iloc[0]

                if len(window_data) > 1:
                    x = window_data[time_col].to_numpy(dtype=float)
                    y = window_data[value_col].to_numpy(dtype=float)
                    finite = np.isfinite(x) & np.isfinite(y)
                    x = x[finite]
                    y = y[finite]

                    # polyfit can fail when x has <2 unique points or degenerate values
                    if len(x) > 1 and np.unique(x).size > 1:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', np.exceptions.RankWarning)
                                poly_fit = np.polyfit(x, y, 1)
                                value_trend = float(poly_fit[0])
                        except Exception:
                            value_trend = 0.0
                    else:
                        value_trend = 0.0

                    value_std = window_data[value_col].std()
                else:
                    value_trend = 0.0
                    value_std = 0.0

                summary_list.append({
                    id_col: group,
                    windowing_col: current_time,
                    f'{value_col}_mean_{lookback_hours}h': value_mean,
                    f'{value_col}_max_{lookback_hours}h': value_max,
                    f'{value_col}_min_{lookback_hours}h': value_min,
                    f'{value_col}_change_{lookback_hours}h': value_change,
                    f'{value_col}_trend_{lookback_hours}h': value_trend,
                    f'{value_col}_std_{lookback_hours}h': value_std,
                })

        if summary_list:
            summary_df_list.append(pd.DataFrame(summary_list))

    return pd.concat(summary_df_list, ignore_index=True) if summary_df_list else pd.DataFrame()


def build_feature_sets(dataset: pd.DataFrame):
    """Build feature set configurations."""
    # Find target column
    target_col = None
    for col in ['target_circulatory_failure', 'target_circ_failure', 'target_circulatory']:
        if col in dataset.columns:
            target_col = col
            break
    if target_col is None:
        raise ValueError('No target column found')

    # Normalize common categorical gender encoding to numeric
    if 'gender' in dataset.columns:
        dataset['gender'] = dataset['gender'].replace(
            {
                'M': 1, 'Male': 1, 'male': 1,
                'F': 0, 'Female': 0, 'female': 0,
            }
        )

    # Static features: demographics and the admission baseline of each target parameter
    static_features = [c for c in ['age', 'gender'] if c in dataset.columns]
    static_features += [c for c in dataset.columns if c.startswith('baseline_')]

    # Trajectory features. Bayes-sourced columns are unsuffixed; bootstrap-sourced
    # columns carry a "_boot" suffix (see load_prediction_dataset()'s dual-backend
    # merge). Kept strictly disjoint from each other and from "Baseline" below.
    lactate_traj = ['lactate_stable', 'lactate_gradual', 'lactate_rapid', 'lactate_worsening']
    heartrate_traj = ['heartrate_stable', 'heartrate_gradual', 'heartrate_rapid', 'heartrate_worsening']
    systolic_traj = ['systolic_stable', 'systolic_gradual', 'systolic_rapid', 'systolic_worsening']
    multi_traj = lactate_traj + heartrate_traj + systolic_traj

    lactate_traj_boot = [f'{c}_boot' for c in lactate_traj]
    heartrate_traj_boot = [f'{c}_boot' for c in heartrate_traj]
    systolic_traj_boot = [f'{c}_boot' for c in systolic_traj]
    multi_traj_boot = lactate_traj_boot + heartrate_traj_boot + systolic_traj_boot

    # Summary statistics
    lactate_summary = [c for c in dataset.columns if c.startswith('lactate_') and c.endswith('h')]
    heartrate_summary = [c for c in dataset.columns if c.startswith('heartrate_') and c.endswith('h')]
    systolic_summary = [c for c in dataset.columns if c.startswith('systolic_') and c.endswith('h')]
    multi_summary = lactate_summary + heartrate_summary + systolic_summary

    # Anything already classified as trajectory (either backend) or summary must
    # not also leak into "dynamic" via the substring-based vital-sign scan below
    # (e.g. "systolic_stable", "systolic_stable_boot", and "systolic_mean_12h"
    # all contain "systolic") — that would silently contaminate "Baseline" with
    # trajectory/summary signal it's supposed to exclude. This previously only
    # guarded dynamic_labs (via exclude_keywords) and not dynamic_vitals, so
    # heartrate_stable/gradual/rapid and systolic_stable/gradual/rapid were
    # leaking straight into "Baseline" whenever a trajectory backend was loaded
    # (mirrors the fix documented in hirid_circulatory_failure_analysis.py's
    # build_feature_sets()).
    non_baseline_cols = set(multi_traj + multi_traj_boot + multi_summary)

    # Dynamic features (summary stats and vitals)
    exclude_keywords = ['stable', 'gradual', 'rapid', 'worsening', 'marker', 'trend', 'change', 'boot']
    dynamic_labs = [
        col for col in dataset.columns
        if any(col.startswith(prefix) for prefix in ['min_', 'mean_', 'max_'])
        and not any(keyword in col.lower() for keyword in exclude_keywords)
        and col not in non_baseline_cols
    ]
    dynamic_vitals = [
        col for col in dataset.columns
        if any(x in col for x in ['heart_rate', 'heartrate', 'respiratory_rate', 'o2_sat', 'systolic', 'diastolic', 'mean_bp', 'temperature'])
        and col not in non_baseline_cols
    ]
    dynamic_features = [
        c for c in dict.fromkeys(dynamic_labs + dynamic_vitals)
        if c not in static_features  # e.g. "baseline_systolic" contains "systolic"
    ]

    baseline_features = static_features + dynamic_features if dynamic_features else static_features

    def _present(cols):
        return [c for c in list(dict.fromkeys(cols)) if c in dataset.columns]

    # Column presence alone isn't enough to detect whether a backend has real
    # data (a column could exist but be all-NaN); check for actual values.
    def _has_data(cols):
        present = _present(cols)
        return len(present) > 0 and bool(dataset[present].notna().any().any())

    has_boot_traj = _has_data(lactate_traj_boot)
    logger.info(f'Bootstrap trajectory backend available for feature sets: {has_boot_traj}')

    feature_sets = {
        'Baseline': _present(baseline_features),
        'Baseline + Single Trajectory': _present(baseline_features + lactate_traj),
        'Baseline + Multi Trajectory': _present(baseline_features + multi_traj),
        'Baseline + Single Summary': _present(baseline_features + lactate_summary),
        'Baseline + Multi Summary': _present(baseline_features + multi_summary),
        'Baseline + Single Trajectory + Summary': _present(baseline_features + lactate_traj + lactate_summary),
        'Baseline + Multi Trajectory + Summary': _present(baseline_features + multi_traj + multi_summary),
    }
    if has_boot_traj:
        feature_sets['Baseline + Single Trajectory (Bootstrap)'] = _present(baseline_features + lactate_traj_boot)
        feature_sets['Baseline + Multi Trajectory (Bootstrap)'] = _present(baseline_features + multi_traj_boot)
        feature_sets['Baseline + Single Trajectory + Summary (Bootstrap)'] = _present(baseline_features + lactate_traj_boot + lactate_summary)
        feature_sets['Baseline + Multi Trajectory + Summary (Bootstrap)'] = _present(baseline_features + multi_traj_boot + multi_summary)

    feature_sets = {k: v for k, v in feature_sets.items() if len(v) > 0}

    return target_col, feature_sets


def run_cv(dataset, target_col, id_col, feature_sets, train_n_jobs=1, train_backend='threading'):
    """Run cross-validated model comparison."""
    dataset_clean = dataset.dropna(subset=[target_col]).copy()

    # Temporal carry-forward of trajectory probabilities (matches HiRID/shared-template practice)
    traj_cols = [c for c in dataset_clean.columns if any(k in c for k in ['_stable', '_gradual', '_rapid', '_worsening'])]
    if traj_cols:
        dataset_clean[traj_cols] = dataset_clean.groupby(id_col)[traj_cols].ffill(limit=2)

    y = dataset_clean[target_col].astype(int)
    groups = dataset_clean[id_col]

    logger.info(f'Dataset: {len(dataset_clean):,} samples, {y.mean():.1%} positive rate')
    logger.info(f'Feature sets: {len(feature_sets)}')

    # Models
    def get_xgb(scale_pos_weight, seed):
        return XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=seed, eval_metric='logloss'
        )

    def get_lr(scale_pos_weight, seed):
        return LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=seed, solver='lbfgs'
        )

    def get_rf(scale_pos_weight, seed):
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight='balanced', random_state=seed, n_jobs=4
        )

    def get_gb(scale_pos_weight, seed):
        return HistGradientBoostingClassifier(
            max_bins=225, max_depth=3, learning_rate=0.1, random_state=seed
        )

    models = {
        'XGBoost': get_xgb,
        'Logistic Regression': get_lr,
        'Random Forest': get_rf,
        'Gradient Boosting': get_gb,
    }

    boosting_models = {'XGBoost', 'Gradient Boosting'}
    n_repeats, n_folds = 10, 5

    fold_rows = []

    for model_name, model_fn in models.items():
        logger.info(f'\n{"=" * 80}\nModel: {model_name}\n{"=" * 80}')

        dataset_model = dataset_clean.copy()

        for feature_set_name, feature_cols in feature_sets.items():
            selected_cols = [c for c in feature_cols if c in dataset_model.columns]
            if len(selected_cols) == 0:
                continue

            logger.info(f'  -> {feature_set_name} ({len(selected_cols)} features)')

            model_fs_results = []

            for repeat_idx in range(1, n_repeats + 1):
                repeat_seed = SEED + repeat_idx - 1
                rng = np.random.RandomState(seed=repeat_seed)
                shuffle_idx = rng.permutation(len(dataset_model))

                dataset_repeat = dataset_model.iloc[shuffle_idx].reset_index(drop=True)
                y_repeat = y.iloc[shuffle_idx].reset_index(drop=True)
                groups_repeat = groups.iloc[shuffle_idx].reset_index(drop=True)

                gkf = GroupKFold(n_splits=n_folds)
                fold_aucs, fold_auprs = [], []

                for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(dataset_repeat, y_repeat, groups_repeat), 1):
                    X_train = dataset_repeat.iloc[train_idx][selected_cols].copy()
                    X_test = dataset_repeat.iloc[test_idx][selected_cols].copy()
                    y_train = y_repeat.iloc[train_idx]
                    y_test = y_repeat.iloc[test_idx]

                    # Ensure all model inputs are numeric (coerce strings like "Female"/"Male")
                    X_train = X_train.apply(pd.to_numeric, errors='coerce')
                    X_test = X_test.apply(pd.to_numeric, errors='coerce')

                    # Use train medians for imputation; if a column is entirely NaN, fall back to 0
                    train_medians = X_train.median(numeric_only=True).fillna(0.0)
                    X_train = X_train.fillna(train_medians).fillna(0.0)
                    X_test = X_test.fillna(train_medians).fillna(0.0)

                    # Preprocessing
                    if model_name not in boosting_models:
                        imputer = SimpleImputer(strategy='median')
                        X_train_proc = imputer.fit_transform(X_train)
                        X_test_proc = imputer.transform(X_test)
                    else:
                        X_train_proc = X_train.values
                        X_test_proc = X_test.values

                    scaler = StandardScaler()
                    X_train_proc = scaler.fit_transform(X_train_proc)
                    X_test_proc = scaler.transform(X_test_proc)

                    # Train
                    pos = y_train.sum()
                    neg = len(y_train) - pos
                    scale_pos_weight = neg / max(pos, 1)

                    model = model_fn(scale_pos_weight, repeat_seed)
                    model.fit(X_train_proc, y_train)

                    y_pred = model.predict_proba(X_test_proc)[:, 1]
                    auroc = roc_auc_score(y_test, y_pred)
                    aupr = average_precision_score(y_test, y_pred)

                    fold_aucs.append(auroc)
                    fold_auprs.append(aupr)

                    fold_rows.append({
                        'Model': model_name,
                        'Feature Set': feature_set_name,
                        'Repeat': repeat_idx,
                        'Fold': fold_idx,
                        'AUROC': auroc,
                        'AUPR': aupr,
                    })

                logger.info(f'     repeat {repeat_idx:02d}/{n_repeats}: '
                           f'AUROC={np.mean(fold_aucs):.4f} | AUPR={np.mean(fold_auprs):.4f}')

    results_df = pd.DataFrame(fold_rows)

    summary_df = (
        results_df.groupby(['Model', 'Feature Set'], as_index=False)
        .agg(
            ROC_AUC_mean=('AUROC', 'mean'),
            ROC_AUC_std=('AUROC', 'std'),
            AUPR_mean=('AUPR', 'mean'),
            AUPR_std=('AUPR', 'std'),
            n_folds=('AUROC', 'count'),
        )
        .sort_values(['Model', 'ROC_AUC_mean'], ascending=[True, False])
    )
    summary_df['ROC-AUC'] = summary_df['ROC_AUC_mean']
    summary_df['ROC-AUC std'] = summary_df['ROC_AUC_std']
    summary_df['AUPR'] = summary_df['AUPR_mean']
    summary_df['AUPR std'] = summary_df['AUPR_std']

    baseline_metrics = (
        results_df[results_df['Feature Set'] == 'Baseline'][['Model', 'Repeat', 'Fold', 'AUROC', 'AUPR']]
        .rename(columns={'AUROC': 'Baseline AUROC', 'AUPR': 'Baseline AUPR'})
    )

    delta_df = results_df.merge(
        baseline_metrics,
        on=['Model', 'Repeat', 'Fold'],
        how='left',
        validate='many_to_one',
    )
    delta_df['Delta AUROC'] = delta_df['AUROC'] - delta_df['Baseline AUROC']
    delta_df['Delta AUPR'] = delta_df['AUPR'] - delta_df['Baseline AUPR']

    delta_summary_df = (
        delta_df[delta_df['Feature Set'] != 'Baseline']
        .groupby(['Model', 'Feature Set'], as_index=False)
        .agg(
            Delta_ROC_AUC_mean=('Delta AUROC', 'mean'),
            Delta_ROC_AUC_std=('Delta AUROC', 'std'),
            Delta_AUPR_mean=('Delta AUPR', 'mean'),
            Delta_AUPR_std=('Delta AUPR', 'std'),
            n_folds=('Delta AUROC', 'count'),
        )
        .sort_values(['Model', 'Delta_ROC_AUC_mean'], ascending=[True, False])
    )

    return results_df, summary_df, delta_summary_df


def main():
    parser = argparse.ArgumentParser(description='eICU Circulatory Failure Analysis')
    parser.add_argument('--base-dir', type=str, default=os.path.join(os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet"), "eicu", "circulatory_failure"),
                       help='Base directory for data')
    parser.add_argument('--output-dir', type=str, default='results/eicu/circulatory_failure',
                       help='Output directory for results')
    parser.add_argument('--train-n-jobs', type=int, default=1,
                       help='Number of parallel jobs for CV')
    parser.add_argument('--train-backend', type=str, choices=['threading', 'loky', 'multiprocessing', 'sequential', 'processes'],
                       default='threading', help='Joblib backend for parallelization')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('✓ eICU Circulatory Failure Analysis')
    logger.info(f'  Base dir: {base_dir}')
    logger.info(f'  Output dir: {output_dir}')
    logger.info('')

    # Load data
    logger.info('1. Loading data...')
    dataset, lactate_ts, heartrate_ts, systolic_ts, has_bootstrap = load_data(base_dir)

    # Compute summary stats
    logger.info('\n2. Computing summary statistics...')
    id_col = _pick_id_col(dataset)
    time_col, windowing_col = _pick_time_cols(dataset)

    lactate_summary = biomarker_summary_stats(lactate_ts, 'lactate', id_col, windowing_col, time_col, lookback_hours=12)
    heartrate_summary = biomarker_summary_stats(heartrate_ts, 'heartrate', id_col, windowing_col, time_col, lookback_hours=12)
    systolic_summary = biomarker_summary_stats(systolic_ts, 'systolic', id_col, windowing_col, time_col, lookback_hours=12)

    logger.info(f'Lactate summary: {len(lactate_summary):,} rows')
    logger.info(f'Heartrate summary: {len(heartrate_summary):,} rows')
    logger.info(f'Systolic summary: {len(systolic_summary):,} rows')

    # Merge summary stats
    dataset = dataset.merge(lactate_summary, on=[id_col, windowing_col], how='left')
    dataset = dataset.merge(heartrate_summary, on=[id_col, windowing_col], how='left')
    dataset = dataset.merge(systolic_summary, on=[id_col, windowing_col], how='left')

    # Build feature sets
    logger.info('\n3. Building feature sets...')
    target_col, feature_sets = build_feature_sets(dataset)
    logger.info(f'Target: {target_col}')
    for name, cols in feature_sets.items():
        logger.info(f'  {name}: {len(cols)} features')

    # Run CV
    logger.info('\n4. Running cross-validated model comparison...')
    results_df, summary_df, delta_summary_df = run_cv(dataset, target_col, id_col, feature_sets, 
                       train_n_jobs=args.train_n_jobs, train_backend=args.train_backend)

    # Save results
    legacy_fold_path = output_dir / 'circulatory_failure_cv_results.csv'
    aligned_fold_path = output_dir / 'eicu_circulatory_failure_fold_results.parquet'
    summary_path = output_dir / 'eicu_circulatory_failure_summary.csv'
    delta_path = output_dir / 'eicu_circulatory_failure_delta_summary.csv'

    results_df.to_csv(legacy_fold_path, index=False)
    results_df.to_parquet(aligned_fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    delta_summary_df.to_csv(delta_path, index=False)
    logger.info(f'✓ Results saved to {legacy_fold_path}')
    logger.info(f'✓ Results saved to {aligned_fold_path}')
    logger.info(f'✓ Results saved to {summary_path}')
    logger.info(f'✓ Results saved to {delta_path}')

    # Summary
    summary = summary_df.groupby('Feature Set').agg({
        'ROC-AUC': 'mean',
        'AUPR': 'mean'
    }).round(4)
    logger.info('\nSummary by Feature Set:')
    logger.info(summary)


if __name__ == '__main__':
    main()
