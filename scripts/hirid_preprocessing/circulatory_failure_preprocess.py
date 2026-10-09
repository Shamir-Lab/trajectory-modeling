"""
HiRiD Circulatory Failure Preprocessing Script

Creates:
- Lactate, Heart Rate, and Systolic BP time series
- Prediction dataset with circulatory failure outcome

Outcome definition (Hyland et al., 2020):
- Future window contains lactate >= 2 mmol/L AND (MAP <= 65 mmHg OR vasopressor/inotrope start)
- Excludes windows where patient is already in circulatory failure at prediction time

Usage:
    python scripts/hirid_preprocessing/circulatory_failure_preprocess.py
"""

from __future__ import annotations

import os
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

_TRAJ_DATA_ROOT = os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet")
OUTPUT_DIR = Path(_TRAJ_DATA_ROOT) / "hirid" / "circulatory_failure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = os.path.join(_TRAJ_DATA_ROOT, "HiRiD", "hirid.duckdb")
MIN_LOS_DAYS = 2.0
AGE_MIN, AGE_MAX = 18, 90
MIN_TARGET_MEASUREMENTS = 5   # lactate measurements per stay
FFILL_LIMIT = 3               # consecutive fills for vitals and labs
MISSINGNESS_THRESHOLD = 0.7   # drop feature columns with more missing values than this
PREDICTION_GAP_HOURS = 1.0
PREDICTION_WINDOW_HOURS = 8.0
MAP_THRESHOLD = 65.0
LACTATE_THRESHOLD = 2.0

print("=" * 80)
print("HiRiD Circulatory Failure Preprocessing")
print("=" * 80)

conn = duckdb.connect(DB_PATH, read_only=True)

cohort_query = f"""
SELECT p.patientid, p.admission_time, p.discharge_time, p.los_days,
       CAST(g.age AS INTEGER) AS age, g.sex AS gender
FROM patient_info p
INNER JOIN ref_general_table g ON p.patientid = g.patientid
WHERE p.los_days >= {MIN_LOS_DAYS}
  AND CAST(g.age AS INTEGER) BETWEEN {AGE_MIN} AND {AGE_MAX}
"""

cohort = conn.execute(cohort_query).fetchdf()
print(f"  Total ICU stays: {len(cohort):,}")

patient_ids = cohort['patientid'].tolist()

# Variable IDs (from data/hirid/hirid_variable_reference.csv)
VARS = {
    200: 'HeartRate',
    100: 'SysBP',   # Invasive systolic arterial pressure
    600: 'SysBP',   # Non-invasive systolic arterial pressure
    120: 'DiasBP',  # Invasive diastolic arterial pressure
    620: 'DiasBP',  # Non-invasive diastolic arterial pressure
    110: 'MAP',     # Invasive mean arterial pressure
    610: 'MAP',     # Non-invasive mean arterial pressure
    24000524: 'Lactate',  # Arterial blood lactate
    24000732: 'Lactate',  # Venous blood lactate
    24000485: 'Lactate',  # Venous blood lactate
}

# Ranges for sanity filtering
RANGES = {
    'HeartRate': (20, 300),
    'SysBP': (40, 300),
    'DiasBP': (20, 200),
    'MAP': (20, 200),
    'Lactate': (0.1, 20),
}

# Filter cohort to just the cohort we're interested in (saves memory in join)
cohort_subset = cohort[['patientid', 'admission_time']].copy()

# Manually filter to the variables we want (more efficient than large WHERE IN)
obs_query = f"""
SELECT 
    o.patientid,
    o.datetime as charttime,
    o.variableid,
    CAST(o.value AS DOUBLE) as valuenum,
    p.admission_time as admittime
FROM observations o
INNER JOIN patient_info p ON o.patientid = p.patientid
WHERE 
    o.variableid IN (200, 100, 600, 120, 620, 110, 610, 24000524, 24000732, 24000485)
    AND o.value IS NOT NULL
    AND o.patientid IN ({','.join(str(pid) for pid in patient_ids)})
"""

obs_df = conn.execute(obs_query).fetchdf()
print(f"✓ Fetched observations: {len(obs_df):,} rows")

# Ensure variableid is int for proper mapping
obs_df['variableid'] = obs_df['variableid'].astype(int)

obs_df['feature'] = obs_df['variableid'].map(VARS)

# Filter by ranges - remove rows where feature is in RANGES and value is outside the valid range
for feat, (min_v, max_v) in RANGES.items():
    mask = (obs_df['feature'] == feat) & (~obs_df['valuenum'].between(min_v, max_v))
    obs_df = obs_df[~mask]

obs_df['charttime'] = pd.to_datetime(obs_df['charttime'])
obs_df['admittime'] = pd.to_datetime(obs_df['admittime'])
obs_df['time_hours'] = (obs_df['charttime'] - obs_df['admittime']).dt.total_seconds() / 3600.0
obs_df['time_hour'] = np.floor(obs_df['time_hours']).astype(int)

# Time series outputs
lactate_ts = obs_df[obs_df['feature'] == 'Lactate'].copy()
lactate_ts = lactate_ts.rename(columns={'patientid': 'patientid', 'valuenum': 'lactate'})

heartrate_ts = obs_df[obs_df['feature'] == 'HeartRate'].copy()
heartrate_ts = heartrate_ts.rename(columns={'patientid': 'patientid', 'valuenum': 'heartrate'})

sbp_ts = obs_df[obs_df['feature'] == 'SysBP'].copy()
sbp_ts = sbp_ts.rename(columns={'patientid': 'patientid', 'valuenum': 'systolic'})

# Baselines (first 24h)

def add_baseline(df: pd.DataFrame, value_col: str, baseline_col: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    baseline_window = df[df['time_hours'] <= 24.0]
    baseline = (
        baseline_window.sort_values('time_hours')
        .groupby('patientid')[value_col]
        .first()
        .reset_index()
        .rename(columns={value_col: baseline_col})
    )
    return df.merge(baseline, on='patientid', how='left')


# Keep stays with enough measurements of the target clinical parameter (lactate)
lactate_counts = lactate_ts.groupby('patientid').size()
valid_ids = set(lactate_counts[lactate_counts >= MIN_TARGET_MEASUREMENTS].index)
obs_df = obs_df[obs_df['patientid'].isin(valid_ids)].copy()
lactate_ts = lactate_ts[lactate_ts['patientid'].isin(valid_ids)].copy()
heartrate_ts = heartrate_ts[heartrate_ts['patientid'].isin(valid_ids)].copy()
sbp_ts = sbp_ts[sbp_ts['patientid'].isin(valid_ids)].copy()
cohort = cohort[cohort['patientid'].isin(valid_ids)].copy()
print(f"✓ {len(valid_ids):,} stays with >= {MIN_TARGET_MEASUREMENTS} lactate measurements")

lactate_ts = add_baseline(lactate_ts, 'lactate', 'baseline_lactate')
heartrate_ts = add_baseline(heartrate_ts, 'heartrate', 'baseline_heartrate')
sbp_ts = add_baseline(sbp_ts, 'systolic', 'baseline_systolic')

lactate_ts.to_csv(OUTPUT_DIR / "lactate_timeseries.csv", index=False)
heartrate_ts.to_csv(OUTPUT_DIR / "heartrate_timeseries.csv", index=False)
sbp_ts.to_csv(OUTPUT_DIR / "systolic_timeseries.csv", index=False)
cohort.to_csv(OUTPUT_DIR / "patient_cohort.csv", index=False)

print("✓ Saved circulatory time series and cohort")

# Hourly aggregation
obs_daily = obs_df.groupby(['patientid', 'time_hour', 'feature'])['valuenum'].agg(['min', 'max', 'mean']).reset_index()

obs_daily = obs_daily.pivot_table(
    index=['patientid', 'time_hour'],
    columns='feature',
    values=['min', 'max', 'mean'],
    aggfunc='first'
).reset_index()

obs_daily.columns = ['_'.join(col).strip('_') for col in obs_daily.columns]

# Compute MAP if SysBP & DiasBP available and MAP not already present
if 'mean_MAP' not in obs_daily.columns and 'mean_SysBP' in obs_daily.columns and 'mean_DiasBP' in obs_daily.columns:
    obs_daily['mean_MAP'] = (2 * obs_daily['mean_DiasBP'] + obs_daily['mean_SysBP']) / 3.0

# Vasopressor detection (HiRiD pharma IDs)
VASOPRESSOR_IDS = [
    1000462, 1000656, 1000657, 1000658,  # Noradrenalin
    71, 1000750, 1000649, 1000650, 1000655,  # Adrenalin
    112, 113,  # Vasopressin
    426,  # Dobutrex (inotrope)
]

vaso_query = f"""
SELECT r.patientid, r.givenat as charttime, p.admission_time as admittime
FROM pharma_records r
INNER JOIN patient_info p ON r.patientid = p.patientid
WHERE r.pharmaid IN {tuple(VASOPRESSOR_IDS)}
AND r.patientid IN ({','.join(str(pid) for pid in patient_ids)})
"""

vaso_df = conn.execute(vaso_query).fetchdf()
if len(vaso_df) > 0:
    vaso_df['charttime'] = pd.to_datetime(vaso_df['charttime'])
    vaso_df['admittime'] = pd.to_datetime(vaso_df['admittime'])
    vaso_df['time_hour'] = ((vaso_df['charttime'] - vaso_df['admittime']).dt.total_seconds() / 3600.0).astype(int)
    vaso_start = vaso_df.groupby('patientid')['time_hour'].min().reset_index().rename(columns={'time_hour': 'vaso_start_hour'})
else:
    vaso_start = pd.DataFrame(columns=['patientid', 'vaso_start_hour'])

# Prepare lactate lookup from raw time series (not pivoted, which has many gaps)
lactate_hourly = obs_df[obs_df['feature'] == 'Lactate'].groupby(['patientid', 'time_hour'])['valuenum'].mean().reset_index()
lactate_hourly = lactate_hourly.rename(columns={'valuenum': 'lactate'})

failure_events = []
excluded = {'already_failure': 0, 'no_future_data': 0, 'ambiguous': 0}
total_windows = 0

for patientid, grp in obs_daily.groupby('patientid'):
    grp = grp.sort_values('time_hour')
    patient_lactate = lactate_hourly[lactate_hourly['patientid'] == patientid]
    total_windows += len(grp)
    
    for i in range(len(grp)):
        current_time = grp.iloc[i]['time_hour']
        current_map = grp.iloc[i].get('mean_MAP', np.nan)
        
        # Get current lactate from hourly aggregation (may be NA if not measured this hour)
        current_lactate_row = patient_lactate[patient_lactate['time_hour'] == current_time]
        current_lactate = current_lactate_row['lactate'].iloc[0] if len(current_lactate_row) > 0 else np.nan

        current_vaso = False
        if patientid in vaso_start['patientid'].values:
            start_hour = vaso_start[vaso_start['patientid'] == patientid]['vaso_start_hour'].iloc[0]
            current_vaso = start_hour <= current_time

        # Need MAP for prediction; lactate can be missing at current time
        if pd.isna(current_map):
            excluded['ambiguous'] += 1
            continue

        # Skip if already in circulatory failure (lactate high AND (hypotension OR vasopressor))
        if not pd.isna(current_lactate) and current_lactate >= LACTATE_THRESHOLD and (current_map <= MAP_THRESHOLD or current_vaso):
            excluded['already_failure'] += 1
            continue

        prediction_start = current_time + PREDICTION_GAP_HOURS
        prediction_end = current_time + PREDICTION_GAP_HOURS + PREDICTION_WINDOW_HOURS

        future_grp = grp[(grp['time_hour'] >= prediction_start) & (grp['time_hour'] <= prediction_end)]
        if len(future_grp) == 0:
            excluded['no_future_data'] += 1
            continue

        # Check future hypotension from vitals
        future_hypotension = False
        if 'mean_MAP' in future_grp.columns:
            future_hypotension = (future_grp['mean_MAP'].dropna() <= MAP_THRESHOLD).any()

        # Check future lactate from raw time series
        future_lactate_df = patient_lactate[
            (patient_lactate['time_hour'] >= prediction_start) & 
            (patient_lactate['time_hour'] <= prediction_end)
        ]
        future_lactate = (future_lactate_df['lactate'] >= LACTATE_THRESHOLD).any() if len(future_lactate_df) > 0 else False

        vasopressor_in_window = False
        if patientid in vaso_start['patientid'].values:
            start_hour = vaso_start[vaso_start['patientid'] == patientid]['vaso_start_hour'].iloc[0]
            vasopressor_in_window = prediction_start <= start_hour <= prediction_end

        # Circulatory failure (Hyland et al., 2020): lactate >= 2 mmol/L AND
        # (MAP <= 65 mmHg OR vasopressor/inotrope) within the prediction window
        target = int(future_lactate and (future_hypotension or vasopressor_in_window))
        row = grp.iloc[i].to_dict()
        row['target_circulatory_failure'] = target
        failure_events.append(row)

prediction_dataset = pd.DataFrame(failure_events)
print(f"\n\nOutcome Processing:")
print(f"  Total time windows examined: {total_windows:,}")
print(f"  Excluded (already failure): {excluded['already_failure']:,}")
print(f"  Excluded (no future data): {excluded['no_future_data']:,}")
print(f"  Excluded (ambiguous/no MAP): {excluded['ambiguous']:,}")
print(f"  Valid prediction windows: {len(prediction_dataset):,}")

if prediction_dataset.empty:
    prediction_dataset = pd.DataFrame(columns=list(obs_daily.columns) + ['target_circulatory_failure'])
    prediction_dataset.to_csv(OUTPUT_DIR / "circulatory_failure_prediction_dataset.csv", index=False)
    print("⚠️ No valid prediction windows; saved empty circulatory_failure_prediction_dataset.csv")
else:
    positive_first = prediction_dataset[prediction_dataset['target_circulatory_failure'] == 1].groupby('patientid').first().reset_index()
    all_negatives = prediction_dataset[prediction_dataset['target_circulatory_failure'] == 0]
    prediction_dataset = pd.concat([all_negatives, positive_first], ignore_index=True).sort_values(['patientid', 'time_hour']).reset_index(drop=True)

    # Static features (demographics, admission baselines) and forward-filled
    # hourly features (labels above use the raw hourly values); then drop
    # feature columns with >70% missing values
    features = obs_daily.sort_values(['patientid', 'time_hour']).copy()
    feat_cols = [c for c in features.columns if c not in ('patientid', 'time_hour')]
    features[feat_cols] = features.groupby('patientid', sort=False)[feat_cols].ffill(limit=FFILL_LIMIT)
    static = cohort[['patientid', 'age', 'gender']].copy()
    for ts, col in [(lactate_ts, 'baseline_lactate'), (heartrate_ts, 'baseline_heartrate'), (sbp_ts, 'baseline_systolic')]:
        static = static.merge(ts[['patientid', col]].drop_duplicates('patientid'), on='patientid', how='left')
    prediction_dataset = (
        prediction_dataset[['patientid', 'time_hour', 'target_circulatory_failure']]
        .merge(features, on=['patientid', 'time_hour'], how='left')
        .merge(static, on='patientid', how='left')
    )
    missing_pct = prediction_dataset[feat_cols].isnull().mean()
    cols_to_drop = [c for c in missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index if c != 'mean_MAP']
    prediction_dataset = prediction_dataset.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} columns with >70% missing values")

    prediction_dataset.to_csv(OUTPUT_DIR / "circulatory_failure_prediction_dataset.csv", index=False)

    print(f"✓ Saved {len(prediction_dataset):,} rows")
    print(f"   Positive events: {prediction_dataset['target_circulatory_failure'].sum():,} ({100*prediction_dataset['target_circulatory_failure'].mean():.1f}%)")
    print(f"   Skipped (already failure): {excluded['already_failure']:,}")
    (f"   Skipped (no future data): {excluded['no_future_data']:,}")

conn.close()
print("\n✓ Circulatory failure preprocessing complete!")
