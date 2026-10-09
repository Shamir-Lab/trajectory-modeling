"""
eICU Circulatory Failure Preprocessing Script

Creates:
- Lactate, Heart Rate, and Systolic BP time series
- Prediction dataset with circulatory failure outcome

Outcome definition (default):
- Future window contains vasopressor use OR hypotension (MAP/SBP below threshold)
- Excludes windows where patient is already in circulatory failure at prediction time

Usage:
    python scripts/eicu_preprocessing/circulatory_failure_preprocess.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('src'))

from eicu_loader import EICULoader, ffill_within_stay, MIN_TARGET_MEASUREMENTS, MISSINGNESS_THRESHOLD

OUTPUT_DIR = Path("/home/gaga/data/physionet/eicu/circulatory_failure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Outcome configuration (Nature Medicine 2020): predict CF within next 8h
PREDICTION_GAP_HOURS = 1.0
PREDICTION_WINDOW_HOURS = 8.0
MAP_THRESHOLD = 65.0
LACTATE_THRESHOLD = 2.0
MIN_MEASUREMENTS = MIN_TARGET_MEASUREMENTS  # lactate measurements per stay

print("=" * 80)
print("eICU Circulatory Failure Preprocessing")
print("=" * 80)

loader = EICULoader()
cohort = loader.load_general_cohort()
print(f"  Total ICU stays: {len(cohort):,}")
print(f"  Mean age: {cohort['age'].mean():.1f}")
print(f"  Mortality: {cohort['hospital_expire_flag'].mean():.1%}")

all_ids = cohort['patientunitstayid'].tolist()

# Load time series in batches
BATCH_SIZE = 1000
lactate_list = []
hr_list = []
sbp_list = []
map_list = []

for i in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="  Batches"):
    batch_ids = all_ids[i:i + BATCH_SIZE]

    batch_lact = loader.load_labs(['lactate'], batch_ids)
    if len(batch_lact) > 0:
        batch_lact['time_hours'] = batch_lact['lab_time_hours']
        batch_lact['time_hour'] = np.floor(batch_lact['time_hours']).astype(int)
        lactate_list.append(batch_lact)

    batch_vitals = loader.load_vitals(batch_ids)
    if len(batch_vitals) > 0:
        batch_vitals['time_hours'] = batch_vitals['vital_time_hours']
        batch_vitals['time_hour'] = np.floor(batch_vitals['time_hours']).astype(int)
        hr_list.append(batch_vitals)

    batch_bp = loader.load_vitals_bp(batch_ids)
    if len(batch_bp) > 0:
        batch_bp['time_hours'] = batch_bp['vital_time_hours']
        batch_bp['time_hour'] = np.floor(batch_bp['time_hours']).astype(int)
        sbp_list.append(batch_bp)

lactate_ts = pd.concat(lactate_list, ignore_index=True) if lactate_list else pd.DataFrame()
hr_ts = pd.concat(hr_list, ignore_index=True) if hr_list else pd.DataFrame()
sbp_ts = pd.concat(sbp_list, ignore_index=True) if sbp_list else pd.DataFrame()

if len(lactate_ts) == 0:
    print("  ERROR: No lactate measurements found!")
    loader.close()
    sys.exit(1)

lactate_ts = lactate_ts.rename(columns={
    'patientunitstayid': 'stay_id',
    'labresult': 'lactate'
})
hr_ts = hr_ts.rename(columns={
    'patientunitstayid': 'stay_id',
    'heart_rate': 'heartrate'
})
sbp_ts = sbp_ts.rename(columns={
    'patientunitstayid': 'stay_id',
    'systolic': 'systolic',
    'mean_bp': 'mean_bp'
})

print(f"  Total lactate measurements: {len(lactate_ts):,}")
print(f"  Total HR measurements: {len(hr_ts):,}")
print(f"  Total BP measurements: {len(sbp_ts):,}")

# Baselines (first 24h)

def baseline_first(df: pd.DataFrame, value_col: str, baseline_col: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    df = df.copy()
    baseline_window = df[df['time_hours'] <= 24.0]
    baseline = (
        baseline_window.sort_values('time_hours')
        .groupby('stay_id')[value_col]
        .first()
        .reset_index()
        .rename(columns={value_col: baseline_col})
    )
    return df.merge(baseline, on='stay_id', how='left')


lactate_ts = baseline_first(lactate_ts, 'lactate', 'baseline_lactate')
hr_ts = baseline_first(hr_ts, 'heartrate', 'baseline_heartrate')
sbp_ts = baseline_first(sbp_ts, 'systolic', 'baseline_systolic')

lactate_ts['time_hour'] = lactate_ts['time_hour']
hr_ts['time_hour'] = hr_ts['time_hour']
sbp_ts['time_hour'] = sbp_ts['time_hour']

lactate_ts.to_csv(OUTPUT_DIR / "lactate_timeseries.csv", index=False)
hr_ts.to_csv(OUTPUT_DIR / "heartrate_timeseries.csv", index=False)
sbp_ts.to_csv(OUTPUT_DIR / "systolic_timeseries.csv", index=False)

print("✓ Saved circulatory time series")

# Filter cohort with minimum measurements (using lactate availability)
measurement_counts = lactate_ts.groupby('stay_id').size()
valid_patients = measurement_counts[measurement_counts >= MIN_MEASUREMENTS].index.tolist()
cohort_filtered = cohort[cohort['patientunitstayid'].isin(valid_patients)].copy()
cohort_filtered = cohort_filtered.rename(columns={'patientunitstayid': 'stay_id'})
print(f"  Final cohort size: {len(cohort_filtered):,}")

# Load vitals and labs for daily aggregation
lab_names = [
    'lactate', 'creatinine', 'bun', 'glucose', 'potassium', 'sodium',
    'chloride', 'bicarbonate', 'anion gap', 'hemoglobin', 'hematocrit',
    'platelet', 'wbc', 'inr', 'pt', 'ptt', 'magnesium', 'phosphate',
    'bilirubin', 'albumin'
]

stay_ids_list = cohort_filtered['stay_id'].tolist()
BATCH_SIZE_VITALS = 500
labs_data = []
vitals_data = []
bp_data = []
meds_data = []

for i in tqdm(range(0, len(stay_ids_list), BATCH_SIZE_VITALS), desc="  Batches"):
    batch_ids = stay_ids_list[i:i + BATCH_SIZE_VITALS]

    batch_labs = loader.load_labs(lab_names, batch_ids)
    if len(batch_labs) > 0:
        labs_data.append(batch_labs)

    batch_vitals = loader.load_vitals(batch_ids)
    if len(batch_vitals) > 0:
        vitals_data.append(batch_vitals)

    batch_bp = loader.load_vitals_bp(batch_ids)
    if len(batch_bp) > 0:
        bp_data.append(batch_bp)

    vasopressor_names = ['norepinephrine', 'epinephrine', 'vasopressin', 'dopamine', 'phenylephrine', 'dobutamine']
    batch_meds = loader.load_medications(vasopressor_names, batch_ids)
    if len(batch_meds) > 0:
        meds_data.append(batch_meds)

labs = pd.concat(labs_data, ignore_index=True) if labs_data else pd.DataFrame()
vitals = pd.concat(vitals_data, ignore_index=True) if vitals_data else pd.DataFrame()
bp = pd.concat(bp_data, ignore_index=True) if bp_data else pd.DataFrame()
meds = pd.concat(meds_data, ignore_index=True) if meds_data else pd.DataFrame()

if len(meds) > 0:
    meds['time_hour'] = np.floor(meds['med_time_hours']).astype(int)
    meds['vasopressor_any'] = 1

if len(labs) > 0:
    labs['time_hour'] = np.floor(labs['lab_time_hours']).astype(int)
    labs_daily = labs.groupby(['patientunitstayid', 'time_hour', 'labname'])['labresult'].agg(['min', 'max', 'mean']).reset_index()
    labs_daily = labs_daily.pivot_table(
        index=['patientunitstayid', 'time_hour'],
        columns='labname',
        values=['min', 'max', 'mean'],
        aggfunc='first'
    )
    labs_daily.columns = ['_'.join(col).replace('result', '') for col in labs_daily.columns]
    labs_daily = labs_daily.reset_index().rename(columns={'patientunitstayid': 'stay_id'})
else:
    labs_daily = pd.DataFrame()

if len(vitals) > 0:
    vitals['time_hour'] = np.floor(vitals['vital_time_hours']).astype(int)
    vitals_daily = vitals.groupby(['patientunitstayid', 'time_hour']).agg({
        'heart_rate': ['min', 'max', 'mean'],
        'respiratory_rate': ['min', 'max', 'mean'],
        'temperature': ['min', 'max', 'mean'],
        'o2_sat': ['min', 'max', 'mean']
    })
    vitals_daily.columns = ['_'.join(col).strip() for col in vitals_daily.columns]
    vitals_daily = vitals_daily.reset_index().rename(columns={'patientunitstayid': 'stay_id'})
else:
    vitals_daily = pd.DataFrame()

if len(bp) > 0:
    bp['time_hour'] = np.floor(bp['vital_time_hours']).astype(int)
    bp_daily = bp.groupby(['patientunitstayid', 'time_hour']).agg({
        'systolic': ['min', 'max', 'mean'],
        'diastolic': ['min', 'max', 'mean'],
        'mean_bp': ['min', 'max', 'mean']
    })
    bp_daily.columns = ['_'.join(col).strip() for col in bp_daily.columns]
    bp_daily = bp_daily.reset_index().rename(columns={'patientunitstayid': 'stay_id'})
else:
    bp_daily = pd.DataFrame()

if len(meds) > 0:
    meds_daily = meds.groupby(['patientunitstayid', 'time_hour'])['vasopressor_any'].max().reset_index()
    meds_daily = meds_daily.rename(columns={'patientunitstayid': 'stay_id'})
else:
    meds_daily = pd.DataFrame()

if len(labs_daily) > 0:
    daily_features = labs_daily
    if len(vitals_daily) > 0:
        daily_features = daily_features.merge(vitals_daily, on=['stay_id', 'time_hour'], how='outer')
    if len(bp_daily) > 0:
        daily_features = daily_features.merge(bp_daily, on=['stay_id', 'time_hour'], how='outer')
else:
    daily_features = vitals_daily
    if len(bp_daily) > 0:
        daily_features = daily_features.merge(bp_daily, on=['stay_id', 'time_hour'], how='outer')

if len(meds_daily) > 0:
    daily_features = daily_features.merge(meds_daily, on=['stay_id', 'time_hour'], how='left')
    daily_features['vasopressor_any'] = daily_features['vasopressor_any'].fillna(0)

# Merge demographics
cohort_final = cohort_filtered[['stay_id', 'age', 'gender', 'hospital_expire_flag']]
prediction_base = daily_features.merge(cohort_final, on='stay_id', how='left')

baseline_lactate = lactate_ts[['stay_id', 'baseline_lactate']].drop_duplicates()
baseline_hr = hr_ts[['stay_id', 'baseline_heartrate']].drop_duplicates()
baseline_sbp = sbp_ts[['stay_id', 'baseline_systolic']].drop_duplicates()

prediction_base = prediction_base.merge(baseline_lactate, on='stay_id', how='left')
prediction_base = prediction_base.merge(baseline_hr, on='stay_id', how='left')
prediction_base = prediction_base.merge(baseline_sbp, on='stay_id', how='left')

# Define outcome
failure_events = []
excluded = {'already_failure': 0, 'no_future_data': 0, 'ambiguous': 0}

for stay_id, grp in prediction_base.groupby('stay_id'):
    grp = grp.sort_values('time_hour')
    for i in range(len(grp)):
        current_time = grp.iloc[i]['time_hour']
        current_bp = grp.iloc[i].get('mean_bp_mean', np.nan)
        current_lactate = grp.iloc[i].get('mean_lactate', np.nan)

        current_vasopressor = grp.iloc[i].get('vasopressor_any', 0) == 1

        if pd.isna(current_bp) or pd.isna(current_lactate):
            excluded['ambiguous'] += 1
            continue

        if current_lactate >= LACTATE_THRESHOLD and (current_bp <= MAP_THRESHOLD or current_vasopressor):
            excluded['already_failure'] += 1
            continue

        prediction_start = current_time + PREDICTION_GAP_HOURS
        prediction_end = current_time + PREDICTION_GAP_HOURS + PREDICTION_WINDOW_HOURS

        future_grp = grp[(grp['time_hour'] >= prediction_start) & (grp['time_hour'] <= prediction_end)]
        if len(future_grp) == 0:
            excluded['no_future_data'] += 1
            continue

        if 'mean_bp_mean' not in future_grp.columns or 'mean_lactate' not in future_grp.columns:
            excluded['ambiguous'] += 1
            continue

        future_hypotension = (future_grp['mean_bp_mean'] <= MAP_THRESHOLD).any()
        future_lactate = (future_grp['mean_lactate'] >= LACTATE_THRESHOLD).any()

        future_vasopressor = future_grp.get('vasopressor_any', pd.Series(dtype=int)).max() == 1

        target = int(future_lactate and (future_hypotension or future_vasopressor))
        row = grp.iloc[i].to_dict()
        row['target_circulatory_failure'] = target
        failure_events.append(row)

prediction_dataset = pd.DataFrame(failure_events)
print(f"  Excluded (already failure): {excluded['already_failure']:,}")
print(f"  Excluded (no future data): {excluded['no_future_data']:,}")

positive_first = prediction_dataset[prediction_dataset['target_circulatory_failure'] == 1].groupby('stay_id').first().reset_index()
all_negatives = prediction_dataset[prediction_dataset['target_circulatory_failure'] == 0]
prediction_dataset = pd.concat([all_negatives, positive_first], ignore_index=True).sort_values(['stay_id', 'time_hour']).reset_index(drop=True)

# Features: forward-filled within stay (labels above use the raw hourly values),
# then columns with >70% missing values dropped
features_filled = ffill_within_stay(prediction_base.sort_values(['stay_id', 'time_hour']), 'stay_id')
prediction_dataset = prediction_dataset[['stay_id', 'time_hour', 'target_circulatory_failure']].merge(
    features_filled, on=['stay_id', 'time_hour'], how='left'
)
missing_pct = prediction_dataset.isnull().mean()
keep_cols = {'stay_id', 'time_hour', 'target_circulatory_failure', 'age', 'gender', 'hospital_expire_flag',
             'mean_lactate', 'mean_bp_mean'}
cols_to_drop = [c for c in missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index if c not in keep_cols]
prediction_dataset = prediction_dataset.drop(columns=cols_to_drop)
print(f"  Dropped {len(cols_to_drop)} columns with >70% missing values")

prediction_dataset.to_csv(OUTPUT_DIR / "circulatory_failure_prediction_dataset.csv", index=False)
print(f"✓ Saved {len(prediction_dataset):,} rows")
print(f"   Positive events: {prediction_dataset['target_circulatory_failure'].sum():,} ({100*prediction_dataset['target_circulatory_failure'].mean():.1f}%)")

loader.close()
print("\n✓ Circulatory failure preprocessing complete!")
