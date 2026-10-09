"""
eICU Ventilator Weaning Preprocessing Script

Mirrors MIMIC ventilator preprocessing:
1. Load ventilator cohort (mechanically ventilated patients)
2. Extract P/F ratio time series (PaO2/FiO2)
3. Calculate baseline P/F ratio (first 24h)
4. Filter patients with ≥5 P/F ratio measurements
5. Load vitals/labs and aggregate daily
6. Define weaning success outcome (gap=0.5d, window=4d)
   - P/F ratio ≥300 AND survived
7. Forward-fill imputation
8. Drop >70% missing columns
9. First positive event only per patient
10. Save prediction dataset

Usage:
    python scripts/eicu_preprocessing/ventilator_preprocess.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import pandas as pd
import numpy as np
from eicu_loader import EICULoader, ffill_within_stay, MIN_TARGET_MEASUREMENTS, MISSINGNESS_THRESHOLD
from tqdm import tqdm

# Configuration
PREDICTION_GAP_DAYS = 0.5      # Don't predict events within 12h
PREDICTION_WINDOW_DAYS = 2.0   # Predict weaning success within 2 days after gap
LOOKBACK_DAYS = 3.0            # Use 3 days of historical data for features
MIN_PF_MEASUREMENTS = MIN_TARGET_MEASUREMENTS  # Minimum measurements of the target clinical parameter
_TRAJ_DATA_ROOT = os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet")
OUTPUT_PATH = os.path.join(_TRAJ_DATA_ROOT, "eicu", "ventilator", "ventilator_prediction_dataset.csv")

MAX_PATIENTS = None            # Process full cohort

print("=" * 80)
print("eICU Ventilator Weaning Preprocessing Pipeline")
print("=" * 80)

# Load cohort
print("\n[1/10] Loading ventilator cohort...")
loader = EICULoader()
cohort = loader.load_ventilator_cohort()
if MAX_PATIENTS is not None:
    cohort = cohort.head(MAX_PATIENTS)
print(f"  Total ICU stays: {len(cohort):,}")
print(f"  Mean age: {cohort['age'].mean():.1f}")
print(f"  Mortality: {cohort['hospital_expire_flag'].mean():.1%}")

# Load PaO2 and FiO2 to calculate P/F ratio
print("\n[2/10] Loading PaO2/FiO2 for P/F ratio calculation...")
all_ids = cohort['patientunitstayid'].tolist()

# Process in batches to avoid memory issues
BATCH_SIZE = 1000
pao2_list = []
fio2_list = []

for i in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="  Batches"):
    batch_ids = all_ids[i:i+BATCH_SIZE]
    
    # Load PaO2 (arterial oxygen)
    batch_pao2 = loader.load_labs(['pao2', 'paO2'], batch_ids)
    if len(batch_pao2) > 0:
        batch_pao2['time_days'] = batch_pao2['lab_time_hours'] / 24.0
        pao2_list.append(batch_pao2)
    
    # Load FiO2 (fraction inspired oxygen)
    batch_fio2 = loader.load_labs(['fio2', 'FiO2'], batch_ids)
    if len(batch_fio2) > 0:
        batch_fio2['time_days'] = batch_fio2['lab_time_hours'] / 24.0
        fio2_list.append(batch_fio2)

if len(pao2_list) == 0 or len(fio2_list) == 0:
    print("  ERROR: Insufficient PaO2/FiO2 measurements found!")
    loader.close()
    sys.exit(1)

pao2_ts = pd.concat(pao2_list, ignore_index=True)
fio2_ts = pd.concat(fio2_list, ignore_index=True)

pao2_ts = pao2_ts.rename(columns={
    'patientunitstayid': 'stay_id',
    'labresult': 'pao2'
})
fio2_ts = fio2_ts.rename(columns={
    'patientunitstayid': 'stay_id',
    'labresult': 'fio2'
})

# Merge PaO2 and FiO2 by timestamp (±1 hour tolerance)
print("  Merging PaO2 and FiO2 measurements...")
merged_list = []

for stay_id in tqdm(cohort['patientunitstayid'].unique(), desc="  Patients"):
    pao2_patient = pao2_ts[pao2_ts['stay_id'] == stay_id]
    fio2_patient = fio2_ts[fio2_ts['stay_id'] == stay_id]
    
    if len(pao2_patient) == 0 or len(fio2_patient) == 0:
        continue
    
    # Merge by approximate time (within 1 hour)
    for _, pao2_row in pao2_patient.iterrows():
        time_match = fio2_patient[
            abs(fio2_patient['time_days'] - pao2_row['time_days']) <= (1/24.0)  # 1 hour = 1/24 days
        ]
        if len(time_match) > 0:
            fio2_val = time_match.iloc[0]['fio2']
            # FiO2 should be between 0.21 and 1.0 (or 21-100 if percentage)
            if fio2_val > 1.0:
                fio2_val = fio2_val / 100.0
            if 0.21 <= fio2_val <= 1.0 and pao2_row['pao2'] > 0:
                pf_ratio = pao2_row['pao2'] / fio2_val
                merged_list.append({
                    'stay_id': stay_id,
                    'time_days': pao2_row['time_days'],
                    'pao2': pao2_row['pao2'],
                    'fio2': fio2_val,
                    'pf_ratio': pf_ratio
                })

pf_ts = pd.DataFrame(merged_list)

if len(pf_ts) == 0:
    print("  ERROR: No valid P/F ratio pairs found!")
    loader.close()
    sys.exit(1)

# Filter physiologically plausible P/F ratios
pf_ts = pf_ts[(pf_ts['pf_ratio'] >= 50) & (pf_ts['pf_ratio'] <= 600)]
if len(pf_ts) == 0:
    print("  ERROR: No valid P/F ratio pairs after filtering!")
    loader.close()
    sys.exit(1)

print(f"  Total P/F ratio measurements: {len(pf_ts):,}")
print(f"  Patients with P/F ratio: {pf_ts['stay_id'].nunique():,}")
print(f"  Mean P/F ratio: {pf_ts['pf_ratio'].mean():.1f}")

# Calculate baseline P/F ratio (first 24h)
print("\n[3/10] Calculating baseline P/F ratio (first 24h)...")
baseline_pf = (
    pf_ts[pf_ts['time_days'] <= 1.0]
    .sort_values('time_days')
    .groupby('stay_id')['pf_ratio']
    .first()
    .reset_index()
    .rename(columns={'pf_ratio': 'baseline_pf_ratio'})
)

print(f"  Patients with baseline: {len(baseline_pf):,}")
print(f"  Mean baseline P/F: {baseline_pf['baseline_pf_ratio'].mean():.1f}")

# Filter patients with minimum measurements
print(f"\n[4/10] Filtering patients with ≥{MIN_PF_MEASUREMENTS} measurements...")
measurement_counts = pf_ts.groupby('stay_id').size()
valid_patients = measurement_counts[measurement_counts >= MIN_PF_MEASUREMENTS].index.tolist()
print(f"  Patients with ≥{MIN_PF_MEASUREMENTS} measurements: {len(valid_patients):,}")

# Filter cohort and merge baseline
cohort_filtered = cohort[cohort['patientunitstayid'].isin(valid_patients)].copy()
cohort_filtered = cohort_filtered.rename(columns={'patientunitstayid': 'stay_id'})
cohort_filtered = cohort_filtered.merge(baseline_pf, on='stay_id', how='inner')
print(f"  Final cohort size: {len(cohort_filtered):,}")

# Load vitals and labs for feature generation
print(f"\n[5/10] Loading vitals and labs for daily aggregation...")
print("  (This will take several minutes...)")

# Common lab names in eICU (matching MIMIC feature set)
lab_names = [
    'pao2', 'paco2', 'ph', 'fio2', 'peep', 'tidal volume',
    'creatinine', 'bun', 'glucose', 'potassium', 'sodium',
    'chloride', 'bicarbonate', 'hemoglobin', 'hematocrit',
    'platelet', 'wbc', 'lactate'
]

# Prepare for batch loading
stay_ids_list = cohort_filtered['stay_id'].tolist()
print(f"  Loading vitals/labs for {len(stay_ids_list)} patients in batches...")

BATCH_SIZE_VITALS = 500
labs_data = []
vitals_data = []
bp_data = []

for i in tqdm(range(0, len(stay_ids_list), BATCH_SIZE_VITALS), desc="  Batches"):
    batch_ids = stay_ids_list[i:i+BATCH_SIZE_VITALS]
    
    # Load labs for batch
    batch_labs = loader.load_labs(lab_names, batch_ids)
    if len(batch_labs) > 0:
        labs_data.append(batch_labs)
    
    # Load vitals for batch
    batch_vitals = loader.load_vitals(batch_ids)
    if len(batch_vitals) > 0:
        vitals_data.append(batch_vitals)
    
    # Load BP for batch
    batch_bp = loader.load_vitals_bp(batch_ids)
    if len(batch_bp) > 0:
        bp_data.append(batch_bp)

# Concat all batches
if len(labs_data) > 0:
    labs = pd.concat(labs_data, ignore_index=True)
else:
    labs = pd.DataFrame()

if len(vitals_data) > 0:
    vitals = pd.concat(vitals_data, ignore_index=True)
else:
    vitals = pd.DataFrame()

if len(bp_data) > 0:
    bp = pd.concat(bp_data, ignore_index=True)
else:
    bp = pd.DataFrame()

# Vectorized aggregation across all patients
print("  Aggregating all measurements by day (vectorized)...")

if len(labs) > 0:
    labs['time_day'] = (labs['lab_time_hours'] / 24.0).astype(int)
    labs_daily = labs.groupby(['patientunitstayid', 'time_day', 'labname'])['labresult'].agg(['min', 'max', 'mean']).reset_index()
    labs_daily = labs_daily.pivot_table(
        index=['patientunitstayid', 'time_day'],
        columns='labname',
        values=['min', 'max', 'mean'],
        aggfunc='first'
    )
    labs_daily.columns = ['_'.join(col).replace('result', '') for col in labs_daily.columns]
    labs_daily = labs_daily.reset_index()
    labs_daily = labs_daily.rename(columns={'patientunitstayid': 'stay_id'})
else:
    labs_daily = pd.DataFrame()

if len(vitals) > 0:
    vitals['time_day'] = (vitals['vital_time_hours'] / 24.0).astype(int)
    vitals_daily = vitals.groupby(['patientunitstayid', 'time_day']).agg({
        'heart_rate': ['min', 'max', 'mean'],
        'respiratory_rate': ['min', 'max', 'mean'],
        'temperature': ['min', 'max', 'mean'],
        'o2_sat': ['min', 'max', 'mean']
    })
    vitals_daily.columns = ['_'.join(col).strip() for col in vitals_daily.columns]
    vitals_daily = vitals_daily.reset_index()
    vitals_daily = vitals_daily.rename(columns={'patientunitstayid': 'stay_id'})
else:
    vitals_daily = pd.DataFrame()

if len(bp) > 0:
    bp['time_day'] = (bp['vital_time_hours'] / 24.0).astype(int)
    bp_daily = bp.groupby(['patientunitstayid', 'time_day']).agg({
        'systolic': ['min', 'max', 'mean'],
        'diastolic': ['min', 'max', 'mean'],
        'mean_bp': ['min', 'max', 'mean']
    })
    bp_daily.columns = ['_'.join(col).strip() for col in bp_daily.columns]
    bp_daily = bp_daily.reset_index()
    bp_daily = bp_daily.rename(columns={'patientunitstayid': 'stay_id'})
else:
    bp_daily = pd.DataFrame()

# Aggregate P/F ratio by day
pf_ts['time_day'] = pf_ts['time_days'].astype(int)
pf_daily = pf_ts.groupby(['stay_id', 'time_day']).agg({
    'pf_ratio': ['min', 'max', 'mean']
}).reset_index()
pf_daily.columns = ['stay_id', 'time_day', 'pf_ratio_min', 'pf_ratio_max', 'pf_ratio_mean']

# Merge all daily features
if len(labs_daily) > 0:
    daily_features = labs_daily
    if len(vitals_daily) > 0:
        daily_features = daily_features.merge(vitals_daily, on=['stay_id', 'time_day'], how='outer')
    if len(bp_daily) > 0:
        daily_features = daily_features.merge(bp_daily, on=['stay_id', 'time_day'], how='outer')
    daily_features = daily_features.merge(pf_daily, on=['stay_id', 'time_day'], how='outer')
else:
    daily_features = vitals_daily if len(vitals_daily) > 0 else bp_daily if len(bp_daily) > 0 else pf_daily

print(f"  Generated {len(daily_features):,} daily feature rows for {daily_features['stay_id'].nunique():,} patients")

# Continue with outcome definition
daily_features_list = [daily_features] if len(daily_features) > 0 else []

# Flatten results
if len(daily_features_list) > 0:
    daily_features = pd.concat(daily_features_list, ignore_index=True)
    print(f"\n  Daily feature rows: {len(daily_features):,}")
    print(f"  Patients: {daily_features['stay_id'].nunique():,}")
else:
    print("  ERROR: No daily features extracted!")
    loader.close()
    sys.exit(1)

# Forward-fill imputation within each patient
print("\n[6/10] Forward-fill imputation...")
daily_features = daily_features.sort_values(['stay_id', 'time_day'])
daily_features = ffill_within_stay(daily_features, 'stay_id')

# Drop columns with >70% missing
print("\n[7/10] Dropping columns with >70% missing...")
missing_pct = daily_features.isnull().mean()
cols_to_drop = missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index.tolist()
if 'stay_id' in cols_to_drop:
    cols_to_drop.remove('stay_id')
if 'time_day' in cols_to_drop:
    cols_to_drop.remove('time_day')
print(f"  Columns dropped: {len(cols_to_drop)}")
daily_features = daily_features.drop(columns=cols_to_drop)

# Merge cohort info
daily_features = daily_features.merge(
    cohort_filtered[['stay_id', 'age', 'gender', 'hospital_expire_flag', 'baseline_pf_ratio']],
    on='stay_id',
    how='left'
)

# Define weaning success outcome (P/F ratio ≥300 AND survived)
print(f"\n[8/10] Defining weaning success outcome (gap={PREDICTION_GAP_DAYS}d, window={PREDICTION_WINDOW_DAYS}d)...")
weaning_events = []
excluded_counts = {'already_weaned': 0, 'no_future_data': 0}

for stay_id, grp in tqdm(daily_features.groupby('stay_id'), desc="  Patients"):
    grp = grp.sort_values('time_day')
    baseline = grp['baseline_pf_ratio'].iloc[0]
    mortality_flag = grp['hospital_expire_flag'].iloc[0]
    
    # Get P/F ratio time series for this patient
    patient_pf = pf_ts[pf_ts['stay_id'] == stay_id].copy()
    
    for i in range(len(grp)):
        current_time = grp.iloc[i]['time_day']
        
        # Get current P/F ratio (use mean from daily aggregation if available)
        if 'pf_ratio_mean' in grp.columns:
            current_pf = grp.iloc[i]['pf_ratio_mean']
        else:
            # Fallback to time series
            current_pf_ts = patient_pf[patient_pf['time_days'] <= current_time]
            if len(current_pf_ts) > 0:
                current_pf = current_pf_ts['pf_ratio'].iloc[-1]
            else:
                current_pf = baseline
        
        # Skip index days with P/F already >= 400 (as in MIMIC and the thesis Methods)
        if pd.notna(current_pf) and current_pf >= 400.0:
            excluded_counts['already_weaned'] += 1
            continue
        
        # Define prediction window (+1 day shift to match end-of-day prediction)
        prediction_start = current_time + PREDICTION_GAP_DAYS + 1
        prediction_end = current_time + PREDICTION_GAP_DAYS + PREDICTION_WINDOW_DAYS + 1
        
        future_window = patient_pf[
            (patient_pf['time_days'] >= prediction_start) &
            (patient_pf['time_days'] <= prediction_end)
        ]
        
        if len(future_window) == 0:
            excluded_counts['no_future_data'] += 1
            continue
        
        # Check weaning success: P/F ratio ≥300 AND survived
        weaning_success = (
            (future_window['pf_ratio'] >= 300.0).any() and
            mortality_flag == 0
        )
        
        # Create prediction row
        row = grp.iloc[i].to_dict()
        row['target_weaning_success'] = int(weaning_success)
        row['current_pf_ratio'] = current_pf
        row['pf_ratio_change'] = current_pf - baseline
        weaning_events.append(row)

prediction_dataset = pd.DataFrame(weaning_events)
print(f"  Total prediction windows: {len(prediction_dataset):,}")
print(f"  Excluded (already weaned): {excluded_counts['already_weaned']:,}")
print(f"  Excluded (no future data): {excluded_counts['no_future_data']:,}")

# First positive event only per patient
print("\n[9/10] Filtering to first positive event per patient...")
positive_first = prediction_dataset[prediction_dataset['target_weaning_success'] == 1].groupby('stay_id').first().reset_index()
all_negatives = prediction_dataset[prediction_dataset['target_weaning_success'] == 0]
prediction_dataset = pd.concat([all_negatives, positive_first], ignore_index=True).sort_values(['stay_id', 'time_day']).reset_index(drop=True)

# Save raw P/F ratio time series ONLY for patients in final prediction dataset
print(f"\n[9.5/10] Saving raw P/F ratio time series...")
final_stay_ids = prediction_dataset['stay_id'].unique()
pf_ts_filtered = pf_ts[pf_ts['stay_id'].isin(final_stay_ids)].copy()
# Add time_day column (integer day for windowing in trajectory computation)
pf_ts_filtered['time_day'] = pf_ts_filtered['time_days'].astype(int)
pf_ts_output = os.path.join(_TRAJ_DATA_ROOT, "eicu", "ventilator", "pf_ratio_timeseries.csv")
os.makedirs(os.path.dirname(pf_ts_output), exist_ok=True)
pf_ts_filtered.to_csv(pf_ts_output, index=False)
print(f"  ✓ Saved raw time series: {len(pf_ts_filtered):,} measurements for {len(final_stay_ids):,} patients")

print(f"\n📊 Final Dataset Summary:")
print(f"  Total samples: {len(prediction_dataset):,}")
print(f"  Unique patients: {prediction_dataset['stay_id'].nunique():,}")
print(f"  Positive events: {prediction_dataset['target_weaning_success'].sum():,} ({100*prediction_dataset['target_weaning_success'].mean():.1f}%)")
print(f"  Features: {len(prediction_dataset.columns)}")

# Save
print(f"\n[10/10] Saving to {OUTPUT_PATH}...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
prediction_dataset.to_csv(OUTPUT_PATH, index=False)
print(f"  ✓ Saved {len(prediction_dataset):,} rows")

loader.close()
print("\n✓ Ventilator preprocessing complete!")
