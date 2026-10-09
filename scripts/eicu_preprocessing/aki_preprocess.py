"""
eICU AKI Preprocessing Script

Mirrors MIMIC AKI preprocessing:
1. Load AKI cohort (age 18-90, LOS≥2d)
2. Extract creatinine time series
3. Calculate baseline creatinine (first measurement)
4. Filter patients with ≥5 creatinine measurements
5. Load vitals/labs and aggregate daily
6. Define AKI Stage 3 outcome (gap=0.5d, window=4d)
   - Creatinine ≥3× baseline OR ≥4 mg/dL + rise ≥0.5
   - OR dialysis initiation
7. Forward-fill imputation
8. Drop >70% missing columns
9. First positive event only per patient
10. Save prediction dataset

Usage:
    python scripts/eicu_preprocessing/aki_preprocess.py
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
PREDICTION_WINDOW_DAYS = 4.0   # Predict AKI Stage 3 within 4 days after gap
LOOKBACK_DAYS = 7.0            # Use 7 days of historical data for features
MIN_CREAT_MEASUREMENTS = MIN_TARGET_MEASUREMENTS  # Minimum measurements of the target clinical parameter
_TRAJ_DATA_ROOT = os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet")
OUTPUT_PATH = os.path.join(_TRAJ_DATA_ROOT, "eicu", "aki", "aki_prediction_dataset.csv")

MAX_PATIENTS = None            # Process full cohort

print("=" * 80)
print("eICU AKI Preprocessing Pipeline")
print("=" * 80)

# Load cohort
print("\n[1/10] Loading AKI cohort...")
loader = EICULoader()
cohort = loader.load_aki_cohort()
if MAX_PATIENTS is not None:
    cohort = cohort.head(MAX_PATIENTS)
print(f"  Total ICU stays: {len(cohort):,}")
print(f"  Mean age: {cohort['age'].mean():.1f}")
print(f"  Mortality: {cohort['hospital_expire_flag'].mean():.1%}")

# Load creatinine time series for all patients
print("\n[2/10] Loading creatinine time series...")
all_ids = cohort['patientunitstayid'].tolist()

# Process in batches to avoid memory issues
BATCH_SIZE = 1000
creatinine_list = []

for i in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="  Batches"):
    batch_ids = all_ids[i:i+BATCH_SIZE]
    batch_labs = loader.load_labs(['creatinine'], batch_ids)
    if len(batch_labs) > 0:
        batch_labs['time_days'] = batch_labs['lab_time_hours'] / 24.0
        creatinine_list.append(batch_labs)

creatinine_ts = pd.concat(creatinine_list, ignore_index=True)
creatinine_ts = creatinine_ts.rename(columns={
    'patientunitstayid': 'stay_id',
    'labresult': 'creatinine'
})
print(f"  Total creatinine measurements: {len(creatinine_ts):,}")
print(f"  Patients with creatinine: {creatinine_ts['stay_id'].nunique():,}")

# Calculate baseline creatinine (first 24h)
print("\n[3/10] Calculating baseline creatinine (first 24h)...")
baseline_window = creatinine_ts[creatinine_ts['time_days'] <= 1.0]
baseline_creat = (
    baseline_window.sort_values('time_days')
    .groupby('stay_id')['creatinine']
    .first()
    .reset_index()
    .rename(columns={'creatinine': 'baseline_creatinine'})
)

print(f"  Patients with baseline: {len(baseline_creat):,}")
print(f"  Mean baseline: {baseline_creat['baseline_creatinine'].mean():.2f} mg/dL")

# Filter patients with minimum measurements
print(f"\n[4/10] Filtering patients with ≥{MIN_CREAT_MEASUREMENTS} measurements...")
measurement_counts = creatinine_ts.groupby('stay_id').size()
valid_patients = measurement_counts[measurement_counts >= MIN_CREAT_MEASUREMENTS].index.tolist()
print(f"  Patients with ≥{MIN_CREAT_MEASUREMENTS} measurements: {len(valid_patients):,}")

# Filter cohort and merge baseline
cohort_filtered = cohort[cohort['patientunitstayid'].isin(valid_patients)].copy()
cohort_filtered = cohort_filtered.rename(columns={'patientunitstayid': 'stay_id'})
cohort_filtered = cohort_filtered.merge(baseline_creat, on='stay_id', how='inner')
print(f"  Final cohort size: {len(cohort_filtered):,}")

# Prepare for batch loading
stay_ids_list = cohort_filtered['stay_id'].tolist()
BATCH_SIZE_VITALS = 500

# Load dialysis events (best-effort via infusionDrug)
print(f"\n[5/10] Loading dialysis events...")
dialysis_terms = [
    'dialysis', 'hemodialysis', 'haemodialysis', 'hemofiltration',
    'cvvh', 'cvvhd', 'cvvhdf', 'crrt', 'rrt', 'renal replacement'
]
dialysis_list = []
for i in tqdm(range(0, len(stay_ids_list), BATCH_SIZE_VITALS), desc="  Dialysis batches"):
    batch_ids = stay_ids_list[i:i+BATCH_SIZE_VITALS]
    batch_dialysis = loader.load_medications(dialysis_terms, batch_ids, match_mode="contains")
    if len(batch_dialysis) > 0:
        dialysis_list.append(batch_dialysis)

if len(dialysis_list) > 0:
    dialysis_events = pd.concat(dialysis_list, ignore_index=True)
    dialysis_events['time_days'] = dialysis_events['med_time_hours'] / 24.0
    dialysis_events = dialysis_events.rename(columns={'patientunitstayid': 'stay_id'})
else:
    dialysis_events = pd.DataFrame(columns=['stay_id', 'time_days'])

# Load vitals and labs for feature generation
print(f"\n[5/10] Loading vitals and labs for daily aggregation...")
print("  (This will take several minutes...)")

# Common lab names in eICU (matching MIMIC feature set)
lab_names = [
    'creatinine', 'bun', 'glucose', 'potassium', 'sodium', 'chloride',
    'bicarbonate', 'anion gap', 'hemoglobin', 'hematocrit', 'platelet',
    'wbc', 'inr', 'pt', 'ptt', 'magnesium', 'phosphate'
]

 # Prepare for batch loading

# Load all at once in batches for efficiency
print(f"  Loading vitals/labs for {len(stay_ids_list)} patients in batches...")
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

# Per-patient aggregation removed; using vectorized aggregation below.
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

# Merge all daily features
if len(labs_daily) > 0:
    daily_features = labs_daily
    if len(vitals_daily) > 0:
        daily_features = daily_features.merge(vitals_daily, on=['stay_id', 'time_day'], how='outer')
    if len(bp_daily) > 0:
        daily_features = daily_features.merge(bp_daily, on=['stay_id', 'time_day'], how='outer')
else:
    daily_features = vitals_daily if len(vitals_daily) > 0 else bp_daily if len(bp_daily) > 0 else pd.DataFrame()

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
    cohort_filtered[['stay_id', 'age', 'gender', 'hospital_expire_flag', 'baseline_creatinine']],
    on='stay_id',
    how='left'
)

# Define AKI Stage 3 outcome
print(f"\n[8/10] Defining AKI Stage 3 outcome (gap={PREDICTION_GAP_DAYS}d, window={PREDICTION_WINDOW_DAYS}d)...")
aki_events = []
excluded_counts = {'already_aki': 0, 'no_future_data': 0}

for stay_id, grp in tqdm(daily_features.groupby('stay_id'), desc="  Patients"):
    grp = grp.sort_values('time_day')
    baseline = grp['baseline_creatinine'].iloc[0]
    
    # Get creatinine time series for this patient
    patient_creat = creatinine_ts[creatinine_ts['stay_id'] == stay_id].copy()
    
    for i in range(len(grp)):
        current_time = grp.iloc[i]['time_day']
        
        # Get current creatinine (use mean from daily aggregation if available)
        creat_col = [c for c in grp.columns if 'creatinine' in c.lower() and 'mean' in c]
        if len(creat_col) > 0:
            current_creat = grp.iloc[i][creat_col[0]]
        else:
            # Fallback to time series
            current_creat_ts = patient_creat[patient_creat['time_days'] <= current_time]
            if len(current_creat_ts) > 0:
                current_creat = current_creat_ts['creatinine'].iloc[-1]
            else:
                current_creat = baseline
        
        # Skip if already in AKI Stage 3
        if pd.notna(current_creat) and (current_creat >= 3 * baseline or current_creat >= 4.0):
            excluded_counts['already_aki'] += 1
            continue
        
        # Define prediction window (+1 day shift to match end-of-day prediction)
        prediction_start = current_time + PREDICTION_GAP_DAYS + 1
        prediction_end = current_time + PREDICTION_GAP_DAYS + PREDICTION_WINDOW_DAYS + 1
        
        future_window = patient_creat[
            (patient_creat['time_days'] >= prediction_start) &
            (patient_creat['time_days'] <= prediction_end)
        ]
        
        if len(future_window) == 0:
            excluded_counts['no_future_data'] += 1
            continue
        
        # Check AKI Stage 3 criteria
        aki_stage3 = (
            (future_window['creatinine'] >= 3 * baseline).any() |
            ((future_window['creatinine'] >= 4.0) &
             (future_window['creatinine'] - current_creat >= 0.5)).any()
        )

        dialysis_in_window = False
        if len(dialysis_events) > 0:
            dialysis_window = dialysis_events[
                (dialysis_events['stay_id'] == stay_id) &
                (dialysis_events['time_days'] >= prediction_start) &
                (dialysis_events['time_days'] <= prediction_end)
            ]
            dialysis_in_window = len(dialysis_window) > 0
        
        # Create prediction row
        row = grp.iloc[i].to_dict()
        row['target_aki_stage3'] = int(aki_stage3 or dialysis_in_window)
        row['current_creatinine'] = current_creat
        row['creat_fold_change'] = current_creat / baseline if baseline > 0 else np.nan
        aki_events.append(row)

prediction_dataset = pd.DataFrame(aki_events)
print(f"  Total prediction windows: {len(prediction_dataset):,}")
print(f"  Excluded (already AKI): {excluded_counts['already_aki']:,}")
print(f"  Excluded (no future data): {excluded_counts['no_future_data']:,}")

# First positive event only per patient
print("\n[9/10] Filtering to first positive event per patient...")
positive_first = prediction_dataset[prediction_dataset['target_aki_stage3'] == 1].groupby('stay_id').first().reset_index()
all_negatives = prediction_dataset[prediction_dataset['target_aki_stage3'] == 0]
prediction_dataset = pd.concat([all_negatives, positive_first], ignore_index=True).sort_values(['stay_id', 'time_day']).reset_index(drop=True)

# Save raw creatinine time series ONLY for patients in final prediction dataset
print(f"\n[9.5/10] Saving raw creatinine time series...")
final_stay_ids = prediction_dataset['stay_id'].unique()
creatinine_ts_filtered = creatinine_ts[creatinine_ts['stay_id'].isin(final_stay_ids)].copy()
# Add time_day column (integer day for windowing in trajectory computation)
creatinine_ts_filtered['time_day'] = creatinine_ts_filtered['time_days'].astype(int)
creatinine_ts_output = os.path.join(_TRAJ_DATA_ROOT, "eicu", "aki", "creatinine_timeseries.csv")
os.makedirs(os.path.dirname(creatinine_ts_output), exist_ok=True)
creatinine_ts_filtered.to_csv(creatinine_ts_output, index=False)
print(f"  ✓ Saved raw time series: {len(creatinine_ts_filtered):,} measurements for {len(final_stay_ids):,} patients")

print(f"\n📊 Final Dataset Summary:")
print(f"  Total samples: {len(prediction_dataset):,}")
print(f"  Unique patients: {prediction_dataset['stay_id'].nunique():,}")
print(f"  Positive events: {prediction_dataset['target_aki_stage3'].sum():,} ({100*prediction_dataset['target_aki_stage3'].mean():.1f}%)")
print(f"  Features: {len(prediction_dataset.columns)}")

# Save
print(f"\n[10/10] Saving to {OUTPUT_PATH}...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
prediction_dataset.to_csv(OUTPUT_PATH, index=False)
print(f"  ✓ Saved {len(prediction_dataset):,} rows")

loader.close()
print("\n✓ AKI preprocessing complete!")
