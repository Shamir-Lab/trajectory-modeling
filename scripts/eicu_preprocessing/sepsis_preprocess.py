"""
eICU Sepsis Preprocessing Script

Mirrors MIMIC sepsis preprocessing:
1. Load sepsis cohort (sepsis/septic shock patients)
2. Extract lactate time series
3. Calculate baseline lactate (first measurement)
4. Filter patients with ≥5 lactate measurements
5. Load vitals/labs and aggregate daily
6. Define septic shock outcome (gap=0.5d, window=4d)
   - Lactate ≥4 mmol/L + vasopressor OR mortality
7. Forward-fill imputation
8. Drop >70% missing columns
9. First positive event only per patient
10. Save prediction dataset

Usage:
    python scripts/eicu_preprocessing/sepsis_preprocess.py
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
PREDICTION_WINDOW_DAYS = 2.0   # Predict septic shock within 2 days after gap
LOOKBACK_DAYS = 3.0            # Use 3 days of historical data for features
MIN_LACTATE_MEASUREMENTS = MIN_TARGET_MEASUREMENTS  # Minimum measurements of the target clinical parameter
_TRAJ_DATA_ROOT = os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet")
OUTPUT_PATH = os.path.join(_TRAJ_DATA_ROOT, "eicu", "sepsis", "sepsis_prediction_dataset.csv")

MAX_PATIENTS = None            # Process full cohort

print("=" * 80)
print("eICU Sepsis Preprocessing Pipeline")
print("=" * 80)

# Load cohort
print("\n[1/10] Loading sepsis cohort...")
loader = EICULoader()
cohort = loader.load_sepsis_cohort()
if MAX_PATIENTS is not None:
    cohort = cohort.head(MAX_PATIENTS)
print(f"  Total ICU stays: {len(cohort):,}")
print(f"  Mean age: {cohort['age'].mean():.1f}")
print(f"  Mortality: {cohort['hospital_expire_flag'].mean():.1%}")

# Load lactate time series for all patients
print("\n[2/10] Loading lactate time series...")
all_ids = cohort['patientunitstayid'].tolist()

# Process in batches to avoid memory issues
BATCH_SIZE = 1000
lactate_list = []
wbc_list = []
platelet_list = []

for i in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="  Batches"):
    batch_ids = all_ids[i:i+BATCH_SIZE]
    # Lactate
    batch_lact = loader.load_labs(['lactate'], batch_ids)
    if len(batch_lact) > 0:
        batch_lact['time_days'] = batch_lact['lab_time_hours'] / 24.0
        lactate_list.append(batch_lact)
    # WBC
    batch_wbc = loader.load_labs(['WBC x 1000'], batch_ids)
    if len(batch_wbc) > 0:
        batch_wbc['time_days'] = batch_wbc['lab_time_hours'] / 24.0
        wbc_list.append(batch_wbc)
    # Platelets
    batch_plt = loader.load_labs(['platelets x 1000'], batch_ids)
    if len(batch_plt) > 0:
        batch_plt['time_days'] = batch_plt['lab_time_hours'] / 24.0
        platelet_list.append(batch_plt)

if len(lactate_list) == 0:
    print("  ERROR: No lactate measurements found!")
    loader.close()
    sys.exit(1)

lactate_ts = pd.concat(lactate_list, ignore_index=True)
lactate_ts = lactate_ts.rename(columns={
    'patientunitstayid': 'stay_id',
    'labresult': 'lactate'
})
wbc_ts = pd.concat(wbc_list, ignore_index=True) if len(wbc_list) > 0 else pd.DataFrame(columns=['stay_id','time_days','labresult'])
wbc_ts = wbc_ts.rename(columns={
    'patientunitstayid': 'stay_id',
    'labresult': 'wbc'
})
platelet_ts = pd.concat(platelet_list, ignore_index=True) if len(platelet_list) > 0 else pd.DataFrame(columns=['stay_id','time_days','labresult'])
platelet_ts = platelet_ts.rename(columns={
    'patientunitstayid': 'stay_id',
    'labresult': 'platelet'
})
print(f"  Total lactate measurements: {len(lactate_ts):,}")
print(f"  Patients with lactate: {lactate_ts['stay_id'].nunique():,}")
print(f"  Total WBC measurements: {len(wbc_ts):,}")
print(f"  Patients with WBC: {wbc_ts['stay_id'].nunique():,}")
print(f"  Total platelet measurements: {len(platelet_ts):,}")
print(f"  Patients with platelets: {platelet_ts['stay_id'].nunique():,}")

# Calculate baseline lactate (first 24h)
print("\n[3/10] Calculating baseline lactate (first 24h)...")
baseline_window = lactate_ts[lactate_ts['time_days'] <= 1.0]
baseline_lactate = (
    baseline_window.sort_values('time_days')
    .groupby('stay_id')['lactate']
    .first()
    .reset_index()
    .rename(columns={'lactate': 'baseline_lactate'})
)

print(f"  Patients with baseline: {len(baseline_lactate):,}")
print(f"  Mean baseline: {baseline_lactate['baseline_lactate'].mean():.2f} mmol/L")

# Filter patients with minimum measurements
print(f"\n[4/10] Filtering patients with ≥{MIN_LACTATE_MEASUREMENTS} measurements...")
measurement_counts = lactate_ts.groupby('stay_id').size()
valid_patients = measurement_counts[measurement_counts >= MIN_LACTATE_MEASUREMENTS].index.tolist()
print(f"  Patients with ≥{MIN_LACTATE_MEASUREMENTS} measurements: {len(valid_patients):,}")

# Filter cohort and merge baseline
cohort_filtered = cohort[cohort['patientunitstayid'].isin(valid_patients)].copy()
cohort_filtered = cohort_filtered.rename(columns={'patientunitstayid': 'stay_id'})
cohort_filtered = cohort_filtered.merge(baseline_lactate, on='stay_id', how='inner')
print(f"  Final cohort size: {len(cohort_filtered):,}")

# Load vitals and labs for feature generation
print(f"\n[5/10] Loading vitals and labs for daily aggregation...")
print("  (This will take several minutes...)")

# Common lab names in eICU (matching MIMIC feature set)
lab_names = [
    'lactate', 'creatinine', 'bun', 'glucose', 'potassium', 'sodium',
    'chloride', 'bicarbonate', 'anion gap', 'hemoglobin', 'hematocrit',
    'platelet', 'wbc', 'inr', 'pt', 'ptt', 'magnesium', 'phosphate',
    'bilirubin', 'albumin'
]

# Prepare for batch loading
stay_ids_list = cohort_filtered['stay_id'].tolist()
print(f"  Loading vitals/labs for {len(stay_ids_list)} patients in batches...")

BATCH_SIZE_VITALS = 500
labs_data = []
vitals_data = []
bp_data = []
meds_data = []

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
    
    # Load vasopressors for batch
    vasopressor_names = ['norepinephrine', 'epinephrine', 'vasopressin', 'dopamine', 'phenylephrine']
    batch_meds = loader.load_medications(vasopressor_names, batch_ids)
    if len(batch_meds) > 0:
        meds_data.append(batch_meds)

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

if len(meds_data) > 0:
    meds = pd.concat(meds_data, ignore_index=True)
    meds['time_day'] = (meds['med_time_hours'] / 24.0).astype(int)
    meds['vasopressor_any'] = 1  # Binary flag for vasopressor use
    # First vasopressor start per stay (days since unit admission)
    vaso_start = (meds.groupby('patientunitstayid')['med_time_hours'].min() / 24.0).to_dict()
else:
    meds = pd.DataFrame()
    vaso_start = {}

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

if len(meds) > 0:
    meds_daily = meds.groupby(['patientunitstayid', 'time_day'])['vasopressor_any'].max().reset_index()
    meds_daily = meds_daily.rename(columns={'patientunitstayid': 'stay_id'})
else:
    meds_daily = pd.DataFrame()

# Merge all daily features
if len(labs_daily) > 0:
    daily_features = labs_daily
    if len(vitals_daily) > 0:
        daily_features = daily_features.merge(vitals_daily, on=['stay_id', 'time_day'], how='outer')
    if len(bp_daily) > 0:
        daily_features = daily_features.merge(bp_daily, on=['stay_id', 'time_day'], how='outer')
    if len(meds_daily) > 0:
        daily_features = daily_features.merge(meds_daily, on=['stay_id', 'time_day'], how='outer')
else:
    daily_features = vitals_daily if len(vitals_daily) > 0 else bp_daily if len(bp_daily) > 0 else pd.DataFrame()

# Fill missing vasopressor flags with 0
if 'vasopressor_any' in daily_features.columns:
    daily_features['vasopressor_any'] = daily_features['vasopressor_any'].fillna(0)

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
if 'vasopressor_any' in cols_to_drop:
    cols_to_drop.remove('vasopressor_any')
print(f"  Columns dropped: {len(cols_to_drop)}")
daily_features = daily_features.drop(columns=cols_to_drop)

# Merge cohort info
daily_features = daily_features.merge(
    cohort_filtered[['stay_id', 'age', 'gender', 'hospital_expire_flag', 'baseline_lactate']],
    on='stay_id',
    how='left'
)

# Define septic shock outcome (lactate ≥4 mmol/L + vasopressor OR mortality)
print(f"\n[8/10] Defining septic shock outcome (gap={PREDICTION_GAP_DAYS}d, window={PREDICTION_WINDOW_DAYS}d)...")
shock_events = []
excluded_counts = {'already_shock': 0, 'no_future_data': 0}

for stay_id, grp in tqdm(daily_features.groupby('stay_id'), desc="  Patients"):
    grp = grp.sort_values('time_day')
    baseline = grp['baseline_lactate'].iloc[0]
    mortality_flag = grp['hospital_expire_flag'].iloc[0]
    
    # Get lactate time series for this patient
    patient_lactate = lactate_ts[lactate_ts['stay_id'] == stay_id].copy()
    
    for i in range(len(grp)):
        current_time = grp.iloc[i]['time_day']
        
        # Get current lactate (use mean from daily aggregation if available)
        lactate_col = [c for c in grp.columns if 'lactate' in c.lower() and 'mean' in c]
        if len(lactate_col) > 0:
            current_lactate = grp.iloc[i][lactate_col[0]]
        else:
            # Fallback to time series
            current_lactate_ts = patient_lactate[patient_lactate['time_days'] <= current_time]
            if len(current_lactate_ts) > 0:
                current_lactate = current_lactate_ts['lactate'].iloc[-1]
            else:
                current_lactate = baseline
        
        # Check current vasopressor use
        current_vasopressor = grp.iloc[i].get('vasopressor_any', 0)
        
        # Skip if already in septic shock (lactate ≥4 + vasopressor)
        if pd.notna(current_lactate) and current_lactate >= 4.0 and current_vasopressor == 1:
            excluded_counts['already_shock'] += 1
            continue
        
        # Define prediction window (+1 day shift to match end-of-day prediction)
        prediction_start = current_time + PREDICTION_GAP_DAYS + 1
        prediction_end = current_time + PREDICTION_GAP_DAYS + PREDICTION_WINDOW_DAYS + 1
        
        future_window = patient_lactate[
            (patient_lactate['time_days'] >= prediction_start) &
            (patient_lactate['time_days'] <= prediction_end)
        ]
        
        # Vasopressor initiation within the prediction window
        first_vaso = vaso_start.get(stay_id)
        future_vasopressor = int(first_vaso is not None and prediction_start <= first_vaso <= prediction_end)
        
        if len(future_window) == 0 and mortality_flag == 0:
            excluded_counts['no_future_data'] += 1
            continue
        
        # Check septic shock criteria: lactate ≥4 mmol/L + vasopressor OR mortality
        shock_event = (
            (len(future_window) > 0 and (future_window['lactate'] >= 4.0).any() and future_vasopressor == 1) or
            mortality_flag == 1
        )
        
        # Create prediction row
        row = grp.iloc[i].to_dict()
        row['target_septic_shock'] = int(shock_event)
        row['current_lactate'] = current_lactate
        row['lactate_fold_change'] = current_lactate / baseline if baseline > 0 else np.nan
        shock_events.append(row)

prediction_dataset = pd.DataFrame(shock_events)
print(f"  Total prediction windows: {len(prediction_dataset):,}")
print(f"  Excluded (already shock): {excluded_counts['already_shock']:,}")
print(f"  Excluded (no future data): {excluded_counts['no_future_data']:,}")

# First positive event only per patient
print("\n[9/10] Filtering to first positive event per patient...")
positive_first = prediction_dataset[prediction_dataset['target_septic_shock'] == 1].groupby('stay_id').first().reset_index()
all_negatives = prediction_dataset[prediction_dataset['target_septic_shock'] == 0]
prediction_dataset = pd.concat([all_negatives, positive_first], ignore_index=True).sort_values(['stay_id', 'time_day']).reset_index(drop=True)

# Save raw biomarker time series ONLY for patients in final prediction dataset
print(f"\n[9.5/10] Saving raw biomarker time series...")
final_stay_ids = prediction_dataset['stay_id'].unique()
for biomarker, ts_df in [('lactate', lactate_ts), ('wbc', wbc_ts), ('platelets', platelet_ts)]:
    ts_filtered = ts_df[ts_df['stay_id'].isin(final_stay_ids)].copy()
    ts_filtered['time_day'] = ts_filtered['time_days'].astype(int)
    ts_output = os.path.join(_TRAJ_DATA_ROOT, "eicu", "sepsis", f"{biomarker}_timeseries.csv")
    os.makedirs(os.path.dirname(ts_output), exist_ok=True)
    ts_filtered.to_csv(ts_output, index=False)
    print(f"  ✓ Saved {biomarker}: {len(ts_filtered):,} measurements for {len(final_stay_ids):,} patients")

print(f"\n📊 Final Dataset Summary:")
print(f"  Total samples: {len(prediction_dataset):,}")
print(f"  Unique patients: {prediction_dataset['stay_id'].nunique():,}")
print(f"  Positive events: {prediction_dataset['target_septic_shock'].sum():,} ({100*prediction_dataset['target_septic_shock'].mean():.1f}%)")
print(f"  Features: {len(prediction_dataset.columns)}")

# Save
print(f"\n[10/10] Saving to {OUTPUT_PATH}...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
prediction_dataset.to_csv(OUTPUT_PATH, index=False)
print(f"  ✓ Saved {len(prediction_dataset):,} rows")

loader.close()
print("\n✓ Sepsis preprocessing complete!")
