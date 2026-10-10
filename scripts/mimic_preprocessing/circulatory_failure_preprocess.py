"""
MIMIC Circulatory Failure Preprocessing Script

Creates:
- Lactate, Heart Rate, and Systolic BP time series
- Prediction dataset with circulatory failure outcome

Outcome definition (default):
- Future window contains vasopressor start OR hypotension (MAP/SBP below threshold)
- Excludes windows where patient is already in circulatory failure at prediction time

Usage:
    python scripts/mimic_preprocessing/circulatory_failure_preprocess.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from common import (
    MIMICPaths,
    connect_db,
    load_metadata,
    fetch_labevents_timeseries,
    fetch_vitals_labs,
    ensure_time_day,
    aggregate_vitals_labs,
    ffill_within_stay,
    ICU_LOS_SQL,
    MIN_TARGET_MEASUREMENTS,
    MISSINGNESS_THRESHOLD,
    fetch_vasopressor_starts,
)

OUTPUT_DIR = Path(os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet")) / "mimic" / "circulatory_failure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Outcome configuration (Nature Medicine 2020): predict CF within next 8h
PREDICTION_GAP_HOURS = 1.0
PREDICTION_WINDOW_HOURS = 8.0
MAP_THRESHOLD = 65.0
LACTATE_THRESHOLD = 2.0

print("=" * 80)
print("MIMIC Circulatory Failure Preprocessing")
print("=" * 80)

paths = MIMICPaths()
conn = connect_db(paths.db_path)

hadm_query = f"""
SELECT 
    a.subject_id::INTEGER AS subject_id,
    a.hadm_id::INTEGER AS hadm_id,
    a.admittime::TIMESTAMP as admittime,
    a.dischtime::TIMESTAMP as dischtime,
    CAST(a.dischtime::DATE - a.admittime::DATE AS REAL) as los_days,
    p.gender,
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) as age,
    a.hospital_expire_flag::INT as hospital_expire_flag,
    a.discharge_location
FROM admissions a
INNER JOIN patients p ON a.subject_id = p.subject_id
WHERE 
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) BETWEEN 18 AND 90
    AND {ICU_LOS_SQL}
"""

hadm_df = conn.execute(hadm_query).fetchdf()
hadm_df.columns = hadm_df.columns.str.lower()
print(f"✓ Loaded {len(hadm_df):,} admissions")

hadm_ids = hadm_df["hadm_id"].astype(int).tolist()

vital_meta, labs_meta = load_metadata(paths)

labs_meta["feature_upper"] = labs_meta["feature name"].str.upper()

lactate_items = labs_meta[labs_meta["feature_upper"].str.contains("LACTATE")]["itemid"].tolist()

vital_meta["feature_upper"] = vital_meta["feature name"].str.upper()
hr_items = vital_meta[vital_meta["feature_upper"] == "HEARTRATE"]["itemid"].tolist()
sbp_items = vital_meta[vital_meta["feature_upper"] == "SYSBP"]["itemid"].tolist()
map_items = vital_meta[vital_meta["feature_upper"] == "MEANBP"]["itemid"].tolist()

lactate_df = fetch_labevents_timeseries(conn, hadm_ids, lactate_items, "lactate", 0.1, 50.0)

# Keep admissions with enough measurements of the target clinical parameter (lactate)
lactate_counts = lactate_df.groupby("hadm_id").size()
hadm_ids = lactate_counts[lactate_counts >= MIN_TARGET_MEASUREMENTS].index.astype(int).tolist()
hadm_df = hadm_df[hadm_df["hadm_id"].isin(hadm_ids)].copy()
lactate_df = lactate_df[lactate_df["hadm_id"].isin(hadm_ids)].copy()
print(f"✓ {len(hadm_ids):,} admissions with >= {MIN_TARGET_MEASUREMENTS} lactate measurements")

vitals_df, _ = fetch_vitals_labs(conn, hadm_ids=hadm_ids, vital_items=list(set(hr_items + sbp_items + map_items)), lab_items=[])

if len(vitals_df) > 0:
    vitals_df["charttime"] = pd.to_datetime(vitals_df["charttime"])
    vitals_df["admittime"] = pd.to_datetime(vitals_df["admittime"])
    vitals_df["time_days"] = (vitals_df["charttime"] - vitals_df["admittime"]).dt.total_seconds() / 86400.0
    vitals_df = vitals_df.merge(vital_meta[["itemid", "feature name"]], on="itemid", how="left")
    vitals_df["feature name"] = vitals_df["feature name"].str.upper()
else:
    vitals_df = pd.DataFrame(columns=["hadm_id", "charttime", "admittime", "itemid", "valuenum", "time_days", "feature name"])

lactate_df["time_hours"] = lactate_df["time_days"] * 24.0
lactate_df["time_hour"] = np.floor(lactate_df["time_hours"]).astype(int)

hr_ts = vitals_df[vitals_df["feature name"] == "HEARTRATE"].copy()
hr_ts = hr_ts.rename(columns={"valuenum": "heartrate"})
hr_ts = hr_ts[["hadm_id", "charttime", "time_days", "heartrate"]].copy()
hr_ts["time_hours"] = hr_ts["time_days"] * 24.0
hr_ts["time_hour"] = np.floor(hr_ts["time_hours"]).astype(int)

sbp_ts = vitals_df[vitals_df["feature name"] == "SYSBP"].copy()
sbp_ts = sbp_ts.rename(columns={"valuenum": "systolic"})
sbp_ts = sbp_ts[["hadm_id", "charttime", "time_days", "systolic"]].copy()
sbp_ts["time_hours"] = sbp_ts["time_days"] * 24.0
sbp_ts["time_hour"] = np.floor(sbp_ts["time_hours"]).astype(int)

map_ts = vitals_df[vitals_df["feature name"] == "MEANBP"].copy()
map_ts = map_ts.rename(columns={"valuenum": "mean_bp"})
map_ts = map_ts[["hadm_id", "charttime", "time_days", "mean_bp"]].copy()
map_ts["time_hours"] = map_ts["time_days"] * 24.0
map_ts["time_hour"] = np.floor(map_ts["time_hours"]).astype(int)


def add_baseline(df: pd.DataFrame, value_col: str, baseline_col: str) -> pd.DataFrame:
    if len(df) == 0:
        return df

    df = df.copy()
    df["admittime"] = df["hadm_id"].map(hadm_df.set_index("hadm_id")["admittime"])
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["charttime"] = pd.to_datetime(df["charttime"])

    first_24h = df[df["charttime"] <= df["admittime"] + pd.Timedelta(hours=24)]
    baseline_df = (
        first_24h.sort_values("charttime")
        .groupby("hadm_id")[value_col]
        .first()
        .reset_index()
        .rename(columns={value_col: baseline_col})
    )

    df = df.merge(baseline_df, on="hadm_id", how="left")
    return df


lactate_df = add_baseline(lactate_df, "lactate", "baseline_lactate")
hr_ts = add_baseline(hr_ts, "heartrate", "baseline_heartrate")
sbp_ts = add_baseline(sbp_ts, "systolic", "baseline_systolic")
map_ts = add_baseline(map_ts, "mean_bp", "baseline_mean_bp")

lactate_df.to_csv(OUTPUT_DIR / "lactate_timeseries.csv", index=False)
hr_ts.to_csv(OUTPUT_DIR / "heartrate_timeseries.csv", index=False)
sbp_ts.to_csv(OUTPUT_DIR / "systolic_timeseries.csv", index=False)
if len(map_ts) > 0:
    map_ts.to_csv(OUTPUT_DIR / "meanbp_timeseries.csv", index=False)

hadm_df.to_csv(OUTPUT_DIR / "hadm_cohort.csv", index=False)

print("✓ Saved circulatory time series and cohort")

# Load vitals/labs and aggregate hourly
vital_items = vital_meta["itemid"].tolist()
lab_items = labs_meta["itemid"].tolist()

vitals_df, labs_df = fetch_vitals_labs(conn, hadm_ids=hadm_ids, vital_items=vital_items, lab_items=lab_items)

if len(vitals_df) > 0 and len(labs_df) > 0:
    vitals_df = vitals_df.merge(vital_meta, on='itemid', how='left')
    vitals_df = vitals_df[vitals_df['valuenum'].between(vitals_df['min'], vitals_df['max'], inclusive='both')]
    if 'units' in vitals_df.columns:
        vitals_df.loc[vitals_df['units'] == 'F', 'valuenum'] = (
            vitals_df.loc[vitals_df['units'] == 'F', 'valuenum'] - 32
        ) * 5.0 / 9.0
        vitals_df.loc[vitals_df['units'] == 'F', 'units'] = 'C'
        vitals_df.loc[vitals_df['feature name'] == 'TempF', 'feature name'] = 'TempC'

    labs_df = labs_df.merge(labs_meta, on='itemid', how='left')
    labs_df = labs_df[labs_df['valuenum'].between(labs_df['min'], labs_df['max'], inclusive='both')]

    vitals_df['time_hours'] = (pd.to_datetime(vitals_df['charttime']) - pd.to_datetime(vitals_df['admittime'])).dt.total_seconds() / 3600.0
    labs_df['time_hours'] = (pd.to_datetime(labs_df['charttime']) - pd.to_datetime(labs_df['admittime'])).dt.total_seconds() / 3600.0
    vitals_df['time_hour'] = np.floor(vitals_df['time_hours']).astype(int)
    labs_df['time_hour'] = np.floor(labs_df['time_hours']).astype(int)

    vitals_labs = pd.concat([vitals_df, labs_df], ignore_index=True)
    vitals_labs_pivot = vitals_labs.pivot_table(
        index=['hadm_id', 'time_hour'],
        columns='feature name',
        values='valuenum',
        aggfunc=['min', 'max', 'mean']
    ).reset_index()
    vitals_labs_pivot.columns = [f'{col[1]}_{col[0]}' if col[1] else col[0] for col in vitals_labs_pivot.columns]
    vitals_labs_pivot.columns = [c.lower() for c in vitals_labs_pivot.columns]

    hadm_final = hadm_df.merge(vitals_labs_pivot, on=["hadm_id"], how="inner")
    hadm_final = hadm_final.drop_duplicates(subset=["hadm_id", "time_hour"])

    hadm_final = hadm_final.sort_values(["hadm_id", "time_hour"])
    hadm_final = ffill_within_stay(hadm_final, "hadm_id")

    missing_pct = hadm_final.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index.tolist()
    required_cols = ["lactate_mean", "meanbp_mean"]
    for col in ["hadm_id", "time_hour", "hospital_expire_flag", "age", "gender"] + required_cols:
        if col in cols_to_drop:
            cols_to_drop.remove(col)
    hadm_final = hadm_final.drop(columns=cols_to_drop)

    baseline_lactate = lactate_df[["hadm_id", "baseline_lactate"]].drop_duplicates()
    baseline_hr = hr_ts[["hadm_id", "baseline_heartrate"]].drop_duplicates()
    baseline_sbp = sbp_ts[["hadm_id", "baseline_systolic"]].drop_duplicates()
    hadm_final = hadm_final.merge(baseline_lactate, on="hadm_id", how="left")
    hadm_final = hadm_final.merge(baseline_hr, on="hadm_id", how="left")
    hadm_final = hadm_final.merge(baseline_sbp, on="hadm_id", how="left")

    # Vasopressors/inotropes (norepinephrine, epinephrine, dopamine, dobutamine;
    # MetaVision and CareVue)
    vasopressor_df = fetch_vasopressor_starts(
        conn,
        hadm_final["hadm_id"].astype(int).unique().tolist(),
        mv_items=(221906, 221289, 221662, 221653),
        cv_items=(30047, 30120, 30044, 30119, 30309),
        out_col="vaso_start_hour",
        seconds_per_unit=3600,
    )
    print(f"✓ Detected vasopressors in {len(vasopressor_df):,} admissions")

    # Prepare lactate time series for outcome lookup (from raw lactate_df, not pivoted)
    lactate_lookup = lactate_df[["hadm_id", "time_hours", "lactate"]].copy()
    lactate_lookup["time_hour"] = np.floor(lactate_lookup["time_hours"]).astype(int)
    # Hourly lactate aggregation for outcome detection
    lactate_hourly = lactate_lookup.groupby(["hadm_id", "time_hour"])["lactate"].mean().reset_index()

    failure_events = []
    excluded = {"already_failure": 0, "no_future_data": 0, "ambiguous": 0}

    for hadm_id, grp in hadm_final.groupby("hadm_id"):
        grp = grp.sort_values("time_hour")
        patient_lactate = lactate_hourly[lactate_hourly["hadm_id"] == hadm_id]
        
        for i in range(len(grp)):
            current_time = grp.iloc[i]["time_hour"]
            current_bp = grp.iloc[i].get("meanbp_mean", np.nan)
            
            # Get current lactate from hourly aggregation (may be NA if not measured this hour)
            current_lactate_row = patient_lactate[patient_lactate["time_hour"] == current_time]
            current_lactate = current_lactate_row["lactate"].iloc[0] if len(current_lactate_row) > 0 else np.nan

            current_vasopressor = False
            if hadm_id in vasopressor_df["hadm_id"].values:
                vaso_start = vasopressor_df[vasopressor_df["hadm_id"] == hadm_id]["vaso_start_hour"].iloc[0]
                current_vasopressor = vaso_start <= current_time

            # Need MAP for prediction; lactate can be missing at current time
            if pd.isna(current_bp):
                excluded["ambiguous"] += 1
                continue

            # Skip if already in circulatory failure (lactate high AND (hypotension OR vasopressor))
            if not pd.isna(current_lactate) and current_lactate >= LACTATE_THRESHOLD and (current_bp <= MAP_THRESHOLD or current_vasopressor):
                excluded["already_failure"] += 1
                continue

            # +1 hour shift: the hourly features summarize the whole index hour, so the gap
            # and prediction window start at its end (as for the daily tasks)
            prediction_start = current_time + PREDICTION_GAP_HOURS + 1
            prediction_end = current_time + PREDICTION_GAP_HOURS + PREDICTION_WINDOW_HOURS + 1

            future_grp = grp[(grp["time_hour"] >= prediction_start) & (grp["time_hour"] <= prediction_end)]
            if len(future_grp) == 0:
                excluded["no_future_data"] += 1
                continue

            # Check future hypotension from vitals
            future_hypotension = False
            if "meanbp_mean" in future_grp.columns:
                future_hypotension = (future_grp["meanbp_mean"].dropna() <= MAP_THRESHOLD).any()

            # Check future lactate from raw time series
            future_lactate_df = patient_lactate[
                (patient_lactate["time_hour"] >= prediction_start) & 
                (patient_lactate["time_hour"] <= prediction_end)
            ]
            future_lactate = (future_lactate_df["lactate"] >= LACTATE_THRESHOLD).any() if len(future_lactate_df) > 0 else False

            vasopressor_in_window = False
            if hadm_id in vasopressor_df["hadm_id"].values:
                vaso_start = vasopressor_df[vasopressor_df["hadm_id"] == hadm_id]["vaso_start_hour"].iloc[0]
                vasopressor_in_window = prediction_start <= vaso_start <= prediction_end

            # Circulatory failure (Hyland et al., 2020): lactate >= 2 mmol/L AND
            # (MAP <= 65 mmHg OR vasopressor/inotrope) within the prediction window
            target = int(future_lactate and (future_hypotension or vasopressor_in_window))

            row = grp.iloc[i].to_dict()
            row["target_circulatory_failure"] = target
            failure_events.append(row)

    outcome_df = pd.DataFrame(failure_events)
    if outcome_df.empty:
        outcome_df = pd.DataFrame(columns=list(hadm_final.columns) + ["target_circulatory_failure"])
        outcome_df.to_csv(OUTPUT_DIR / "circulatory_failure_prediction_dataset.csv", index=False)
        print("⚠️ No valid prediction windows; saved empty circulatory_failure_prediction_dataset.csv")
    else:
        outcome_df = outcome_df.sort_values(["hadm_id", "time_hour"])
        positive_first = outcome_df[outcome_df["target_circulatory_failure"] == 1].groupby("hadm_id").first().reset_index()
        all_negatives = outcome_df[outcome_df["target_circulatory_failure"] == 0]
        outcome_df = (
            pd.concat([all_negatives, positive_first], ignore_index=True)
            .sort_values(["hadm_id", "time_hour"])
            .reset_index(drop=True)
        )

        outcome_df.to_csv(OUTPUT_DIR / "circulatory_failure_prediction_dataset.csv", index=False)
        print("✓ Saved circulatory_failure_prediction_dataset.csv")
        print(f"   Positive events (first per patient): {outcome_df['target_circulatory_failure'].sum():,}")
        print(f"   Skipped (already failure): {excluded['already_failure']:,}")
        print(f"   Skipped (no future data): {excluded['no_future_data']:,}")
else:
    print("⚠️ Skipped prediction dataset (missing vitals/labs)")

print("\n✓ Circulatory failure preprocessing complete!")
