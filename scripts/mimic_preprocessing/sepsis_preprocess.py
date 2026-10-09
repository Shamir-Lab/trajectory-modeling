"""
MIMIC Sepsis Preprocessing Script

Extracts:
- Sepsis cohort (ICD-9 codes)
- Lactate, WBC, Platelet time series

Outputs:
- results/mimic/sepsis/lactate_timeseries.csv
- results/mimic/sepsis/wbc_timeseries.csv
- results/mimic/sepsis/platelet_timeseries.csv
- results/mimic/sepsis/hadm_cohort.csv
- results/mimic/sepsis/septic_shock_outcomes.csv
- results/mimic/sepsis/sepsis_prediction_dataset.csv

Usage:
    python scripts/mimic_preprocessing/sepsis_preprocess.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

from common import (
    MIMICPaths,
    connect_db,
    load_metadata,
    fetch_labevents_timeseries,
    fetch_vitals_labs,
    ensure_time_day,
    aggregate_vitals_labs,
    drop_high_missing_columns,
    ffill_within_stay,
    fetch_vasopressor_starts,
    ICU_LOS_SQL,
    MIN_TARGET_MEASUREMENTS,
    MISSINGNESS_THRESHOLD,
)

OUTPUT_DIR = Path("results/mimic/sepsis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MIMIC Sepsis Preprocessing")
print("=" * 80)

paths = MIMICPaths()
conn = connect_db(paths.db_path)

hadm_query = f"""
WITH sepsis_patients AS (
    SELECT DISTINCT a.hadm_id
    FROM admissions a
    INNER JOIN diagnoses_icd d ON a.hadm_id = d.hadm_id
    WHERE 
        (d.icd9_code LIKE '99591' OR  -- sepsis
         d.icd9_code LIKE '99592' OR  -- severe sepsis
         d.icd9_code LIKE '78552')    -- septic shock
        -- 99594 (SIRS due to a non-infectious process) is not sepsis
)
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
INNER JOIN sepsis_patients sp ON a.hadm_id = sp.hadm_id
WHERE 
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) BETWEEN 18 AND 90
    AND {ICU_LOS_SQL}
"""

hadm_df = conn.execute(hadm_query).fetchdf()
hadm_df.columns = hadm_df.columns.str.lower()
print(f"✓ Loaded {len(hadm_df):,} sepsis admissions")

hadm_ids = hadm_df["hadm_id"].astype(int).tolist()

vital_meta, labs_meta = load_metadata(paths)

labs_meta["feature_upper"] = labs_meta["feature name"].str.upper()

lactate_items = labs_meta[labs_meta["feature_upper"].str.contains("LACTATE")]["itemid"].tolist()
wbc_items = labs_meta[labs_meta["feature_upper"].str.contains("WBC")]["itemid"].tolist()
platelet_items = labs_meta[labs_meta["feature_upper"].str.contains("PLATELET")]["itemid"].tolist()

lactate_df = fetch_labevents_timeseries(conn, hadm_ids, lactate_items, "lactate", 0.1, 50.0)
wbc_df = fetch_labevents_timeseries(conn, hadm_ids, wbc_items, "wbc", 0.1, 300.0)
platelet_df = fetch_labevents_timeseries(conn, hadm_ids, platelet_items, "platelet", 1.0, 2000.0)

lactate_df = ensure_time_day(lactate_df)
wbc_df = ensure_time_day(wbc_df)
platelet_df = ensure_time_day(platelet_df)

# Keep admissions with enough measurements of the target clinical parameter
lactate_counts = lactate_df.groupby("hadm_id").size()
hadm_ids = lactate_counts[lactate_counts >= MIN_TARGET_MEASUREMENTS].index.astype(int).tolist()
hadm_df = hadm_df[hadm_df["hadm_id"].isin(hadm_ids)].copy()
lactate_df = lactate_df[lactate_df["hadm_id"].isin(hadm_ids)].copy()
wbc_df = wbc_df[wbc_df["hadm_id"].isin(hadm_ids)].copy()
platelet_df = platelet_df[platelet_df["hadm_id"].isin(hadm_ids)].copy()
print(f"✓ {len(hadm_ids):,} admissions with >= {MIN_TARGET_MEASUREMENTS} lactate measurements")


def add_baseline_and_demographics(df: pd.DataFrame, value_col: str, baseline_col: str) -> pd.DataFrame:
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

    demo_cols = ["subject_id", "age", "gender", "hospital_expire_flag"]
    df = df.merge(baseline_df, on="hadm_id", how="left")
    df = df.merge(hadm_df[["hadm_id"] + demo_cols], on="hadm_id", how="left")
    if "admittime" in df.columns:
        df = df.drop(columns=["admittime"])
    return df


lactate_df = add_baseline_and_demographics(lactate_df, "lactate", "baseline_lactate")
wbc_df = add_baseline_and_demographics(wbc_df, "wbc", "baseline_wbc")
platelet_df = add_baseline_and_demographics(platelet_df, "platelet", "baseline_platelet")

lactate_df.to_csv(OUTPUT_DIR / "lactate_timeseries.csv", index=False)
wbc_df.to_csv(OUTPUT_DIR / "wbc_timeseries.csv", index=False)
platelet_df.to_csv(OUTPUT_DIR / "platelet_timeseries.csv", index=False)
hadm_df.to_csv(OUTPUT_DIR / "hadm_cohort.csv", index=False)

print("✓ Saved sepsis time series and cohort")

# Load vitals/labs and aggregate daily for outcomes
vital_items = vital_meta["itemid"].tolist()
lab_items = labs_meta["itemid"].tolist()
vitals_df, labs_df = fetch_vitals_labs(conn, hadm_ids=hadm_ids, vital_items=vital_items, lab_items=lab_items)

if len(vitals_df) > 0 and len(labs_df) > 0:
    vitals_labs_pivot = aggregate_vitals_labs(
        vitals_df=vitals_df,
        labs_df=labs_df,
        vital_meta=vital_meta,
        labs_meta=labs_meta,
    ).reset_index()
    vitals_labs_pivot.columns = [c.lower() for c in vitals_labs_pivot.columns]

    if "admittime" in vitals_labs_pivot.columns:
        vitals_labs_pivot = vitals_labs_pivot.drop(columns=["admittime"])

    hadm_final = hadm_df.merge(vitals_labs_pivot, on=["hadm_id"], how="inner")
    hadm_final["time_day"] = ((pd.to_datetime(hadm_final["charttime"]) - pd.to_datetime(hadm_final["admittime"]))
                              .dt.total_seconds() / 86400.0).astype(int)
    hadm_final = hadm_final.drop_duplicates(subset=["hadm_id", "time_day"])

    # Forward-fill imputation within each admission
    hadm_final = hadm_final.sort_values(["hadm_id", "time_day"])
    hadm_final = ffill_within_stay(hadm_final, "hadm_id")

    # Drop columns with >70% missing
    missing_pct = hadm_final.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index.tolist()
    for col in ["hadm_id", "time_day", "lactate_mean", "hospital_expire_flag", "age", "gender"]:
        if col in cols_to_drop:
            cols_to_drop.remove(col)
    hadm_final = hadm_final.drop(columns=cols_to_drop)

    baseline_lactate = lactate_df[["hadm_id", "baseline_lactate"]].drop_duplicates()
    baseline_wbc = wbc_df[["hadm_id", "baseline_wbc"]].drop_duplicates()
    baseline_platelet = platelet_df[["hadm_id", "baseline_platelet"]].drop_duplicates()
    hadm_final = hadm_final.merge(baseline_lactate, on="hadm_id", how="left")
    hadm_final = hadm_final.merge(baseline_wbc, on="hadm_id", how="left")
    hadm_final = hadm_final.merge(baseline_platelet, on="hadm_id", how="left")

    # Detect vasopressor use (norepinephrine, epinephrine; MetaVision and CareVue)
    vasopressor_df = fetch_vasopressor_starts(
        conn,
        hadm_final["hadm_id"].astype(int).unique().tolist(),
        mv_items=(221906, 221289),
        cv_items=(30047, 30120, 30044, 30119, 30309),
        out_col="vaso_start_day",
        seconds_per_unit=86400,
    )
    print(f"✓ Detected vasopressors in {len(vasopressor_df):,} admissions")

    # Define septic shock outcome (matches notebook logic)
    PREDICTION_GAP_DAYS = 0.5
    PREDICTION_WINDOW_DAYS = 2.0

    septic_shock_events = []
    n_at_shock = 0
    n_no_data = 0

    for hadm_id, grp in hadm_final.groupby("hadm_id"):
        grp = grp.sort_values("time_day")
        mortality = hadm_final[hadm_final["hadm_id"] == hadm_id]["hospital_expire_flag"].iloc[0]

        for i in range(len(grp)):
            current_time = grp.iloc[i]["time_day"]
            current_lactate = grp.iloc[i]["lactate_mean"]

            # Skip if already in shock (lactate ≥4 AND vasopressor already started)
            current_vasopressor = False
            if hadm_id in vasopressor_df["hadm_id"].values:
                vaso_start = vasopressor_df[vasopressor_df["hadm_id"] == hadm_id]["vaso_start_day"].iloc[0]
                current_vasopressor = vaso_start <= current_time

            if current_lactate >= 4.0 and current_vasopressor:
                n_at_shock += 1
                continue

            prediction_start = current_time + PREDICTION_GAP_DAYS + 1
            prediction_end = current_time + PREDICTION_GAP_DAYS + PREDICTION_WINDOW_DAYS + 1

            future_window = lactate_df[
                (lactate_df["hadm_id"] == hadm_id)
                & (lactate_df["time_days"].between(prediction_start, prediction_end, inclusive="both"))
            ]

            if len(future_window) == 0:
                n_no_data += 1
                continue

            high_lactate = (future_window["lactate"] >= 4.0).any()

            vasopressor_in_window = False
            if hadm_id in vasopressor_df["hadm_id"].values:
                vaso_start = vasopressor_df[vasopressor_df["hadm_id"] == hadm_id]["vaso_start_day"].iloc[0]
                vasopressor_in_window = prediction_start <= vaso_start <= prediction_end

            target = int((high_lactate and vasopressor_in_window) or mortality)

            septic_shock_events.append({
                "hadm_id": hadm_id,
                "time_day": current_time,
                "target_septic_shock": target,
                "current_lactate": current_lactate,
            })

    outcome_df = pd.DataFrame(septic_shock_events)
    outcome_df = outcome_df.sort_values(["hadm_id", "time_day"])
    positive_first = outcome_df[outcome_df["target_septic_shock"] == 1].groupby("hadm_id").first().reset_index()
    all_negatives = outcome_df[outcome_df["target_septic_shock"] == 0]
    outcome_df = (
        pd.concat([all_negatives, positive_first], ignore_index=True)
        .sort_values(["hadm_id", "time_day"])
        .reset_index(drop=True)
    )

    outcome_df.to_csv(OUTPUT_DIR / "septic_shock_outcomes.csv", index=False)
    print("✓ Saved septic_shock_outcomes.csv")
    print(f"   Septic shock events (first per patient): {outcome_df['target_septic_shock'].sum():,}")
    print(f"   Skipped (already in shock): {n_at_shock:,}")
    print(f"   Skipped (no future data): {n_no_data:,}")

    hadm_final.to_csv(OUTPUT_DIR / "sepsis_prediction_dataset.csv", index=False)
    print("✓ Saved sepsis_prediction_dataset.csv")
else:
    print("⚠️ Skipped outcome and prediction dataset (missing vitals/labs)")

conn.close()
