"""
MIMIC Liver Failure Preprocessing Script

Extracts:
- Liver cohort (chronic liver disease)
- Bilirubin time series
- Baseline bilirubin and cohort metadata
- Daily vitals/labs aggregation

Outputs:
- results/mimic/liver/bilirubin_timeseries.csv
- results/mimic/liver/hadm_cohort.csv
- results/mimic/liver/aclf_outcomes.csv
- results/mimic/liver/liver_prediction_dataset.csv

Usage:
    python scripts/mimic_preprocessing/liver_preprocess.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

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
    drop_high_missing_columns,
)

OUTPUT_DIR = Path("results/mimic/liver")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MIMIC Liver Preprocessing")
print("=" * 80)

paths = MIMICPaths()
conn = connect_db(paths.db_path)

# Cohort selection
hadm_query = f"""
WITH liver_patients AS (
    SELECT DISTINCT a.hadm_id
    FROM admissions a
    INNER JOIN diagnoses_icd d ON a.hadm_id = d.hadm_id
    WHERE 
        (d.icd9_code LIKE '571%' OR
         d.icd9_code LIKE '5722%' OR
         d.icd9_code LIKE '5728%' OR
         d.icd9_code LIKE 'V427%')
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
INNER JOIN liver_patients lp ON a.hadm_id = lp.hadm_id
WHERE 
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) BETWEEN 18 AND 90
    AND {ICU_LOS_SQL}
"""

hadm_df = conn.execute(hadm_query).fetchdf()
hadm_df.columns = hadm_df.columns.str.lower()
print(f"✓ Loaded {len(hadm_df):,} liver admissions")

hadm_ids = hadm_df["hadm_id"].astype(int).tolist()

# Metadata
vital_meta, labs_meta = load_metadata(paths)
lab_items = labs_meta["itemid"].tolist()
vital_items = vital_meta["itemid"].tolist()

# Bilirubin time series
bili_itemids = labs_meta[labs_meta["feature name"] == "BILIRUBIN"]["itemid"].tolist()

bili_df = fetch_labevents_timeseries(
    conn,
    hadm_ids=hadm_ids,
    itemids=bili_itemids,
    value_col="bilirubin",
    min_val=0.1,
    max_val=50.0,
    batch_size=500,
)

bili_df = ensure_time_day(bili_df)
print(f"✓ Loaded bilirubin measurements: {len(bili_df):,}")

# Baseline bilirubin (first 24h)
if len(bili_df) == 0:
    raise RuntimeError("No bilirubin measurements found.")

bili_df["admittime"] = bili_df["hadm_id"].map(hadm_df.set_index("hadm_id")["admittime"])
bili_df["admittime"] = pd.to_datetime(bili_df["admittime"])
bili_df["charttime"] = pd.to_datetime(bili_df["charttime"])

first_24h = bili_df[bili_df["charttime"] <= bili_df["admittime"] + pd.Timedelta(hours=24)]
baseline_bili = (
    first_24h.sort_values("charttime")
    .pivot_table(index="hadm_id", values="bilirubin", aggfunc="first")
    .reset_index()
    .rename(columns={"bilirubin": "baseline_bilirubin"})
)

# Filter admissions with sufficient measurements
bili_counts = bili_df.groupby("hadm_id").size()
valid_hadm_ids = bili_counts[bili_counts >= MIN_TARGET_MEASUREMENTS].index.tolist()

bili_filtered = bili_df[bili_df["hadm_id"].isin(valid_hadm_ids)].copy()
hadm_filtered = hadm_df[hadm_df["hadm_id"].isin(valid_hadm_ids)].merge(
    baseline_bili, on="hadm_id", how="inner"
)

# Time series with baseline features
bili_ts = bili_filtered[["hadm_id", "time_days", "time_day", "bilirubin"]].merge(
    hadm_filtered[["hadm_id", "baseline_bilirubin", "age", "gender", "hospital_expire_flag"]],
    on="hadm_id",
    how="left",
)

bili_ts["bili_fold_change"] = bili_ts["bilirubin"] / bili_ts["baseline_bilirubin"]

# Save time series and cohort
bili_ts.to_csv(OUTPUT_DIR / "bilirubin_timeseries.csv", index=False)
hadm_filtered.to_csv(OUTPUT_DIR / "hadm_cohort.csv", index=False)
print("✓ Saved bilirubin_timeseries.csv and hadm_cohort.csv")

# Load vitals/labs and aggregate daily
vitals_df, labs_df = fetch_vitals_labs(conn, hadm_ids=hadm_filtered["hadm_id"].tolist(), vital_items=vital_items, lab_items=lab_items)

if len(vitals_df) > 0 and len(labs_df) > 0:
    vitals_labs_pivot = aggregate_vitals_labs(
        vitals_df=vitals_df,
        labs_df=labs_df,
        vital_meta=vital_meta,
        labs_meta=labs_meta,
    )
    vitals_labs_pivot = vitals_labs_pivot.reset_index()
    vitals_labs_pivot.columns = [c.lower() for c in vitals_labs_pivot.columns]

    if "admittime" in vitals_labs_pivot.columns:
        vitals_labs_pivot = vitals_labs_pivot.drop(columns=["admittime"])

    # Merge with cohort
    hadm_final = hadm_filtered.merge(vitals_labs_pivot, on=["hadm_id"], how="inner")
    hadm_final["time_day"] = ((pd.to_datetime(hadm_final["charttime"]) - pd.to_datetime(hadm_final["admittime"]))
                              .dt.total_seconds() / 86400.0).astype(int)

    # Forward-fill imputation within each admission
    hadm_final = hadm_final.sort_values(["hadm_id", "time_day"])
    hadm_final = ffill_within_stay(hadm_final, "hadm_id")

    # Drop columns with >70% missing
    missing_pct = hadm_final.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index.tolist()
    for col in ["hadm_id", "time_day", "bilirubin_mean", "hospital_expire_flag", "age", "gender", "baseline_bilirubin"]:
        if col in cols_to_drop:
            cols_to_drop.remove(col)
    hadm_final = hadm_final.drop(columns=cols_to_drop)

    # Define ACLF outcome (matches notebook logic)
    PREDICTION_GAP_DAYS = 1.0
    PREDICTION_WINDOW_DAYS = 5.0

    aclf_events = []
    n_in_aclf = 0
    n_no_data = 0

    hadm_final = hadm_final.drop_duplicates(subset=["hadm_id", "time_day"])

    for hadm_id, grp in hadm_final.groupby("hadm_id"):
        grp = grp.sort_values("time_day")
        mortality = hadm_final[hadm_final["hadm_id"] == hadm_id]["hospital_expire_flag"].iloc[0]

        for i in range(len(grp)):
            current_time = grp.iloc[i]["time_day"]
            current_bili = grp.iloc[i]["bilirubin_mean"]

            # Skip if already in ACLF
            if current_bili >= 12.0:
                n_in_aclf += 1
                continue

            prediction_start = current_time + PREDICTION_GAP_DAYS + 1
            prediction_end = current_time + PREDICTION_GAP_DAYS + PREDICTION_WINDOW_DAYS + 1

            future_window = bili_ts[
                (bili_ts["hadm_id"] == hadm_id)
                & (bili_ts["time_days"] >= prediction_start)
                & (bili_ts["time_days"] <= prediction_end)
            ]

            if len(future_window) == 0:
                n_no_data += 1
                continue

            severe_bili = (future_window["bilirubin"] >= 12.0).any()
            target = int(severe_bili or mortality)

            aclf_events.append({
                "hadm_id": hadm_id,
                "time_day": current_time,
                "target_aclf": target,
                "current_bilirubin": current_bili,
            })

    outcome_df = pd.DataFrame(aclf_events)
    outcome_df = outcome_df.sort_values(["hadm_id", "time_day"])
    positive_first = outcome_df[outcome_df["target_aclf"] == 1].groupby("hadm_id").first().reset_index()
    all_negatives = outcome_df[outcome_df["target_aclf"] == 0]
    outcome_df = (
        pd.concat([all_negatives, positive_first], ignore_index=True)
        .sort_values(["hadm_id", "time_day"])
        .reset_index(drop=True)
    )

    outcome_df.to_csv(OUTPUT_DIR / "aclf_outcomes.csv", index=False)
    print("✓ Saved aclf_outcomes.csv")
    print(f"   ACLF events (first per patient): {outcome_df['target_aclf'].sum():,}")
    print(f"   Windows skipped (in ACLF at baseline): {n_in_aclf:,}")
    print(f"   Windows skipped (no future data): {n_no_data:,}")

    hadm_final.to_csv(OUTPUT_DIR / "liver_prediction_dataset.csv", index=False)
    print("✓ Saved liver_prediction_dataset.csv")
else:
    print("⚠️ Skipped prediction dataset (missing vitals/labs)")

conn.close()
print("Done.")
