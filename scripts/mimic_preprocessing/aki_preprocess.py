"""
MIMIC AKI Preprocessing Script

Extracts:
- ICU cohort
- Creatinine time series

Outputs:
- results/mimic/aki/creatinine_timeseries.csv
- results/mimic/aki/hadm_cohort.csv
- results/mimic/aki/aki_outcomes.csv
- results/mimic/aki/aki_prediction_dataset.csv

Usage:
    python scripts/mimic_preprocessing/aki_preprocess.py
"""

import os
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
    ffill_within_stay,
    ICU_LOS_SQL,
    MIN_TARGET_MEASUREMENTS,
    MISSINGNESS_THRESHOLD,
    drop_high_missing_columns,
)

OUTPUT_DIR = Path(os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet")) / "mimic" / "aki"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MIMIC AKI Preprocessing")
print("=" * 80)

paths = MIMICPaths()
conn = connect_db(paths.db_path)

hadm_query = f"""
SELECT 
    i.subject_id::INTEGER AS subject_id,
    i.hadm_id::INTEGER AS hadm_id,
    a.admittime::TIMESTAMP as admittime,
    a.dischtime::TIMESTAMP as dischtime,
    i.intime::TIMESTAMP as intime,
    i.outtime::TIMESTAMP as outtime,
    CAST(a.dischtime::DATE - a.admittime::DATE AS REAL) as los_days,
    p.gender,
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) as age,
    a.hospital_expire_flag::INT as hospital_expire_flag,
    a.discharge_location
FROM icustays i
INNER JOIN patients p ON i.subject_id = p.subject_id
INNER JOIN admissions a ON i.hadm_id = a.hadm_id
WHERE 
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) BETWEEN 18 AND 90
    AND {ICU_LOS_SQL}
"""

hadm_df = conn.execute(hadm_query).fetchdf()
hadm_df.columns = hadm_df.columns.str.lower()
print(f"✓ Loaded {len(hadm_df):,} ICU admissions")

hadm_unique = hadm_df.sort_values("admittime").drop_duplicates(subset=["hadm_id"])

hadm_ids = hadm_unique["hadm_id"].astype(int).tolist()

vital_meta, labs_meta = load_metadata(paths)
labs_meta["feature_upper"] = labs_meta["feature name"].str.upper()

creat_items = labs_meta[labs_meta["feature_upper"].str.contains("CREATININE")]["itemid"].tolist()

creat_df = fetch_labevents_timeseries(conn, hadm_ids, creat_items, "creatinine", 0.1, 20.0)
creat_df = ensure_time_day(creat_df)

if len(creat_df) == 0:
    raise RuntimeError("No creatinine measurements found.")

# Baseline creatinine (first 24h)
creat_df["admittime"] = creat_df["hadm_id"].map(hadm_unique.set_index("hadm_id")["admittime"])
creat_df["admittime"] = pd.to_datetime(creat_df["admittime"])
creat_df["charttime"] = pd.to_datetime(creat_df["charttime"])

first_24h = creat_df[creat_df["charttime"] <= creat_df["admittime"] + pd.Timedelta(days=1)]
baseline_creat_df = (
    first_24h.sort_values("charttime")
    .pivot_table(index="hadm_id", values="creatinine", aggfunc="first")
    .reset_index()
    .rename(columns={"creatinine": "baseline_creatinine"})
)

hadm_enriched = hadm_unique.merge(baseline_creat_df, on="hadm_id", how="inner")

# Filter patients with sufficient creatinine measurements
creat_counts = creat_df.groupby("hadm_id").size()
valid_hadm_ids = creat_counts[creat_counts >= MIN_TARGET_MEASUREMENTS].index.tolist()

creatinine_ts = creat_df[creat_df["hadm_id"].isin(valid_hadm_ids)].merge(
    hadm_enriched[["hadm_id", "baseline_creatinine", "age", "gender", "hospital_expire_flag"]],
    on="hadm_id",
    how="left",
)
creatinine_ts["creat_fold_change"] = creatinine_ts["creatinine"] / creatinine_ts["baseline_creatinine"]

hadm_final = hadm_enriched[hadm_enriched["hadm_id"].isin(valid_hadm_ids)].copy()

creat_daily = (
    creatinine_ts.groupby(["hadm_id", "time_day"])["creatinine"]
    .mean()
    .reset_index()
    .rename(columns={"creatinine": "creatinine_mean"})
)

hadm_daily = hadm_final.merge(creat_daily, on="hadm_id", how="inner")
hadm_daily = hadm_daily.drop_duplicates(subset=["hadm_id", "time_day"])

# Load dialysis data (matches notebook logic)
hadm_subset = tuple(hadm_final["hadm_id"].tolist())
dialysis_query = """
WITH dialysis_procedures AS (
    -- 1. Procedures (MetaVision)
    SELECT DISTINCT
        i.hadm_id::INT as hadm_id,
        MIN(EXTRACT(EPOCH FROM (pe.starttime::timestamp - a.admittime::timestamp)) / 86400) as dialysis_start_time,
        'procedureevents_mv' as source
    FROM procedureevents_mv pe
    INNER JOIN icustays i ON pe.hadm_id = i.hadm_id
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    WHERE
        pe.itemid IN (225805, 225809, 225802, 225803, 225441)
        AND pe.starttime::timestamp BETWEEN a.admittime::timestamp AND a.dischtime::timestamp
        AND i.hadm_id::INT IN {hadm_ids}
    GROUP BY i.hadm_id

    UNION

    -- 2. Input Events (dialysis solutions)
    SELECT DISTINCT
        i.hadm_id::INT as hadm_id,
        MIN(EXTRACT(EPOCH FROM (ie.starttime::timestamp - a.admittime::timestamp)) / 86400) as dialysis_start_time,
        'inputevents_mv' as source
    FROM inputevents_mv ie
    INNER JOIN icustays i ON ie.hadm_id = i.hadm_id
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    WHERE
        ie.itemid IN (227536, 227525)
        AND ie.starttime::timestamp BETWEEN a.admittime::timestamp AND a.dischtime::timestamp
        AND i.hadm_id::INT IN {hadm_ids}
    GROUP BY i.hadm_id

    UNION

    -- 3. Chart Events (dialysis documentation)
    SELECT DISTINCT
        i.hadm_id::INT as hadm_id,
        MIN(EXTRACT(EPOCH FROM (ce.charttime::timestamp - a.admittime::timestamp)) / 86400) as dialysis_start_time,
        'chartevents' as source
    FROM chartevents ce
    INNER JOIN icustays i ON ce.hadm_id = i.hadm_id
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    WHERE
        ce.itemid IN (226499, 224154, 225810, 225806, 225807, 225959)
        AND ce.charttime::timestamp BETWEEN a.admittime::timestamp AND a.dischtime::timestamp
        AND i.hadm_id::INT IN {hadm_ids}
    GROUP BY i.hadm_id

    -- ICD-9 codes are not used: they carry no timestamp, so they cannot place
    -- dialysis initiation inside a prediction window. (The former ICD block
    -- assigned them time 0.0, which through MIN() below overwrote the real
    -- start time of every admission that also had a timed record; 3995 and
    -- 5498 are procedure codes and never matched in diagnoses_icd anyway.)
)
SELECT
    hadm_id,
    MIN(dialysis_start_time) as dialysis_start_time,
    STRING_AGG(DISTINCT source, ', ' ORDER BY source) as sources
FROM dialysis_procedures
GROUP BY hadm_id
ORDER BY hadm_id
"""

try:
    dialysis_df = conn.execute(dialysis_query.format(hadm_ids=hadm_subset)).fetchdf()
    dialysis_df.columns = dialysis_df.columns.str.lower()
    print(f"✓ Loaded dialysis data: {len(dialysis_df)} patients")
except Exception as e:
    print(f"⚠️  Error loading dialysis data: {e}")
    dialysis_df = pd.DataFrame(columns=["hadm_id", "dialysis_start_time", "sources"])

dialysis_df.columns = dialysis_df.columns.str.lower()

# Define AKI Stage 3 outcome (matches notebook logic)
PREDICTION_GAP_DAYS = 0.5
PREDICTION_WINDOW_DAYS = 4.0

aki_events = []
excluded_counts = {"already_aki": 0, "no_future_data": 0, "within_gap": 0}

for hadm_id, grp in hadm_daily.groupby("hadm_id"):
    grp = grp.sort_values("time_day")
    baseline = grp["baseline_creatinine"].iloc[0]

    for i in range(len(grp)):
        current_time = grp.iloc[i]["time_day"]
        current_creat = grp.iloc[i]["creatinine_mean"]

        if current_creat >= 3 * baseline or current_creat >= 4.0:
            excluded_counts["already_aki"] += 1
            continue

        prediction_start = current_time + PREDICTION_GAP_DAYS + 1
        prediction_end = current_time + PREDICTION_GAP_DAYS + PREDICTION_WINDOW_DAYS + 1

        future_window = creatinine_ts[
            (creatinine_ts["hadm_id"] == hadm_id)
            & (creatinine_ts["time_days"] >= prediction_start)
            & (creatinine_ts["time_days"] <= prediction_end)
        ]

        if len(future_window) == 0:
            excluded_counts["no_future_data"] += 1
            continue

        aki_stage3_creat = (
            (future_window["creatinine"] >= 3 * baseline).any()
            or ((future_window["creatinine"] >= 4.0) & (future_window["creatinine"] - current_creat >= 0.5)).any()
        )

        dialysis_in_window = False
        if hadm_id in dialysis_df["hadm_id"].values:
            dialysis_time = dialysis_df[dialysis_df["hadm_id"] == hadm_id]["dialysis_start_time"].iloc[0]
            dialysis_in_window = prediction_start <= dialysis_time <= prediction_end

        target_aki = int(aki_stage3_creat or dialysis_in_window)

        aki_events.append({
            "hadm_id": hadm_id,
            "time_day": current_time,
            "target_aki_stage3": target_aki,
            "current_creatinine": current_creat,
            "baseline_creatinine": baseline,
            "creat_fold_change": current_creat / baseline,
            "prediction_gap_days": PREDICTION_GAP_DAYS,
            "prediction_window_days": PREDICTION_WINDOW_DAYS,
        })

aki_outcomes_df = pd.DataFrame(aki_events)
aki_outcomes_df = aki_outcomes_df.sort_values(["hadm_id", "time_day"])
positive_first = aki_outcomes_df[aki_outcomes_df["target_aki_stage3"] == 1].groupby("hadm_id").first().reset_index()
all_negatives = aki_outcomes_df[aki_outcomes_df["target_aki_stage3"] == 0]
aki_outcomes_df = (
    pd.concat([all_negatives, positive_first], ignore_index=True)
    .sort_values(["hadm_id", "time_day"])
    .reset_index(drop=True)
)

creatinine_ts.to_csv(OUTPUT_DIR / "creatinine_timeseries.csv", index=False)
hadm_final.to_csv(OUTPUT_DIR / "hadm_cohort.csv", index=False)
aki_outcomes_df.to_csv(OUTPUT_DIR / "aki_outcomes.csv", index=False)

# Build prediction dataset from vitals/labs
vital_items = vital_meta["itemid"].tolist()
lab_items = labs_meta["itemid"].tolist()
vitals_df, labs_df = fetch_vitals_labs(conn, hadm_ids=hadm_final["hadm_id"].tolist(), vital_items=vital_items, lab_items=lab_items)

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

    hadm_pred = hadm_final.merge(vitals_labs_pivot, on=["hadm_id"], how="inner")
    hadm_pred["time_day"] = ((pd.to_datetime(hadm_pred["charttime"]) - pd.to_datetime(hadm_pred["admittime"]))
                             .dt.total_seconds() / 86400.0).astype(int)
    hadm_pred = hadm_pred.drop_duplicates(subset=["hadm_id", "time_day"])

    # Forward-fill imputation within each admission
    hadm_pred = hadm_pred.sort_values(["hadm_id", "time_day"])
    hadm_pred = ffill_within_stay(hadm_pred, "hadm_id")

    # Drop columns with >70% missing
    missing_pct = hadm_pred.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index.tolist()
    for col in ["hadm_id", "time_day", "creatinine_mean", "hospital_expire_flag", "age", "gender", "baseline_creatinine"]:
        if col in cols_to_drop:
            cols_to_drop.remove(col)
    hadm_pred = hadm_pred.drop(columns=cols_to_drop)

    hadm_pred.to_csv(OUTPUT_DIR / "aki_prediction_dataset.csv", index=False)
    print("✓ Saved aki_prediction_dataset.csv")
else:
    print("⚠️ Skipped prediction dataset (missing vitals/labs)")

print("✓ Saved creatinine_timeseries.csv, hadm_cohort.csv, and aki_outcomes.csv")
print(f"   AKI Stage 3 events (first per patient): {aki_outcomes_df['target_aki_stage3'].sum():,}")
print(f"   Excluded already AKI: {excluded_counts['already_aki']:,}")
print(f"   Excluded no future data: {excluded_counts['no_future_data']:,}")

conn.close()
