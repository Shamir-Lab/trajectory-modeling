"""
MIMIC Ventilator Preprocessing Script

Extracts:
- Mechanically ventilated cohort
- P/F ratio time series (PaO2 / FiO2 daily means)

Outputs:
- results/mimic/ventilator/pf_ratio_timeseries.csv
- results/mimic/ventilator/hadm_cohort.csv
- results/mimic/ventilator/ventilator_outcomes.csv
- results/mimic/ventilator/ventilator_prediction_dataset.csv

Usage:
    python scripts/mimic_preprocessing/ventilator_preprocess.py
"""

import os
import pandas as pd
from pathlib import Path

from common import (
    MIMICPaths,
    connect_db,
    load_metadata,
    fetch_vitals_labs,
    ensure_time_day,
    aggregate_vitals_labs,
    ffill_within_stay,
    ICU_LOS_SQL,
    MIN_TARGET_MEASUREMENTS,
    MISSINGNESS_THRESHOLD,
    drop_high_missing_columns,
)

OUTPUT_DIR = Path(os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet")) / "mimic" / "ventilator"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MIMIC Ventilator Preprocessing")
print("=" * 80)

paths = MIMICPaths()
conn = connect_db(paths.db_path)

hadm_query = f"""
WITH ventilated_patients AS (
    SELECT DISTINCT
        i.hadm_id::INTEGER AS hadm_id,
        MIN(ce.charttime::TIMESTAMP) as vent_start
    FROM chartevents ce
    INNER JOIN icustays i ON ce.icustay_id = i.icustay_id
    WHERE 
        ce.itemid IN (720, 223849)
        AND ce.value IS NOT NULL
    GROUP BY i.hadm_id
)
SELECT 
    i.subject_id::INTEGER AS subject_id,
    i.hadm_id::INTEGER AS hadm_id,
    a.admittime::TIMESTAMP as admittime,
    a.dischtime::TIMESTAMP as dischtime,
    vp.vent_start::TIMESTAMP as vent_start,
    CAST(a.dischtime::DATE - a.admittime::DATE AS REAL) as los_days,
    p.gender,
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) as age,
    a.hospital_expire_flag::INT as hospital_expire_flag,
    a.discharge_location
FROM icustays i
INNER JOIN patients p ON i.subject_id = p.subject_id
INNER JOIN admissions a ON i.hadm_id = a.hadm_id
INNER JOIN ventilated_patients vp ON i.hadm_id = vp.hadm_id
WHERE 
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) BETWEEN 18 AND 90
    AND {ICU_LOS_SQL}
"""

hadm_df = conn.execute(hadm_query).fetchdf()
hadm_df.columns = hadm_df.columns.str.lower()
print(f"✓ Loaded {len(hadm_df):,} ventilated admissions")

hadm_ids = hadm_df["hadm_id"].astype(int).tolist()

vital_meta, labs_meta = load_metadata(paths)

labs_meta["feature_upper"] = labs_meta["feature name"].str.upper()

# Use labevents for PaO2/FiO2 to match notebook logic
pao2_items = labs_meta[labs_meta["feature_upper"].str.contains("PAO2")]["itemid"].tolist()
fio2_items = labs_meta[labs_meta["feature_upper"].str.contains("FIO2")]["itemid"].tolist()

# Fallback to known MIMIC-III itemids if metadata search misses
if len(pao2_items) == 0:
    pao2_items = [50821]
if len(fio2_items) == 0:
    fio2_items = [50816]

all_items = sorted(set(pao2_items + fio2_items))

bg_df = pd.DataFrame()
if hadm_ids and all_items:
    bg_query = f"""
    SELECT
        le.hadm_id::INTEGER AS hadm_id,
        le.charttime::TIMESTAMP AS charttime,
        EXTRACT(EPOCH FROM (le.charttime::TIMESTAMP - a.admittime::TIMESTAMP)) / 86400.0 as time_days,
        MAX(CASE WHEN le.itemid IN {tuple(pao2_items)} THEN CAST(le.valuenum AS REAL) END) as pao2,
        MAX(CASE WHEN le.itemid IN {tuple(fio2_items)} THEN CAST(le.valuenum AS REAL) END) as fio2
    FROM labevents le
    INNER JOIN admissions a ON le.hadm_id = a.hadm_id
    WHERE
        le.itemid IN {tuple(all_items)}
        AND le.hadm_id::INTEGER IN {tuple(hadm_ids)}
        AND le.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.dischtime::TIMESTAMP
    GROUP BY le.hadm_id, le.charttime, a.admittime
    HAVING MAX(CASE WHEN le.itemid IN {tuple(pao2_items)} THEN CAST(le.valuenum AS REAL) END) IS NOT NULL
        AND MAX(CASE WHEN le.itemid IN {tuple(fio2_items)} THEN CAST(le.valuenum AS REAL) END) IS NOT NULL
    """
    bg_df = conn.execute(bg_query).fetchdf()
    if len(bg_df) > 0:
        bg_df.columns = bg_df.columns.str.lower()

if len(bg_df) == 0:
    print("⚠️ Missing PaO2 or FiO2 data. Skipping P/F ratio computation.")
else:
    # Normalize FiO2 to fraction if reported as percentage
    bg_df.loc[bg_df["fio2"] > 1.0, "fio2"] = bg_df.loc[bg_df["fio2"] > 1.0, "fio2"] / 100.0
    bg_df = bg_df[(bg_df["fio2"] >= 0.21) & (bg_df["fio2"] <= 1.0)]
    bg_df["pf_ratio"] = bg_df["pao2"] / bg_df["fio2"]
    bg_df = bg_df[(bg_df["pf_ratio"] >= 50) & (bg_df["pf_ratio"] <= 600)]
    bg_df = ensure_time_day(bg_df)

    # Keep admissions with enough measurements of the target clinical parameter
    pf_counts = bg_df.groupby("hadm_id").size()
    pf_valid = pf_counts[pf_counts >= MIN_TARGET_MEASUREMENTS].index
    bg_df = bg_df[bg_df["hadm_id"].isin(pf_valid)].copy()
    hadm_df = hadm_df[hadm_df["hadm_id"].isin(pf_valid)].copy()
    hadm_ids = hadm_df["hadm_id"].astype(int).tolist()
    print(f"✓ {len(pf_valid):,} admissions with >= {MIN_TARGET_MEASUREMENTS} P/F measurements")

    # Baseline P/F ratio (first 24h after vent start)
    bg_df = bg_df.merge(hadm_df[["hadm_id", "vent_start"]], on="hadm_id", how="left")
    bg_df["charttime"] = pd.to_datetime(bg_df["charttime"])
    bg_df["vent_start"] = pd.to_datetime(bg_df["vent_start"])

    first_24h = bg_df[bg_df["charttime"] <= bg_df["vent_start"] + pd.Timedelta(hours=24)]
    baseline_pf = (
        first_24h.sort_values("charttime")
        .groupby("hadm_id")["pf_ratio"]
        .first()
        .reset_index()
        .rename(columns={"pf_ratio": "baseline_pf_ratio"})
    )

    pf_daily = bg_df.groupby(["hadm_id", "time_day"])["pf_ratio"].mean().reset_index()
    pf_daily["time_days"] = pf_daily["time_day"].astype(float)

    pf_ts = pf_daily.merge(baseline_pf, on="hadm_id", how="left")
    pf_ts = pf_ts.merge(hadm_df[["hadm_id", "age", "gender", "hospital_expire_flag"]], on="hadm_id", how="left")
    pf_ts["pf_improvement"] = pf_ts["pf_ratio"] - pf_ts["baseline_pf_ratio"]

    pf_ts.to_csv(OUTPUT_DIR / "pf_ratio_timeseries.csv", index=False)
    print("✓ Saved pf_ratio_timeseries.csv")

    # Define weaning success outcome (matches notebook logic)
    PREDICTION_GAP_DAYS = 0.5
    PREDICTION_WINDOW_DAYS = 2.0

    weaning_events = []
    n_good = 0
    n_no_data = 0

    hadm_final = pf_ts.copy()

    for hadm_id, grp in hadm_final.groupby("hadm_id"):
        grp = grp.sort_values("time_day")
        mortality = hadm_final[hadm_final["hadm_id"] == hadm_id]["hospital_expire_flag"].iloc[0]

        for i in range(len(grp)):
            current_time = grp.iloc[i]["time_day"]
            current_pf = grp.iloc[i]["pf_ratio"]

            if current_pf >= 400:
                n_good += 1
                continue

            prediction_start = current_time + PREDICTION_GAP_DAYS + 1
            prediction_end = current_time + PREDICTION_GAP_DAYS + PREDICTION_WINDOW_DAYS + 1

            future_window = pf_ts[
                (pf_ts["hadm_id"] == hadm_id)
                & (pf_ts["time_days"] >= prediction_start)
                & (pf_ts["time_days"] <= prediction_end)
            ]

            if len(future_window) == 0:
                n_no_data += 1
                continue

            weaning_ready = (future_window["pf_ratio"] >= 300).any()
            target = int(weaning_ready and not mortality)

            weaning_events.append({
                "hadm_id": hadm_id,
                "time_day": current_time,
                "target_weaning_success": target,
                "current_pf_ratio": current_pf,
            })

    outcome_df = pd.DataFrame(weaning_events)
    outcome_df = outcome_df.sort_values(["hadm_id", "time_day"])
    positive_first = outcome_df[outcome_df["target_weaning_success"] == 1].groupby("hadm_id").first().reset_index()
    all_negatives = outcome_df[outcome_df["target_weaning_success"] == 0]
    outcome_df = (
        pd.concat([all_negatives, positive_first], ignore_index=True)
        .sort_values(["hadm_id", "time_day"])
        .reset_index(drop=True)
    )

    outcome_df.to_csv(OUTPUT_DIR / "ventilator_outcomes.csv", index=False)
    print("✓ Saved ventilator_outcomes.csv")
    print(f"   Successful weaning (first per patient): {outcome_df['target_weaning_success'].sum():,}")
    print(f"   Skipped due to good PF: {n_good:,}")
    print(f"   Skipped due to no future data: {n_no_data:,}")

    # Build prediction dataset from vitals/labs
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

        hadm_pred = hadm_df.merge(vitals_labs_pivot, on=["hadm_id"], how="inner")
        hadm_pred["time_day"] = ((pd.to_datetime(hadm_pred["charttime"]) - pd.to_datetime(hadm_pred["admittime"]))
                                 .dt.total_seconds() / 86400.0).astype(int)
        hadm_pred = hadm_pred.drop_duplicates(subset=["hadm_id", "time_day"])

        hadm_pred = hadm_pred.merge(baseline_pf, on="hadm_id", how="left")
        pf_day = pf_ts[["hadm_id", "time_day", "pf_ratio", "pf_improvement"]].drop_duplicates()
        hadm_pred = hadm_pred.merge(pf_day, on=["hadm_id", "time_day"], how="left")

        # Forward-fill imputation within each admission
        hadm_pred = hadm_pred.sort_values(["hadm_id", "time_day"])
        hadm_pred = ffill_within_stay(hadm_pred, "hadm_id")

        # Drop columns with >70% missing
        missing_pct = hadm_pred.isnull().mean()
        cols_to_drop = missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index.tolist()
        for col in ["hadm_id", "time_day", "pf_ratio", "pf_improvement", "hospital_expire_flag", "age", "gender", "baseline_pf_ratio"]:
            if col in cols_to_drop:
                cols_to_drop.remove(col)
        hadm_pred = hadm_pred.drop(columns=cols_to_drop)

        hadm_pred.to_csv(OUTPUT_DIR / "ventilator_prediction_dataset.csv", index=False)
        print("✓ Saved ventilator_prediction_dataset.csv")
    else:
        print("⚠️ Skipped prediction dataset (missing vitals/labs)")

hadm_df.to_csv(OUTPUT_DIR / "hadm_cohort.csv", index=False)
print("✓ Saved hadm_cohort.csv")
conn.close()
