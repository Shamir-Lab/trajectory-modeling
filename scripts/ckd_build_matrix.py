#!/usr/bin/env python3
"""
CKD patient-year outcome matrix construction for trajectory_representation_chronic
(All of Us).

Assembles a continuous patient-year feature matrix from raw All of Us extracts
(agg_df.parquet, egfr_df.parquet, raw_df_initial.parquet):

  continuous yearly spine -> pivoted yearly summary stats -> limited LOCF ->
  static demographics -> 3-year rolling eGFR stats -> MCMC/Bootstrap
  trajectory probabilities -> ESKD/dialysis/death targets.

Two corrections relative to the original outcome-matrix script this replaces:
  1. A continuous per-patient year spine (build_patient_year_spine()) so LOCF
     and prediction points aren't silently skipped for years with zero
     measurements.
  2. Competing-risk censoring (death censors ESKD/dialysis, not just death
     itself) plus a combined ESKD-or-death outcome — see
     generate_dynamic_targets().

Input:   raw All of Us extracts under --raw-data-dir (agg_df.parquet,
         egfr_df.parquet, raw_df_initial.parquet, optional
         egfr_mcmc_probs_*.csv / egfr_boot_probs_*.csv trajectory files).
Output:  a single parquet feature matrix at --output, consumed by
         ckd_trajectory_analysis.py.
"""

from __future__ import annotations

import argparse
import gc
import glob
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed as jdelayed

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_RAW_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "final_dynamic_features.parquet"

# Maximum allowed age (in years) for a carried-forward last-observation value.
# Values older than this revert to NaN rather than being carried forward
# indefinitely (limited LOCF).
MAX_LOCF_YEARS = 1.0

# Feature names (agg_df.feature_name values) dropped entirely from the
# matrix — found via ckd_investigate_missingness.py to be >90% missing with
# no evidence of a pipeline bug (i.e. genuinely rarely-ordered labs in this
# cohort), so kept only as noise/dimensionality for downstream models.
EXCLUDED_FEATURES = {"WBC", "CRP", "INR"}

# Feature names exempt from the MAX_LOCF_YEARS expiration: their "_value_last"
# is forward-filled with no limit instead of reverting to NaN after
# MAX_LOCF_YEARS. Body height was flagged by ckd_investigate_missingness.py
# as a static candidate (near-zero within-patient variation) — in adults it
# doesn't meaningfully change, so a 1-year staleness cutoff destroys a value
# that was never actually stale.
UNLIMITED_LOCF_FEATURES = {"Body height"}

# Applied to MCMC/Bootstrap trajectory probability columns the same way as
# MAX_LOCF_YEARS is applied to lab "_value_last" columns — trajectory columns
# are merged onto the continuous per-patient year spine at years_lower
# granularity (no day-level timestamp), so "up to N years old" here just
# means an unbroken forward-fill of at most N rows on that gap-free grid.
TRAJECTORY_LOCF_YEARS = 1.0

# Minimum fraction of feature columns that must be missing for a patient-year
# row to be dropped, evaluated once after all LOCF (lab + trajectory) has
# been applied. Rows this sparse would be dominated by median/zero-imputed
# values rather than real signal, biasing models toward the imputation
# strategy instead of the data.
ROW_MISSING_THRESHOLD = 0.70


# ============================================================================
# Memory / Merge Utilities
# ============================================================================


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric dtypes in-place to reduce memory footprint."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif str(col_type)[:5] == "float":
                df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(
        f"  Memory usage: {start_mem:.1f} MB -> {end_mem:.1f} MB "
        f"({100 * (start_mem - end_mem) / max(start_mem, 1e-9):.1f}% reduction)"
    )
    return df


def safe_left_merge(
    base_df: pd.DataFrame,
    add_df: pd.DataFrame,
    key_cols: list[str],
    add_name: str = "add_df",
) -> pd.DataFrame:
    base_dup = int(base_df.duplicated(subset=key_cols).sum())
    add_dup = int(add_df.duplicated(subset=key_cols).sum())

    if base_dup > 0:
        logger.warning(f"  Base table has {base_dup:,} duplicate keys on {key_cols}; keeping first")
        base_df = base_df.drop_duplicates(subset=key_cols, keep="first")

    if add_dup > 0:
        prob_cols = [c for c in add_df.columns if c not in key_cols]
        logger.warning(f"  {add_name} has {add_dup:,} duplicate keys; aggregating means")
        add_df = add_df.groupby(key_cols, observed=False, as_index=False)[prob_cols].mean()

    return base_df.merge(add_df, on=key_cols, how="left", validate="one_to_one")


def _patient_egfr_stats(pid, pid_df, time_col, windowing_col, value_col, lookback_years):
    """3-year rolling eGFR stats (mean/max/min/change/trend/std) per patient-year."""
    pid_df = pid_df.sort_values(time_col)
    times = pid_df[time_col].to_numpy(dtype=float)
    values = pid_df[value_col].to_numpy(dtype=float)
    windows = pid_df[windowing_col].to_numpy(dtype=float)

    rows = []
    for w in np.unique(windows):
        mask = (times >= w - lookback_years) & (times <= w + 1)
        wv, wt = values[mask], times[mask]
        n = len(wv)
        if n == 0:
            rows.append((pid, w, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        # A 2-point "trend" is just the line through two values — noisy and
        # easily dominated by a single erroneous draw. Require >= 3 points
        # before trusting a fitted slope; otherwise report unknown, not flat.
        trend = np.nan
        if n >= 3:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", np.exceptions.RankWarning)
                    trend = float(np.polyfit(wt, wv, 1)[0])
            except Exception:
                trend = np.nan
        rows.append((
            pid, w,
            float(np.mean(wv)), float(np.max(wv)), float(np.min(wv)),
            float(wv[-1] - wv[0]), trend,
            float(np.std(wv)) if n > 1 else 0.0,
        ))
    return rows


# ============================================================================
# Patient-Year Spine
# ============================================================================


def build_patient_year_spine(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Continuous per-patient year spine: one row per integer year from each
    patient's first measurement year through their maximum follow-up year,
    with no gaps — even for years with zero measurements.

    Without this, pivoting yearly_stats directly only yields rows for years
    that had at least one measurement, which silently skips valid prediction
    points and breaks LOCF across gap years.
    """
    year_range = agg_df.groupby("person_id", observed=False)["years_lower"].agg(["min", "max"])
    year_range = year_range.reset_index().rename(columns={"min": "min_year", "max": "max_year"})

    n_years = (year_range["max_year"] - year_range["min_year"] + 1).astype(int).to_numpy()
    person_id_rep = np.repeat(year_range["person_id"].to_numpy(), n_years)
    offsets = np.concatenate([np.arange(n) for n in n_years]) if len(n_years) else np.array([], dtype=int)
    years_lower_rep = np.repeat(year_range["min_year"].to_numpy(), n_years) + offsets

    spine_df = pd.DataFrame({"person_id": person_id_rep, "years_lower": years_lower_rep.astype(np.int16)})
    logger.info(f"  Patient-year spine: {len(spine_df):,} rows across {spine_df['person_id'].nunique():,} patients")
    return spine_df


# ============================================================================
# Target Definition (competing-risk censoring, combined outcome)
# ============================================================================


def generate_dynamic_targets(
    data_path: Path, dynamic_df: pd.DataFrame, patient_followup: pd.DataFrame
) -> pd.DataFrame:
    """
    Define ESKD / dialysis / death targets at 1-3 and 2-5 year horizons from
    raw_df_initial.parquet, streamed in chunks to bound memory use.

    Competing-risk censoring: if a patient dies within a target's outcome
    window and the ESKD/dialysis event itself did not occur in that window,
    target_eskd_{window} / target_dialysis_{window} are set to pd.NA rather
    than 0 — death makes it impossible to observe whether ESKD/dialysis would
    have occurred, so labeling it a negative would bias the target. Death
    occurring *after* the ESKD/dialysis event within the same window does not
    censor — the event was already observed.

    target_eskd_or_death_{window} is 1 if either event occurs in the window,
    0 if neither occurs and follow-up is sufficient, and is computed from raw
    event times (not from the already-censored ESKD column) since death is
    itself one of the two events of interest for this combined outcome.
    """
    columns_to_load = ["person_id", "ckd_index_date", "death_date", "eskd_date", "dialysis_date"]
    logger.info("  Streaming outcome dates from raw_df_initial.parquet in chunks...")

    parquet_file = pq.ParquetFile(data_path / "raw_df_initial.parquet")
    chunks = []
    for batch in parquet_file.iter_batches(columns=columns_to_load, batch_size=250_000):
        chunk = batch.to_pandas(ignore_metadata=True)
        chunk = chunk.drop_duplicates(subset=["person_id"])

        date_cols = ["ckd_index_date", "death_date", "eskd_date", "dialysis_date"]
        for col in date_cols:
            chunk[col] = pd.to_datetime(chunk[col], errors="coerce")

        for event in ("death", "eskd", "dialysis"):
            chunk[f"days_to_{event}"] = (
                chunk[f"{event}_date"] - chunk["ckd_index_date"]
            ).dt.days.astype(np.float32)

        chunk.drop(columns=date_cols, inplace=True, errors="ignore")
        chunks.append(chunk)

    outcomes_raw = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["person_id"])
    del chunks, parquet_file
    gc.collect()

    logger.info("  Merging outcome dates into the patient-year spine...")
    targets = dynamic_df[["person_id", "years_lower"]].copy()
    targets = targets.merge(outcomes_raw, on="person_id", how="left")
    del outcomes_raw
    gc.collect()

    targets = targets.merge(patient_followup, on="person_id", how="left")
    targets["prediction_time_days"] = (targets["years_lower"] + 1) * 365.25

    windows = {
        "1_to_3_years": {"start_gap_days": 1 * 365.25, "end_gap_days": 3 * 365.25},
        "2_to_5_years": {"start_gap_days": 2 * 365.25, "end_gap_days": 5 * 365.25},
    }

    logger.info("  Computing time-horizon targets (with competing-risk censoring)...")
    for win_name, win_params in windows.items():
        window_start = targets["prediction_time_days"] + win_params["start_gap_days"]
        window_end = targets["prediction_time_days"] + win_params["end_gap_days"]
        days_to_fu = targets["max_followup_days"]
        days_to_death = targets["days_to_death"]

        for event in ("death", "eskd", "dialysis"):
            col_name = f"target_{event}_{win_name}"
            days_to_event = targets[f"days_to_{event}"]

            col = pd.Series(pd.NA, dtype=pd.Int8Dtype(), index=targets.index)
            mask_event_in_window = (days_to_event >= window_start) & (days_to_event <= window_end)
            col.loc[mask_event_in_window] = 1
            resolved = mask_event_in_window.copy()

            if event in ("eskd", "dialysis"):
                mask_death_censor = (~resolved) & (days_to_death >= window_start) & (days_to_death <= window_end)
                col.loc[mask_death_censor] = pd.NA  # explicit: competing-risk censored, not a negative
                resolved = resolved | mask_death_censor

            mask_event_after = (~resolved) & (days_to_event > window_end)
            col.loc[mask_event_after] = 0
            resolved = resolved | mask_event_after

            mask_no_event_sufficient_fu = (~resolved) & days_to_event.isna() & (days_to_fu >= window_end)
            col.loc[mask_no_event_sufficient_fu] = 0

            targets[col_name] = col

        # Combined outcome: 1 if ESKD or death occurs in-window, computed from
        # raw event times so it is independent of the ESKD column's censoring.
        days_to_eskd = targets["days_to_eskd"]
        combined_col = f"target_eskd_or_death_{win_name}"
        combined = pd.Series(pd.NA, dtype=pd.Int8Dtype(), index=targets.index)

        mask_combined_event = (
            ((days_to_eskd >= window_start) & (days_to_eskd <= window_end))
            | ((days_to_death >= window_start) & (days_to_death <= window_end))
        )
        combined.loc[mask_combined_event] = 1

        eskd_after_or_absent = days_to_eskd.isna() | (days_to_eskd > window_end)
        death_after_or_absent = days_to_death.isna() | (days_to_death > window_end)
        mask_combined_neg = (~mask_combined_event) & eskd_after_or_absent & death_after_or_absent & (days_to_fu >= window_end)
        combined.loc[mask_combined_neg] = 0

        targets[combined_col] = combined

    drop_cols = [
        "days_to_death", "days_to_eskd", "days_to_dialysis",
        "max_followup_days", "prediction_time_days",
    ]
    return targets.drop(columns=[c for c in drop_cols if c in targets.columns])


# ============================================================================
# Trajectory-Based Subject Filtering
# ============================================================================


def exclude_subjects_without_trajectories(dynamic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop every row for a patient who has no MCMC or Bootstrap trajectory
    probability at all, across any of their patient-year rows.

    Keeps the cohort identical across all three downstream feature-set
    comparisons (Static+Dynamic / +MCMC / +Bootstrap) — without this, the
    Static+Dynamic arm would be evaluated on a larger, systematically
    different population than the trajectory arms, and any AUROC/AUPR
    difference could reflect the cohort rather than the added features.

    If neither trajectory source was found at all (no mcmc_*/*_boot columns
    in the matrix), this is a no-op — there is nothing to filter on, and
    dropping every subject would silently empty the matrix.
    """
    traj_prob_cols = [c for c in dynamic_df.columns if c.startswith("mcmc_") or c.startswith("boot_")]
    if not traj_prob_cols:
        logger.warning(
            "  No MCMC or Bootstrap trajectory columns present — skipping "
            "trajectory-based subject exclusion (nothing to filter on)."
        )
        return dynamic_df

    has_any_traj = dynamic_df[traj_prob_cols].notna().any(axis=1)
    ids_with_traj = set(dynamic_df.loc[has_any_traj, "person_id"].unique())

    n_before = dynamic_df["person_id"].nunique()
    n_after = len(ids_with_traj)
    n_dropped = n_before - n_after

    if n_dropped == 0:
        logger.info("  All patients have at least one trajectory value — no subjects excluded.")
        return dynamic_df

    rows_before = len(dynamic_df)
    dynamic_df = dynamic_df[dynamic_df["person_id"].isin(ids_with_traj)].reset_index(drop=True)
    logger.info(
        f"  Excluding {n_dropped:,} / {n_before:,} patients with no trajectory "
        f"features at all (rows: {rows_before:,} -> {len(dynamic_df):,})."
    )
    return dynamic_df


def apply_trajectory_locf(
    dynamic_df: pd.DataFrame, max_locf_years: float = TRAJECTORY_LOCF_YEARS
) -> pd.DataFrame:
    """
    Forward-fill MCMC/Bootstrap trajectory probability columns within each
    patient, limited to `max_locf_years` years old — the same limited-LOCF
    treatment already applied to the yearly lab/biomarker "_value_last"
    columns, extended to trajectory columns.

    Trajectory columns have no day-level measurement timestamp — they're
    merged onto the continuous per-patient year spine (built gap-free by
    build_patient_year_spine()) at years_lower granularity only. So "up to
    N years old" here is exactly an N-row-limited forward-fill on that grid,
    rather than the day-precise days-since-last-measurement comparison used
    for lab values. Fractional years are floored to whole rows.
    """
    traj_cols = [c for c in dynamic_df.columns if c.startswith("mcmc_") or c.startswith("boot_")]
    if not traj_cols:
        return dynamic_df

    limit = int(max_locf_years)
    logger.info(f"Applying {limit}-year LOCF to {len(traj_cols)} trajectory column(s)...")
    dynamic_df = dynamic_df.sort_values(["person_id", "years_lower"])
    dynamic_df[traj_cols] = dynamic_df.groupby("person_id")[traj_cols].ffill(limit=limit)
    return dynamic_df


def drop_sparse_feature_rows(
    dynamic_df: pd.DataFrame,
    exclude_cols: set[str],
    threshold: float = ROW_MISSING_THRESHOLD,
) -> pd.DataFrame:
    """
    Drop patient-year rows where >= `threshold` fraction of feature columns
    are missing, evaluated after all LOCF (lab + trajectory) has already
    been applied — rows this sparse would be dominated by median/zero-
    imputed values rather than real signal downstream, biasing models toward
    the imputation strategy instead of the data.

    exclude_cols: columns NOT counted as features when computing each row's
    missing fraction (e.g. person_id, years_lower — structural spine
    coordinates, not measurements; target_* columns, if present, since
    target availability reflects censoring/follow-up, not feature quality).
    """
    feature_cols = [c for c in dynamic_df.columns if c not in exclude_cols]
    if not feature_cols:
        return dynamic_df

    missing_frac = dynamic_df[feature_cols].isna().mean(axis=1)
    keep_mask = missing_frac < threshold
    n_dropped = int((~keep_mask).sum())

    if n_dropped == 0:
        logger.info(f"  No rows have >= {threshold:.0%} missing features — none dropped.")
        return dynamic_df

    logger.info(
        f"  Dropping {n_dropped:,} / {len(dynamic_df):,} patient-year rows with "
        f">= {threshold:.0%} missing features (post-LOCF, over {len(feature_cols)} feature columns)."
    )
    return dynamic_df.loc[keep_mask].reset_index(drop=True)


# ============================================================================
# Main Orchestrator
# ============================================================================


def create_dynamic_prediction_table(
    data_dir: str | Path,
    excluded_features: set[str] | None = None,
    unlimited_locf_features: set[str] | None = None,
    trajectory_locf_years: float = TRAJECTORY_LOCF_YEARS,
    row_missing_threshold: float = ROW_MISSING_THRESHOLD,
) -> pd.DataFrame:
    """
    Build the full patient-year feature matrix from raw All of Us extracts:
    continuous yearly spine -> pivoted summary stats -> limited LOCF ->
    static demographics -> 3-year rolling eGFR stats -> MCMC/Bootstrap
    trajectory probabilities (limited LOCF) -> subject/row filtering ->
    ESKD/dialysis/death targets.

    excluded_features: feature_name values dropped from agg_df before any
        aggregation (default: EXCLUDED_FEATURES).
    unlimited_locf_features: feature_name values whose "_value_last" is
        forward-filled with no staleness limit instead of expiring after
        MAX_LOCF_YEARS (default: UNLIMITED_LOCF_FEATURES).
    trajectory_locf_years: limited-LOCF window (in years) applied to MCMC/
        Bootstrap trajectory columns (default: TRAJECTORY_LOCF_YEARS). Pass 0
        to disable.
    row_missing_threshold: patient-year rows with a feature-missing fraction
        >= this value are dropped after all LOCF is applied (default:
        ROW_MISSING_THRESHOLD). Pass 1.0 (or higher) to disable.
    """
    excluded_features = EXCLUDED_FEATURES if excluded_features is None else set(excluded_features)
    unlimited_locf_features = (
        UNLIMITED_LOCF_FEATURES if unlimited_locf_features is None else set(unlimited_locf_features)
    )

    data_path = Path(data_dir)

    logger.info("Loading agg_df.parquet...")
    agg_df = pd.read_parquet(data_path / "agg_df.parquet")
    if excluded_features:
        n_before = len(agg_df)
        agg_df = agg_df[~agg_df["feature_name"].isin(excluded_features)]
        logger.info(
            f"  Excluding feature(s) {sorted(excluded_features)}: "
            f"{n_before - len(agg_df):,} raw rows dropped ({len(agg_df):,} remain)"
        )
    agg_df["years_since_index"] = agg_df["days_since_index"] / 365.25
    agg_df["years_lower"] = np.floor(agg_df["years_since_index"]).astype(np.int16)
    agg_df = agg_df.sort_values(["person_id", "feature_name", "days_since_index"])

    logger.info("Extracting max follow-up days per patient for censoring...")
    patient_followup = (
        agg_df.groupby("person_id", observed=False)["days_since_index"].max()
        .reset_index().rename(columns={"days_since_index": "max_followup_days"})
    )
    patient_followup["max_followup_days"] = patient_followup["max_followup_days"].astype(np.float32)

    logger.info("Building continuous patient-year spine...")
    spine_df = build_patient_year_spine(agg_df)

    logger.info("Aggregating yearly features (mean/min/max/std/last/days_since_index_max)...")
    grouped = agg_df.groupby(["person_id", "years_lower", "feature_name"], observed=False)
    yearly_stats = grouped.agg(
        value_min=("value_as_number", "min"),
        value_max=("value_as_number", "max"),
        value_mean=("value_as_number", "mean"),
        value_std=("value_as_number", "std"),
        value_last=("value_as_number", "last"),
        days_since_index_max=("days_since_index", "max"),
    ).reset_index()
    unique_features = yearly_stats["feature_name"].unique()

    logger.info("Pivoting to wide format...")
    pivot_df = yearly_stats.pivot_table(
        index=["person_id", "years_lower"],
        columns="feature_name",
        values=["value_min", "value_max", "value_mean", "value_std", "value_last", "days_since_index_max"],
        observed=False,
    )
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    logger.info("Merging pivoted aggregates onto the continuous spine (no-signal years become NaN, not skipped)...")
    dynamic_df = spine_df.merge(pivot_df, on=["person_id", "years_lower"], how="left")

    del agg_df, yearly_stats, pivot_df, grouped, spine_df
    gc.collect()

    # ------------------------------------------------------------------
    # Limited Last-Observation-Carried-Forward (LOCF)
    # ------------------------------------------------------------------
    logger.info(
        f"Applying LOCF for last values with a strict limit of {MAX_LOCF_YEARS} years "
        f"(unlimited for {sorted(unlimited_locf_features)})..."
    )
    dynamic_df = dynamic_df.sort_values(["person_id", "years_lower"])

    locf_cols = [c for c in dynamic_df.columns if c.endswith("_value_last") or c.endswith("_days_since_index_max")]
    dynamic_df[locf_cols] = dynamic_df.groupby("person_id")[locf_cols].ffill()

    logger.info("Computing time_from_last_measurement and enforcing LOCF expiration...")
    prediction_time_days = (dynamic_df["years_lower"] + 1) * 365.25
    max_locf_days = MAX_LOCF_YEARS * 365.25

    for feature in unique_features:
        max_col = f"{feature}_days_since_index_max"
        time_col = f"{feature}_time_from_last_measurement"
        val_col = f"{feature}_value_last"

        if max_col in dynamic_df.columns:
            dynamic_df[time_col] = (prediction_time_days - dynamic_df[max_col]).astype(np.float32)
            # Unlimited-LOCF features (e.g. Body height) were already
            # forward-filled above with no pandas-level limit — skip the
            # staleness cutoff so that fill is never reverted to NaN.
            if feature not in unlimited_locf_features:
                stale_mask = dynamic_df[time_col] > max_locf_days
                dynamic_df.loc[stale_mask, time_col] = np.nan
                if val_col in dynamic_df.columns:
                    dynamic_df.loc[stale_mask, val_col] = np.nan
            dynamic_df.drop(columns=[max_col], inplace=True)

    logger.info("Downcasting dynamic_df to save memory...")
    dynamic_df = reduce_mem_usage(dynamic_df)

    logger.info("Merging static demographics...")
    static_df = pd.read_parquet(data_path / "agg_df.parquet", columns=["person_id", "sex_at_birth", "age_at_index"])
    static_df = static_df.drop_duplicates(subset=["person_id"])
    dynamic_df = dynamic_df.merge(static_df, on="person_id", how="left")
    del static_df
    gc.collect()

    egfr_path = data_path / "egfr_df.parquet"
    if egfr_path.exists():
        logger.info("Calculating 3-year eGFR rolling statistics...")
        egfr_df = pd.read_parquet(egfr_path)
        if "years_since_index" not in egfr_df.columns:
            egfr_df["years_since_index"] = egfr_df["days_since_index"] / 365.25
        if "years_lower" not in egfr_df.columns:
            egfr_df["years_lower"] = np.floor(egfr_df["years_since_index"]).astype(np.int16)

        # backend="threading": Parallel's default "loky" backend forks worker
        # processes, which can deadlock indefinitely (no error, no output) when
        # forking from a process that has already initialized a threaded BLAS
        # (numpy/OpenBLAS/MKL) — the same fork-safety hazard fixed for
        # RandomForest in ckd_trajectory_analysis.py. Threading avoids it.
        all_rows = Parallel(n_jobs=-1, backend="threading")(
            jdelayed(_patient_egfr_stats)(
                pid, pid_df, "years_since_index", "years_lower", "value_as_number", lookback_years=3.0
            )
            for pid, pid_df in egfr_df.groupby("person_id", observed=False)
        )
        del egfr_df
        gc.collect()

        flat_rows = [row for rows in all_rows for row in rows]
        egfr_stats_df = pd.DataFrame(
            flat_rows,
            columns=[
                "person_id", "years_lower",
                "egfr_3yr_mean", "egfr_3yr_max", "egfr_3yr_min",
                "egfr_3yr_change", "egfr_3yr_trend", "egfr_3yr_std",
            ],
        )
        egfr_stats_df = reduce_mem_usage(egfr_stats_df)
        del all_rows, flat_rows
        gc.collect()

        dynamic_df = safe_left_merge(
            dynamic_df, egfr_stats_df, key_cols=["person_id", "years_lower"], add_name="eGFR 3yr stats"
        )
        del egfr_stats_df
        gc.collect()
    else:
        logger.warning(f"  {egfr_path} not found — skipping eGFR rolling statistics.")

    logger.info("Loading and merging trajectory probabilities...")

    def _load_concat_trajs(file_pattern: str, prefix: str = "", suffix: str = "") -> pd.DataFrame:
        files = glob.glob(str(data_path / "**" / file_pattern), recursive=True)
        if not files:
            logger.warning(f"  No files found matching pattern: {file_pattern}")
            return pd.DataFrame()
        df_list = []
        for f in files:
            try:
                df_list.append(pd.read_csv(f))
            except Exception:
                pass
        if not df_list:
            return pd.DataFrame()
        traj_df = pd.concat(df_list, ignore_index=True)
        # Drop stray pandas row-index columns (e.g. "Unnamed: 0") left behind
        # when a trajectory CSV was written with index=True — these are not
        # features and would otherwise leak into the model matrix.
        unnamed_cols = [c for c in traj_df.columns if str(c).startswith("Unnamed:")]
        if unnamed_cols:
            logger.warning(f"  Dropping stray index column(s) from {file_pattern}: {unnamed_cols}")
            traj_df = traj_df.drop(columns=unnamed_cols)

        # Guard against silently blending two inconsistent trajectory
        # computations (e.g. a stale file left alongside a rerun) — mean-
        # averaging duplicate (person_id, years_lower) keys via
        # safe_left_merge's generic fallback would hide exactly the kind of
        # data-integrity problem this needs to surface, not paper over.
        dup_mask = traj_df.duplicated(subset=["person_id", "years_lower"], keep=False)
        if dup_mask.any():
            dup_keys = traj_df.loc[dup_mask, ["person_id", "years_lower"]].drop_duplicates()
            raise ValueError(
                f"Duplicate (person_id, years_lower) keys found across {len(files)} file(s) "
                f"matching {file_pattern!r}: {len(dup_keys)} duplicated key(s), "
                f"{int(dup_mask.sum())} affected row(s). Sample:\n{dup_keys.head(10).to_string(index=False)}\n"
                f"This usually means two inconsistent trajectory computations (e.g. a stale run left "
                f"alongside a rerun) are both present under the matched files — resolve which is "
                f"authoritative before rebuilding the matrix."
            )

        traj_df = reduce_mem_usage(traj_df)
        rename_dict = {
            c: f"{prefix}{c}{suffix}" for c in traj_df.columns if c not in ("person_id", "years_lower")
        }
        return traj_df.rename(columns=rename_dict)

    # MCMC (Bayesian) trajectory probabilities — kept strictly separate from
    # Bootstrap. Both sources get a readable prefix ("mcmc_" / "boot_"), and
    # ckd_trajectory_analysis.py's organize_features() classifies a
    # trajectory column as Bootstrap specifically by the "boot_" prefix — so
    # this naming is load-bearing, not cosmetic; changing it silently
    # misclassifies Bootstrap columns as Bayesian (see organize_features()'s
    # traj_boot detection).
    mcmc_df = _load_concat_trajs("egfr_mcmc_probs_*.csv", prefix="mcmc_")
    if not mcmc_df.empty:
        dynamic_df = safe_left_merge(
            dynamic_df, mcmc_df, key_cols=["person_id", "years_lower"], add_name="MCMC trajectories"
        )
        del mcmc_df
        gc.collect()

    boot_df = _load_concat_trajs("egfr_boot_probs_*.csv", prefix="boot_")
    if not boot_df.empty:
        dynamic_df = safe_left_merge(
            dynamic_df, boot_df, key_cols=["person_id", "years_lower"], add_name="Bootstrap trajectories"
        )
        del boot_df
        gc.collect()

    if trajectory_locf_years > 0:
        dynamic_df = apply_trajectory_locf(dynamic_df, max_locf_years=trajectory_locf_years)

    dynamic_df = exclude_subjects_without_trajectories(dynamic_df)

    if row_missing_threshold < 1.0:
        # Evaluated on feature columns only — person_id/years_lower are
        # structural spine coordinates, not measurements, and target_*
        # columns don't exist yet at this point in the pipeline (generated
        # below), so this is exactly "all columns except id and years_lower".
        dynamic_df = drop_sparse_feature_rows(
            dynamic_df, exclude_cols={"person_id", "years_lower"}, threshold=row_missing_threshold
        )

    logger.info("Generating dynamic targets (ESKD / dialysis / death, competing-risk censored)...")
    if (data_path / "raw_df_initial.parquet").exists():
        target_df = generate_dynamic_targets(data_path, dynamic_df, patient_followup)
        dynamic_df = safe_left_merge(
            dynamic_df, target_df, key_cols=["person_id", "years_lower"], add_name="targets"
        )
        del target_df
        gc.collect()
    else:
        logger.warning("  raw_df_initial.parquet not found — skipping outcome generation.")

    logger.info(f"Final patient-year matrix: {dynamic_df.shape}")
    return dynamic_df


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the CKD patient-year outcome matrix from raw All of Us extracts."
    )
    parser.add_argument(
        "--raw-data-dir", type=str, default=str(DEFAULT_RAW_DATA_DIR),
        help="Directory containing agg_df.parquet, egfr_df.parquet, raw_df_initial.parquet, "
             "and optional egfr_mcmc_probs_*.csv / egfr_boot_probs_*.csv trajectory files.",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help="Output parquet path for the assembled feature matrix "
             "(consumed by ckd_trajectory_analysis.py's --data argument).",
    )
    parser.add_argument(
        "--exclude-features", nargs="*", default=None,
        help=f"Feature names to drop entirely before aggregation (default: {sorted(EXCLUDED_FEATURES)}). "
             "Pass with no values to disable exclusion.",
    )
    parser.add_argument(
        "--unlimited-locf-features", nargs="*", default=None,
        help=f"Feature names exempt from the {MAX_LOCF_YEARS}-year LOCF expiration — their last value is "
             f"forward-filled with no limit (default: {sorted(UNLIMITED_LOCF_FEATURES)}). "
             "Pass with no values to disable.",
    )
    parser.add_argument(
        "--trajectory-locf-years", type=float, default=TRAJECTORY_LOCF_YEARS,
        help=f"Limited-LOCF window (years) for MCMC/Bootstrap trajectory columns "
             f"(default: {TRAJECTORY_LOCF_YEARS}). Pass 0 to disable.",
    )
    parser.add_argument(
        "--row-missing-threshold", type=float, default=ROW_MISSING_THRESHOLD,
        help=f"Drop patient-year rows with a feature-missing fraction >= this value, evaluated "
             f"after all LOCF (default: {ROW_MISSING_THRESHOLD}). Pass 1.0 to disable.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    excluded_features = set(args.exclude_features) if args.exclude_features is not None else None
    unlimited_locf_features = (
        set(args.unlimited_locf_features) if args.unlimited_locf_features is not None else None
    )

    logger.info(f"Building patient-year matrix from raw extracts in: {args.raw_data_dir}")
    df = create_dynamic_prediction_table(
        args.raw_data_dir,
        excluded_features=excluded_features,
        unlimited_locf_features=unlimited_locf_features,
        trajectory_locf_years=args.trajectory_locf_years,
        row_missing_threshold=args.row_missing_threshold,
    )

    df.to_parquet(output_path, index=False)
    logger.info(f"✓ Matrix saved → {output_path}  (shape={df.shape})")


if __name__ == "__main__":
    main()
