#!/usr/bin/env python3
"""
Investigate missingness in the CKD patient-year matrix (output of
ckd_build_matrix.py) beyond a flat missing-rate table.

A flat "% missing per column" table can't distinguish between:
  - genuine sparsity (a lab that's rarely ordered — expected, handle via
    imputation/indicator features, not a bug)
  - a static/dynamic misclassification (a value that's constant per patient
    but got bucketed as a yearly-varying "dynamic" column, or vice versa)
  - a coverage/threshold artifact (e.g. trajectory probabilities that are
    only computed when a patient has enough observations in the lookback
    window, which looks like "missing" but is actually "not eligible")
  - a data-extraction cliff (missingness that jumps sharply after some
    years_lower value, suggesting the extract's coverage window ended
    partway through follow-up rather than patients simply not being tested)

This script checks all four and prints an actionable recommendation per
column group: keep as-is, add a "was measured" indicator, treat as static,
or investigate the extract further.

Usage:
    python ckd_investigate_missingness.py --data ../data/final_dynamic_features.parquet
    python ckd_investigate_missingness.py --data ... --raw-data-dir ../data  # adds trajectory-coverage cross-check
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

HIGH_MISSING_THRESHOLD = 0.90
NEAR_CONSTANT_THRESHOLD = 0.02  # coefficient of variation below this => candidate "static"
CLIFF_JUMP_THRESHOLD = 0.25     # missing-rate jump between consecutive years_lower => candidate extraction cliff


def _base_feature(col: str) -> str | None:
    """Strip a known summary/trajectory suffix to recover the underlying biomarker name."""
    suffixes = [
        "_value_mean", "_value_max", "_value_min", "_value_std", "_value_last",
        "_time_from_last_measurement", "_change", "_trend",
    ]
    for suf in suffixes:
        if col.endswith(suf):
            return col[: -len(suf)]
    return None


def check_unnamed_columns(df: pd.DataFrame) -> list[str]:
    stray = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if stray:
        logger.warning(f"Stray index columns still present (rebuild with the fixed ckd_build_matrix.py): {stray}")
    return stray


def missing_rate_table(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    exclude = {id_col, "years_lower"}
    exclude |= {c for c in df.columns if c.startswith("target_")}
    cols = [c for c in df.columns if c not in exclude]
    n = len(df)
    rows = [
        {"column": c, "n_missing": int(df[c].isna().sum()), "pct_missing": round(100 * df[c].isna().sum() / n, 2)}
        for c in cols
    ]
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def missingness_by_year(df: pd.DataFrame, value_last_cols: list[str]) -> pd.DataFrame:
    """
    Per-feature missing rate broken out by years_lower. A roughly flat curve
    is consistent with routine intermittent testing (expected sparsity). A
    sharp jump after some year suggests the extract's coverage window ended
    or a source table changed partway through the study period — that's an
    extraction issue, not patient-level sparsity, and warrants checking the
    raw source rather than just imputing around it.
    """
    rows = []
    for col in value_last_cols:
        by_year = df.groupby("years_lower", observed=True)[col].apply(lambda s: s.isna().mean())
        by_year = by_year.sort_index()
        if len(by_year) < 2:
            continue
        max_jump = float(by_year.diff().abs().max())
        rows.append({
            "feature": _base_feature(col) or col,
            "min_year_missing_rate": round(float(by_year.min()), 3),
            "max_year_missing_rate": round(float(by_year.max()), 3),
            "max_year_over_year_jump": round(max_jump, 3),
            "likely_extraction_cliff": max_jump >= CLIFF_JUMP_THRESHOLD,
        })
    return pd.DataFrame(rows).sort_values("max_year_over_year_jump", ascending=False)


def per_patient_constancy(df: pd.DataFrame, id_col: str, value_last_cols: list[str]) -> pd.DataFrame:
    """
    For each "_value_last" column, restrict to patients with >=2 non-null
    yearly observations and measure how much the value varies within a
    patient (coefficient of variation). A column with near-zero
    within-patient variation despite being bucketed as "dynamic" is a
    candidate for a static/dynamic misclassification (the height/weight bug
    found in a prior run is the general case of this pattern) — or, for
    genuinely-varying labs, confirms the dynamic bucketing is correct.
    """
    rows = []
    for col in value_last_cols:
        sub = df[[id_col, col]].dropna()
        counts = sub.groupby(id_col, observed=True)[col].count()
        multi_obs_ids = counts[counts >= 2].index
        if len(multi_obs_ids) < 5:
            continue
        multi = sub[sub[id_col].isin(multi_obs_ids)]
        grp = multi.groupby(id_col, observed=True)[col]
        means = grp.mean()
        stds = grp.std().fillna(0.0)
        cv = (stds / means.replace(0, np.nan)).abs()
        median_cv = float(cv.median()) if len(cv) else np.nan
        rows.append({
            "feature": _base_feature(col) or col,
            "n_patients_with_repeat_measurements": int(len(multi_obs_ids)),
            "median_within_patient_cv": round(median_cv, 4) if pd.notna(median_cv) else np.nan,
            "static_candidate": pd.notna(median_cv) and median_cv < NEAR_CONSTANT_THRESHOLD,
        })
    return pd.DataFrame(rows).sort_values("median_within_patient_cv", ascending=True, na_position="last")


def trajectory_coverage(df: pd.DataFrame, id_col: str, raw_data_dir: str | None) -> pd.DataFrame:
    """
    Report what fraction of patient-year rows have non-null MCMC / Bootstrap
    trajectory probabilities, and — if --raw-data-dir is given — cross-tab
    coverage against the number of eGFR readings available in each row's
    3-year lookback window. Trajectory classification typically requires a
    minimum number of observations to fit; if coverage tracks that
    threshold, the ~90% missingness is a legitimate "not enough data to
    classify" outcome rather than a pipeline defect.
    """
    traj_prob_cols = [c for c in df.columns if c.endswith("_prob") or "trajtype_" in c]
    mcmc_cols = [c for c in traj_prob_cols if not c.endswith("_boot")]
    boot_cols = [c for c in traj_prob_cols if c.endswith("_boot")]

    summary_rows = []
    for name, cols in (("mcmc", mcmc_cols), ("boot", boot_cols)):
        if not cols:
            continue
        covered = df[cols].notna().any(axis=1)
        summary_rows.append({
            "source": name,
            "n_rows_total": len(df),
            "n_rows_covered": int(covered.sum()),
            "pct_covered": round(100 * covered.mean(), 2),
        })
    coverage_df = pd.DataFrame(summary_rows)

    if raw_data_dir and mcmc_cols:
        egfr_path = Path(raw_data_dir) / "egfr_df.parquet"
        if egfr_path.exists():
            egfr_df = pd.read_parquet(egfr_path)
            if "years_since_index" not in egfr_df.columns:
                egfr_df["years_since_index"] = egfr_df["days_since_index"] / 365.25
            if "years_lower" not in egfr_df.columns:
                egfr_df["years_lower"] = np.floor(egfr_df["years_since_index"]).astype(int)

            n_obs_in_window = []
            for pid, pid_df in egfr_df.groupby("person_id", observed=True):
                times = pid_df["years_since_index"].to_numpy()
                for w in pid_df["years_lower"].unique():
                    mask = (times >= w - 3.0) & (times <= w + 1)
                    n_obs_in_window.append((pid, w, int(mask.sum())))
            obs_df = pd.DataFrame(n_obs_in_window, columns=["person_id", "years_lower", "n_egfr_obs_in_window"])

            merged = df[[id_col, "years_lower"] + mcmc_cols].merge(
                obs_df, left_on=[id_col, "years_lower"], right_on=["person_id", "years_lower"], how="left"
            )
            merged["n_egfr_obs_in_window"] = merged["n_egfr_obs_in_window"].fillna(0)
            merged["covered"] = merged[mcmc_cols].notna().any(axis=1)

            by_n_obs = merged.groupby("n_egfr_obs_in_window", observed=True)["covered"].agg(["mean", "count"])
            by_n_obs.columns = ["pct_covered", "n_rows"]
            by_n_obs["pct_covered"] = (by_n_obs["pct_covered"] * 100).round(2)
            logger.info("\nTrajectory coverage vs. # eGFR observations in the 3yr lookback window:")
            logger.info("\n" + by_n_obs.reset_index().to_string(index=False))
        else:
            logger.info(f"  (skipping obs-count cross-check: {egfr_path} not found)")

    return coverage_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate missingness in the CKD patient-year matrix.")
    parser.add_argument("--data", type=str, required=True, help="Path to the built matrix parquet.")
    parser.add_argument("--raw-data-dir", type=str, default=None,
                         help="Optional: raw extract dir, enables the trajectory-coverage cross-check against egfr_df.parquet.")
    parser.add_argument("--high-missing-threshold", type=float, default=HIGH_MISSING_THRESHOLD,
                         help="Columns above this missing fraction are flagged as drop/indicator candidates.")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="If given, write the detail tables as CSVs here in addition to printing summaries.")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    id_col = "person_id" if "person_id" in df.columns else df.columns[0]
    logger.info(f"Loaded {args.data}: shape={df.shape}, id_col={id_col!r}")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stray index columns
    stray = check_unnamed_columns(df)

    # 2. Missing-rate table
    miss_df = missing_rate_table(df, id_col)
    if out_dir:
        miss_df.to_csv(out_dir / "missing_rates.csv", index=False)

    value_last_cols = [c for c in df.columns if c.endswith("_value_last")]

    # 3. Missingness by year (extraction-cliff check)
    by_year_df = missingness_by_year(df, value_last_cols)
    if out_dir:
        by_year_df.to_csv(out_dir / "missingness_by_year.csv", index=False)
    cliffs = by_year_df[by_year_df["likely_extraction_cliff"]]

    # 4. Per-patient constancy (static-misclassification check)
    constancy_df = per_patient_constancy(df, id_col, value_last_cols)
    if out_dir:
        constancy_df.to_csv(out_dir / "per_patient_constancy.csv", index=False)
    static_candidates = constancy_df[constancy_df["static_candidate"]]

    # 5. Trajectory coverage
    traj_cov_df = trajectory_coverage(df, id_col, args.raw_data_dir)
    if out_dir and not traj_cov_df.empty:
        traj_cov_df.to_csv(out_dir / "trajectory_coverage.csv", index=False)

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("CKD MATRIX MISSINGNESS INVESTIGATION")
    print("=" * 78)

    if stray:
        print(f"\n[ACTION] {len(stray)} stray index column(s) found: {stray}")
        print("  -> Rebuild the matrix with the fixed ckd_build_matrix.py (drops 'Unnamed: *').")

    high_missing = miss_df[miss_df["pct_missing"] >= args.high_missing_threshold * 100]
    print(f"\n[INFO] {len(high_missing)} / {len(miss_df)} columns are >= "
          f"{args.high_missing_threshold:.0%} missing.")
    if len(high_missing):
        print(high_missing.head(20).to_string(index=False))
        print("  -> For genuinely rare labs, consider: (a) dropping the raw value columns and keeping only a")
        print("     binary 'was measured this year' indicator, since a single scalar with >90% imputed values")
        print("     usually contributes noise rather than signal to tree/linear models alike; or (b) leaving")
        print("     them in for boosting models (which handle NaN as its own split) but dropping for LR/RF.")

    if len(cliffs):
        print(f"\n[ACTION] {len(cliffs)} feature(s) show a sharp missing-rate jump across years_lower "
              f"(>= {CLIFF_JUMP_THRESHOLD:.0%} jump year-over-year):")
        print(cliffs.to_string(index=False))
        print("  -> This pattern (rather than a flat, gradually-rising curve) usually means the raw extract's")
        print("     coverage window ended partway through follow-up, not that patients stopped being tested.")
        print("     Check the source table's date range for these concept_ids before trusting late-year rows.")
    else:
        print("\n[OK] No feature shows a sharp missing-rate cliff across years_lower — sparsity looks like it "
              "accumulates gradually, consistent with routine intermittent lab ordering rather than an "
              "extraction artifact.")

    if len(static_candidates):
        print(f"\n[ACTION] {len(static_candidates)} feature(s) bucketed as 'dynamic' vary almost not at all "
              f"within a patient (median CV < {NEAR_CONSTANT_THRESHOLD}):")
        print(static_candidates.to_string(index=False))
        print("  -> These are candidates to compute once per patient (like age_at_index) instead of running")
        print("     through the yearly pivot + 1-year LOCF-expiry pipeline, which otherwise wipes a value that")
        print("     was never actually stale. 'Body height' is the known example; check this list for others.")

    if not traj_cov_df.empty:
        print("\n[INFO] Trajectory probability coverage:")
        print(traj_cov_df.to_string(index=False))
        print("  -> If coverage tracks the number of eGFR observations in the lookback window (see the")
        print("     cross-tab above, when --raw-data-dir is passed), the ~90% missingness reflects a minimum-")
        print("     observations threshold in trajectory classification, not a merge bug.")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
