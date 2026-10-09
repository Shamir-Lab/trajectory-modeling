#!/usr/bin/env python3
"""
Standalone CKD trajectory analysis for trajectory_representation_chronic (All of Us).

Evaluates CKD progression targets (ESKD / dialysis / death / combined
ESKD-or-death, at 1-3 and 2-5 year horizons) using Static+Dynamic,
Static+Dynamic+MCMC-Trajectories, and Static+Dynamic+Bootstrap-Trajectories
feature sets (kept strictly disjoint — see organize_features() /
build_feature_sets()), with the same fold-level preprocessing convention used
across this repo (fillna(0) for trajectory/summary columns, median imputation
for clinical labs, StandardScaler; OHE applied before imputation).

This script only consumes a pre-assembled patient-year feature matrix — it
does not build one. Run ckd_build_matrix.py first to produce that matrix from
raw All of Us extracts.

Input:   ../data/final_dynamic_features.parquet (produced by ckd_build_matrix.py)
Outputs: ../results/trajectory_representation_chronic/
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed as jdelayed
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 920
np.random.seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Standard CKD target set produced by ckd_build_matrix.py's
# generate_dynamic_targets(): ESKD, dialysis, and death at each horizon, plus
# the combined ESKD-or-death outcome. Used only as a fallback when --targets
# is omitted and no target_* columns can be auto-detected in the loaded data.
DEFAULT_TARGETS = [
    "target_eskd_1_to_3_years",
    "target_eskd_2_to_5_years",
    "target_dialysis_1_to_3_years",
    "target_dialysis_2_to_5_years",
    "target_death_1_to_3_years",
    "target_death_2_to_5_years",
    "target_eskd_or_death_1_to_3_years",
    "target_eskd_or_death_2_to_5_years",
]

# Low priority for routine analysis — death alone, without ESKD/dialysis
# context, is a secondary outcome for this study. Excluded from auto-detected
# default runs; only included when explicitly requested, either by naming
# them in --targets or via --include-low-priority-targets.
LOW_PRIORITY_TARGETS = {
    "target_death_1_to_3_years",
    "target_death_2_to_5_years",
}

DEFAULT_DATA = REPO_ROOT / "data" / "final_dynamic_features.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "trajectory_representation_chronic"

CV_N_REPEATS = 10
CV_N_SPLITS = 5

# Mirrors analysis_template.py: only LR and RF get fill+impute+scale;
# boosting models (HistGB, XGBoost) handle NaN natively and are not scaled.
NON_BOOSTING_MODELS = {"LogisticRegression", "RandomForest"}


# ============================================================================
# ID Column Detection
# ============================================================================


def detect_id_col(df: pd.DataFrame) -> str:
    for col in ("person_id", "patient_id", "patientid", "subject_id", "mrn", "hadm_id", "stay_id"):
        if col in df.columns:
            return col
    raise ValueError(
        f"No patient ID column found. First 20 columns: {list(df.columns[:20])}"
    )


# ============================================================================
# Feature Organization
# ============================================================================
# Mirrors analysis_utils.organize_feature_sets() with CKD-specific extensions
# marked with inline comments.


def organize_features(df: pd.DataFrame, id_col: str) -> dict[str, list[str]]:
    """
    Categorise columns into the canonical groups used across this repo.

    Column patterns are copied verbatim from analysis_utils.organize_feature_sets();
    CKD-specific additions are noted inline.

    Keys returned
    -------------
    trajectory   prob_*, trajtype_*, _stable/gradual/rapid/increase/decline, _boot
    summary      "3yr" rolling-window features (egfr_3yr_mean/max/min/change/
                 trend/std) — the genuinely *derived* summary statistics,
                 analogous to biomarker_summary_stats() in the ICU template.
                 Used both for descriptive-stats labeling and to decide which
                 fold columns get fillna(0) before imputation (0 = "no signal
                 detected" is a meaningful value for a rolling trend/change
                 feature the way it isn't for a raw lab value).
    baseline     everything non-trajectory, non-summary numeric (legacy grouping,
                 kept only for descriptive-stats completeness)
    static       demographic/time-invariant subset of all non-trajectory numeric columns
    dynamic      non-static, non-summary subset of all non-trajectory numeric
                 columns — the regular yearly lab aggregates (_mean/_max/_min/
                 _std/_last/_time_from_last_measurement). Disjoint from
                 `summary`: "Static+Dynamic" and "Static+Dynamic+Summary" are
                 meant to be genuinely different feature sets, not one a
                 no-op superset of the other.
    cat_cols     string columns needing one-hot encoding
    all_numeric  all numeric columns (trajectory + static + dynamic + summary,
                 all mutually disjoint)
    traj_bayes   trajectory columns WITHOUT _boot  (Bayesian / MCMC source)
    traj_boot    trajectory columns WITH _boot     (Bootstrap source)
    """
    # --- Exclusions (mirrors analysis_utils.organize_feature_sets) ---
    exclude_cols = {
        id_col,
        "charttime", "admittime", "dischtime", "subject_id",
        "los_days", "hospital_expire_flag", "in_hospital_mortality",
        "icu_expire_flag", "icu_mortality",
        "outcome", "label", "y",
    }
    # Exclude all target columns generically
    exclude_cols |= {c for c in df.columns if c.startswith("target_")}

    # String columns handled via OHE (not part of organize_feature_sets in the repo,
    # but required here because the CKD matrix carries raw categoricals)
    cat_cols = [
        c for c in df.columns
        if c not in exclude_cols and df[c].dtype == object
    ]

    # --- Numeric columns (mirrors repo exactly) ---
    numeric_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and c not in cat_cols
        and not c.startswith("target_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # --- Trajectory columns (copied verbatim from analysis_utils, plus the
    # CKD "boot_"/"mcmc_" prefixes added by ckd_build_matrix.py) ---
    trajectory_cols = [
        c for c in numeric_cols
        if (c.endswith("_stable")
            or c.endswith("_gradual")
            or c.endswith("_rapid")
            or c.endswith("_increase")
            or c.endswith("_decline")
            or c.startswith("prob_")
            or c.startswith("trajtype_")
            or c.endswith("_prob")
            or c.startswith("boot_")   # CKD Bootstrap trajectory prefix
            or c.startswith("mcmc_")   # CKD MCMC/Bayesian trajectory prefix
            or "_boot" in c)
    ]

    # --- Summary statistic columns ---
    # Only the "3yr" rolling-window features (egfr_3yr_mean/max/min/change/
    # trend/std) count as "summary" here — these are genuinely derived,
    # lookback-window statistics, the CKD analogue of biomarker_summary_stats()
    # in the ICU template. The regular yearly lab aggregates (_mean/_max/_min/
    # _std/_last/_time_from_last_measurement) are NOT summary stats in this
    # sense — they're just each biomarker's own yearly-varying value, which is
    # why they belong in `dynamic` below, not here.
    summary_cols = [
        c for c in numeric_cols
        if "3yr" in c
    ]

    # --- Baseline: everything else numeric (mirrors repo) ---
    baseline_cols = [
        c for c in numeric_cols
        if c not in trajectory_cols and c not in summary_cols
    ]

    # --- Static/Dynamic split ---
    static_tokens = {
        "age", "gender", "sex", "baseline", "admit", "admission",
        "ethnicity", "race", "height", "weight", "bmi", "baseline_",
        # CKD-specific additions
        "ckd_stage", "smoke", "smoking", "diabetes", "hypertension",
        "comorbid", "diagnosis", "cause",
    }
    non_traj_numeric = [c for c in numeric_cols if c not in trajectory_cols]
    # A column is only eligible for "static" if it is NOT already a per-year
    # aggregate (summary OR the regular yearly lab aggregates that will end up
    # in `dynamic`) — otherwise a token like "height"/"weight" matching inside
    # e.g. "Body height_value_std" would misclassify a longitudinal,
    # yearly-varying column as time-invariant. Aggregate suffixes always win.
    static_candidates = [c for c in non_traj_numeric if c not in summary_cols]
    static_cols = [
        c for c in static_candidates
        if any(tok in c.lower() for tok in static_tokens)
    ]
    # Disjoint from summary_cols: "Static+Dynamic" and "Static+Dynamic+Summary"
    # must be genuinely different feature sets (see organize_features docstring).
    dynamic_cols = [c for c in non_traj_numeric if c not in static_cols and c not in summary_cols]

    # --- Bayesian vs Bootstrap split within trajectory ---
    # CKD convention (ckd_build_matrix.py): Bootstrap columns carry a "boot_"
    # PREFIX (e.g. "boot_trajtype_stable_prob"), not the "_boot" suffix used
    # elsewhere in this repo (analysis_template.py's _suffix_bootstrap_cols)
    # — check both so either naming is classified correctly.
    traj_boot  = [c for c in trajectory_cols if c.startswith("boot_") or c.endswith("_boot") or "_boot" in c]
    traj_bayes = [c for c in trajectory_cols if c not in traj_boot]

    return {
        "trajectory":  trajectory_cols,
        "summary":     summary_cols,
        "baseline":    baseline_cols,
        "static":      static_cols,
        "dynamic":     dynamic_cols,
        "cat_cols":    cat_cols,
        "all_numeric": numeric_cols,
        "traj_bayes":  traj_bayes,
        "traj_boot":   traj_boot,
    }


def add_missingness_indicators(
    df: pd.DataFrame, groups: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Add a {col}_was_measured binary indicator (1 = value present, 0 = missing)
    for every "dynamic" and "summary" feature column, computed once on the
    raw pre-imputation data.

    This is leakage-safe: presence/absence is a deterministic function of the
    raw data, not something requiring fold-specific fitting the way
    imputation/scaling do. It matters because fold-level fillna(0)/median-
    imputation otherwise discards whether a value was truly measured vs.
    imputed — and in EHR data, measurement frequency is itself informative
    (tied to care/monitoring intensity), so silently treating "never
    measured" and "measured but happened to be near the median" as identical
    throws away real signal.

    Each indicator is appended into the SAME group as its base column, so it
    automatically flows into whichever feature-set combination (Static+
    Dynamic, +Summary, etc.) already includes that base column — no changes
    needed in build_feature_sets().
    """
    new_cols: dict[str, pd.Series] = {}
    for group_name in ("dynamic", "summary"):
        indicator_cols = []
        for col in groups[group_name]:
            ind_col = f"{col}_was_measured"
            new_cols[ind_col] = df[col].notna().astype(np.int8)
            indicator_cols.append(ind_col)
        groups[group_name] = groups[group_name] + indicator_cols

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        logger.info(f"Added {len(new_cols)} missingness indicator column(s)")
    return df, groups


def _group_label(col: str, groups: dict[str, list[str]]) -> str:
    """Return the most specific group name for a column (for descriptive stats)."""
    for g in ("traj_bayes", "traj_boot", "summary", "static", "dynamic", "cat_cols"):
        if col in groups.get(g, []):
            return g
    return "other"


# ============================================================================
# One-Hot Encoding
# ============================================================================


def one_hot_encode(
    df: pd.DataFrame, cat_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """OHE categorical columns; returns encoded dataframe and new dummy column names."""
    present = [c for c in cat_cols if c in df.columns]
    if not present:
        return df.copy(), []
    dummies = pd.get_dummies(df[present], drop_first=False, dtype=float)
    ohe_cols = list(dummies.columns)
    return pd.concat([df.drop(columns=present), dummies], axis=1), ohe_cols


# ============================================================================
# Feature Set Construction
# ============================================================================
# Exactly the 6 feature sets selected for presentation, matching
# src/analysis/poster/feature_set_mapper.py's CANONICAL_LABELS naming so that
# pipeline can match them directly without custom aliases. MCMC and Bootstrap
# trajectories are never combined into the same feature set — mixing them
# would defeat the point of comparing the two representations independently.


def build_feature_sets(
    groups: dict[str, list[str]],
    ohe_cols: list[str],
    df_cols: set[str],
) -> dict[str, list[str]]:
    """
    Build exactly the 6 feature-set configurations selected for presentation:
      - "Static + Dynamic"
      - "Static + Dynamic + Summary"
      - "Static + Dynamic + Trajectory"                        (MCMC/Bayesian)
      - "Static + Dynamic + Summary + Trajectory"              (MCMC/Bayesian)
      - "Static + Dynamic + Trajectory (Bootstrap)"
      - "Static + Dynamic + Summary + Trajectory (Bootstrap)"

    OHE dummy columns are appended to static (they encode time-invariant
    categoricals). `summary` (the "3yr" rolling-window features) is a
    separate ingredient from `dynamic` — organize_features() keeps the two
    disjoint by construction — so "+ Summary" is a genuinely additive
    comparison, not a no-op.

    A trajectory source is omitted from every combination it would appear in
    if it has no columns in the data, rather than silently falling back to
    the other source — MCMC and Bootstrap results must stay independently
    comparable.
    """
    def _valid(cols: list[str]) -> list[str]:
        return list(dict.fromkeys(c for c in cols if c in df_cols))

    static = _valid(groups["static"] + ohe_cols)
    dynamic = _valid(groups["dynamic"])
    summary = _valid(groups["summary"])
    traj_bayes = _valid(groups["traj_bayes"])
    traj_boot = _valid(groups["traj_boot"])

    if not traj_bayes:
        logger.warning("No MCMC/Bayesian trajectory columns found — omitting MCMC feature sets.")
    if not traj_boot:
        logger.warning("No Bootstrap trajectory columns found — omitting Bootstrap feature sets.")

    sd = static + dynamic
    sd_sum = sd + summary

    configs: dict[str, list[str]] = {
        "Static + Dynamic": _valid(sd),
        "Static + Dynamic + Summary": _valid(sd_sum),
    }
    if traj_bayes:
        configs["Static + Dynamic + Trajectory"] = _valid(sd + traj_bayes)
        configs["Static + Dynamic + Summary + Trajectory"] = _valid(sd_sum + traj_bayes)
    if traj_boot:
        configs["Static + Dynamic + Trajectory (Bootstrap)"] = _valid(sd + traj_boot)
        configs["Static + Dynamic + Summary + Trajectory (Bootstrap)"] = _valid(sd_sum + traj_boot)

    return {name: cols for name, cols in configs.items() if cols}


# ============================================================================
# Descriptive Statistics
# ============================================================================


def save_descriptive_stats(
    df_raw: pd.DataFrame,
    target_col: str,
    id_col: str,
    groups: dict[str, list[str]],
    output_dir: Path,
    label: str,
) -> None:
    """
    Save four pre-imputation diagnostic tables to output_dir/descriptive_stats/.

    1. {label}_outcome_distribution.csv   — positive-rate summary
    2. {label}_patient_counts.csv         — patient / row counts by outcome
    3. {label}_missing_rates.csv          — per-column NaN rate
    4. {label}_feature_statistics.csv     — Mean±SD (continuous) or % (categorical),
                                            stratified by outcome (all / pos / neg)
    """
    out = output_dir / "descriptive_stats"
    out.mkdir(parents=True, exist_ok=True)

    pos = df_raw[df_raw[target_col] == 1]
    neg = df_raw[df_raw[target_col] == 0]
    n   = max(1, len(df_raw))

    # 1. Outcome distribution
    pd.DataFrame([
        {"metric": "n_rows_total",    "value": len(df_raw)},
        {"metric": "n_rows_positive", "value": len(pos)},
        {"metric": "n_rows_negative", "value": len(neg)},
        {"metric": "positive_rate",   "value": float(df_raw[target_col].mean())},
    ]).to_csv(out / f"{label}_outcome_distribution.csv", index=False)

    # 2. Patient counts
    pd.DataFrame([
        {"metric": "n_patients_total",    "value": int(df_raw[id_col].nunique())},
        {"metric": "n_patients_positive", "value": int(pos[id_col].nunique())},
        {"metric": "n_patients_negative", "value": int(neg[id_col].nunique())},
        {"metric": "n_rows_total",        "value": len(df_raw)},
        {"metric": "n_rows_positive",     "value": len(pos)},
        {"metric": "n_rows_negative",     "value": len(neg)},
    ]).to_csv(out / f"{label}_patient_counts.csv", index=False)

    # 3. Missing rates
    all_feat_cols = (
        groups["trajectory"]
        + groups["summary"]
        + groups["baseline"]
        + groups["cat_cols"]
    )
    present = [c for c in all_feat_cols if c in df_raw.columns]
    missing_rows = sorted(
        [
            {
                "column":        c,
                "feature_group": _group_label(c, groups),
                "n_missing":     int(df_raw[c].isna().sum()),
                "pct_missing":   round(100.0 * df_raw[c].isna().sum() / n, 2),
            }
            for c in present
        ],
        key=lambda r: r["pct_missing"],
        reverse=True,
    )
    pd.DataFrame(missing_rows).to_csv(out / f"{label}_missing_rates.csv", index=False)

    # 4. Feature statistics stratified by outcome
    stat_rows = []
    for c in present:
        g = _group_label(c, groups)
        if c in groups["cat_cols"] or df_raw[c].dtype == object:
            for val in sorted(df_raw[c].dropna().unique()):
                stat_rows.append({
                    "column": c, "feature_group": g, "type": "categorical",
                    "category_value": str(val),
                    "all_pct":  round(100.0 * (df_raw[c] == val).sum() / n, 2),
                    "pos_pct":  round(100.0 * (pos[c] == val).sum() / max(1, len(pos)), 2),
                    "neg_pct":  round(100.0 * (neg[c] == val).sum() / max(1, len(neg)), 2),
                    "all_mean": np.nan, "all_sd":  np.nan,
                    "pos_mean": np.nan, "pos_sd":  np.nan,
                    "neg_mean": np.nan, "neg_sd":  np.nan,
                })
        elif pd.api.types.is_numeric_dtype(df_raw[c]):
            stat_rows.append({
                "column": c, "feature_group": g, "type": "continuous",
                "category_value": np.nan,
                "all_pct": np.nan, "pos_pct": np.nan, "neg_pct": np.nan,
                "all_mean": round(float(df_raw[c].mean()), 4),
                "all_sd":   round(float(df_raw[c].std()),  4),
                "pos_mean": round(float(pos[c].mean()), 4) if len(pos) else np.nan,
                "pos_sd":   round(float(pos[c].std()),  4) if len(pos) else np.nan,
                "neg_mean": round(float(neg[c].mean()), 4) if len(neg) else np.nan,
                "neg_sd":   round(float(neg[c].std()),  4) if len(neg) else np.nan,
            })
    pd.DataFrame(stat_rows).to_csv(out / f"{label}_feature_statistics.csv", index=False)

    logger.info(f"  Descriptive stats → {out}")


# ============================================================================
# Models
# ============================================================================
# Mirrors get_default_models() in analysis_utils.py exactly.


def get_models() -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=SEED, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=SEED, n_jobs=-1, class_weight="balanced"
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=SEED, class_weight="balanced"
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=SEED,
        )
    except ImportError:
        logger.info("XGBoost not installed — skipping")
    return models


# ============================================================================
# Core Evaluation (single fold)
# ============================================================================
# Mirrors _eval_one_split() in analysis_template.py exactly.
#
# Non-boosting models (LogisticRegression, RandomForest):
#   1. Fill trajectory-column NaNs with 0
#   2. Fill summary-stat-column NaNs with 0
#   3. Median-impute remaining NaNs (fit on train, apply to test)
#   4. Standard-scale (fit on train, apply to test)
#
# Boosting models (HistGradientBoosting, XGBoost):
#   No explicit imputation — these models handle NaN natively.
#   Standard-scale is still applied (NaN propagates transparently).


def _eval_one_split(
    model_name: str,
    model,
    selected_cols: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    traj_col_set: set[str],
    summary_col_set: set[str],
    log_ctx: str = "",
) -> tuple[float, float, np.ndarray]:
    # Stage-level progress logging: if the process hangs, the last line
    # printed identifies exactly which stage (impute/scale/fit/predict) of
    # which (repeat, fold, model, feature_set) combination is stuck — logged
    # BEFORE each blocking call, since a log line queued after a hang never
    # prints.
    prefix = f"{log_ctx} {model_name}" if log_ctx else model_name

    X_tr_df = X_train[selected_cols].copy()
    X_te_df = X_test[selected_cols].copy()

    if model_name in NON_BOOSTING_MODELS:
        # Fill trajectory and summary NaNs with 0, then median-impute remainder
        traj_in_fs    = [c for c in selected_cols if c in traj_col_set]
        summary_in_fs = [c for c in selected_cols if c in summary_col_set]
        if traj_in_fs:
            X_tr_df.loc[:, traj_in_fs] = X_tr_df[traj_in_fs].fillna(0)
            X_te_df.loc[:, traj_in_fs] = X_te_df[traj_in_fs].fillna(0)
        if summary_in_fs:
            X_tr_df.loc[:, summary_in_fs] = X_tr_df[summary_in_fs].fillna(0)
            X_te_df.loc[:, summary_in_fs] = X_te_df[summary_in_fs].fillna(0)
        logger.info(f"{prefix} | impute: starting (train={len(X_tr_df):,} x {len(selected_cols)})")
        t0 = time.perf_counter()
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr_df)
        X_te = imputer.transform(X_te_df)
        logger.info(f"{prefix} | impute: done in {time.perf_counter() - t0:.1f}s")
    else:
        # Boosting models handle NaN natively — pass raw arrays
        X_tr = X_tr_df.to_numpy()
        X_te = X_te_df.to_numpy()

    logger.info(f"{prefix} | scale: starting")
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    logger.info(f"{prefix} | scale: done in {time.perf_counter() - t0:.1f}s")

    est = clone(model)
    if model_name == "XGBoost":
        est.set_params(
            scale_pos_weight=(len(y_train) - y_train.sum()) / max(1, y_train.sum())
        )
    # Always force single-threaded fitting on the estimator itself.
    # RandomForest's internal tree-building parallelism defaults to joblib's
    # process-based ("loky") backend, which can deadlock indefinitely (no
    # error, no output) when it forks from a process that has already
    # initialized a threaded BLAS (numpy/OpenBLAS/MKL) — a well-known Linux
    # fork-safety hazard, not something that only shows up under outer CV
    # parallelism. Parallelism across CV repeats is handled once, safely, at
    # the outer level via run_cv's Parallel(..., backend="threading", ...).
    if hasattr(est, "get_params") and "n_jobs" in est.get_params(deep=False):
        est.set_params(n_jobs=1)

    logger.info(f"{prefix} | fit: starting (n_train={len(X_tr):,}, n_features={X_tr.shape[1]})")
    t0 = time.perf_counter()
    est.fit(X_tr, y_train)
    logger.info(f"{prefix} | fit: done in {time.perf_counter() - t0:.1f}s")

    logger.info(f"{prefix} | predict: starting (n_test={len(X_te):,})")
    t0 = time.perf_counter()
    if hasattr(est, "predict_proba"):
        y_pred = est.predict_proba(X_te)[:, 1]
    elif hasattr(est, "decision_function"):
        y_pred = est.decision_function(X_te)
    else:
        y_pred = est.predict(X_te)
    logger.info(f"{prefix} | predict: done in {time.perf_counter() - t0:.1f}s")

    return (
        float(roc_auc_score(y_test, y_pred)),
        float(average_precision_score(y_test, y_pred)),
        y_pred,
    )


# ============================================================================
# Cross-Validation
# ============================================================================


def run_cv(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    feature_configs: dict[str, list[str]],
    traj_col_set: set[str],
    summary_col_set: set[str],
    n_repeats: int = CV_N_REPEATS,
    n_splits: int = CV_N_SPLITS,
    n_jobs: int = 1,
) -> tuple[dict, list[dict], pd.DataFrame]:
    """
    GroupKFold cross-validation (groups = patient ID) across all feature sets and models.

    Returns
    -------
    summary   {model: {feature_set: {auroc_mean, auroc_std, aupr_mean, aupr_std, n_folds}}}
    fold_rows per-fold detail records
    oof_df    out-of-fold predictions as a single compact DataFrame (id, model,
              feature_set [category], repeat [int16], fold [int8], y_true/
              y_pred [float32]) — built once via columnar concatenation, not
              accumulated as a list of per-row dicts, to keep memory bounded
              at O(rows x models x feature_sets) instead of O(that x ~600
              bytes of CPython dict overhead per row).
    """
    models = get_models()

    feature_union = list(dict.fromkeys(
        c for cols in feature_configs.values() for c in cols
    ))

    X_all  = df[feature_union].copy()
    y_all  = df[target_col].to_numpy(dtype=float)
    groups = df[id_col].to_numpy()

    valid  = np.isfinite(y_all)
    X_all, y_all, groups = X_all[valid], y_all[valid], groups[valid]

    logger.info(f"  {len(X_all):,} rows | positive rate {y_all.mean():.1%}")

    def _run_one_repeat(repeat_idx: int) -> dict:
        logger.info(f"[repeat {repeat_idx}] starting ({n_splits}-fold StratifiedGroupKFold split)")
        rng  = np.random.RandomState(SEED + repeat_idx)
        perm = rng.permutation(len(X_all))
        X_rep = X_all.iloc[perm].reset_index(drop=True)
        y_rep = y_all[perm]
        g_rep = groups[perm]

        # Stratified (not just grouped): plain GroupKFold ignores the outcome
        # distribution entirely when assigning patients to folds, so under
        # severe class imbalance a fold can land with zero positives by pure
        # chance — silently reducing effective power via the single-class
        # skip below. StratifiedGroupKFold balances the outcome per fold
        # while still respecting patient (group) boundaries.
        gkf = StratifiedGroupKFold(n_splits=n_splits)
        n_folds_skipped = 0
        metrics: dict[str, dict[str, list]] = {
            m: {fs: [] for fs in feature_configs} for m in models
        }
        # One small DataFrame per (fold, model, feature_set) combo, built
        # directly from the existing numpy arrays — NOT a Python dict per
        # row. With ~96K rows x 10 repeats x 4 models x 11 feature sets,
        # a per-row dict (~500-700 bytes of CPython object overhead each)
        # would be tens of GB; a columnar chunk with compact dtypes is
        # ~20 bytes/row, concatenated once at the very end in run_cv.
        oof_chunks: list[pd.DataFrame] = []

        for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X_rep, y_rep, g_rep)):
            X_tr, X_te = X_rep.iloc[tr_idx], X_rep.iloc[te_idx]
            y_tr, y_te = y_rep[tr_idx], y_rep[te_idx]
            g_te        = g_rep[te_idx]

            logger.info(
                f"[repeat {repeat_idx} fold {fold_idx}] train={len(tr_idx):,} test={len(te_idx):,} "
                f"— {len(models)} models x {len(feature_configs)} feature sets to run"
            )

            if len(np.unique(y_te)) < 2:
                n_folds_skipped += 1
                logger.info(f"[repeat {repeat_idx} fold {fold_idx}] skipped (single-class test fold)")
                continue

            for model_name, model in models.items():
                for fs_name, fs_cols in feature_configs.items():
                    sel = [c for c in fs_cols if c in X_tr.columns]
                    if not sel:
                        continue
                    log_ctx = f"[repeat {repeat_idx} fold {fold_idx}] {fs_name!r} ({len(sel)} features) —"
                    t_combo = time.perf_counter()
                    try:
                        auroc, aupr, y_pred = _eval_one_split(
                            model_name=model_name,
                            model=model,
                            selected_cols=sel,
                            X_train=X_tr,
                            X_test=X_te,
                            y_train=y_tr,
                            y_test=y_te,
                            traj_col_set=traj_col_set,
                            summary_col_set=summary_col_set,
                            log_ctx=log_ctx,
                        )
                        logger.info(
                            f"{log_ctx} {model_name} complete in "
                            f"{time.perf_counter() - t_combo:.1f}s (auroc={auroc:.3f}, aupr={aupr:.3f})"
                        )
                        metrics[model_name][fs_name].append(
                            {"repeat": repeat_idx, "fold": fold_idx,
                             "auroc": auroc, "aupr": aupr}
                        )
                        oof_chunks.append(pd.DataFrame({
                            "id": g_te,
                            "model": model_name,
                            "feature_set": fs_name,
                            "repeat": np.int16(repeat_idx),
                            "fold": np.int8(fold_idx),
                            "y_true": y_te.astype(np.float32),
                            "y_pred": np.asarray(y_pred, dtype=np.float32),
                        }))
                    except Exception as exc:
                        logger.warning(
                            f"    Error {model_name}/{fs_name} "
                            f"r{repeat_idx}f{fold_idx}: {exc}"
                        )
        if n_folds_skipped:
            logger.info(
                f"[repeat {repeat_idx}] complete — {n_folds_skipped}/{n_splits} fold(s) skipped (single-class test)"
            )
        else:
            logger.info(f"[repeat {repeat_idx}] complete")

        return {"metrics": metrics, "oof": oof_chunks}

    repeat_results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
        jdelayed(_run_one_repeat)(i) for i in range(n_repeats)
    )

    fold_rows: list[dict] = []
    oof_chunks_all: list[pd.DataFrame] = []
    raw: dict[str, dict[str, list]] = {m: {fs: [] for fs in feature_configs} for m in models}

    for res in repeat_results:
        oof_chunks_all.extend(res["oof"])
        for m in models:
            for fs in feature_configs:
                batch = res["metrics"][m][fs]
                raw[m][fs].extend(batch)
                for r in batch:
                    fold_rows.append({
                        "model": m, "feature_set": fs,
                        "repeat": r["repeat"], "fold": r["fold"],
                        "auroc": r["auroc"], "aupr": r["aupr"],
                    })
    del repeat_results

    # Concatenate once — this is the only full-size OOF DataFrame materialized
    # for this target; save_results() must reuse it rather than rebuilding it
    # from a list. "model"/"feature_set" stay plain strings (not category)
    # so they merge cleanly against results_df's plain-string columns in
    # save_results() — each combo's string is already a single shared object
    # reference per chunk, so this costs only 8 bytes/row, not a full copy.
    if oof_chunks_all:
        oof_df = pd.concat(oof_chunks_all, ignore_index=True)
    else:
        oof_df = pd.DataFrame(columns=["id", "model", "feature_set", "repeat", "fold", "y_true", "y_pred"])
    del oof_chunks_all

    summary: dict[str, dict[str, dict]] = {}
    for m in models:
        summary[m] = {}
        for fs in feature_configs:
            folds = raw[m][fs]
            if not folds:
                continue
            # Aggregate at the repeat level first, then compute mean/std
            # across those repeat-level means — not pooled across all raw
            # folds. Folds within a repeat share the same underlying patient
            # pool (just partitioned differently), so they aren't independent
            # samples; pooling them into one std understates true uncertainty
            # and can make small differences between feature sets look more
            # significant than they are. Repeat-level means (here, up to 10)
            # are the honest unit of variation. ddof=1 (sample std): we're
            # estimating spread from a small sample of repeat-level
            # estimates, not treating them as the full population.
            per_repeat = pd.DataFrame(folds).groupby("repeat")[["auroc", "aupr"]].mean()
            summary[m][fs] = {
                "auroc_mean": float(per_repeat["auroc"].mean()),
                "auroc_std":  float(per_repeat["auroc"].std()) if len(per_repeat) > 1 else 0.0,
                "aupr_mean":  float(per_repeat["aupr"].mean()),
                "aupr_std":   float(per_repeat["aupr"].std()) if len(per_repeat) > 1 else 0.0,
                "n_folds":    len(folds),
                "n_repeats":  int(len(per_repeat)),
            }

    return summary, fold_rows, oof_df


# ============================================================================
# Saving Results and Plots
# ============================================================================
# ROC/PR curve plotting mirrors _save_notebook_style_plots() in
# analysis_template.py. Bar-chart/boxplot comparison plots are produced by
# save_poster_comparison() below, reusing src/analysis/poster/ directly.


def save_results(
    summary: dict,
    fold_rows: list[dict],
    oof_df: pd.DataFrame,
    output_dir: Path,
    label: str,
    feature_configs: dict[str, list[str]],
    top_k: int = 10,
) -> None:
    """
    Write all outputs for one target.

    Files
    -----
    {label}_cv_results.csv          — per-(model, feature_set) mean/std AUROC + AUPR
    {label}_fold_metrics.csv        — per-fold detail (for boxplots / auditing)
    {label}_oof_predictions.parquet — top-k OOF predictions
    {label}_roc_pr_curves.png       — mean ± SD ROC/PR curves for top-k

    Poster-style comparison plots ({label}_poster_main.png,
    {label}_poster_best_model.png) and the extended pairwise significance
    table ({label}_significance_extended.csv) are produced separately by
    save_poster_comparison() — see main().
    """
    rows = [
        {"model": m, "feature_set": fs, **metrics}
        for m, fs_dict in summary.items()
        for fs, metrics in fs_dict.items()
    ]
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / f"{label}_cv_results.csv", index=False)

    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(
            output_dir / f"{label}_fold_metrics.csv", index=False
        )

    if not oof_df.empty and not results_df.empty:
        ranked    = results_df.sort_values(["aupr_mean", "auroc_mean"], ascending=False)
        top_pairs_df = ranked[["model", "feature_set"]].head(top_k)
        # Vectorized inner-join filter instead of a row-wise .apply(), which
        # would box every one of potentially tens of millions of rows into a
        # Python tuple just to test set membership.
        oof_filt  = oof_df.merge(top_pairs_df, on=["model", "feature_set"], how="inner")
        try:
            oof_filt.to_parquet(
                output_dir / f"{label}_oof_predictions.parquet", index=False
            )
        except Exception:
            oof_filt.to_csv(
                output_dir / f"{label}_oof_predictions.csv.gz", index=False
            )

    if results_df.empty:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(f"Skipping plots (matplotlib unavailable): {exc}")
        return

    # Compute baseline positive rate once (used as reference line in AUPR plots)
    baseline = float(oof_df["y_true"].mean()) if not oof_df.empty else np.nan

    # --- Mean ± SD ROC/PR curves (mirrors _save_notebook_style_plots curve section) ---
    if not oof_df.empty:
        ranked         = results_df.sort_values(["aupr_mean", "auroc_mean"], ascending=False)
        top_pairs_list = list(zip(ranked["model"].head(top_k), ranked["feature_set"].head(top_k)))
        curve_labels   = [f"{m} | {fs}" for m, fs in top_pairs_list]
        grid           = np.linspace(0, 1, 100)

        fig, (axr, axp) = plt.subplots(1, 2, figsize=(22, 8), constrained_layout=True)

        for (m, fs), lbl in zip(top_pairs_list, curve_labels):
            grp = oof_df[(oof_df["model"] == m) & (oof_df["feature_set"] == fs)]
            if grp.empty:
                continue

            roc_interp, pr_interp = [], []
            for _, fd in grp.groupby(["repeat", "fold"]):
                yt = fd["y_true"].to_numpy()
                yp = fd["y_pred"].to_numpy()
                if len(np.unique(yt)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(yt, yp)
                prec, rec, _ = precision_recall_curve(yt, yp)
                roc_interp.append(np.interp(grid, fpr, tpr))
                pr_interp.append(np.interp(grid, rec[::-1], prec[::-1]))

            if not roc_interp:
                continue

            roc_arr = np.vstack(roc_interp)
            pr_arr  = np.vstack(pr_interp)

            axr.plot(grid, roc_arr.mean(0), linewidth=2, label=lbl)
            axr.fill_between(
                grid, roc_arr.mean(0) - roc_arr.std(0),
                roc_arr.mean(0) + roc_arr.std(0), alpha=0.15,
            )
            axp.plot(grid, pr_arr.mean(0), linewidth=2, label=lbl)
            axp.fill_between(
                grid, pr_arr.mean(0) - pr_arr.std(0),
                pr_arr.mean(0) + pr_arr.std(0), alpha=0.15,
            )

        axr.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.7)
        axr.set_xlabel("False Positive Rate")
        axr.set_ylabel("True Positive Rate")
        axr.set_title(f"ROC Curves (mean ± SD) — {label}")
        axr.grid(alpha=0.3)
        axr.legend(fontsize=8)

        if np.isfinite(baseline):
            axp.axhline(baseline, linestyle="--", linewidth=1, color="gray", alpha=0.7)
        axp.set_xlabel("Recall")
        axp.set_ylabel("Precision")
        axp.set_title(f"Precision-Recall Curves (mean ± SD) — {label}")
        axp.grid(alpha=0.3)
        axp.legend(fontsize=8)

        fig.savefig(
            output_dir / f"{label}_roc_pr_curves.png", dpi=160, bbox_inches="tight"
        )
        plt.close(fig)

    logger.info(f"  Saved → {output_dir / (label + '_cv_results.csv')}")
    logger.info("\n" + results_df.to_string(index=False))


# ============================================================================
# Poster-Style Comparison (self-contained)
# ============================================================================
# Mirrors the plotting/statistics style used for the MIMIC/eICU/HiRID poster
# figures (src/analysis/poster/) closely enough to look identical, but is a
# standalone copy rather than an import from there — this script is meant to
# be deployable on its own (e.g. copied to a cloud workbench with no access
# to the rest of the repo), and importing src/analysis/poster/ at runtime
# would silently break that. The two will drift if src/analysis/poster/
# changes independently; if the shared module's plotting style changes,
# resync this block by hand.

POSTER_CANONICAL_LABELS = {
    "sd": "Static + Dynamic",
    "sd_sum": "Static + Dynamic + Summary",
    "sd_traj": "Static + Dynamic + Trajectory",
    "sd_sum_traj": "Static + Dynamic + Summary + Trajectory",
    "sd_traj_boot": "Static + Dynamic + Trajectory (Bootstrap)",
    "sd_sum_traj_boot": "Static + Dynamic + Summary + Trajectory (Bootstrap)",
}

_POSTER_MODEL_ABBR = {
    "LogisticRegression": "LR",
    "RandomForest": "RF",
    "HistGradientBoosting": "HGB",
    "XGBoost": "XGB",
}

_POSTER_FEATURE_ABBR = {
    "Static + Dynamic": "SD",
    "Static + Dynamic + Trajectory": "SD+T",
    "Static + Dynamic + Trajectory (Bootstrap)": "SD+T (B)",
    "Static + Dynamic + Summary": "SD+S",
    "Static + Dynamic + Summary + Trajectory": "SD+S+T",
    "Static + Dynamic + Summary + Trajectory (Bootstrap)": "SD+S+T (B)",
}


def _poster_abbr_feature(label: str) -> str:
    return _POSTER_FEATURE_ABBR.get(label, label)


def _poster_abbr_model(label: str) -> str:
    return _POSTER_MODEL_ABBR.get(label, label)


def _poster_set_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _poster_metric_palettes(n_colors: int) -> dict[str, list[str]]:
    auroc_scale = ["#EAF0F7", "#CCD8E7", "#A8BDD3", "#7C98B3", "#4F6E8C", "#334A63"]
    aupr_scale = ["#F1E5D2", "#D9BE93", "#B88A4A", "#93540A", "#7A620E", "#5D4B10"]

    def _expand(scale: list[str], n: int) -> list[str]:
        return scale[:n] if n <= len(scale) else [scale[i % len(scale)] for i in range(n)]

    return {"auroc": _expand(auroc_scale, n_colors), "aupr": _expand(aupr_scale, n_colors)}


def _poster_save_figure(fig, out_base: Path, formats: list[str], dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.2)


def _poster_select_best_model(fold_df: pd.DataFrame, canonical_ids: list[str]) -> tuple[str, pd.DataFrame]:
    work = fold_df[fold_df["canonical_feature_set"].isin(canonical_ids)]
    if work.empty:
        raise ValueError("No rows available for best-model selection")
    grouped = (
        work.groupby("model", as_index=False)
        .agg(aupr_mean=("aupr", "mean"), auroc_mean=("auroc", "mean"), n_rows=("aupr", "size"))
        .sort_values(["aupr_mean", "auroc_mean", "model"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return str(grouped.iloc[0]["model"]), grouped


def _poster_p_to_stars(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def _poster_compare_pairwise(
    best_model_df: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    metrics: list[str] = ("auroc", "aupr"),
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test per (metric, left, right); falls back
    to a binomial sign test if Wilcoxon fails (e.g. all-zero differences)."""
    from scipy.stats import binomtest, wilcoxon
    rows = []
    for metric in metrics:
        for left_id, right_id in comparisons:
            left = best_model_df.loc[best_model_df["canonical_feature_set"] == left_id, metric]
            right = best_model_df.loc[best_model_df["canonical_feature_set"] == right_id, metric]
            # Positional alignment (not by repeat/fold key) — caller must
            # ensure both slices are already sorted into matching order.
            merged = pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1).dropna()
            if merged.shape[0] < 1:
                p, n = np.nan, 0
            else:
                try:
                    _, p = wilcoxon(merged.iloc[:, 0], merged.iloc[:, 1])
                    n = merged.shape[0]
                except Exception:
                    dif = merged.iloc[:, 1] - merged.iloc[:, 0]
                    n = int(dif.count())
                    p = float(binomtest(int((dif > 0).sum()), n).pvalue) if n > 0 else np.nan
            rows.append({
                "metric": metric, "left": left_id, "right": right_id, "n_pairs": n,
                "p_value": p, "stars": _poster_p_to_stars(p) if pd.notna(p) else "",
            })
    # Explicit columns: when `comparisons` is empty (e.g. only 1-2 canonical
    # feature sets present, so no trajectory-vs-baseline pairs exist), `rows`
    # is empty and pd.DataFrame([]) would have NO columns at all — the caller
    # filters on sig_df["metric"], which would then raise KeyError.
    return pd.DataFrame(rows, columns=["metric", "left", "right", "n_pairs", "p_value", "stars"])


def _poster_plot_main_panel(
    fold_df: pd.DataFrame,
    canonical_order: list[str],
    title_prefix: str,
    out_base: Path,
    formats: list[str],
    dpi: int,
    figsize: tuple[float, float],
    chance_baseline: dict[str, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _poster_set_style()
    sns.set_style("whitegrid")
    metric_palettes = _poster_metric_palettes(len(canonical_order))

    work = fold_df.copy()
    work["feature_label"] = work["canonical_feature_set"].map(POSTER_CANONICAL_LABELS)
    work["model_label"] = work["model"].map(_poster_abbr_model)
    label_order = [POSTER_CANONICAL_LABELS[c] for c in canonical_order if c in POSTER_CANONICAL_LABELS]

    fig_w, fig_h = max(float(figsize[0]), 18.0), max(float(figsize[1]), 6.8)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    for ax, metric in zip(axes, ["auroc", "aupr"]):
        pal = dict(zip(label_order, metric_palettes[metric]))
        sns.boxplot(
            data=work, x="model_label", y=metric, hue="feature_label",
            hue_order=label_order, palette=pal, ax=ax, fliersize=1.8, linewidth=1.0,
        )
        baseline_val = chance_baseline.get(metric) if chance_baseline else None
        if baseline_val is not None:
            ax.axhline(baseline_val, color="black", linestyle="--", linewidth=1.0, alpha=0.6, zorder=0)
        ax.set_title("AUROC" if metric == "auroc" else "AUPR")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.upper())

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.legend(
            handles, [_poster_abbr_feature(lbl) for lbl in labels], title="Feature Set",
            loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, frameon=False,
        )

    fig.suptitle(f"{title_prefix}: model-stratified CV performance", y=1.03)
    fig.subplots_adjust(bottom=0.28, top=0.88, wspace=0.2)
    _poster_save_figure(fig, out_base, formats, dpi)
    plt.close(fig)


def _poster_plot_best_model_panel(
    best_model_df: pd.DataFrame,
    canonical_order: list[str],
    sig_df: pd.DataFrame,
    title_prefix: str,
    best_model: str,
    out_base: Path,
    formats: list[str],
    dpi: int,
    figsize: tuple[float, float],
    chance_baseline: dict[str, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _poster_set_style()
    sns.set_style("whitegrid")
    metric_palettes = _poster_metric_palettes(len(canonical_order))

    work = best_model_df.copy()
    work["feature_label"] = work["canonical_feature_set"].map(POSTER_CANONICAL_LABELS)
    work["feature_abbr"] = work["feature_label"].map(_poster_abbr_feature)
    label_order = [POSTER_CANONICAL_LABELS[c] for c in canonical_order if c in POSTER_CANONICAL_LABELS]
    abbr_order = [_poster_abbr_feature(lbl) for lbl in label_order]

    fig_w, fig_h = max(float(figsize[0]), 16.5), max(float(figsize[1]), 7.2)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    for ax, metric in zip(axes, ["auroc", "aupr"]):
        pal = dict(zip(label_order, metric_palettes[metric]))
        sns.boxplot(
            data=work, x="feature_abbr", y=metric, hue="feature_label",
            order=abbr_order, hue_order=label_order, palette=pal, dodge=False,
            ax=ax, fliersize=2.0, linewidth=1.1,
        )
        ax.set_title("AUROC" if metric == "auroc" else "AUPR")
        ax.set_xlabel("Feature Set")
        ax.set_ylabel(metric.upper())

        handles, labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.legend(
            handles, [_poster_abbr_feature(lbl) for lbl in labels], title="Feature Set",
            loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, frameon=False,
        )

        metric_sig = sig_df[sig_df["metric"] == metric].copy()
        if work.empty:
            continue

        y_max, y_min = float(work[metric].max()), float(work[metric].min())
        baseline_val = chance_baseline.get(metric) if chance_baseline else None
        if baseline_val is not None:
            ax.axhline(baseline_val, color="black", linestyle="--", linewidth=1.0, alpha=0.6, zorder=0)
            y_max, y_min = max(y_max, baseline_val), min(y_min, baseline_val)
        y_span = max(y_max - y_min, 0.02)
        start_offset, step = 0.03 * y_span, 0.22 * y_span

        if not metric_sig.empty:
            metric_sig = metric_sig.reset_index(drop=True)
            drawn = 0
            for _, row in metric_sig.iterrows():
                left_id, right_id = row.get("left"), row.get("right")
                stars = str(row.get("stars", ""))
                if not left_id or not right_id or not stars:
                    continue
                try:
                    left_label = _poster_abbr_feature(POSTER_CANONICAL_LABELS[left_id])
                    right_label = _poster_abbr_feature(POSTER_CANONICAL_LABELS[right_id])
                except Exception:
                    continue
                if left_label not in abbr_order or right_label not in abbr_order:
                    continue
                x_left, x_right = abbr_order.index(left_label), abbr_order.index(right_label)
                if x_left == x_right:
                    continue
                x0, x1 = (x_left, x_right) if x_left < x_right else (x_right, x_left)
                y = y_max + start_offset + (drawn * step)
                ax.plot([x0, x1], [y, y], color="black", linewidth=1.0)
                tick_h = 0.02 * y_span
                ax.plot([x0, x0], [y, y - tick_h], color="black", linewidth=1.0)
                ax.plot([x1, x1], [y, y - tick_h], color="black", linewidth=1.0)
                ax.text(
                    (x0 + x1) / 2.0, y + 0.04 * y_span, stars,
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                )
                drawn += 1
            y_upper = y_max + start_offset + (step * drawn) + (0.08 * y_span)
            y_lower = y_min - (0.04 * y_span)
            ax.set_ylim(y_lower, y_upper)
        elif baseline_val is not None:
            ax.set_ylim(y_min - 0.04 * y_span, y_max + 0.08 * y_span)

    fig.suptitle(f"{title_prefix}: best model ({best_model}) with significance", y=1.03)
    fig.subplots_adjust(bottom=0.30, top=0.86, wspace=0.2)
    _poster_save_figure(fig, out_base, formats, dpi)
    plt.close(fig)


def save_poster_comparison(
    fold_rows: list[dict],
    positive_rate: float | None,
    output_dir: Path,
    label: str,
) -> None:
    """
    Produces:
      {label}_poster_main.png            — model-stratified boxplots, all 6
                                            canonical feature sets
      {label}_poster_best_model.png      — best model (by mean AUPR) across
                                            feature sets, with paired-Wilcoxon
                                            significance stars for the
                                            curated comparisons
      {label}_significance_extended.csv  — paired Wilcoxon tests for ALL
                                            models x all C(6,2)=15 feature-set
                                            pairs (more thorough than what's
                                            annotated on the plot — "more
                                            extensive statistical comparison")

    positive_rate: used for the AUPR panel's dashed baseline reference line.
    Pass None (e.g. when reconstructing plots from saved CSVs without access
    to the original labeled dataset) to omit that line — the AUROC panel's
    0.5 baseline is unaffected either way.
    """
    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        logger.warning(f"  No fold-level results for {label!r} — skipping poster comparison.")
        return

    # Our feature-set names (build_feature_sets) are exact matches for
    # POSTER_CANONICAL_LABELS' values by construction, so a direct lookup is
    # all that's needed here (no fuzzy/regex alias matching required).
    label_to_id = {v: k for k, v in POSTER_CANONICAL_LABELS.items()}
    available = set(fold_df["feature_set"].unique())
    canonical_order = [cid for cid, lbl in POSTER_CANONICAL_LABELS.items() if lbl in available]
    if not canonical_order:
        logger.warning(f"  No feature sets in {label!r} matched the poster's canonical set — skipping.")
        return

    fold_sel = fold_df[fold_df["feature_set"].isin(label_to_id)].copy()
    fold_sel["canonical_feature_set"] = fold_sel["feature_set"].map(label_to_id)

    # Sort so every (model, canonical_feature_set) slice is in the same
    # (repeat, fold) order — _poster_compare_pairwise() pairs rows
    # positionally (reset_index + concat), not by (repeat, fold) key, so
    # misaligned ordering would silently compare mismatched folds. Also warn
    # if any combo is short a fold (e.g. from a per-combo exception during
    # CV) — positional pairing across uneven-length groups is unsafe.
    fold_sel = fold_sel.sort_values(
        ["model", "canonical_feature_set", "repeat", "fold"]
    ).reset_index(drop=True)
    counts = fold_sel.groupby(["model", "canonical_feature_set"], observed=True).size()
    if counts.nunique() > 1:
        logger.warning(
            f"  {label}: uneven fold counts across (model, feature_set) combos — "
            f"paired significance tests may compare misaligned folds:\n{counts.to_string()}"
        )

    best_model, _ = _poster_select_best_model(fold_sel, canonical_order)
    best_model_df = fold_sel[fold_sel["model"] == best_model].copy()

    # Curated comparisons for the plot annotation: each trajectory source vs.
    # its own no-trajectory base, mirroring the poster's own default pairing.
    default_pairs = [
        ("sd", "sd_traj"), ("sd", "sd_traj_boot"), ("sd_traj", "sd_traj_boot"),
        ("sd_sum", "sd_sum_traj"), ("sd_sum", "sd_sum_traj_boot"), ("sd_sum_traj", "sd_sum_traj_boot"),
    ]
    plot_pairs = [(l, r) for l, r in default_pairs if l in canonical_order and r in canonical_order]
    sig_df = _poster_compare_pairwise(best_model_df, plot_pairs, metrics=["auroc", "aupr"])

    chance_baseline = {"auroc": 0.5}
    if positive_rate is not None:
        chance_baseline["aupr"] = float(positive_rate)
    title_prefix = f"CKD {label.replace('_', ' ').title()}"

    _poster_plot_main_panel(
        fold_df=fold_sel, canonical_order=canonical_order, title_prefix=title_prefix,
        out_base=output_dir / f"{label}_poster_main", formats=["png"], dpi=200,
        figsize=(18.0, 6.8), chance_baseline=chance_baseline,
    )
    _poster_plot_best_model_panel(
        best_model_df=best_model_df, canonical_order=canonical_order, sig_df=sig_df,
        title_prefix=title_prefix, best_model=best_model,
        out_base=output_dir / f"{label}_poster_best_model", formats=["png"], dpi=200,
        figsize=(16.5, 7.2), chance_baseline=chance_baseline,
    )

    # Extended significance table: ALL models x all C(6,2) pairs, not just
    # the curated pairs annotated on the plot.
    all_pairs = list(itertools.combinations(canonical_order, 2))
    sig_rows = []
    for model_name in fold_sel["model"].unique():
        model_df = fold_sel[fold_sel["model"] == model_name]
        sig_model = _poster_compare_pairwise(model_df, all_pairs, metrics=["auroc", "aupr"])
        sig_model.insert(0, "model", model_name)
        sig_rows.append(sig_model)
    sig_extended = pd.concat(sig_rows, ignore_index=True) if sig_rows else pd.DataFrame()
    sig_extended.to_csv(output_dir / f"{label}_significance_extended.csv", index=False)

    logger.info(f"  Poster comparison saved for {label!r} (best model: {best_model})")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CKD trajectory analysis: evaluate ESKD/dialysis/death targets "
                    "on a pre-assembled patient-year matrix (build one first with "
                    "ckd_build_matrix.py)."
    )
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA),
        help="Path to the pre-assembled feature parquet, produced by ckd_build_matrix.py.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT),
        help="Output directory for all results.",
    )
    parser.add_argument("--cv-repeats", type=int, default=CV_N_REPEATS)
    parser.add_argument("--cv-splits",  type=int, default=CV_N_SPLITS)
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel jobs for CV repeats (threading backend).",
    )
    parser.add_argument(
        "--targets", nargs="+", default=None,
        help="Target columns to evaluate (default: auto-detect all target_* "
             "columns present in the data, excluding low-priority ones — see "
             "--include-low-priority-targets; falls back to the standard CKD "
             "target set if none are found). Explicitly naming a target here "
             "always includes it, regardless of priority.",
    )
    parser.add_argument(
        "--include-low-priority-targets", action="store_true",
        help=f"Include low-priority targets in auto-detection (default: excluded): "
             f"{sorted(LOW_PRIORITY_TARGETS)}. Ignored if --targets is passed explicitly.",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="K for top-k OOF export and plots.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load the pre-assembled patient-year matrix ────────────────────────────
    data_path = Path(args.data)
    logger.info(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"  Shape: {df.shape}")

    id_col = detect_id_col(df)
    logger.info(f"  Patient ID column: {id_col!r}")

    targets_to_run = args.targets
    if not targets_to_run:
        detected = sorted(c for c in df.columns if c.startswith("target_"))
        candidates = detected if detected else DEFAULT_TARGETS
        if not args.include_low_priority_targets:
            excluded = [t for t in candidates if t in LOW_PRIORITY_TARGETS]
            candidates = [t for t in candidates if t not in LOW_PRIORITY_TARGETS]
            if excluded:
                logger.info(
                    f"  Excluding low-priority targets by default: {excluded} "
                    f"(use --include-low-priority-targets to include)"
                )
        targets_to_run = candidates
        logger.info(f"  Auto-detected targets: {targets_to_run}")

    # ── Organise features on the raw, pre-imputation dataframe ───────────────
    groups = organize_features(df, id_col)
    df, groups = add_missingness_indicators(df, groups)
    logger.info("Feature groups:")
    for g in ("trajectory", "traj_bayes", "traj_boot", "summary",
              "baseline", "static", "dynamic", "cat_cols"):
        logger.info(f"  {g:10s}: {len(groups[g]):4d} columns")

    # ── One-hot encode categorical columns (before any imputation) ────────────
    df_encoded, ohe_cols = one_hot_encode(df, groups["cat_cols"])
    logger.info(
        f"OHE: {len(groups['cat_cols'])} categorical → {len(ohe_cols)} dummy columns"
    )

    # Sets used for NaN-fill decisions inside each fold
    traj_col_set    = set(groups["trajectory"])
    summary_col_set = set(groups["summary"])

    # ── Build feature set configurations ─────────────────────────────────────
    feature_configs = build_feature_sets(
        groups, ohe_cols, set(df_encoded.columns)
    )
    logger.info(f"Feature sets ({len(feature_configs)}):")
    for name, cols in feature_configs.items():
        n_traj_b = sum(1 for c in cols if c in groups["traj_bayes"])
        n_traj_s = sum(1 for c in cols if c in groups["traj_boot"])
        logger.info(
            f"  {name!r}: {len(cols)} features "
            f"(bayes_traj={n_traj_b}, boot_traj={n_traj_s})"
        )

    # ── Iterate over targets ──────────────────────────────────────────────────
    all_meta: dict[str, dict] = {}

    for target_col in targets_to_run:
        if target_col not in df_encoded.columns:
            logger.warning(f"Target {target_col!r} not in data — skipping")
            continue

        label = target_col.removeprefix("target_")
        sep = "=" * 65
        logger.info(f"\n{sep}")
        logger.info(f"  TARGET: {target_col}")
        logger.info(sep)

        df_target = df_encoded.dropna(subset=[target_col]).copy()
        df_target[target_col] = df_target[target_col].astype(int)
        logger.info(
            f"  Rows: {len(df_target):,}  "
            f"Positive: {df_target[target_col].mean():.1%}"
        )

        # Descriptive stats on unencoded data (for human-readable output)
        df_raw_target = df.dropna(subset=[target_col]).copy()
        df_raw_target[target_col] = df_raw_target[target_col].astype(int)
        save_descriptive_stats(
            df_raw=df_raw_target,
            target_col=target_col,
            id_col=id_col,
            groups=groups,
            output_dir=output_dir,
            label=label,
        )

        logger.info(f"  Running {args.cv_repeats}×{args.cv_splits} StratifiedGroupKFold CV …")
        summary, fold_rows, oof_df = run_cv(
            df=df_target,
            target_col=target_col,
            id_col=id_col,
            feature_configs=feature_configs,
            traj_col_set=traj_col_set,
            summary_col_set=summary_col_set,
            n_repeats=args.cv_repeats,
            n_splits=args.cv_splits,
            n_jobs=args.n_jobs,
        )

        save_results(
            summary=summary,
            fold_rows=fold_rows,
            oof_df=oof_df,
            output_dir=output_dir,
            label=label,
            feature_configs=feature_configs,
            top_k=args.top_k,
        )

        try:
            save_poster_comparison(
                fold_rows=fold_rows,
                positive_rate=float(df_target[target_col].mean()),
                output_dir=output_dir,
                label=label,
            )
        except Exception as exc:
            logger.warning(f"  Poster comparison failed for {label!r}: {exc}")

        all_meta[target_col] = {
            "n_rows":        len(df_target),
            "positive_rate": float(df_target[target_col].mean()),
            "n_folds_run":   len(fold_rows),
        }

    # ── Run-level metadata ────────────────────────────────────────────────────
    meta = {
        "timestamp_utc":     datetime.now(timezone.utc).isoformat(),
        "data_path":         str(data_path),
        "id_col":            id_col,
        "targets":           targets_to_run,
        "cv_repeats":        args.cv_repeats,
        "cv_splits":         args.cv_splits,
        "n_jobs":            args.n_jobs,
        "top_k":             args.top_k,
        "feature_groups":    {g: len(groups[g]) for g in groups},
        "feature_sets":      {name: len(cols) for name, cols in feature_configs.items()},
        "ohe_columns":       ohe_cols,
        "targets_detail":    all_meta,
    }
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        meta["git_commit"] = None

    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\n✓ All targets complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
