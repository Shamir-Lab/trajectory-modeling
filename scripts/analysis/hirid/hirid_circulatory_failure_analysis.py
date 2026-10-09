#!/usr/bin/env python
"""
HiRiD circulatory failure analysis script (notebook-to-script version).

This script reproduces the core analysis from:
  scripts/notebooks/hirid/circulatory_failure_analysis.ipynb

Run:
  python scripts/hirid_circulatory_failure_analysis.py
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed as jdelayed

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier


SEED = 920

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _pick_id_col(df: pd.DataFrame) -> str:
    for col in ["hadm_id", "stay_id", "patientid"]:
        if col in df.columns:
            return col
    raise ValueError("No ID column found")


def _pick_time_cols(df: pd.DataFrame) -> tuple[str, str]:
    if "time_hours" in df.columns and "time_hour" in df.columns:
        return "time_hours", "time_hour"
    if "time_days" in df.columns and "time_day" in df.columns:
        return "time_days", "time_day"
    if "time_hour" in df.columns:
        return "time_hour", "time_hour"
    if "time_day" in df.columns:
        return "time_day", "time_day"
    raise ValueError("No time columns found")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_named_table(base_dir: Path, stem_name: str) -> pd.DataFrame:
    candidates = [base_dir / f"{stem_name}.parquet", base_dir / f"{stem_name}.csv"]
    for p in candidates:
        if p.exists():
            logger.info(f"Using: {p}")
            return _read_table(p)
    raise FileNotFoundError(f"Missing table for {stem_name}. Tried: {candidates}")


def _safe_left_merge(base_df: pd.DataFrame, add_df: pd.DataFrame, key_cols: list[str], add_name: str) -> pd.DataFrame:
    base_dup = int(base_df.duplicated(subset=key_cols).sum())
    add_dup = int(add_df.duplicated(subset=key_cols).sum())

    if base_dup > 0:
        logger.warning(f"base dataset has {base_dup:,} duplicate keys; keeping first")
        base_df = base_df.drop_duplicates(subset=key_cols, keep="first")

    if add_dup > 0:
        prob_cols = [c for c in add_df.columns if c not in key_cols]
        logger.warning(f"{add_name} has {add_dup:,} duplicate keys; aggregating means")
        add_df = add_df.groupby(key_cols, as_index=False)[prob_cols].mean()

    return base_df.merge(add_df, on=key_cols, how="left", validate="one_to_one")


def _patient_window_stats(
    id_val,
    pid_df: pd.DataFrame,
    time_col: str,
    windowing_col: str,
    value_col: str,
    lookback_hours: int,
):
    pid_df = pid_df.sort_values(time_col)
    times = pd.to_numeric(pid_df[time_col], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(pid_df[value_col], errors="coerce").to_numpy(dtype=float)
    windows = pd.to_numeric(pid_df[windowing_col], errors="coerce").to_numpy(dtype=float)

    keep = np.isfinite(times) & np.isfinite(values) & np.isfinite(windows)
    times = times[keep]
    values = values[keep]
    windows = windows[keep]

    unique_hours = np.unique(windows)
    rows = []

    for h in unique_hours:
        mask = (times >= h - lookback_hours) & (times <= h)
        wv = values[mask]
        wt = times[mask]
        n = len(wv)

        if n == 0:
            rows.append((id_val, h, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        trend = 0.0
        if n > 1:
            x = wt[np.isfinite(wt) & np.isfinite(wv)]
            y = wv[np.isfinite(wt) & np.isfinite(wv)]
            if len(x) > 1 and np.unique(x).size > 1:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", np.exceptions.RankWarning)
                        trend = float(np.polyfit(x, y, 1)[0])
                except Exception:
                    trend = 0.0

        rows.append(
            (
                id_val,
                h,
                float(np.mean(wv)),
                float(np.max(wv)),
                float(np.min(wv)),
                float(wv[-1] - wv[0]),
                trend,
                float(np.std(wv)) if n > 1 else 0.0,
            )
        )

    return rows


def biomarker_summary_stats(ts_df: pd.DataFrame, value_col: str, lookback_hours: int = 12, n_jobs: int = 4) -> pd.DataFrame:
    loc_id_col = _pick_id_col(ts_df)
    time_col, wc = _pick_time_cols(ts_df)
    suffix = f"_{lookback_hours}h"

    df = ts_df[[loc_id_col, time_col, wc, value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[wc] = pd.to_numeric(df[wc], errors="coerce")
    df = df.dropna(subset=[value_col, time_col, wc])

    dup_exact = int(df.duplicated(subset=[loc_id_col, time_col]).sum())
    if dup_exact > 0:
        n_before = len(df)
        df = df.groupby([loc_id_col, time_col], as_index=False).agg({wc: "first", value_col: "mean"})
        logger.info(f"{value_col}: removed {dup_exact:,} exact duplicate (patient, time) rows ({n_before:,} -> {len(df):,})")

    groups_list = list(df.groupby(loc_id_col))
    logger.info(f"{value_col}: {len(groups_list):,} patients (n_jobs={n_jobs})")

    nested = Parallel(n_jobs=n_jobs)(
        jdelayed(_patient_window_stats)(pid, gdf, time_col, wc, value_col, lookback_hours)
        for pid, gdf in groups_list
    )
    all_rows = [r for sublist in nested for r in sublist]

    cols = [
        loc_id_col,
        wc,
        f"{value_col}_mean{suffix}",
        f"{value_col}_max{suffix}",
        f"{value_col}_min{suffix}",
        f"{value_col}_change{suffix}",
        f"{value_col}_trend{suffix}",
        f"{value_col}_std{suffix}",
    ]
    out = pd.DataFrame(all_rows, columns=cols)
    out = out.drop_duplicates(subset=[loc_id_col, wc], keep="last")
    return out


def build_feature_sets(dataset: pd.DataFrame):
    target_col = next(
        (c for c in ["target_circulatory_failure", "target_circ_failure", "target_circulatory"] if c in dataset.columns),
        None,
    )
    if target_col is None:
        raise ValueError("No target column found")

    if "gender" in dataset.columns:
        dataset["gender"] = dataset["gender"].map({"M": 1, "F": 0})

    static_features = [c for c in ["age", "gender"] if c in dataset.columns]

    # Trajectory columns from both backends, kept strictly disjoint: bayes-sourced
    # columns are unsuffixed, bootstrap-sourced columns carry a "_boot" suffix
    # (see main()'s dual-backend merge). Never combined into the same feature
    # set — mixing them would defeat the point of comparing the two trajectory
    # representations independently (mirrors build_feature_sets() in
    # ckd_trajectory_analysis.py).
    lactate_traj = ["lactate_stable", "lactate_gradual", "lactate_rapid", "lactate_worsening"]
    heartrate_traj = ["heartrate_stable", "heartrate_gradual", "heartrate_rapid", "heartrate_worsening"]
    systolic_traj = ["systolic_stable", "systolic_gradual", "systolic_rapid", "systolic_worsening"]
    multi_traj = lactate_traj + heartrate_traj + systolic_traj + ["multi_marker_mean_worsening"]

    lactate_traj_boot = [f"{c}_boot" for c in lactate_traj]
    heartrate_traj_boot = [f"{c}_boot" for c in heartrate_traj]
    systolic_traj_boot = [f"{c}_boot" for c in systolic_traj]
    multi_traj_boot = lactate_traj_boot + heartrate_traj_boot + systolic_traj_boot + ["multi_marker_mean_worsening_boot"]

    lactate_summary = [c for c in dataset.columns if c.startswith("lactate_") and c.endswith("h")]
    heartrate_summary = [c for c in dataset.columns if c.startswith("heartrate_") and c.endswith("h")]
    systolic_summary = [c for c in dataset.columns if c.startswith("systolic_") and c.endswith("h")]
    multi_summary = lactate_summary + heartrate_summary + systolic_summary

    # Anything already classified as trajectory (either backend) or summary must
    # not also leak into "dynamic" via the substring-based vital-sign scan below
    # (e.g. "systolic_stable", "systolic_stable_boot", and "systolic_mean_12h"
    # all contain "systolic") — that would silently contaminate "Baseline" with
    # trajectory/summary signal it's supposed to exclude, and would do so
    # asymmetrically depending on which backend(s) happen to be loaded.
    non_baseline_cols = set(multi_traj + multi_traj_boot + multi_summary)

    dynamic_labs = [
        col
        for col in dataset.columns
        if any(col.startswith(prefix) for prefix in ["min_", "mean_", "max_"])
        and col not in non_baseline_cols
    ]
    dynamic_vitals = [
        col
        for col in dataset.columns
        if any(x in col for x in ["heart_rate", "respiratory_rate", "o2_sat", "systolic", "diastolic", "mean_bp", "temperature"])
        and col not in non_baseline_cols
    ]
    dynamic_features = list(dict.fromkeys(dynamic_labs + dynamic_vitals))

    baseline_features = static_features + dynamic_features
    if len(baseline_features) == 0:
        baseline_features = static_features

    def _present(cols):
        return [c for c in list(dict.fromkeys(cols)) if c in dataset.columns]

    # Column presence alone isn't enough to detect whether a backend has real
    # data: main() fills any missing bayes-convention trajectory column with
    # an all-NaN placeholder for uniformity, so the columns always exist even
    # when that backend was never loaded.
    def _has_data(cols):
        present = _present(cols)
        return len(present) > 0 and bool(dataset[present].notna().any().any())

    has_bayes_traj = _has_data(lactate_traj)
    has_boot_traj = _has_data(lactate_traj_boot)
    logger.info(f"Trajectory backends available for feature sets: bayes={has_bayes_traj}, bootstrap={has_boot_traj}")

    feature_sets = {
        "Baseline": _present(baseline_features),
        "Baseline + Single Summary": _present(baseline_features + lactate_summary),
        "Baseline + Multi Summary": _present(baseline_features + multi_summary),
    }
    if has_bayes_traj:
        feature_sets["Baseline + Single Trajectory"] = _present(baseline_features + lactate_traj)
        feature_sets["Baseline + Multi Trajectory"] = _present(baseline_features + multi_traj)
        feature_sets["Baseline + Single Trajectory + Summary"] = _present(baseline_features + lactate_traj + lactate_summary)
        feature_sets["Baseline + Multi Trajectory + Summary"] = _present(baseline_features + multi_traj + multi_summary)
    if has_boot_traj:
        feature_sets["Baseline + Single Trajectory (Bootstrap)"] = _present(baseline_features + lactate_traj_boot)
        feature_sets["Baseline + Multi Trajectory (Bootstrap)"] = _present(baseline_features + multi_traj_boot)
        feature_sets["Baseline + Single Trajectory + Summary (Bootstrap)"] = _present(baseline_features + lactate_traj_boot + lactate_summary)
        feature_sets["Baseline + Multi Trajectory + Summary (Bootstrap)"] = _present(baseline_features + multi_traj_boot + multi_summary)

    feature_sets = {k: v for k, v in feature_sets.items() if len(v) > 0}

    return target_col, feature_sets


def _run_one_repeat(
    dataset_model: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_cols: list[str],
    model_name: str,
    model_fn,
    boosting_models: set[str],
    n_folds: int,
    repeat_seed: int,
):
    rng = np.random.RandomState(seed=repeat_seed)
    shuffle_idx = rng.permutation(len(dataset_model))

    dataset_repeat = dataset_model.iloc[shuffle_idx].reset_index(drop=True)
    y_repeat = y.iloc[shuffle_idx].reset_index(drop=True)
    groups_repeat = groups.iloc[shuffle_idx].reset_index(drop=True)

    gkf = GroupKFold(n_splits=n_folds)
    rep_auc, rep_aupr = [], []
    rep_fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(dataset_repeat, y_repeat, groups_repeat), start=1):
        X_train = dataset_repeat.iloc[train_idx][feature_cols]
        X_test = dataset_repeat.iloc[test_idx][feature_cols]
        y_train = y_repeat.iloc[train_idx]
        y_test = y_repeat.iloc[test_idx]

        if model_name not in boosting_models:
            imputer = SimpleImputer(strategy="median")
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)
        else:
            X_train_imputed = X_train
            X_test_imputed = X_test

        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train_imputed)
        X_test_proc = scaler.transform(X_test_imputed)

        pos = y_train.sum()
        neg = len(y_train) - pos
        scale_pos_weight = neg / max(pos, 1)

        model = model_fn(scale_pos_weight, repeat_seed)
        model.fit(X_train_proc, y_train)

        y_pred_proba = model.predict_proba(X_test_proc)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        aupr = average_precision_score(y_test, y_pred_proba)

        rep_auc.append(auc)
        rep_aupr.append(aupr)

        rep_fold_rows.append(
            {
                "repeat": (repeat_seed - SEED) + 1,
                "fold": fold_idx,
                "n_features": len(feature_cols),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "pos_rate_test": float(y_test.mean()),
                "ROC-AUC": auc,
                "AUPR": aupr,
            }
        )

    return rep_auc, rep_aupr, rep_fold_rows


def run_cv(
    dataset: pd.DataFrame,
    target_col: str,
    id_col: str,
    feature_sets: dict[str, list[str]],
    eval_profile: str,
    train_n_jobs: int,
    train_backend: str,
):
    max_rows_quick = 250_000

    dataset_clean = dataset.dropna(subset=[target_col]).copy()
    if eval_profile == "quick" and len(dataset_clean) > max_rows_quick:
        rng = np.random.default_rng(SEED)
        sample_idx = rng.choice(dataset_clean.index.values, size=max_rows_quick, replace=False)
        dataset_clean = dataset_clean.loc[sample_idx].copy()
        logger.info(f"Quick profile: sampled {len(dataset_clean):,} rows")

    y = dataset_clean[target_col].astype(int)
    groups = dataset_clean[id_col]

    traj_cols = [c for c in dataset_clean.columns if any(k in c for k in ["_stable", "_gradual", "_rapid", "_worsening"])]
    summary_cols = [
        c
        for c in dataset_clean.columns
        if c.endswith("h") and any(prefix in c for prefix in ["lactate_", "heartrate_", "systolic_"])
    ]

    dataset_clean[traj_cols] = dataset_clean.groupby(id_col)[traj_cols].ffill(limit=2)

    boosting_models = {"XGBoost", "Gradient Boosting"}

    models_to_evaluate = {
        "XGBoost": lambda pos_weight, seed: XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            random_state=seed,
            eval_metric="logloss",
        ),
        "Logistic Regression": lambda pos_weight, seed: LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        ),
        "Random Forest": lambda pos_weight, seed: RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=seed,
            n_jobs=4,
        ),
        "Gradient Boosting": lambda pos_weight, seed: HistGradientBoostingClassifier(
            max_bins=225,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
        ),
    }

    n_repeats, n_folds = (2, 3) if eval_profile == "quick" else (10, 5)
    logger.info(
        f"Starting CV: models={len(models_to_evaluate)}, feature_sets={len(feature_sets)}, repeats={n_repeats}, folds={n_folds}, train_n_jobs={train_n_jobs}, backend={train_backend}"
    )

    fold_rows = []

    for model_name, model_fn in models_to_evaluate.items():
        logger.info(f"{'=' * 80}\nModel: {model_name}\n{'=' * 80}")

        dataset_model = dataset_clean.copy()
        if model_name not in boosting_models:
            dataset_model[traj_cols] = dataset_model[traj_cols].fillna(0)
            dataset_model[summary_cols] = dataset_model[summary_cols].fillna(0)

        for feature_set_name, feature_cols in feature_sets.items():
            logger.info(f"  -> Feature set: {feature_set_name} ({len(feature_cols)} features)")
            fs_auc, fs_aupr = [], []

            repeat_seeds = [SEED + repeat for repeat in range(n_repeats)]
            if train_n_jobs > 1:
                repeat_outputs = Parallel(n_jobs=train_n_jobs, prefer=train_backend)(
                    jdelayed(_run_one_repeat)(
                        dataset_model=dataset_model,
                        y=y,
                        groups=groups,
                        feature_cols=feature_cols,
                        model_name=model_name,
                        model_fn=model_fn,
                        boosting_models=boosting_models,
                        n_folds=n_folds,
                        repeat_seed=repeat_seed,
                    )
                    for repeat_seed in repeat_seeds
                )
            else:
                repeat_outputs = [
                    _run_one_repeat(
                        dataset_model=dataset_model,
                        y=y,
                        groups=groups,
                        feature_cols=feature_cols,
                        model_name=model_name,
                        model_fn=model_fn,
                        boosting_models=boosting_models,
                        n_folds=n_folds,
                        repeat_seed=repeat_seed,
                    )
                    for repeat_seed in repeat_seeds
                ]

            for repeat_idx, (rep_auc, rep_aupr, rep_fold_rows) in enumerate(repeat_outputs, start=1):
                fs_auc.extend(rep_auc)
                fs_aupr.extend(rep_aupr)

                for row in rep_fold_rows:
                    row["Model"] = model_name
                    row["Feature Set"] = feature_set_name
                    fold_rows.append(row)

                logger.info(
                    f"     repeat {repeat_idx:02d}/{n_repeats}: "
                    f"AUROC={np.mean(rep_auc):.4f} | AUPR={np.mean(rep_aupr):.4f}"
                )

            logger.info(
                f"     done {feature_set_name}: "
                f"AUROC={np.mean(fs_auc):.4f}±{np.std(fs_auc):.4f} | "
                f"AUPR={np.mean(fs_aupr):.4f}±{np.std(fs_aupr):.4f}"
            )

    results_df = pd.DataFrame(fold_rows)

    summary_df = (
        results_df.groupby(["Model", "Feature Set"], as_index=False)
        .agg(
            ROC_AUC_mean=("ROC-AUC", "mean"),
            ROC_AUC_std=("ROC-AUC", "std"),
            AUPR_mean=("AUPR", "mean"),
            AUPR_std=("AUPR", "std"),
            n_folds=("ROC-AUC", "count"),
        )
    )

    baseline_metrics = (
        results_df[results_df["Feature Set"] == "Baseline"][["Model", "repeat", "fold", "ROC-AUC", "AUPR"]]
        .rename(columns={"ROC-AUC": "Baseline ROC-AUC", "AUPR": "Baseline AUPR"})
    )

    delta_df = results_df.merge(
        baseline_metrics,
        on=["Model", "repeat", "fold"],
        how="left",
        validate="many_to_one",
    )
    delta_df["Delta ROC-AUC"] = delta_df["ROC-AUC"] - delta_df["Baseline ROC-AUC"]
    delta_df["Delta AUPR"] = delta_df["AUPR"] - delta_df["Baseline AUPR"]

    delta_summary_df = (
        delta_df[delta_df["Feature Set"] != "Baseline"]
        .groupby(["Model", "Feature Set"], as_index=False)
        .agg(
            Delta_ROC_AUC_mean=("Delta ROC-AUC", "mean"),
            Delta_ROC_AUC_std=("Delta ROC-AUC", "std"),
            Delta_AUPR_mean=("Delta AUPR", "mean"),
            Delta_AUPR_std=("Delta AUPR", "std"),
            n_folds=("Delta ROC-AUC", "count"),
        )
    )

    logger.info(f"Finished CV. Fold rows: {len(results_df):,}")
    return results_df, summary_df, delta_summary_df


def main():
    parser = argparse.ArgumentParser(description="HiRiD circulatory failure trajectory analysis")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.join(os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet"), "hirid", "circulatory_failure"),
        help="Directory containing HiRiD circulatory failure data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.environ.get("TRAJ_RESULTS_ROOT", str(Path(__file__).resolve().parents[3] / "results")), "hirid"),
        help="Directory to save output tables",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=12,
        help="Lookback window in hours for summary stats",
    )
    parser.add_argument(
        "--eval-profile",
        type=str,
        default="full",
        choices=["quick", "full"],
        help="quick=2x3 CV on sampled rows, full=10x5 CV",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel workers for summary-stat computation",
    )
    parser.add_argument(
        "--train-n-jobs",
        type=int,
        default=1,
        help="Parallel workers for CV repeats during training (1 disables training parallelism)",
    )
    parser.add_argument(
        "--train-backend",
        type=str,
        default="threads",
        choices=["threads", "processes"],
        help="Parallel backend for training repeats. Use processes only if memory allows.",
    )
    args = parser.parse_args()

    np.random.seed(SEED)

    base_candidates = [
        Path("../../../results/hirid/circulatory_failure"),
        Path(args.base_dir),
    ]
    base_dir = next((p for p in base_candidates if p.exists()), Path(args.base_dir))
    logger.info(f"BASE_DIR: {base_dir}")

    pred_stem_bootstrap = "circulatory_failure_prediction_dataset_with_bootstrap_probs"
    pred_stem_bayes = "circulatory_failure_prediction_dataset_with_probs"
    pred_stem_base = "circulatory_failure_prediction_dataset"

    expected_traj_cols = [
        "lactate_stable",
        "lactate_gradual",
        "lactate_rapid",
        "heartrate_stable",
        "heartrate_gradual",
        "heartrate_rapid",
        "systolic_stable",
        "systolic_gradual",
        "systolic_rapid",
    ]

    def _try_load(stem):
        try:
            return _load_named_table(base_dir, stem)
        except FileNotFoundError:
            return None

    dataset_boot = _try_load(pred_stem_bootstrap)
    dataset_bayes = _try_load(pred_stem_bayes)

    has_boot_merged = dataset_boot is not None and all(c in dataset_boot.columns for c in expected_traj_cols)
    has_bayes_merged = dataset_bayes is not None and all(c in dataset_bayes.columns for c in expected_traj_cols)
    logger.info(f"Pre-merged bootstrap trajectory dataset present: {has_boot_merged}")
    logger.info(f"Pre-merged bayes trajectory dataset present: {has_bayes_merged}")

    # need_raw_bootstrap: only set when no pre-merged bootstrap dataset was
    # found, so raw per-biomarker bootstrap files are tried as a fallback
    # below (mirrors the pre-existing single-backend fallback behavior).
    need_raw_bootstrap = False

    if has_boot_merged and has_bayes_merged:
        # Both backends available: merge into one working dataframe, keeping the
        # two trajectory representations disjoint (bayes columns unsuffixed,
        # bootstrap columns "_boot"-suffixed) so build_feature_sets() can build
        # independently comparable "Baseline + ..." / "Baseline + ... (Bootstrap)"
        # feature sets, matching the poster pipeline's (Bootstrap)-suffix
        # convention. Never blended into the same column/feature set.
        logger.info("Both trajectory backends found — merging into one working dataframe.")
        id_col = _pick_id_col(dataset_boot)
        _, windowing_col = _pick_time_cols(dataset_boot)
        key_cols = [id_col, windowing_col]
        dataset = dataset_boot.rename(columns={c: f"{c}_boot" for c in expected_traj_cols})
        dataset = _safe_left_merge(dataset, dataset_bayes[key_cols + expected_traj_cols], key_cols, "bayes_traj")
        pred_source_stem = f"{pred_stem_bootstrap} + {pred_stem_bayes}"
    elif has_boot_merged:
        dataset = dataset_boot
        pred_source_stem = pred_stem_bootstrap
        id_col = _pick_id_col(dataset)
        _, windowing_col = _pick_time_cols(dataset)
        key_cols = [id_col, windowing_col]
    elif has_bayes_merged:
        dataset = dataset_bayes
        pred_source_stem = pred_stem_bayes
        id_col = _pick_id_col(dataset)
        _, windowing_col = _pick_time_cols(dataset)
        key_cols = [id_col, windowing_col]
        need_raw_bootstrap = True
    else:
        dataset = _load_named_table(base_dir, pred_stem_base)
        pred_source_stem = pred_stem_base
        id_col = _pick_id_col(dataset)
        _, windowing_col = _pick_time_cols(dataset)
        key_cols = [id_col, windowing_col]
        need_raw_bootstrap = True

    logger.info(f"Loaded prediction dataset ({pred_source_stem}): {len(dataset):,} samples")

    lactate_ts = _load_named_table(base_dir, "lactate_timeseries")
    heartrate_ts = _load_named_table(base_dir, "heartrate_timeseries")
    systolic_ts = _load_named_table(base_dir, "systolic_timeseries")

    if need_raw_bootstrap:
        # Raw per-biomarker bootstrap files already use the dataset's final
        # column names directly (lactate_stable, ...), no "prob_*" renaming
        # needed (unlike the bayes per-cohort files handled by
        # merge_circulatory_failure_bayes.sh). Suffix with "_boot" only if
        # bayes columns are already present unsuffixed (has_bayes_merged) so
        # the two backends stay disjoint; otherwise keep bootstrap unsuffixed
        # for backward compatibility with existing single-backend runs.
        suffix = "_boot" if has_bayes_merged else ""
        try:
            lactate_boot = _load_named_table(base_dir, "lactate_trajectory_probs_bootstrap")
            heartrate_boot = _load_named_table(base_dir, "heartrate_trajectory_probs_bootstrap")
            systolic_boot = _load_named_table(base_dir, "systolic_trajectory_probs_bootstrap")
            if suffix:
                for boot_df in (lactate_boot, heartrate_boot, systolic_boot):
                    boot_df.rename(columns={c: f"{c}{suffix}" for c in boot_df.columns if c not in key_cols}, inplace=True)
            dataset = _safe_left_merge(dataset, lactate_boot, key_cols, "lactate_boot")
            dataset = _safe_left_merge(dataset, heartrate_boot, key_cols, "heartrate_boot")
            dataset = _safe_left_merge(dataset, systolic_boot, key_cols, "systolic_boot")
        except FileNotFoundError as exc:
            logger.warning(f"Raw bootstrap trajectory files not found, skipping: {exc}")

    for bm in ["lactate", "heartrate", "systolic"]:
        for comp in ["stable", "gradual", "rapid"]:
            col = f"{bm}_{comp}"
            if col not in dataset.columns:
                dataset[col] = np.nan

    dataset["lactate_worsening"] = dataset["lactate_gradual"] + dataset["lactate_rapid"]
    dataset["heartrate_worsening"] = dataset["heartrate_gradual"] + dataset["heartrate_rapid"]
    dataset["systolic_worsening"] = dataset["systolic_gradual"] + dataset["systolic_rapid"]
    dataset["multi_marker_mean_worsening"] = dataset[["lactate_worsening", "heartrate_worsening", "systolic_worsening"]].mean(axis=1)

    # Same worsening/multi-marker derived features for the bootstrap backend,
    # only computed when bootstrap-sourced ("_boot"-suffixed) columns are
    # actually present — otherwise these stay absent, matching the
    # single-backend case where no Bootstrap feature sets are built at all.
    boot_traj_cols = [f"{bm}_{comp}_boot" for bm in ["lactate", "heartrate", "systolic"] for comp in ["stable", "gradual", "rapid"]]
    if any(c in dataset.columns for c in boot_traj_cols):
        for bm in ["lactate", "heartrate", "systolic"]:
            for comp in ["stable", "gradual", "rapid"]:
                col = f"{bm}_{comp}_boot"
                if col not in dataset.columns:
                    dataset[col] = np.nan
        dataset["lactate_worsening_boot"] = dataset["lactate_gradual_boot"] + dataset["lactate_rapid_boot"]
        dataset["heartrate_worsening_boot"] = dataset["heartrate_gradual_boot"] + dataset["heartrate_rapid_boot"]
        dataset["systolic_worsening_boot"] = dataset["systolic_gradual_boot"] + dataset["systolic_rapid_boot"]
        dataset["multi_marker_mean_worsening_boot"] = dataset[
            ["lactate_worsening_boot", "heartrate_worsening_boot", "systolic_worsening_boot"]
        ].mean(axis=1)

    logger.info(f"Dataset ready with trajectory probabilities: {len(dataset):,} samples")

    logger.info(f"Computing summary statistics (lookback={args.lookback_hours}h)")
    lactate_sum_df = biomarker_summary_stats(lactate_ts, "lactate", lookback_hours=args.lookback_hours, n_jobs=args.n_jobs)
    heartrate_sum_df = biomarker_summary_stats(heartrate_ts, "heartrate", lookback_hours=args.lookback_hours, n_jobs=args.n_jobs)
    systolic_sum_df = biomarker_summary_stats(systolic_ts, "systolic", lookback_hours=args.lookback_hours, n_jobs=args.n_jobs)

    dataset = _safe_left_merge(dataset, lactate_sum_df, key_cols, "lactate_summary")
    dataset = _safe_left_merge(dataset, heartrate_sum_df, key_cols, "heartrate_summary")
    dataset = _safe_left_merge(dataset, systolic_sum_df, key_cols, "systolic_summary")

    target_col, feature_sets = build_feature_sets(dataset)

    results_df, summary_df, delta_summary_df = run_cv(
        dataset,
        target_col=target_col,
        id_col=id_col,
        feature_sets=feature_sets,
        eval_profile=args.eval_profile,
        train_n_jobs=args.train_n_jobs,
        train_backend=args.train_backend,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "hirid_circulatory_failure_fold_results.parquet"
    summary_path = output_dir / "hirid_circulatory_failure_summary.csv"
    delta_path = output_dir / "hirid_circulatory_failure_delta_summary.csv"

    results_df.to_parquet(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    delta_summary_df.to_csv(delta_path, index=False)

    logger.info(f"Saved: {results_path}")
    logger.info(f"Saved: {summary_path}")
    logger.info(f"Saved: {delta_path}")


if __name__ == "__main__":
    main()
