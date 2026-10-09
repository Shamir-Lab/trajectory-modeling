#!/usr/bin/env python
"""
Build the two thesis summary tables from the current data + results on disk.

    Table 1  results/tables/prediction_tasks_overview.csv
             One row per (task, dataset): unique stays, prediction windows,
             positives, negatives, prevalence.

    Table 2  results/tables/cv_metrics_summary.csv
             One row per (task, dataset, model, feature set): AUROC / AUPR
             mean and std over the repeated-CV folds. Replaces the manually
             maintained results/compiled_cv_metrics.csv.

Cohort counts are recomputed by replaying each analysis script's own dataset
loading precedence and target filter, so the numbers match what was modelled
rather than the raw extraction.

Run from the workspace root:
    python scripts/analysis/tables/build_summary_tables.py
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.poster.feature_set_mapper import resolve_feature_mapping  # noqa: E402
from analysis.poster.io_loader import _normalize_task_columns, _read_task_frame  # noqa: E402

logger = logging.getLogger("summary_tables")

DATA_ROOT = Path("/home/gaga/data/physionet")
RESULTS_ROOT = Path("results")

TASK_LABELS = {
    "aki": "AKI Stage 3",
    "sepsis": "Septic Shock",
    "liver": "Liver ACLF",
    "ventilator": "Ventilator Weaning",
    "circulatory_failure": "Circulatory Failure",
}
DATASET_LABELS = {"mimic": "MIMIC-III", "eicu": "eICU", "hirid": "HiRID"}
ID_COLS = {"mimic": "hadm_id", "eicu": "stay_id", "hirid": "patientid"}

# Canonical feature-set order used by the poster pipeline / compiled_cv_metrics.
CANONICAL_IDS = ["sd", "sd_sum", "sd_traj", "sd_sum_traj", "sd_traj_boot", "sd_sum_traj_boot"]
SHORT_LABELS = {
    "sd": "SD",
    "sd_sum": "SD+S",
    "sd_traj": "SD+T",
    "sd_sum_traj": "SD+S+T",
    "sd_traj_boot": "SD+T (B)",
    "sd_sum_traj_boot": "SD+S+T (B)",
}


@dataclass
class CohortSpec:
    """How to rebuild the modelled cohort for one (dataset, task)."""

    dataset: str
    task: str
    subdir: str
    # Prediction-dataset stems in the order the analysis script tries them.
    pred_stems: list[str]
    target_col: str
    windowing_col: str
    window_desc: str
    outcome_file: str | None = None
    outcome_merge_cols: list[str] = field(default_factory=list)


# Stems mirror scripts/analysis/{dataset}/{...}_analysis.py (and, for the
# non-CF tasks, src/analysis/analysis_template.py load_data()).
CF_STEMS = [
    "circulatory_failure_prediction_dataset_with_bootstrap_probs",
    "circulatory_failure_prediction_dataset_with_probs",
    "circulatory_failure_prediction_dataset",
]


def _std_stems(task: str) -> list[str]:
    return [f"{task}_prediction_dataset_with_bootstrap_probs", f"{task}_prediction_dataset"]


COHORT_SPECS: list[CohortSpec] = [
    CohortSpec("mimic", "aki", "aki", _std_stems("aki"), "target_aki_stage3", "time_day",
               "daily windows, 7-day lookback", "aki_outcomes.csv",
               ["hadm_id", "time_day", "target_aki_stage3"]),
    CohortSpec("mimic", "sepsis", "sepsis", _std_stems("sepsis"), "target_septic_shock", "time_day",
               "daily windows, 3-day lookback", "septic_shock_outcomes.csv",
               ["hadm_id", "time_day", "target_septic_shock"]),
    CohortSpec("mimic", "liver", "liver", _std_stems("liver"), "target_aclf", "time_day",
               "daily windows, 5-day lookback", "aclf_outcomes.csv",
               ["hadm_id", "time_day", "target_aclf"]),
    CohortSpec("mimic", "ventilator", "ventilator", _std_stems("ventilator"), "target_weaning_success",
               "time_day", "daily windows, 3-day lookback", "ventilator_outcomes.csv",
               ["hadm_id", "time_day", "target_weaning_success"]),
    CohortSpec("mimic", "circulatory_failure", "circulatory_failure", CF_STEMS,
               "target_circulatory_failure", "time_hour", "hourly windows, 12-hour lookback"),
    CohortSpec("eicu", "aki", "aki", _std_stems("aki"), "target_aki_stage3", "time_day",
               "daily windows, 7-day lookback"),
    CohortSpec("eicu", "sepsis", "sepsis", _std_stems("sepsis"), "target_septic_shock", "time_day",
               "daily windows, 3-day lookback"),
    CohortSpec("eicu", "liver", "liver", _std_stems("liver"), "target_aclf", "time_day",
               "daily windows, 5-day lookback", "aclf_outcomes.csv",
               ["stay_id", "time_day", "target_aclf"]),
    CohortSpec("eicu", "ventilator", "ventilator", _std_stems("ventilator"), "target_weaning_success",
               "time_day", "daily windows, 3-day lookback"),
    CohortSpec("eicu", "circulatory_failure", "circulatory_failure", CF_STEMS,
               "target_circulatory_failure", "time_hour", "hourly windows, 12-hour lookback"),
    CohortSpec("hirid", "circulatory_failure", "circulatory_failure", CF_STEMS,
               "target_circulatory_failure", "time_hour", "hourly windows, 12-hour lookback"),
]


# ---------------------------------------------------------------- table 1


def _resolve_table(base_dir: Path, stem: str) -> Path | None:
    """Parquet-first, CSV fallback — the precedence used by every loader here."""
    for path in (base_dir / f"{stem}.parquet", base_dir / f"{stem}.csv"):
        if path.exists():
            return path
    return None


def _available_columns(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    return pd.read_csv(path, nrows=0).columns.tolist()


def _read_subset(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=cols)
    return pd.read_csv(path, usecols=cols)


def cohort_row(spec: CohortSpec) -> dict | None:
    base_dir = DATA_ROOT / spec.dataset / spec.subdir
    id_col = ID_COLS[spec.dataset]

    path = stem = None
    for candidate in spec.pred_stems:
        path = _resolve_table(base_dir, candidate)
        if path is not None:
            stem = candidate
            break
    if path is None:
        logger.warning("%s/%s: no prediction dataset found under %s", spec.dataset, spec.task, base_dir)
        return None

    cols = _available_columns(path)
    wanted = [c for c in (id_col, spec.windowing_col, spec.target_col) if c in cols]
    if id_col not in wanted:
        logger.warning("%s/%s: id column %s missing from %s", spec.dataset, spec.task, id_col, path.name)
        return None

    logger.info("%s/%s: reading %s", spec.dataset, spec.task, path.name)
    df = _read_subset(path, wanted)

    # Merge the outcome table when the target is not already carried on the
    # prediction dataset (mirrors AnalysisTemplate.load_data).
    merged_outcomes = False
    if spec.target_col not in df.columns and spec.outcome_file:
        outcome_path = base_dir / spec.outcome_file
        if not outcome_path.exists():
            logger.warning("%s/%s: outcome file %s missing", spec.dataset, spec.task, outcome_path)
            return None
        outcomes = pd.read_csv(outcome_path, usecols=spec.outcome_merge_cols)
        merge_keys = [c for c in spec.outcome_merge_cols if c != spec.target_col]
        df = df.merge(outcomes, on=merge_keys, how="left")
        merged_outcomes = True

    if spec.target_col not in df.columns:
        logger.warning("%s/%s: target %s not found", spec.dataset, spec.task, spec.target_col)
        return None

    n_before = len(df)
    df = df[df[spec.target_col].notna()]
    y = df[spec.target_col].astype(int)

    n_windows = len(df)
    n_pos = int(y.sum())
    # MIMIC ventilator's outcome table has repeated (id, window) keys, so the
    # left merge above duplicates some windows. Report both counts rather than
    # silently deduplicating: the CV runs on the merged rows as-is.
    n_distinct = int(df.groupby([id_col, spec.windowing_col]).ngroups) if spec.windowing_col in df.columns else n_windows
    return {
        "Task": TASK_LABELS[spec.task],
        "Dataset": DATASET_LABELS[spec.dataset],
        "Outcome": spec.target_col,
        "ID_Unit": id_col,
        "Windowing": spec.window_desc,
        "Unique_Stays": int(df[id_col].nunique()),
        "Prediction_Windows": n_windows,
        "Distinct_ID_Window_Pairs": n_distinct,
        "Positive_Windows": n_pos,
        "Negative_Windows": n_windows - n_pos,
        "Prevalence_Pct": round(100.0 * n_pos / n_windows, 3) if n_windows else float("nan"),
        "Windows_Dropped_No_Label": n_before - n_windows,
        "Source_File": str(path.relative_to(DATA_ROOT)),
        "Outcome_Merged_From": spec.outcome_file if merged_outcomes else "",
    }


def build_cohort_table() -> pd.DataFrame:
    rows = [r for r in (cohort_row(s) for s in COHORT_SPECS) if r is not None]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- table 2

_STD_ALIASES = {
    "auroc_std": ["auroc_std", "ROC_AUC_std", "ROC-AUC std"],
    "aupr_std": ["aupr_std", "AUPR_std", "AUPR std"],
}


def _pick(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return df[name]
    return None


def resolve_summary_path(dataset: str, task: str) -> Path | None:
    """Newest CV summary for this task.

    Several cohorts have both a legacy root-level `results/{ds}/{ds}_{task}_summary.csv`
    and a newer `results/{ds}/{task}/...` copy from a later re-run. io_loader's fixed
    precedence prefers the root-level file, which can be the stale one, so pick by
    modification time instead.
    """
    task_dir = RESULTS_ROOT / dataset / task
    root_dir = RESULTS_ROOT / dataset
    candidates = [
        root_dir / f"{dataset}_{task}_summary.csv",
        task_dir / f"{dataset}_{task}_summary.csv",
        task_dir / f"{task}_cv_results.csv",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    newest = max(existing, key=lambda p: p.stat().st_mtime)
    for other in existing:
        if other != newest:
            logger.info(
                "%s/%s: using newer %s over %s", dataset, task, newest, other
            )
    return newest


def metrics_rows(dataset: str, task: str) -> pd.DataFrame | None:
    summary_path = resolve_summary_path(dataset, task)
    if summary_path is None:
        logger.warning("%s/%s: no CV summary file found under %s", dataset, task, RESULTS_ROOT / dataset)
        return None

    summary = _normalize_task_columns(_read_task_frame(summary_path), kind="summary")
    # Some CF summaries carry both "ROC_AUC_mean" and a legacy "ROC-AUC" copy,
    # which the loader renames onto the same name; keep the first.
    summary = summary.loc[:, ~summary.columns.duplicated()]
    if "feature_set" not in summary.columns or "model" not in summary.columns:
        logger.warning("%s/%s: unexpected summary columns %s", dataset, task, list(summary.columns))
        return None

    available = summary["feature_set"].astype(str).unique().tolist()
    mapping = resolve_feature_mapping(available, CANONICAL_IDS, multi_marker_preference=True)
    raw_to_canonical = {raw: cid for cid, raw in mapping.items()}

    missing = [c for c in CANONICAL_IDS if c not in mapping]
    if missing:
        logger.warning(
            "%s/%s: no feature set matched %s (available: %s)", dataset, task, missing, available
        )

    out = pd.DataFrame(
        {
            "Task": TASK_LABELS.get(task, task),
            "Dataset": DATASET_LABELS.get(dataset, dataset),
            "Model": summary["model"].astype(str),
            "Feature_Set_Raw": summary["feature_set"].astype(str),
            "AUROC_Mean": summary["auroc_mean"],
            "AUROC_Std": _pick(summary, _STD_ALIASES["auroc_std"]),
            "AUPR_Mean": summary["aupr_mean"],
            "AUPR_Std": _pick(summary, _STD_ALIASES["aupr_std"]),
            "n_folds": summary["n_folds"] if "n_folds" in summary.columns else pd.NA,
            "Source_File": str(summary_path),
            "Results_Written": datetime.fromtimestamp(summary_path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            ),
        }
    )
    canonical = out["Feature_Set_Raw"].map(raw_to_canonical)
    out.insert(3, "Feature_Set", canonical.map(SHORT_LABELS))
    out.insert(4, "Feature_Set_Canonical", canonical)
    # Keep only the six canonical sets that the poster/thesis figures compare.
    out = out[out["Feature_Set_Canonical"].notna()].copy()
    order = {cid: i for i, cid in enumerate(CANONICAL_IDS)}
    out["_fs_order"] = out["Feature_Set_Canonical"].map(order)
    out = out.sort_values(["Model", "_fs_order"]).drop(columns="_fs_order")
    return out


def build_metrics_table() -> pd.DataFrame:
    frames = []
    for spec in COHORT_SPECS:
        frame = metrics_rows(spec.dataset, spec.task)
        if frame is not None and len(frame):
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    task_order = {TASK_LABELS[t]: i for i, t in enumerate(TASK_LABELS)}
    ds_order = {DATASET_LABELS[d]: i for i, d in enumerate(DATASET_LABELS)}
    combined["_t"] = combined["Task"].map(task_order)
    combined["_d"] = combined["Dataset"].map(ds_order)
    combined = combined.sort_values(["_t", "_d"], kind="stable").drop(columns=["_t", "_d"])
    for col in ("AUROC_Mean", "AUROC_Std", "AUPR_Mean", "AUPR_Std"):
        combined[col] = pd.to_numeric(combined[col], errors="coerce").round(4)
    # Provenance columns last so the metric block stays readable.
    tail = ["Source_File", "Results_Written"]
    combined = combined[[c for c in combined.columns if c not in tail] + tail]
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------- entrypoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="results/tables", help="Directory for the two CSVs")
    parser.add_argument("--skip-cohort", action="store_true", help="Only rebuild the CV metrics table")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_cohort:
        cohort = build_cohort_table()
        cohort_path = out_dir / "prediction_tasks_overview.csv"
        cohort.to_csv(cohort_path, index=False)
        logger.info("Wrote %s (%d rows)", cohort_path, len(cohort))
        print("\n=== Table 1: prediction tasks ===")
        print(cohort.to_string(index=False))

    metrics = build_metrics_table()
    metrics_path = out_dir / "cv_metrics_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    logger.info("Wrote %s (%d rows)", metrics_path, len(metrics))
    print("\n=== Table 2: CV metrics (%d rows) ===" % len(metrics))
    print(metrics.groupby(["Task", "Dataset"]).size().to_string())


if __name__ == "__main__":
    main()
