from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TaskTables:
    dataset: str
    task: str
    summary_df: pd.DataFrame
    fold_df: pd.DataFrame
    summary_path: Path
    fold_path: Path
    metadata_path: Path | None


def discover_tasks(results_root: Path, dataset: str) -> list[str]:
    dataset_dir = results_root / dataset
    if not dataset_dir.exists():
        return []

    out = []
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue
        task = child.name
        if _task_files_exist(child, dataset, task):
            out.append(task)

    # Some legacy outputs write files directly under results/{dataset} instead of a task subdir.
    legacy_tasks = _discover_root_level_tasks(dataset_dir, dataset)
    for task in legacy_tasks:
        if task not in out:
            out.append(task)
    return out


def _discover_root_level_tasks(dataset_dir: Path, dataset: str) -> list[str]:
    tasks: set[str] = set()
    summary_suffix = "_summary.csv"
    fold_suffixes = ["_fold_results.parquet", "_cv_results_fold_results.parquet", "_cv_results.csv"]

    for path in dataset_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.startswith(f"{dataset}_") and name.endswith(summary_suffix) and not name.endswith("_delta_summary.csv"):
            task = name[len(dataset) + 1 : -len(summary_suffix)]
            tasks.add(task)
        for suffix in fold_suffixes:
            if name.startswith(f"{dataset}_") and name.endswith(suffix):
                task = name[len(dataset) + 1 : -len(suffix)]
                tasks.add(task)
    return sorted(tasks)


def _task_files_exist(task_dir: Path, dataset: str, task: str) -> bool:
    summary_candidates = [
        task_dir / f"{task}_cv_results.csv",
        task_dir / f"{dataset}_{task}_summary.csv",
    ]
    fold_candidates = [
        task_dir / f"{task}_cv_results_fold_metrics.csv",
        task_dir / f"{dataset}_{task}_fold_results.parquet",
        task_dir / f"{task}_cv_results_fold_results.parquet",
    ]
    return any(p.exists() for p in summary_candidates) and any(p.exists() for p in fold_candidates)


def _read_task_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalize_task_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "summary":
        rename_map = {
            "Model": "model",
            "Feature Set": "feature_set",
            "ROC_AUC_mean": "auroc_mean",
            "ROC-AUC": "auroc_mean",
            "AUPR_mean": "aupr_mean",
            "AUPR": "aupr_mean",
        }
    else:
        rename_map = {
            "Model": "model",
            "Feature Set": "feature_set",
            "Repeat": "repeat",
            "Fold": "fold",
            "AUROC": "auroc",
            "ROC-AUC": "auroc",
            "AUPR": "aupr",
        }
    normalized = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return normalized


def _assert_cols(df: pd.DataFrame, cols: list[str], file_path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_path}: {missing}")


def load_task_tables(results_root: Path, dataset: str, task: str) -> TaskTables:
    task_dir = results_root / dataset / task
    root_dir = results_root / dataset
    summary_candidates = [
        root_dir / f"{dataset}_{task}_summary.csv",
        task_dir / f"{dataset}_{task}_summary.csv",
        task_dir / f"{task}_cv_results.csv",
    ]
    fold_candidates = [
        root_dir / f"{dataset}_{task}_fold_results.parquet",
        root_dir / f"{task}_cv_results_fold_results.parquet",
        root_dir / f"{task}_cv_results.csv",
        task_dir / f"{task}_cv_results_fold_metrics.csv",
        task_dir / f"{dataset}_{task}_fold_results.parquet",
        task_dir / f"{task}_cv_results_fold_results.parquet",
    ]
    metadata_candidates = [
        root_dir / f"{dataset}_{task}_metadata.json",
        task_dir / f"{task}_cv_results_metadata.json",
    ]

    summary_path = next((p for p in summary_candidates if p.exists()), None)
    fold_path = next((p for p in fold_candidates if p.exists()), None)
    metadata_path = next((p for p in metadata_candidates if p.exists()), None)

    if summary_path is None:
        raise FileNotFoundError(f"Missing summary file for {dataset}/{task}: tried {summary_candidates}")
    if fold_path is None:
        raise FileNotFoundError(f"Missing fold metrics file for {dataset}/{task}: tried {fold_candidates}")

    summary_df = _normalize_task_columns(_read_task_frame(summary_path), kind="summary")
    fold_df = _normalize_task_columns(_read_task_frame(fold_path), kind="fold")

    _assert_cols(
        summary_df,
        ["model", "feature_set", "auroc_mean", "aupr_mean"],
        summary_path,
    )
    _assert_cols(
        fold_df,
        ["model", "feature_set", "fold", "repeat", "auroc", "aupr"],
        fold_path,
    )

    return TaskTables(
        dataset=dataset,
        task=task,
        summary_df=summary_df,
        fold_df=fold_df,
        summary_path=summary_path,
        fold_path=fold_path,
        metadata_path=metadata_path,
    )
