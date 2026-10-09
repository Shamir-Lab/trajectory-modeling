from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .config_schema import PosterConfig, load_poster_config
from .feature_set_mapper import CANONICAL_LABELS, resolve_feature_mapping
from .io_loader import discover_tasks, load_task_tables
from .model_selector import select_best_model_by_aupr
from .plot_panels import plot_best_model_panel, plot_main_panel
from .stats_significance import compare_pairwise

logger = logging.getLogger(__name__)


def _normalize_feature_sets(
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    mapping: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv_map = {v: k for k, v in mapping.items()}

    fold = fold_df[fold_df["feature_set"].isin(inv_map)].copy()
    summary = summary_df[summary_df["feature_set"].isin(inv_map)].copy()

    fold["canonical_feature_set"] = fold["feature_set"].map(inv_map)
    summary["canonical_feature_set"] = summary["feature_set"].map(inv_map)

    return fold, summary


def _task_output_dir(cfg: PosterConfig, dataset: str, task: str) -> Path:
    return Path(cfg.outputs.root) / dataset / task / cfg.run.run_name


def _save_tables(out_dir: Path, fold_df: pd.DataFrame, summary_df: pd.DataFrame, leaderboard_df: pd.DataFrame, sig_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / "fold_metrics_selected_models.csv", index=False)
    summary_df.to_csv(out_dir / "metrics_selected_models.csv", index=False)
    leaderboard_df.to_csv(out_dir / "best_model_summary.csv", index=False)
    sig_df.to_csv(out_dir / "best_model_significance.csv", index=False)


def _save_manifest(
    out_dir: Path,
    cfg: PosterConfig,
    dataset: str,
    task: str,
    mapping: dict[str, str],
    best_model: str,
    source_paths: dict[str, str],
    n_rows: int,
) -> None:
    manifest = {
        "dataset": dataset,
        "task": task,
        "run_name": cfg.run.run_name,
        "repeat_policy": cfg.run.repeat_policy,
        "metric_for_best_model": cfg.selection.metric_for_best_model,
        "selected_feature_mapping": mapping,
        "best_model": best_model,
        "source_files": source_paths,
        "n_fold_rows": int(n_rows),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run_poster_pipeline(config_path: str | Path) -> None:
    cfg = load_poster_config(config_path)
    results_root = Path(cfg.inputs.results_root)
    logger.info("Running poster pipeline: %s", cfg.run.run_name)

    for dataset in cfg.inputs.datasets:
        tasks = list(cfg.inputs.tasks) if cfg.inputs.tasks else discover_tasks(results_root, dataset)
        if not tasks:
            logger.warning("No tasks discovered for dataset=%s", dataset)
            continue

        for task in tasks:
            logger.info("Processing %s/%s", dataset, task)
            try:
                tables = load_task_tables(results_root, dataset, task)
            except Exception as e:
                logger.warning("Skipping %s/%s: %s", dataset, task, e)
                continue

            available_feature_sets = sorted(set(tables.fold_df["feature_set"].astype(str).tolist()))
            mapping = resolve_feature_mapping(
                available_feature_sets=available_feature_sets,
                canonical_ids=cfg.selection.feature_sets_canonical,
                custom_aliases=cfg.selection.feature_set_aliases,
                multi_marker_preference=cfg.selection.multi_marker_preference,
            )

            if not mapping:
                logger.warning("Skipping %s/%s: no selected feature sets found", dataset, task)
                continue

            fold_sel, summary_sel = _normalize_feature_sets(
                fold_df=tables.fold_df,
                summary_df=tables.summary_df,
                mapping=mapping,
            )
            if fold_sel.empty:
                logger.warning("Skipping %s/%s: no fold rows after mapping", dataset, task)
                continue

            canonical_order = [c for c in cfg.selection.feature_sets_canonical if c in mapping]

            best_model, leaderboard_df = select_best_model_by_aupr(
                fold_df=fold_sel,
                canonical_feature_ids=canonical_order,
            )

            best_model_df = fold_sel[fold_sel["model"] == best_model].copy()
            # Determine pairwise comparisons: prefer config list, otherwise use sensible defaults
            if getattr(cfg.statistics, "pairwise_comparisons", None):
                comparisons = [(a, b) for a, b in cfg.statistics.pairwise_comparisons]
            else:
                # default comparisons requested by user
                comparisons = [
                    ("sd", "sd_traj"),
                    ("sd", "sd_traj_boot"),
                    ("sd_traj", "sd_traj_boot"),
                    ("sd_sum", "sd_sum_traj"),
                    ("sd_sum", "sd_sum_traj_boot"),
                    ("sd_sum_traj", "sd_sum_traj_boot"),
                ]

            sig_df = compare_pairwise(
                best_model_df=best_model_df,
                comparisons=comparisons,
            )

            out_dir = _task_output_dir(cfg, dataset, task)
            out_dir.mkdir(parents=True, exist_ok=True)

            title_prefix = f"{dataset.upper()} {task.replace('_', ' ').title()}"
            plot_main_panel(
                fold_df=fold_sel,
                canonical_order=canonical_order,
                title_prefix=title_prefix,
                out_base=out_dir / "panel_main_auroc_aupr",
                formats=cfg.plot.formats,
                dpi=cfg.plot.dpi,
                figsize=(float(cfg.plot.figsize_main[0]), float(cfg.plot.figsize_main[1])),
            )
            plot_best_model_panel(
                best_model_df=best_model_df,
                canonical_order=canonical_order,
                sig_df=sig_df,
                title_prefix=title_prefix,
                best_model=best_model,
                out_base=out_dir / "panel_best_model_stars",
                formats=cfg.plot.formats,
                dpi=cfg.plot.dpi,
                figsize=(float(cfg.plot.figsize_best[0]), float(cfg.plot.figsize_best[1])),
            )

            if cfg.outputs.write_tables:
                _save_tables(out_dir, fold_sel, summary_sel, leaderboard_df, sig_df)

            if cfg.outputs.write_manifest:
                _save_manifest(
                    out_dir=out_dir,
                    cfg=cfg,
                    dataset=dataset,
                    task=task,
                    mapping=mapping,
                    best_model=best_model,
                    source_paths={
                        "summary": str(tables.summary_path),
                        "fold": str(tables.fold_path),
                        "metadata": str(tables.metadata_path) if tables.metadata_path else None,
                    },
                    n_rows=len(fold_sel),
                )

            logger.info(
                "Done %s/%s | best model (AUPR): %s | mapped sets: %s",
                dataset,
                task,
                best_model,
                {k: mapping[k] for k in canonical_order},
            )
