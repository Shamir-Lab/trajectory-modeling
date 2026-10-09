#!/usr/bin/env python3
"""
Regenerate CKD poster-style comparison plots from already-completed CV
results — no cross-validation rerun needed.

Use this when a full ckd_trajectory_analysis.py run finished before the
poster-plot fix landed (or whenever you just want to regenerate/tweak the
poster figures for existing results without paying the cost of rerunning
CV).

Scans --results-dir for every {label}_fold_metrics.csv and regenerates, per
target label found:
  {label}_poster_main.png
  {label}_poster_best_model.png
  {label}_significance_extended.csv

Input:  a results directory produced by ckd_trajectory_analysis.py — needs
        at minimum {label}_fold_metrics.csv per target. Optionally
        descriptive_stats/{label}_outcome_distribution.csv or
        {label}_oof_predictions.parquet, used for the AUPR panel's dashed
        baseline reference line (omitted, with a warning, if neither is
        found — the AUROC panel's 0.5 baseline is unaffected either way).

This script does not vendor the poster-plotting code itself — it imports
save_poster_comparison() from ckd_trajectory_analysis.py (expected to sit
alongside this script, or pointed at via --ckd-trajectory-analysis-path),
so there is exactly one copy of that logic to keep in sync.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Matches ckd_trajectory_analysis.py's DEFAULT_OUTPUT exactly, so this script
# defaults to the same results directory the analysis script writes to.
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "trajectory_representation_chronic"

REQUIRED_FOLD_METRICS_COLS = {"model", "feature_set", "repeat", "fold", "auroc", "aupr"}


def _load_fold_rows(fold_metrics_path: Path) -> list[dict]:
    df = pd.read_csv(fold_metrics_path)
    missing = REQUIRED_FOLD_METRICS_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{fold_metrics_path} is missing required column(s): {sorted(missing)}")
    return df.to_dict("records")


def _find_positive_rate(results_dir: Path, label: str) -> float | None:
    """
    Try descriptive_stats/{label}_outcome_distribution.csv's positive_rate
    row first (the layout save_descriptive_stats() writes), then a flat
    {label}_outcome_distribution.csv in --results-dir in case of a different
    layout, then fall back to {label}_oof_predictions.parquet's y_true mean
    (only saved for top-k feature-set/model pairs, so may be absent or
    incomplete). Returns None (with a warning) if nothing is found.
    """
    candidates = [
        results_dir / "descriptive_stats" / f"{label}_outcome_distribution.csv",
        results_dir / f"{label}_outcome_distribution.csv",
    ]
    for dist_path in candidates:
        if not dist_path.exists():
            continue
        try:
            dist_df = pd.read_csv(dist_path)
            row = dist_df.loc[dist_df["metric"] == "positive_rate", "value"]
            if not row.empty:
                return float(row.iloc[0])
        except Exception as exc:
            logger.warning(f"  Could not read positive_rate from {dist_path}: {exc}")

    oof_path = results_dir / f"{label}_oof_predictions.parquet"
    if oof_path.exists():
        try:
            oof_df = pd.read_parquet(oof_path, columns=["y_true"])
            if not oof_df.empty:
                return float(oof_df["y_true"].mean())
        except Exception as exc:
            logger.warning(f"  Could not compute positive_rate from {oof_path}: {exc}")

    logger.warning(
        f"  No positive_rate source found for {label!r} (looked for "
        f"descriptive_stats/{label}_outcome_distribution.csv and "
        f"{label}_oof_predictions.parquet) — AUPR panel will omit its baseline line."
    )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate CKD poster-style plots from existing fold_metrics.csv "
                    "results, without rerunning CV."
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing {label}_fold_metrics.csv files produced by ckd_trajectory_analysis.py "
             f"(default: {DEFAULT_RESULTS_DIR}, matching ckd_trajectory_analysis.py's own default --output-dir).",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Only regenerate these labels (default: every *_fold_metrics.csv found in --results-dir).",
    )
    parser.add_argument(
        "--ckd-trajectory-analysis-path", type=str, default=None,
        help="Path to ckd_trajectory_analysis.py to import save_poster_comparison() from "
             "(default: same directory as this script).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"--results-dir not found or not a directory: {results_dir}")

    analysis_script_dir = (
        Path(args.ckd_trajectory_analysis_path).resolve().parent
        if args.ckd_trajectory_analysis_path else SCRIPT_DIR
    )
    sys.path.insert(0, str(analysis_script_dir))
    try:
        from ckd_trajectory_analysis import save_poster_comparison
    except ImportError as exc:
        raise SystemExit(
            f"Could not import save_poster_comparison from ckd_trajectory_analysis.py "
            f"(looked in {analysis_script_dir}). Pass --ckd-trajectory-analysis-path if it "
            f"lives elsewhere. Original error: {exc}"
        )

    fold_metrics_files = sorted(results_dir.glob("*_fold_metrics.csv"))
    if not fold_metrics_files:
        raise SystemExit(f"No *_fold_metrics.csv files found in {results_dir}")

    labels_filter = set(args.labels) if args.labels else None
    n_done = 0
    for fm_path in fold_metrics_files:
        label = fm_path.name.removesuffix("_fold_metrics.csv")
        if labels_filter and label not in labels_filter:
            continue

        logger.info(f"=== {label} ===")
        try:
            fold_rows = _load_fold_rows(fm_path)
        except Exception as exc:
            logger.warning(f"  Skipping {label!r}: {exc}")
            continue

        positive_rate = _find_positive_rate(results_dir, label)

        try:
            save_poster_comparison(
                fold_rows=fold_rows,
                positive_rate=positive_rate,
                output_dir=results_dir,
                label=label,
            )
            n_done += 1
        except Exception as exc:
            logger.warning(f"  Poster comparison failed for {label!r}: {exc}")

    logger.info(f"\n✓ Regenerated poster plots for {n_done} / {len(fold_metrics_files)} target(s) in {results_dir}")


if __name__ == "__main__":
    main()
