"""
Template for Trajectory Analysis Scripts (Circulatory Failure, Sepsis, Liver, AKI, Ventilator).

Provides a unified interface for:
1. Loading prediction datasets + biomarker time series
2. Computing summary statistics for biomarkers
3. Building feature sets (trajectory + summary stats)
4. Running cross-validated model comparisons with joblib parallelization
5. Saving results to CSV

Subclasses should override:
- DATASET (str): 'mimic', 'hirid', 'eicu'
- TASK (str): 'circulatory_failure', 'sepsis', 'liver', 'aki', 'ventilator'
- BIOMARKERS (dict): biomarker name -> {'file': '...csv', 'value_col': '...', ...}
- TARGET_COL (str): outcome column name
- OUTCOME_COLS (list): columns in outcome file to merge
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed as jdelayed

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from .analysis_utils import (
    pick_id_col,
    pick_time_cols,
    load_named_table,
    safe_left_merge,
    biomarker_summary_stats,
    encode_gender,
    organize_feature_sets,
    get_dataset_config,
    get_default_models,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 920
np.random.seed(SEED)


# ============================================================================
# Configuration (override in subclasses)
# ============================================================================


class AnalysisConfig:
    """Base configuration class for trajectory analysis tasks."""
    
    DATASET: str = None  # 'mimic', 'hirid', 'eicu'
    TASK: str = None  # 'circulatory_failure', 'sepsis', 'liver', 'aki', 'ventilator'
    
    # Biomarkers configuration: name -> {'file': '...csv', 'value_col': '...', ...}
    BIOMARKERS: dict = {}
    
    # Outcome configuration
    TARGET_COL: str = None  # e.g., 'target_septic_shock'
    OUTCOME_FILE: str = None  # e.g., 'septic_shock_outcomes.csv'
    OUTCOME_MERGE_COLS: list[str] = []  # e.g., ['hadm_id', 'time_day', 'target_septic_shock']
    
    # Cross-validation configuration
    CV_N_REPEATS: int = 10
    CV_N_SPLITS: int = 5
    LOOKAHEAD_HOURS: int = 12
    # Unit-aware aliases (preferred). Values are interpreted in the same unit
    # as the time column in the timeseries (days for time_day, hours for time_hour).
    LOOKBACK_WINDOW: float = 12
    LOOKBACK_UNIT: str = "hours"
    # Feature-set construction mode:
    # - auto: use single-vs-multi when multiple biomarker groups exist, else pooled sets
    # - single-multi: force single-vs-multi marker families when possible
    # - pooled: use pooled trajectory/summary/static/dynamic families
    MARKER_MODE: str = "auto"
    PARALLEL_AXIS: str = "repeat"
    SAVE_OOF: bool = True
    OOF_SCOPE: str = "top-k"  # one of: all, top-k, representative-k
    OOF_K: int = 10
    PLOT_K: int = 10
    TRAJ_FFILL_LIMIT: int = 2
    SAVE_NOTEBOOK_STYLE_PLOTS: bool = True
    # Runtime bootstrap trajectory generation (used when bootstrap files are absent)
    COMPUTE_BOOTSTRAP_TRAJ: bool = True
    BOOTSTRAP_N: int = 200
    BOOTSTRAP_N_JOBS: int = 4
    RECOMPUTE_BOOTSTRAP_TRAJ: bool = False
    # If True, treat legacy "*_with_probs" and standalone "*trajectory_probs*"
    # files as bootstrap trajectory sources and suffix trajectory columns with `_boot`.
    WITH_PROBS_IS_BOOTSTRAP: bool = False
    
    def __post_init__(self):
        if not self.DATASET or not self.TASK:
            raise NotImplementedError("Subclass must set DATASET and TASK")


# ============================================================================
# Main Analysis Class
# ============================================================================


class TrajectoryAnalysis:
    """Unified interface for trajectory-based prediction analysis."""
    
    def __init__(self, config: AnalysisConfig, base_dir: Path, output_dir: Path):
        """
        Initialize analysis.
        
        Args:
            config: AnalysisConfig instance
            base_dir: Base directory for input data (e.g., /home/gaga/data/physionet/mimic/circulatory_failure)
            output_dir: Directory for results (e.g., /home/gaga/trajectory-modeling/results/mimic)
        """
        self.config = config
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect from dataset
        self.dataset_config = get_dataset_config(config.DATASET)
        self.id_col = self.dataset_config["id_col"]
        self._last_fold_rows: list[dict] = []
        self._last_oof_rows: list[dict] = []
        self._last_feature_configs: dict[str, list[str]] = {}
        self._last_run_meta: dict = {}
        
        logger.info(f"✓ Initialized {config.DATASET.upper()} {config.TASK.upper()} analysis")
        logger.info(f"  Base dir: {self.base_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load prediction dataset and merge with outcomes.
        
        Returns:
            DataFrame with prediction data and outcome column
        """
        logger.info("\n1. Loading data...")
        logger.info(f"Base directory: {self.base_dir}")
        
        # Load prediction dataset
        pred_stems = [
            f"{self.config.TASK}_prediction_dataset_with_bootstrap_probs",
            f"{self.config.TASK}_prediction_dataset",
        ]
        
        dataset = None
        loaded_stem = None
        for stem in pred_stems:
            try:
                dataset = load_named_table(self.base_dir, stem, verbose=True)
                loaded_stem = stem
                break
            except FileNotFoundError:
                continue
        
        if dataset is None:
            raise FileNotFoundError(f"Could not load prediction dataset. Tried: {pred_stems}")

        force_with_probs_bootstrap = bool(getattr(self.config, "WITH_PROBS_IS_BOOTSTRAP", False))

        def _is_bootstrap_source(stem: str) -> bool:
            stem_l = stem.lower()
            return ("bootstrap" in stem_l) or (
                force_with_probs_bootstrap and stem_l.endswith("_with_probs")
            )

        # Augment with additional trajectory representations (Bayesian / Bootstrap)
        # when available as separate files.
        id_col = pick_id_col(dataset)
        _, windowing_col = pick_time_cols(dataset)
        key_cols = [id_col, windowing_col]

        def _traj_cols(df: pd.DataFrame) -> list[str]:
            return [
                c for c in df.columns
                if c not in key_cols
                and (
                    c.startswith("prob_")
                    or c.startswith("trajtype_")
                    or c.endswith("_stable")
                    or c.endswith("_gradual")
                    or c.endswith("_rapid")
                    or c.endswith("_increase")
                    or c.endswith("_decline")
                    or c.endswith("_prob")
                    or "_boot" in c
                )
            ]

        def _suffix_bootstrap_cols(df: pd.DataFrame) -> pd.DataFrame:
            rename_map = {}
            for col in _traj_cols(df):
                if col.endswith("_boot"):
                    continue
                rename_map[col] = f"{col}_boot"
            if not rename_map:
                return df
            return df.rename(columns=rename_map)

        if loaded_stem and _is_bootstrap_source(loaded_stem):
            dataset = _suffix_bootstrap_cols(dataset)

        extra_stems = [
            f"{self.config.TASK}_prediction_dataset_with_probs",
            f"{self.config.TASK}_prediction_dataset_with_bootstrap_probs",
        ]

        for stem in extra_stems:
            if stem == loaded_stem:
                continue
            try:
                extra_df = load_named_table(self.base_dir, stem, verbose=False)
            except FileNotFoundError:
                continue

            missing_keys = [k for k in key_cols if k not in extra_df.columns]
            if missing_keys:
                continue

            if _is_bootstrap_source(stem):
                extra_df = _suffix_bootstrap_cols(extra_df)

            traj_cols = _traj_cols(extra_df)
            traj_cols = [c for c in traj_cols if c not in dataset.columns]
            if not traj_cols:
                continue

            logger.info(f"  Merging additional trajectories from: {stem} ({len(traj_cols)} cols)")
            dataset = safe_left_merge(
                dataset,
                extra_df[key_cols + traj_cols].copy(),
                key_cols=key_cols,
                add_name=stem,
            )

        # Also merge standalone trajectory probability files if present
        for p in sorted(self.base_dir.glob("*trajectory_probs*.csv")):
            try:
                tdf = pd.read_csv(p)
            except Exception:
                continue

            missing_keys = [k for k in key_cols if k not in tdf.columns]
            if missing_keys:
                continue

            if "bootstrap" in p.name.lower() or (
                force_with_probs_bootstrap and "trajectory_probs" in p.name.lower()
            ):
                tdf = _suffix_bootstrap_cols(tdf)

            traj_cols = _traj_cols(tdf)
            traj_cols = [c for c in traj_cols if c not in dataset.columns]
            if not traj_cols:
                continue

            logger.info(f"  Merging trajectory file: {p.name} ({len(traj_cols)} cols)")
            dataset = safe_left_merge(
                dataset,
                tdf[key_cols + traj_cols].copy(),
                key_cols=key_cols,
                add_name=p.name,
            )
        
        logger.info(f"  Samples: {len(dataset):,}")
        
        # Merge outcomes if provided
        if self.config.OUTCOME_FILE and self.config.OUTCOME_MERGE_COLS:
            outcome_path = self.base_dir / self.config.OUTCOME_FILE
            if outcome_path.exists():
                logger.info(f"  Merging outcomes from: {outcome_path}")
                outcomes = pd.read_csv(outcome_path)
                
                # Identify merge keys
                merge_keys = [c for c in self.config.OUTCOME_MERGE_COLS if c != self.config.TARGET_COL]
                
                dataset = dataset.merge(
                    outcomes[self.config.OUTCOME_MERGE_COLS],
                    on=merge_keys,
                    how="left",
                )
                
                # Filter to rows with outcome
                dataset = dataset[dataset[self.config.TARGET_COL].notna()].copy()
                dataset[self.config.TARGET_COL] = dataset[self.config.TARGET_COL].astype(int)
                
                logger.info(f"  After filtering: {len(dataset):,}")
                logger.info(f"  Outcome rate: {dataset[self.config.TARGET_COL].mean():.1%}")
        
        return dataset

    def _compute_or_load_bootstrap_trajectory(
        self,
        ts_df: pd.DataFrame,
        biomarker: str,
        bio_config: dict,
        lookback_value: float,
    ) -> Optional[pd.DataFrame]:
        """Load cached bootstrap trajectory probs or compute them from timeseries."""
        if not getattr(self.config, "COMPUTE_BOOTSTRAP_TRAJ", True):
            return None

        value_col = str(bio_config.get("value_col", biomarker))
        ts_file = str(bio_config.get("file", f"{biomarker}_timeseries.csv"))
        ts_stem = Path(ts_file).stem
        if ts_stem.endswith("_timeseries"):
            ts_stem = ts_stem[:-len("_timeseries")]

        boot_cfg = (bio_config.get("bootstrap") or {}).copy()
        output_name = boot_cfg.get("output_file", f"{ts_stem}_trajectory_probs_bootstrap.csv")
        output_path = self.base_dir / output_name

        id_col_ts = pick_id_col(ts_df)
        time_col_ts, windowing_col_ts = pick_time_cols(ts_df)

        marker_token = value_col.lower().replace(" ", "_")
        dst_cols = [
            f"{marker_token}_stable_boot",
            f"{marker_token}_gradual_boot",
            f"{marker_token}_rapid_boot",
        ]

        force_recompute = bool(getattr(self.config, "RECOMPUTE_BOOTSTRAP_TRAJ", False))

        if output_path.exists() and not force_recompute:
            try:
                boot_df = pd.read_csv(output_path)
                if id_col_ts in boot_df.columns and windowing_col_ts in boot_df.columns and all(c in boot_df.columns for c in dst_cols):
                    return boot_df[[id_col_ts, windowing_col_ts] + dst_cols].copy()
            except Exception:
                pass
        elif output_path.exists() and force_recompute:
            logger.info(f"    Recomputing bootstrap trajectories and overwriting: {output_path.name}")

        try:
            from traj_features.backends.bootstrap import BootstrapTraj, BootstrapConfig
            from traj_features.backends.bayes.classify import flags_from_traj, pos_flags_from_traj
        except Exception as e:
            logger.warning(f"    Bootstrap backend unavailable for {biomarker}: {e}")
            return None

        class_mode = str(boot_cfg.get("class_func", "auto")).lower()
        if class_mode in {"pos", "positive", "increase", "increasing"}:
            class_func = pos_flags_from_traj
            default_traj_types = ("stable", "gradual_increase", "rapid_increase")
        elif class_mode in {"neg", "negative", "decline", "decrease", "decreasing"}:
            class_func = flags_from_traj
            default_traj_types = ("stable", "gradual_decline", "rapid_decline")
        else:
            # Heuristic: platelets commonly modeled as declining trajectory for worsening.
            if "platelet" in marker_token:
                class_func = flags_from_traj
                default_traj_types = ("stable", "gradual_decline", "rapid_decline")
            else:
                class_func = pos_flags_from_traj
                default_traj_types = ("stable", "gradual_increase", "rapid_increase")

        traj_types = tuple(boot_cfg.get("traj_types", default_traj_types))
        label_map = boot_cfg.get(
            "label_map",
            {
                "nonprogression": traj_types[0],
                "linear": traj_types[1],
                "nonlinear": traj_types[2],
            },
        )

        traj_input = ts_df[[id_col_ts, time_col_ts, windowing_col_ts, value_col]].copy()
        traj_input = traj_input.rename(
            columns={
                id_col_ts: "patientid",
                time_col_ts: "time",
                windowing_col_ts: "time_window",
                value_col: "lab_value",
            }
        )
        traj_input = traj_input.dropna(subset=["lab_value", "time", "time_window"]).sort_values(["patientid", "time"])

        if traj_input.empty:
            return None

        logger.info(f"    Computing bootstrap trajectories ({biomarker})...")
        model = BootstrapTraj(
            BootstrapConfig(
                window_years=float(boot_cfg.get("window", lookback_value)),
                n_bootstrap=int(boot_cfg.get("n_bootstrap", getattr(self.config, "BOOTSTRAP_N", 1000))),
                flat_thr=float(boot_cfg.get("flat_thr", 0.1)),
                decline_thr=float(boot_cfg.get("decline_thr", 0.3)),
                nonlinear_gap=float(boot_cfg.get("nonlinear_gap", 0.5)),
                class_func=class_func,
                traj_types=traj_types,
                label_map=label_map,
                pids="patientid",
                values="lab_value",
                time_col="time",
                windowing_col="time_window",
                n_jobs=int(boot_cfg.get("n_jobs", getattr(self.config, "BOOTSTRAP_N_JOBS", 4))),
                progressbar=False,
            )
        )

        boot_df = model.embed(traj_input)
        if boot_df.empty:
            return None

        rename_map = {
            f"trajtype_{traj_types[0]}_prob": dst_cols[0],
            f"trajtype_{traj_types[1]}_prob": dst_cols[1],
            f"trajtype_{traj_types[2]}_prob": dst_cols[2],
        }
        for src in rename_map:
            if src not in boot_df.columns:
                boot_df[src] = np.nan

        boot_df = boot_df.rename(columns={"patientid": id_col_ts, "time_window": windowing_col_ts, **rename_map})
        boot_df = boot_df[[id_col_ts, windowing_col_ts] + dst_cols].copy()
        boot_df = boot_df.drop_duplicates(subset=[id_col_ts, windowing_col_ts], keep="last")

        try:
            boot_df.to_csv(output_path, index=False)
            logger.info(f"    Cached bootstrap trajectories: {output_path.name}")
        except Exception as e:
            logger.warning(f"    Failed to cache bootstrap trajectories for {biomarker}: {e}")

        return boot_df
    
    def prepare_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare trajectory and summary statistic features.
        
        Args:
            dataset: Prediction dataset (from load_data)
        
        Returns:
            Dataset with added features
        """
        logger.info("\n2. Preparing features...")
        
        # Detect columns
        time_col, windowing_col = pick_time_cols(dataset)

        # Backward-compatible lookback resolution
        lookback_value = getattr(self.config, "LOOKBACK_WINDOW", getattr(self.config, "LOOKAHEAD_HOURS", 12))
        lookback_unit = getattr(self.config, "LOOKBACK_UNIT", "hours")
        
        # Add summary statistics for each biomarker
        for biomarker, bio_config in self.config.BIOMARKERS.items():
            logger.info(f"\n  {biomarker.upper()}:")
            
            ts_path = self.base_dir / bio_config.get("file", f"{biomarker}_timeseries.csv")
            if not ts_path.exists():
                logger.warning(f"    Missing: {ts_path}")
                continue
            
            logger.info(f"    Loading: {ts_path}")
            ts_df = pd.read_csv(ts_path)
            
            value_col = bio_config.get("value_col", biomarker)
            
            # Compute summary stats
            logger.info(f"    Computing summary stats (lookback={lookback_value}{lookback_unit[0]})...")
            stats_df = biomarker_summary_stats(
                ts_df,
                value_col=value_col,
                # Historical parameter name; value is interpreted in native time units.
                lookback_hours=lookback_value,
                n_jobs=4,
            )
            
            # Merge into dataset
            logger.info(f"    Merging {len(stats_df):,} stat rows...")
            dataset = safe_left_merge(
                dataset,
                stats_df,
                key_cols=[self.id_col, windowing_col],
                add_name=f"{biomarker} summary stats",
            )
            logger.info(f"    ✓ Merged: {len(stats_df):,} rows")

            # Compute/load bootstrap trajectory probabilities (notebook-style)
            boot_df = self._compute_or_load_bootstrap_trajectory(
                ts_df=ts_df,
                biomarker=biomarker,
                bio_config=bio_config,
                lookback_value=float(lookback_value),
            )
            if boot_df is not None and not boot_df.empty:
                logger.info(f"    Merging bootstrap trajectories ({len(boot_df):,} rows)...")
                dataset = safe_left_merge(
                    dataset,
                    boot_df,
                    key_cols=[self.id_col, windowing_col],
                    add_name=f"{biomarker} bootstrap trajectories",
                )
                logger.info("    ✓ Added bootstrap trajectory probabilities")
        
        return dataset
    
    def run_cv(
        self,
        dataset: pd.DataFrame,
        n_repeats: Optional[int] = None,
        n_splits: Optional[int] = None,
        n_jobs: int = 1,
        backend: str = "threading",
        parallel_axis: str = "repeat",
    ) -> dict:
        """
        Run cross-validated model comparisons.
        
        Args:
            dataset: Feature dataframe (from prepare_features)
            n_repeats: Number of CV repeats (default: config.CV_N_REPEATS)
            n_splits: Number of CV folds (default: config.CV_N_SPLITS)
            n_jobs: Number of parallel jobs for repeats
            backend: joblib backend ('threading' or 'processes')
            parallel_axis: parallelization axis ('repeat' or 'feature-set')
        
        Returns:
            Dictionary: {model_name: {feature_set_name: fold_metrics}}
        """
        n_repeats = n_repeats or self.config.CV_N_REPEATS
        n_splits = n_splits or self.config.CV_N_SPLITS
        
        if parallel_axis not in {"repeat", "feature-set"}:
            parallel_axis = "repeat"

        logger.info(
            f"\n3. Running cross-validated model comparison ({n_repeats}x{n_splits}, "
            f"parallel_axis={parallel_axis}, n_jobs={n_jobs})..."
        )
        
        # Organize features
        windowing_col = self._detect_windowing_col(dataset)
        dataset = encode_gender(dataset.copy())
        feature_sets = organize_feature_sets(dataset, self.id_col, windowing_col)
        for group in ("static", "dynamic", "summary", "trajectory"):
            logger.info(f"  {group} features ({len(feature_sets[group])}): {feature_sets[group]}")

        # Notebook-style trajectory preprocessing: temporal carry-forward by patient.
        dataset_model = dataset.copy()
        traj_ffill_limit = int(getattr(self.config, "TRAJ_FFILL_LIMIT", 2))
        if traj_ffill_limit > 0 and len(feature_sets["trajectory"]) > 0:
            traj_cols_all = [c for c in feature_sets["trajectory"] if c in dataset_model.columns]
            if traj_cols_all:
                dataset_model = dataset_model.sort_values([self.id_col, windowing_col]).reset_index(drop=True)
                dataset_model.loc[:, traj_cols_all] = (
                    dataset_model.groupby(self.id_col)[traj_cols_all].ffill(limit=traj_ffill_limit)
                )
        
        # Define notebook-aligned feature configurations
        traj_all = list(dict.fromkeys(feature_sets["trajectory"]))
        summary_all = list(dict.fromkeys(feature_sets["summary"]))
        static_all = list(dict.fromkeys(feature_sets["static"]))
        dynamic_all = list(dict.fromkeys(feature_sets["dynamic"]))
        static_dynamic = list(dict.fromkeys(static_all + dynamic_all))

        traj_boot = [c for c in traj_all if c.endswith("_boot") or "_boot" in c]
        traj_bayes = [c for c in traj_all if c not in traj_boot]

        # Choose primary trajectory source for non-bootstrap feature families.
        # If Bayesian columns are absent, use bootstrap trajectories as primary.
        traj_primary = traj_bayes if len(traj_bayes) > 0 else traj_boot
        if len(traj_bayes) == 0 and len(traj_boot) > 0:
            logger.warning(
                "No Bayesian trajectory columns were found; plain trajectory feature sets "
                "are using bootstrap columns as the fallback source."
            )

        # Marker-specific subsets (used by sepsis notebook-style comparisons)
        biomarkers_cfg = getattr(self.config, "BIOMARKERS", {}) or {}
        biomarker_items = list(biomarkers_cfg.items())
        primary_key = biomarker_items[0][0] if biomarker_items else None
        primary_value_col = biomarker_items[0][1].get("value_col", primary_key) if biomarker_items else None

        def _marker_tokens(name_key: Optional[str], value_col: Optional[str]) -> list[str]:
            toks = []
            for t in (name_key, value_col):
                if not t:
                    continue
                toks.append(str(t).lower())
                if str(t).lower().endswith("s"):
                    toks.append(str(t).lower()[:-1])
            return list(dict.fromkeys([t for t in toks if t]))

        primary_tokens = _marker_tokens(primary_key, primary_value_col)

        def _subset_for_tokens(cols: list[str], tokens: list[str]) -> list[str]:
            out = []
            for c in cols:
                lc = c.lower()
                if any(lc.startswith(f"{t}_") or f"_{t}_" in lc or lc.startswith(t) for t in tokens):
                    out.append(c)
            return list(dict.fromkeys(out))

        single_traj_primary = _subset_for_tokens(traj_primary, primary_tokens)
        single_summary = _subset_for_tokens(summary_all, primary_tokens)
        single_traj_boot = _subset_for_tokens(traj_boot, primary_tokens)

        marker_mode = getattr(self.config, "MARKER_MODE", "auto")
        if marker_mode not in {"auto", "single-multi", "pooled"}:
            marker_mode = "auto"

        # Sepsis-style single-vs-multi marker framing should be available whenever
        # multiple biomarkers are configured, even if trajectory columns are absent.
        task_is_sepsis = str(getattr(self.config, "TASK", "")).lower() == "sepsis"
        can_single_multi = (
            len(biomarker_items) >= 2
            and (
                (len(single_traj_primary) > 0 and len(traj_primary) > len(single_traj_primary))
                or (len(single_summary) > 0 and len(summary_all) > len(single_summary))
                or task_is_sepsis
            )
        )
        use_single_multi = (
            can_single_multi if marker_mode == "auto" else (marker_mode == "single-multi")
        )

        if use_single_multi and (len(single_traj_primary) > 0 or len(single_summary) > 0):
            marker_label = (primary_value_col or primary_key or "Single Marker").capitalize()
            feature_configs = {
                f"{marker_label} Trajectory Only": single_traj_primary,
                "Multi-Marker Trajectories": traj_primary,
                f"{marker_label} Summary Stats Only": single_summary,
                "Multi-Marker Summary Stats": summary_all,
                f"{marker_label} Trajectory + Summary Stats": single_traj_primary + single_summary,
                "Multi-Marker Trajectories + Summary Stats": traj_primary + summary_all,
                f"Static + {marker_label} Trajectory": static_all + single_traj_primary,
                "Static + Multi-Marker Trajectories": static_all + traj_primary,
                f"Static + {marker_label} Summary Stats": static_all + single_summary,
                "Static + Multi-Marker Summary Stats": static_all + summary_all,
            }
            if len(static_dynamic) > 0:
                feature_configs.update(
                    {
                        "Static + Dynamic": static_dynamic,
                        f"Static + Dynamic + {marker_label} Trajectory": static_dynamic + single_traj_primary,
                        "Static + Dynamic + Multi-Marker Trajectories": static_dynamic + traj_primary,
                        f"Static + Dynamic + {marker_label} Summary Stats": static_dynamic + single_summary,
                        "Static + Dynamic + Multi-Marker Summary Stats": static_dynamic + summary_all,
                        f"Static + Dynamic + {marker_label} Trajectory + Summary Stats": static_dynamic + single_traj_primary + single_summary,
                        "Static + Dynamic + Multi-Marker Trajectories + Summary Stats": static_dynamic + traj_primary + summary_all,
                    }
                )

            if len(traj_boot) > 0:
                feature_configs.update(
                    {
                        f"{marker_label} Trajectory Only (Bootstrap)": single_traj_boot,
                        "Multi-Marker Trajectories (Bootstrap)": traj_boot,
                        f"{marker_label} Trajectory + Summary Stats (Bootstrap)": single_traj_boot + single_summary,
                        "Multi-Marker Trajectories + Summary Stats (Bootstrap)": traj_boot + summary_all,
                        f"Static + {marker_label} Trajectory (Bootstrap)": static_all + single_traj_boot,
                        "Static + Multi-Marker Trajectories (Bootstrap)": static_all + traj_boot,
                    }
                )
                if len(static_dynamic) > 0:
                    feature_configs.update(
                        {
                            f"Static + Dynamic + {marker_label} Trajectory (Bootstrap)": static_dynamic + single_traj_boot,
                            "Static + Dynamic + Multi-Marker Trajectories (Bootstrap)": static_dynamic + traj_boot,
                            f"Static + Dynamic + {marker_label} Trajectory + Summary Stats (Bootstrap)": static_dynamic + single_traj_boot + single_summary,
                            "Static + Dynamic + Multi-Marker Trajectories + Summary Stats (Bootstrap)": static_dynamic + traj_boot + summary_all,
                        }
                    )
        else:
            feature_configs = {
                "Trajectory Only": traj_primary,
                "Summary Stats Only": summary_all,
                "Trajectory + Summary Stats": traj_primary + summary_all,
                "Trajectory + Static": traj_primary + static_all,
                "Summary Stats + Static": summary_all + static_all,
                "Trajectory + Summary Stats + Static": traj_primary + summary_all + static_all,
            }

            if len(static_dynamic) > 0:
                feature_configs.update(
                    {
                        "Static + Dynamic": static_dynamic,
                        "Trajectory + Static + Dynamic": traj_primary + static_dynamic,
                        "Summary Stats + Static + Dynamic": summary_all + static_dynamic,
                        "Trajectory + Summary Stats + Static + Dynamic": traj_primary + summary_all + static_dynamic,
                    }
                )

            if len(traj_boot) > 0:
                feature_configs.update(
                    {
                        "Trajectory Only (Bootstrap)": traj_boot,
                        "Trajectory + Summary Stats (Bootstrap)": traj_boot + summary_all,
                        "Trajectory + Static (Bootstrap)": traj_boot + static_all,
                        "Trajectory + Summary Stats + Static (Bootstrap)": traj_boot + summary_all + static_all,
                    }
                )
                if len(static_dynamic) > 0:
                    feature_configs.update(
                        {
                            "Trajectory + Static + Dynamic (Bootstrap)": traj_boot + static_dynamic,
                            "Trajectory + Summary Stats + Static + Dynamic (Bootstrap)": traj_boot + summary_all + static_dynamic,
                        }
                    )
        
        # Filter to available features
        feature_configs = {
            name: list(dict.fromkeys(cols)) for name, cols in feature_configs.items()
            if len(cols) > 0
        }
        self._last_feature_configs = feature_configs
        
        logger.info(f"  Feature sets: {list(feature_configs.keys())}")
        
        # Get models
        models = get_default_models(include_boosting=True)
        non_boosting_models = {"LogisticRegression", "RandomForest"}
        
        # Prepare data for CV
        X_all = dataset_model[feature_sets["all_numeric"]].copy()
        y_all = dataset_model[self.config.TARGET_COL].values
        groups = dataset_model[self.id_col].values
        id_all = dataset_model[self.id_col].values
        win_all = dataset_model[windowing_col].values
        
        # Remove rows with missing outcome or outcome-like columns
        y_series = pd.Series(y_all)
        valid = (~y_series.isna().to_numpy()) & np.isfinite(y_all)
        X_all = X_all[valid]
        y_all = y_all[valid]
        groups = groups[valid]
        id_all = id_all[valid]
        win_all = win_all[valid]
        
        logger.info(f"  Total samples: {len(X_all):,}")
        logger.info(f"  Positive rate: {y_all.mean():.1%}")
        
        # Core evaluation helper
        def _eval_one_split(
            model_name: str,
            model,
            selected_cols: list[str],
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: np.ndarray,
            y_test: np.ndarray,
        ) -> tuple[float, float, np.ndarray]:
            X_tr_df = X_train[selected_cols].copy()
            X_te_df = X_test[selected_cols].copy()

            # Notebook-consistent preprocessing:
            # - Non-boosting models: fill trajectory/summary NaNs with 0, then median impute
            # - Boosting models: no explicit imputation step
            if model_name in non_boosting_models:
                traj_cols_fs = [c for c in selected_cols if c in feature_sets["trajectory"]]
                summary_cols_fs = [c for c in selected_cols if c in feature_sets["summary"]]
                if traj_cols_fs:
                    X_tr_df.loc[:, traj_cols_fs] = X_tr_df[traj_cols_fs].fillna(0)
                    X_te_df.loc[:, traj_cols_fs] = X_te_df[traj_cols_fs].fillna(0)
                if summary_cols_fs:
                    X_tr_df.loc[:, summary_cols_fs] = X_tr_df[summary_cols_fs].fillna(0)
                    X_te_df.loc[:, summary_cols_fs] = X_te_df[summary_cols_fs].fillna(0)

                imputer = SimpleImputer(strategy="median")
                X_tr = imputer.fit_transform(X_tr_df)
                X_te = imputer.transform(X_te_df)
            else:
                X_tr = X_tr_df.to_numpy()
                X_te = X_te_df.to_numpy()

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            est = clone(model)
            if model_name == "XGBoost":
                est.set_params(scale_pos_weight=(len(y_train) - y_train.sum()) / max(1, y_train.sum()))
                
            if n_jobs != 1 and hasattr(est, "get_params"):
                params = est.get_params(deep=False)
                if "n_jobs" in params:
                    est.set_params(n_jobs=1)
            est.fit(X_tr, y_train)

            if hasattr(est, "predict_proba"):
                y_pred = est.predict_proba(X_te)[:, 1]
            elif hasattr(est, "decision_function"):
                y_pred = est.decision_function(X_te)
            else:
                y_pred = est.predict(X_te)

            auroc = roc_auc_score(y_test, y_pred)
            aupr = average_precision_score(y_test, y_pred)
            return auroc, aupr, y_pred

        # Repeat-level worker
        def _run_one_repeat(repeat_idx: int) -> dict:
            """Run one CV repeat across all feature sets and models."""
            results = {}
            oof_rows = []

            rng = np.random.RandomState(SEED + repeat_idx)
            shuffle_idx = rng.permutation(len(X_all))
            X_rep = X_all.iloc[shuffle_idx].reset_index(drop=True)
            y_rep = y_all[shuffle_idx]
            groups_rep = groups[shuffle_idx]
            id_rep = id_all[shuffle_idx]
            win_rep = win_all[shuffle_idx]

            # GroupKFold ignores row order, so each repeat shuffles the groups itself.
            gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED + repeat_idx)

            for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_rep, y_rep, groups_rep)):
                X_train, X_test = X_rep.iloc[train_idx], X_rep.iloc[test_idx]
                y_train, y_test = y_rep[train_idx], y_rep[test_idx]
                
                for model_name, model in models.items():
                    if model_name not in results:
                        results[model_name] = {fs: [] for fs in feature_configs}
                    
                    for feature_set_name, feature_cols in feature_configs.items():
                        selected_cols = [c for c in feature_cols if c in X_train.columns]
                        if len(selected_cols) == 0:
                            continue
                        
                        # Train
                        try:
                            auroc, aupr, y_pred = _eval_one_split(
                                model_name=model_name,
                                model=model,
                                selected_cols=selected_cols,
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                y_test=y_test,
                            )

                            if getattr(self.config, "SAVE_OOF", True):
                                for pid, wv, yt, yp in zip(id_rep[test_idx], win_rep[test_idx], y_test, y_pred):
                                    oof_rows.append(
                                        {
                                            "id": pid,
                                            "window": wv,
                                            "model": model_name,
                                            "feature_set": feature_set_name,
                                            "repeat": repeat_idx,
                                            "fold": fold_idx,
                                            "y_true": float(yt),
                                            "y_pred": float(yp),
                                        }
                                    )
                            
                            results[model_name][feature_set_name].append({
                                "fold": fold_idx,
                                "repeat": repeat_idx,
                                "auroc": auroc,
                                "aupr": aupr,
                            })
                        except Exception as e:
                            logger.warning(f"    Error in {model_name}/{feature_set_name}/fold {fold_idx}: {e}")
            
            return {"metrics": results, "oof": oof_rows}

        # Feature-set-level worker
        def _run_one_feature_set(item: tuple[str, list[str]]) -> dict:
            feature_set_name, feature_cols = item
            fs_results = {mname: {feature_set_name: []} for mname in models}
            oof_rows = []

            for repeat_idx in range(n_repeats):
                rng = np.random.RandomState(SEED + repeat_idx)
                shuffle_idx = rng.permutation(len(X_all))
                X_rep = X_all.iloc[shuffle_idx].reset_index(drop=True)
                y_rep = y_all[shuffle_idx]
                groups_rep = groups[shuffle_idx]
                id_rep = id_all[shuffle_idx]
                win_rep = win_all[shuffle_idx]

                gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED + repeat_idx)
                for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_rep, y_rep, groups_rep)):
                    X_train, X_test = X_rep.iloc[train_idx], X_rep.iloc[test_idx]
                    y_train, y_test = y_rep[train_idx], y_rep[test_idx]

                    selected_cols = [c for c in feature_cols if c in X_train.columns]
                    if len(selected_cols) == 0:
                        continue

                    for model_name, model in models.items():
                        try:
                            auroc, aupr, y_pred = _eval_one_split(
                                model_name=model_name,
                                model=model,
                                selected_cols=selected_cols,
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                y_test=y_test,
                            )
                            if getattr(self.config, "SAVE_OOF", True):
                                for pid, wv, yt, yp in zip(id_rep[test_idx], win_rep[test_idx], y_test, y_pred):
                                    oof_rows.append(
                                        {
                                            "id": pid,
                                            "window": wv,
                                            "model": model_name,
                                            "feature_set": feature_set_name,
                                            "repeat": repeat_idx,
                                            "fold": fold_idx,
                                            "y_true": float(yt),
                                            "y_pred": float(yp),
                                        }
                                    )
                            fs_results[model_name][feature_set_name].append(
                                {
                                    "fold": fold_idx,
                                    "repeat": repeat_idx,
                                    "auroc": auroc,
                                    "aupr": aupr,
                                }
                            )
                        except Exception as e:
                            logger.warning(
                                f"    Error in {model_name}/{feature_set_name}/fold {fold_idx}: {e}"
                            )

            return {"metrics": fs_results, "oof": oof_rows}
        
        if parallel_axis == "feature-set" and n_jobs > 1:
            # Use threading backend to avoid loky hanging issues
            fs_results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
                jdelayed(_run_one_feature_set)(item) for item in feature_configs.items()
            )
            repeat_results = []
            for fs_dict in fs_results:
                repeat_results.append(fs_dict)
        else:
            # Run repeats in parallel (use threading to avoid loky hanging)
            repeat_results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
                jdelayed(_run_one_repeat)(i) for i in range(n_repeats)
            )
        
        # Aggregate results
        aggregated = {}
        fold_rows: list[dict] = []
        oof_rows_all: list[dict] = []
        # Collect OOF rows once per worker result (not once per model x feature set)
        for repeat_dict in repeat_results:
            if isinstance(repeat_dict, dict) and "oof" in repeat_dict:
                oof_rows_all.extend(repeat_dict.get("oof", []))
        for model_name in models:
            aggregated[model_name] = {}
            for fs_name in feature_configs:
                all_metrics = []
                for repeat_dict in repeat_results:
                    metrics_root = repeat_dict.get("metrics", repeat_dict)
                    if model_name in metrics_root and fs_name in metrics_root[model_name]:
                        all_metrics.extend(metrics_root[model_name][fs_name])
                
                if all_metrics:
                    metrics_df = pd.DataFrame(all_metrics)
                    for m in all_metrics:
                        fold_rows.append(
                            {
                                "model": model_name,
                                "feature_set": fs_name,
                                "fold": int(m.get("fold", -1)),
                                "repeat": int(m.get("repeat", -1)),
                                "auroc": float(m.get("auroc", np.nan)),
                                "aupr": float(m.get("aupr", np.nan)),
                            }
                        )
                    aggregated[model_name][fs_name] = {
                        "auroc_mean": metrics_df["auroc"].mean(),
                        "auroc_std": metrics_df["auroc"].std(),
                        "aupr_mean": metrics_df["aupr"].mean(),
                        "aupr_std": metrics_df["aupr"].std(),
                        "n_folds": len(metrics_df),
                    }

        self._last_fold_rows = fold_rows
        self._last_oof_rows = oof_rows_all
        self._last_run_meta = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": self.config.DATASET,
            "task": self.config.TASK,
            "base_dir": str(self.base_dir),
            "output_dir": str(self.output_dir),
            "cv_repeats": int(n_repeats),
            "cv_splits": int(n_splits),
            "train_n_jobs": int(n_jobs),
            "train_backend": backend,
            "parallel_axis": parallel_axis,
            "lookback_window": float(getattr(self.config, "LOOKBACK_WINDOW", getattr(self.config, "LOOKAHEAD_HOURS", 12))),
            "lookback_unit": str(getattr(self.config, "LOOKBACK_UNIT", "hours")),
            "marker_mode": str(getattr(self.config, "MARKER_MODE", "auto")),
            "with_probs_is_bootstrap": bool(getattr(self.config, "WITH_PROBS_IS_BOOTSTRAP", False)),
            "save_oof": bool(getattr(self.config, "SAVE_OOF", True)),
            "oof_scope": str(getattr(self.config, "OOF_SCOPE", "top-k")),
            "oof_k": int(getattr(self.config, "OOF_K", 10)),
            "plot_k": int(getattr(self.config, "PLOT_K", 10)),
            "recompute_bootstrap_trajectories": bool(getattr(self.config, "RECOMPUTE_BOOTSTRAP_TRAJ", False)),
            "traj_ffill_limit": int(getattr(self.config, "TRAJ_FFILL_LIMIT", 2)),
            "save_notebook_style_plots": bool(getattr(self.config, "SAVE_NOTEBOOK_STYLE_PLOTS", True)),
            "feature_sets": {
                name: {
                    "n_features": int(len(cols)),
                    "columns": cols,
                }
                for name, cols in feature_configs.items()
            },
            "trajectory_source": {
                "bayesian_columns": int(len(traj_bayes)),
                "bootstrap_columns": int(len(traj_boot)),
                "primary_source": "bayesian" if len(traj_bayes) > 0 else "bootstrap",
            },
        }

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[2]),
                text=True,
            ).strip()
            self._last_run_meta["git_commit"] = git_commit
        except Exception:
            self._last_run_meta["git_commit"] = None
        
        return aggregated

    def _pick_feature_model_pairs(self, results_df: pd.DataFrame, mode: str, k: int) -> set[tuple[str, str]]:
        if results_df.empty:
            return set()
        if mode == "all":
            return set(zip(results_df["model"], results_df["feature_set"]))

        ranked = results_df.sort_values(["auroc_mean", "aupr_mean"], ascending=False).copy()
        if mode == "top-k":
            chosen = ranked.head(max(k, 1))
            return set(zip(chosen["model"], chosen["feature_set"]))

        # representative-k: nearest to global median performance point
        med_auc = float(results_df["auroc_mean"].median())
        med_aupr = float(results_df["aupr_mean"].median())
        rep = results_df.copy()
        rep["_dist"] = np.sqrt((rep["auroc_mean"] - med_auc) ** 2 + (rep["aupr_mean"] - med_aupr) ** 2)
        chosen = rep.sort_values("_dist", ascending=True).head(max(k, 1))
        return set(zip(chosen["model"], chosen["feature_set"]))

    def _save_png_plots(self, results_df: pd.DataFrame, results_name: str):
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            logger.warning(f"Skipping plots (matplotlib unavailable): {e}")
            return

        if results_df.empty:
            return

        def _palette(n: int):
            if n <= 20:
                cmap = plt.get_cmap("tab20")
                return [cmap(i) for i in range(n)]
            cmap1 = plt.get_cmap("tab20")
            cmap2 = plt.get_cmap("hsv")
            base = [cmap1(i % 20) for i in range(min(n, 20))]
            if n > 20:
                base.extend([cmap2(i / max(1, n - 20)) for i in range(n - 20)])
            return base[:n]

        def _plot_subset(df_sub: pd.DataFrame, tag: str):
            if df_sub.empty:
                return
            d = df_sub.copy()
            d["label"] = d["model"] + " | " + d["feature_set"]
            d_auc = d.sort_values("auroc_mean", ascending=True)
            d_aupr = d.sort_values("aupr_mean", ascending=True)

            fig_h = max(6, 0.32 * len(d))
            fig, axes = plt.subplots(1, 2, figsize=(18, fig_h), constrained_layout=True)

            colors_auc = _palette(len(d_auc))
            colors_aupr = _palette(len(d_aupr))

            axes[0].barh(d_auc["label"], d_auc["auroc_mean"], xerr=d_auc["auroc_std"], color=colors_auc, alpha=0.9)
            axes[0].set_title(f"AUROC ({tag})")
            axes[0].set_xlabel("AUROC")

            axes[1].barh(d_aupr["label"], d_aupr["aupr_mean"], xerr=d_aupr["aupr_std"], color=colors_aupr, alpha=0.9)
            axes[1].set_title(f"AUPR ({tag})")
            axes[1].set_xlabel("AUPR")

            out = self.output_dir / f"{results_name}_{tag}.png"
            fig.savefig(out, dpi=160)
            plt.close(fig)

        k = int(getattr(self.config, "PLOT_K", 10))
        _plot_subset(results_df, "all")

        top_pairs = self._pick_feature_model_pairs(results_df, "top-k", k)
        rep_pairs = self._pick_feature_model_pairs(results_df, "representative-k", k)

        top_df = results_df[
            results_df.apply(lambda r: (r["model"], r["feature_set"]) in top_pairs, axis=1)
        ]
        rep_df = results_df[
            results_df.apply(lambda r: (r["model"], r["feature_set"]) in rep_pairs, axis=1)
        ]

        _plot_subset(top_df, f"top_k_{k}")
        _plot_subset(rep_df, f"representative_k_{k}")

    def _save_notebook_style_plots(self, results_df: pd.DataFrame, results_name: str):
        if not bool(getattr(self.config, "SAVE_NOTEBOOK_STYLE_PLOTS", True)):
            return

        if results_df.empty or not self._last_fold_rows or not self._last_oof_rows:
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            logger.warning(f"Skipping notebook-style plots (matplotlib unavailable): {e}")
            return

        fold_df = pd.DataFrame(self._last_fold_rows)
        oof_df = pd.DataFrame(self._last_oof_rows)
        if fold_df.empty or oof_df.empty:
            return

        k = int(getattr(self.config, "PLOT_K", 10))
        ranked = results_df.sort_values(["aupr_mean", "auroc_mean"], ascending=False)
        chosen = ranked.head(max(1, k)).copy()
        pairs = list(zip(chosen["model"], chosen["feature_set"]))
        if not pairs:
            return

        labels = [f"{m} | {fs}" for m, fs in pairs]

        # Fold-wise boxplots for AUROC/AUPR
        roc_data = []
        ap_data = []
        for m, fs in pairs:
            d = fold_df[(fold_df["model"] == m) & (fold_df["feature_set"] == fs)]
            roc_data.append(d["auroc"].dropna().to_numpy())
            ap_data.append(d["aupr"].dropna().to_numpy())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
        ax1.boxplot(roc_data, labels=labels, patch_artist=True, showmeans=True)
        ax1.axhline(0.5, linestyle="--", linewidth=1, color="gray", alpha=0.8)
        ax1.set_ylabel("AUROC")
        ax1.set_title("AUROC Distribution Across Folds")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(axis="y", alpha=0.3)

        baseline = float(oof_df["y_true"].mean()) if "y_true" in oof_df.columns else np.nan
        ax2.boxplot(ap_data, labels=labels, patch_artist=True, showmeans=True)
        if np.isfinite(baseline):
            ax2.axhline(baseline, linestyle="--", linewidth=1, color="gray", alpha=0.8)
        ax2.set_ylabel("AUPR")
        ax2.set_title("AUPR Distribution Across Folds")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(axis="y", alpha=0.3)

        out_box = self.output_dir / f"{results_name}_notebook_boxplots.png"
        fig.savefig(out_box, dpi=160)
        plt.close(fig)

        # Mean ROC/PR curves with ±1 SD shading across fold-level curves.
        fig, (axr, axp) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
        grid = np.linspace(0, 1, 100)

        for (m, fs), label in zip(pairs, labels):
            group_df = oof_df[(oof_df["model"] == m) & (oof_df["feature_set"] == fs)]
            if group_df.empty:
                continue

            roc_curves = []
            pr_curves = []
            for (_, _), fd in group_df.groupby(["repeat", "fold"]):
                y_true = fd["y_true"].to_numpy()
                y_pred = fd["y_pred"].to_numpy()
                if len(np.unique(y_true)) < 2:
                    continue

                fpr, tpr, _ = roc_curve(y_true, y_pred)
                precision, recall, _ = precision_recall_curve(y_true, y_pred)

                roc_curves.append(np.interp(grid, fpr, tpr))
                pr_curves.append(np.interp(grid, recall[::-1], precision[::-1]))

            if len(roc_curves) == 0 or len(pr_curves) == 0:
                continue

            roc_arr = np.vstack(roc_curves)
            pr_arr = np.vstack(pr_curves)

            roc_mean = roc_arr.mean(axis=0)
            roc_std = roc_arr.std(axis=0)
            pr_mean = pr_arr.mean(axis=0)
            pr_std = pr_arr.std(axis=0)

            axr.plot(grid, roc_mean, linewidth=2, label=label)
            axr.fill_between(grid, roc_mean - roc_std, roc_mean + roc_std, alpha=0.2)

            axp.plot(grid, pr_mean, linewidth=2, label=label)
            axp.fill_between(grid, pr_mean - pr_std, pr_mean + pr_std, alpha=0.2)

        axr.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.8)
        axr.set_xlabel("False Positive Rate")
        axr.set_ylabel("True Positive Rate")
        axr.set_title("ROC Curves (mean ± SD)")
        axr.grid(alpha=0.3)
        axr.legend(fontsize=8)

        if np.isfinite(baseline):
            axp.plot([0, 1], [baseline, baseline], "k--", linewidth=1, alpha=0.8)
        axp.set_xlabel("Recall")
        axp.set_ylabel("Precision")
        axp.set_title("Precision-Recall Curves (mean ± SD)")
        axp.grid(alpha=0.3)
        axp.legend(fontsize=8)

        out_rocpr = self.output_dir / f"{results_name}_notebook_roc_pr.png"
        fig.savefig(out_rocpr, dpi=160)
        plt.close(fig)
    
    def save_results(self, results: dict, results_name: Optional[str] = None):
        """
        Save CV results to CSV.
        
        Args:
            results: Output from run_cv()
            results_name: Base name for output file (default: {TASK}_cv_results.csv)
        """
        if results_name is None:
            results_name = f"{self.config.TASK}_cv_results"
        
        output_path = self.output_dir / f"{results_name}.csv"
        
        rows = []
        for model_name, feature_results in results.items():
            for fs_name, metrics in feature_results.items():
                row = {
                    "model": model_name,
                    "feature_set": fs_name,
                    **metrics,
                }
                rows.append(row)
        
        results_df = pd.DataFrame(rows)
        results_df.to_csv(output_path, index=False)

        # Save per-fold/per-repeat detailed metrics for auditability
        detail_path = self.output_dir / f"{results_name}_fold_metrics.csv"
        if self._last_fold_rows:
            pd.DataFrame(self._last_fold_rows).to_csv(detail_path, index=False)

        # Save OOF predictions (size-aware selection)
        if getattr(self.config, "SAVE_OOF", True) and self._last_oof_rows:
            oof_mode = str(getattr(self.config, "OOF_SCOPE", "top-k"))
            oof_k = int(getattr(self.config, "OOF_K", 10))
            if oof_mode not in {"all", "top-k", "representative-k"}:
                oof_mode = "top-k"

            keep_pairs = self._pick_feature_model_pairs(results_df, oof_mode, oof_k)
            oof_df = pd.DataFrame(self._last_oof_rows)
            if len(keep_pairs) > 0 and oof_mode != "all":
                oof_df = oof_df[
                    oof_df.apply(lambda r: (r["model"], r["feature_set"]) in keep_pairs, axis=1)
                ]

            oof_path = self.output_dir / f"{results_name}_oof_predictions.parquet"
            try:
                oof_df.to_parquet(oof_path, index=False)
            except Exception:
                oof_path = self.output_dir / f"{results_name}_oof_predictions.csv.gz"
                oof_df.to_csv(oof_path, index=False)
            self._last_run_meta["oof_rows_saved"] = int(len(oof_df))
            self._last_run_meta["oof_file"] = str(oof_path)

        # Save PNG plots (all / top-k / representative-k)
        self._save_png_plots(results_df, results_name)
        self._save_notebook_style_plots(results_df, results_name)

        # Save metadata/config snapshot for reproducibility
        meta_path = self.output_dir / f"{results_name}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._last_run_meta, f, indent=2)
        
        logger.info(f"\n4. Results saved to: {output_path}")
        logger.info(f"   Fold metrics: {detail_path}")
        logger.info(f"   Metadata: {meta_path}")
        logger.info(f"\n{results_df.to_string(index=False)}")
    
    def _detect_windowing_col(self, df: pd.DataFrame) -> str:
        """Detect windowing column name."""
        _, windowing_col = pick_time_cols(df)
        return windowing_col


# ============================================================================
# CLI Entry Point
# ============================================================================


def main_cli(
    config_class: type[AnalysisConfig],
    description: str,
):
    """
    Generic CLI entry point for analysis scripts.
    
    Args:
        config_class: AnalysisConfig subclass to instantiate
        description: Script description
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Data directories
    parser.add_argument(
        "--base-dir",
        type=str,
        required=False,
        help="Base directory for input data (auto-detected from dataset/task if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Output directory for results (default: results/{dataset}/{task})",
    )
    
    # CV configuration
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=None,
        help=f"Number of CV repeats (default: {config_class.CV_N_REPEATS})",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=None,
        help=f"Number of CV folds (default: {config_class.CV_N_SPLITS})",
    )
    parser.add_argument(
        "--lookahead-hours",
        type=float,
        default=None,
        help=f"Deprecated alias for --lookback-window (default: {config_class.LOOKAHEAD_HOURS}).",
    )
    parser.add_argument(
        "--lookback-window",
        type=float,
        default=None,
        help="Lookback window in native time units (days for time_day, hours for time_hour).",
    )
    parser.add_argument(
        "--lookback-unit",
        type=str,
        choices=["hours", "days"],
        default=None,
        help="Lookback unit label for logging.",
    )
    parser.add_argument(
        "--marker-mode",
        type=str,
        choices=["auto", "single-multi", "pooled"],
        default=None,
        help="Feature-set mode: auto (default), single-multi, or pooled.",
    )
    parser.add_argument(
        "--with-probs-source",
        type=str,
        choices=["default", "bootstrap", "bayesian"],
        default="default",
        help=(
            "Interpretation of *_with_probs and standalone *trajectory_probs* files: "
            "default=use config, bootstrap=treat as bootstrap and suffix with _boot, "
            "bayesian=treat as non-bootstrap."
        ),
    )
    
    # Parallelization
    parser.add_argument(
        "--train-n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for CV repeats (default: 1)",
    )
    parser.add_argument(
        "--train-backend",
        type=str,
        choices=["threading", "loky", "multiprocessing", "sequential", "processes"],
        default="threading",
        help=(
            "Joblib backend for CV parallelization "
            "(threading, loky, multiprocessing, sequential). "
            "'processes' is accepted as alias for 'loky'."
        ),
    )
    parser.add_argument(
        "--parallel-axis",
        type=str,
        choices=["repeat", "feature-set"],
        default=None,
        help="Parallelize across CV repeats (default) or feature sets.",
    )
    parser.add_argument(
        "--save-oof",
        action="store_true",
        help="Save out-of-fold predictions.",
    )
    parser.add_argument(
        "--no-save-oof",
        action="store_true",
        help="Disable out-of-fold prediction export.",
    )
    parser.add_argument(
        "--oof-scope",
        type=str,
        choices=["all", "top-k", "representative-k"],
        default=None,
        help="Which model/feature-set pairs to keep in OOF output.",
    )
    parser.add_argument(
        "--oof-k",
        type=int,
        default=None,
        help="K value used for OOF scope when using top-k or representative-k.",
    )
    parser.add_argument(
        "--plot-k",
        type=int,
        default=None,
        help="K value for top-k and representative-k PNG plots.",
    )
    parser.add_argument(
        "--recompute-bootstrap-trajectories",
        action="store_true",
        help="Force recomputation of bootstrap trajectory probabilities even when cached files exist.",
    )
    
    args = parser.parse_args()
    
    # Instantiate config
    config = config_class()
    
    # Determine directories
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        dataset_config = get_dataset_config(config.DATASET)
        base_dir = dataset_config["base_path"] / config.TASK
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results") / config.DATASET / config.TASK
    
    # Update config from CLI
    if args.cv_repeats:
        config.CV_N_REPEATS = args.cv_repeats
    if args.cv_splits:
        config.CV_N_SPLITS = args.cv_splits
    if args.lookback_window is not None:
        config.LOOKBACK_WINDOW = args.lookback_window
        config.LOOKAHEAD_HOURS = args.lookback_window
    elif args.lookahead_hours is not None:
        config.LOOKBACK_WINDOW = args.lookahead_hours
        config.LOOKAHEAD_HOURS = args.lookahead_hours

    if args.lookback_unit is not None:
        config.LOOKBACK_UNIT = args.lookback_unit
    if args.marker_mode is not None:
        config.MARKER_MODE = args.marker_mode
    if args.with_probs_source == "bootstrap":
        config.WITH_PROBS_IS_BOOTSTRAP = True
    elif args.with_probs_source == "bayesian":
        config.WITH_PROBS_IS_BOOTSTRAP = False
    if args.parallel_axis is not None:
        config.PARALLEL_AXIS = args.parallel_axis
    if args.no_save_oof:
        config.SAVE_OOF = False
    elif args.save_oof:
        config.SAVE_OOF = True
    if args.oof_scope is not None:
        config.OOF_SCOPE = args.oof_scope
    if args.oof_k is not None:
        config.OOF_K = args.oof_k
    if args.plot_k is not None:
        config.PLOT_K = args.plot_k
    if args.recompute_bootstrap_trajectories:
        config.RECOMPUTE_BOOTSTRAP_TRAJ = True
    
    # Run analysis
    try:
        train_backend = "loky" if args.train_backend == "processes" else args.train_backend
        analysis = TrajectoryAnalysis(config, base_dir, output_dir)
        dataset = analysis.load_data()
        dataset = analysis.prepare_features(dataset)
        results = analysis.run_cv(
            dataset,
            n_repeats=config.CV_N_REPEATS,
            n_splits=config.CV_N_SPLITS,
            n_jobs=args.train_n_jobs,
            backend=train_backend,
            parallel_axis=getattr(config, "PARALLEL_AXIS", "repeat"),
        )
        analysis.save_results(results)
        logger.info("\n✓ Analysis complete")
    
    except Exception as e:
        logger.error(f"✗ Analysis failed: {e}", exc_info=True)
        raise
