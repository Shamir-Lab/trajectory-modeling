"""
Bootstrap trajectory model.

Fast alternative to Bayesian MCMC using bootstrap resampling.
"""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Callable
from .pipeline import compute_bootstrap_trajectory_covariates_parallel
from ..bayes.classify import flags_from_traj
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap trajectory modeling."""
    window_years: float = 2.0
    n_bootstrap: int = 500
    smoothing: float | None = None  # Auto-select if None
    min_points_per_window: int = 5
    grid_freq: int = 12
    flat_thr: float = -1.0
    decline_thr: float = -2.0
    nonlinear_gap: float = 3.0
    pids: str = "patient_id"
    values: str = "lab_value"
    time_col: str = "time"
    windowing_col: str = "time"
    n_jobs: int = -1
    progressbar: bool = False
    traj_types: tuple[str, ...] = ('prolonged_nonprogression', 'linear_decline', 'nonlinear')
    class_func: Callable = flags_from_traj
    label_map: Optional[Dict[str, str]] = None


class BootstrapTraj:
    """Bootstrap-based trajectory probability computation.

    Fast alternative to BayesianTraj using:
    - Bootstrap resampling for uncertainty quantification
    - Cubic splines for smooth fits
    - Same classification logic as Bayesian backend

    Typically 10-100x faster than MCMC with comparable results.

    Example
    -------
    >>> config = BootstrapConfig(
    ...     window_years=3.0,
    ...     n_bootstrap=500,
    ...     flat_thr=10.0,
    ...     decline_thr=30.0
    ... )
    >>> model = BootstrapTraj(config)
    >>> probs = model.embed(lab_data)
    """

    name = "bootstrap"

    def __init__(self, cfg: BootstrapConfig | None = None):
        """Initialize bootstrap trajectory model.

        Parameters
        ----------
        cfg : BootstrapConfig, optional
            Configuration (uses defaults if None)
        """
        self.cfg = cfg or BootstrapConfig()

    def embed(self, lab_long_df: pd.DataFrame) -> pd.DataFrame:
        """Compute trajectory probabilities using bootstrap resampling.

        Parameters
        ----------
        lab_long_df : DataFrame
            Long-format lab data with columns [pids, time_col, values]

        Returns
        -------
        DataFrame
            Trajectory probabilities per (patient, time_day)
        """
        return compute_bootstrap_trajectory_covariates_parallel(
            lab_df=lab_long_df,
            window_years=self.cfg.window_years,
            flat_thr=self.cfg.flat_thr,
            decline_thr=self.cfg.decline_thr,
            nonlinear_gap=self.cfg.nonlinear_gap,
            n_bootstrap=self.cfg.n_bootstrap,
            smoothing=self.cfg.smoothing,
            min_points_per_window=self.cfg.min_points_per_window,
            grid_freq=self.cfg.grid_freq,
            pids=self.cfg.pids,
            values=self.cfg.values,
            time_col=self.cfg.time_col,
            windowing_col=self.cfg.windowing_col,
            n_jobs=self.cfg.n_jobs,
            traj_types=self.cfg.traj_types,
            class_func=self.cfg.class_func,
            label_map=self.cfg.label_map,
            progressbar=self.cfg.progressbar
        )


# Backward-compatibility alias — will be removed in a future version.
BootstrapTrajPS = BootstrapTraj
