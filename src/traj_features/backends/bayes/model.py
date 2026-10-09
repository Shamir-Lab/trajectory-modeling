from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, Callable
from .pipeline import compute_time_varying_trajectory_covariates_parallel
from .classify import flags_from_traj


@dataclass
class BayesConfig:
    window_years: float = 2.0
    df_basis: int = 4
    n_samples: int = 1000
    tune: int = 500
    min_points_per_window: int = 5
    grid_freq: int = 12
    flat_thr: float = -1.0
    decline_thr: float = -2.0
    nonlinear_gap: float = 3.0
    pids: str = "patient_id"
    values: str = "lab_value"
    time_col: str = "time"
    windowing_col: str = "time"
    use_gpu: bool = False
    sampler: Literal["pymc", "numpyro", "nutpie"] = "pymc"
    chains: int = 4
    # "vectorized": all chains share one process (less memory, lower throughput).
    # "parallel": one chain per process (more memory, higher throughput).
    chain_method: Literal["parallel", "vectorized", "sequential"] = "vectorized"
    cores: int = 1
    n_jobs: int = -1
    progressbar: bool = False
    available_gpus: list[int] | None = None
    target_accept: Optional[float] = 0.95
    # batch_size: number of (patient, timepoint) windows processed per joblib batch.
    # Larger values reduce scheduling overhead; smaller values allow more frequent
    # cache flushes (useful on SLURM with limited /tmp space).
    batch_size: int = 500
    traj_types: tuple[str, ...] = ('prolonged_nonprogression', 'linear_decline', 'nonlinear')
    class_func: Callable = flags_from_traj
    label_map: Optional[Dict[str, str]] = None


class BayesianTraj:
    """Bayesian MCMC trajectory probability model.

    Uses PyMC with PyTensor (C-compiled ops via g++) to compute posterior
    trajectory-type probabilities per (patient, time) window.

    Example
    -------
    >>> cfg = BayesConfig(sampler="pymc", use_gpu=False, n_samples=500)
    >>> model = BayesianTraj(cfg)
    >>> probs = model.embed(lab_data)  # DataFrame with trajtype_* probability columns
    """

    name = "bayes"

    def __init__(self, cfg: BayesConfig | None = None):
        self.cfg = cfg or BayesConfig()
        # Auto-select sampler when left at default "pymc":
        #   GPU requested → try numpyro (JAX), then pymc
        #   CPU only      → use pymc (accelerated by g++ via PyTensor when available)
        if self.cfg.sampler == "pymc":
            if self.cfg.use_gpu:
                try:
                    import pymc.sampling.jax  # noqa: F401
                    self.cfg.sampler = "numpyro"
                    print("Using 'numpyro' sampler (JAX/GPU).")
                except Exception:
                    pass

    def embed(self, lab_long_df: pd.DataFrame) -> pd.DataFrame:
        """Compute posterior trajectory-type probabilities per (pid, time).

        Parameters
        ----------
        lab_long_df : DataFrame
            Long-format lab data with columns [pids, time_col, values]

        Returns
        -------
        DataFrame
            Trajectory probabilities per (patient, time_day)
        """
        return compute_time_varying_trajectory_covariates_parallel(
            lab_df=lab_long_df,
            window_years=self.cfg.window_years,
            flat_thr=self.cfg.flat_thr,
            decline_thr=self.cfg.decline_thr,
            nonlinear_gap=self.cfg.nonlinear_gap,
            df_basis=self.cfg.df_basis,
            n_samples=self.cfg.n_samples,
            tune=self.cfg.tune,
            n_jobs=self.cfg.n_jobs,
            min_points_per_window=self.cfg.min_points_per_window,
            grid_freq=self.cfg.grid_freq,
            batch_size=self.cfg.batch_size,
            pids=self.cfg.pids,
            values=self.cfg.values,
            time_col=self.cfg.time_col,
            sampler=self.cfg.sampler,
            chains=self.cfg.chains,
            chain_method=self.cfg.chain_method,
            cores=self.cfg.cores,
            target_accept=self.cfg.target_accept,
            progressbar=self.cfg.progressbar,
            traj_types=self.cfg.traj_types,
            class_func=self.cfg.class_func,
            label_map=self.cfg.label_map,
            windowing_col=self.cfg.windowing_col
        )


# Backward-compatibility alias — will be removed in a future version.
BayesianTrajPS = BayesianTraj
