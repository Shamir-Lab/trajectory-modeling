#!/usr/bin/env python
"""
Sensitivity analysis for the eICU Septic Shock trajectory-archetype thresholds.

Scope: eICU Septic Shock cohort, multi-marker (lactate, wbc, platelet). One marker's
archetype thresholds (tau_flat, tau_decline, gamma_gap) are perturbed at a time; the
other markers keep their already-computed trajectory probabilities. Direction is
marker-specific: lactate/wbc rules assume "increase = worse"; platelet assumes
"decrease = worse" (mirrors `pos_flags_from_traj` vs `flags_from_traj` in
`traj_features.backends.bayes.classify`, and BIOMARKERS below matches
scripts/trajectory/eicu/sepsis_all_biomarkers.py exactly).

Two interchangeable trajectory backends supply the cached inter-point slopes:
  - bootstrap : fast, CPU-only (spline + bootstrap resampling). Safe to build
                interactively, no container needed.
  - bayes     : real PyMC posterior sampling (`_sample_post_trajs_scaled`), matching
                the production Bayesian backend used for lactate/wbc/platelet in
                scripts/trajectory/eicu/sepsis_all_biomarkers.py. Without g++, PyTensor
                falls back to slow pure-Python compilation (see docs/CONTAINER_GUIDE.md)
                — build this cache inside the trajectory-modeling udocker/Docker
                container via scripts/slurm/eicu/run_eicu_sepsis_bayes_slope_cache_container.slurm.

Pipeline
--------
1. Load (or build once, per backend) a cache of sampled inter-point slopes s_k for
   every (stay_id, time_day) prediction window of the chosen marker. Building the
   cache is the ONLY step that fits a spline / runs MCMC; it runs once and is
   skipped on subsequent invocations.
2. `phi_sensitivity` maps cached slopes -> [P_stable, P_gradual, P_nonlinear] for a
   given (tau_flat, tau_decline, gamma_gap, direction). Pure numpy, no
   spline/MCMC refitting. Handles both a fixed local-slope count per window
   (bootstrap: grid_freq is a literal point count) and a ragged one (Bayesian:
   grid_freq is a points-per-unit-time rate, so window span changes the count) by
   NaN-padding to a common width and excluding failed draws / padding from the
   per-window denominator.
3. For each combination in a one-at-a-time perturbation grid around the chosen
   marker's baseline (default lactate: tau_flat=0.1, tau_decline=0.3, gamma_gap=0.5):
     - swap that marker's T (trajectory) columns for the phi_sensitivity output
     - measure upstream stability vs. the baseline probabilities (mean TVD,
       % of windows whose primary/dominant archetype flips)
     - measure downstream performance by running the *exact* eICU Sepsis analysis
       pipeline (`eICUSepsisConfig` + `TrajectoryAnalysis.run_cv` from
       eicu_sepsis_analysis.py / analysis_template.py — identical feature-set
       construction incl. multi-marker single-vs-multi framing, imputation,
       scaling, and model training) and reading off one (model, feature-set) cell's
       fold-level AUPR/AUROC: median AUPR, mean AUROC, 95% CIs.
4. Write sensitivity_results_eicu_sepsis.csv and an AUPR-stability plot.

Run from workspace root:
    # Fast, interactive (bootstrap backend):
    python scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py --backend bootstrap

    # Bayesian backend, single-shot (small cohort / already-cheap window count):
    python scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py --backend bayes --bayes-sampler pymc

    # Bayesian backend, cohort-parallel via SLURM + udocker (see the .slurm script for the full flow):
    python scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py --backend bayes \\
        --build-cache-only --cohort-splits 10 --cohort-index 0
    # ... after all cohort shards finish:
    python scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py --backend bayes --merge-cohort-caches
    python scripts/analysis/eicu/eicu_sepsis_sensitivity_analysis.py --backend bayes   # normal run, reuses merged cache
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed as jdelayed
from scipy.interpolate import UnivariateSpline

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from analysis.analysis_template import TrajectoryAnalysis  # noqa: E402
from analysis.analysis_utils import (  # noqa: E402
    get_dataset_config,
    load_named_table,
    pick_time_cols,
)
from analysis.poster.palette import set_poster_style  # noqa: E402
from eicu_sepsis_analysis import eICUSepsisConfig  # noqa: E402 - same directory; guarantees identical config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEED = 920

DATASET = "eicu"
TASK = "sepsis"

# Mirrors BIOMARKERS in scripts/trajectory/eicu/sepsis_all_biomarkers.py exactly
# (file/value_col/thresholds) plus the direction convention each marker's
# class_func encodes there (pos_flags_from_traj -> "increase", flags_from_traj -> "decrease").
BIOMARKERS = {
    "lactate": {
        "file": "lactate_timeseries.csv",
        "value_col": "lactate",
        "direction": "increase",
        "tau_flat": 0.1,
        "tau_decline": 0.3,
        "gamma_gap": 0.5,
    },
    "wbc": {
        "file": "wbc_timeseries.csv",
        "value_col": "wbc",
        "direction": "increase",
        "tau_flat": 1.0,
        "tau_decline": 3.0,
        "gamma_gap": 2.0,
    },
    "platelet": {
        "file": "platelets_timeseries.csv",
        "value_col": "platelet",
        "direction": "decrease",
        "tau_flat": -20.0,
        "tau_decline": -50.0,
        "gamma_gap": 30.0,
    },
}

DEFAULT_FEATURE_SET_NAME = "Static + Dynamic + Multi-Marker Trajectories + Summary Stats"
# Same feature set with the trajectory-archetype features removed — everything
# else (static, dynamic, summary stats) held fixed. Ceteris-paribus baseline
# for asking whether the trajectory features' contribution survives the
# threshold sensitivity sweep, not just whether they help at the reported ×1.0
# threshold.
TRAJECTORY_ABLATED_FEATURE_SET_NAME = "Static + Dynamic + Multi-Marker Summary Stats"
DEFAULT_MODEL_NAME = "XGBoost"

# First three slots of the validated categorical palette (blue/orange/aqua) —
# clears CVD + normal-vision separation floors in both light and dark modes,
# in fixed order, one hue per perturbed parameter.
PARAM_COLORS = {
    "tau_flat": "#2a78d6",
    "tau_decline": "#eb6834",
    "gamma_gap": "#1baf7a",
}


# ============================================================================
# Shared window construction (used by both backends)
# ============================================================================


def _build_window_list(
    ts_df: pd.DataFrame,
    id_col: str,
    time_col: str,
    windowing_col: str,
    value_col: str,
    window: float,
    min_points_per_window: int,
) -> list[tuple]:
    """Returns [(pid, window_id, window_df[[time_col, value_col]]), ...]."""
    windows = []
    for pid, pid_df in ts_df.groupby(id_col):
        pid_df = pid_df.sort_values(time_col)
        for window_id, _ in pid_df.groupby(windowing_col):
            window_start = window_id - window
            wdf = pid_df[
                (pid_df[windowing_col] >= window_start) & (pid_df[windowing_col] <= window_id)
            ]
            if len(wdf) >= min_points_per_window:
                windows.append((pid, window_id, wdf[[time_col, value_col]].copy()))
    return windows


def _clean_timeseries(
    ts_df: pd.DataFrame, time_col: str, windowing_col: str, value_col: str
) -> pd.DataFrame:
    ts_df = ts_df.copy()
    ts_df[time_col] = pd.to_numeric(ts_df[time_col], errors="coerce")
    ts_df[windowing_col] = pd.to_numeric(ts_df[windowing_col], errors="coerce")
    ts_df[value_col] = pd.to_numeric(ts_df[value_col], errors="coerce")
    return ts_df.dropna(subset=[time_col, windowing_col, value_col])


# ============================================================================
# Bootstrap backend — fast, CPU-only, no container needed
# ============================================================================


def _bootstrap_spline_slopes(
    time: np.ndarray,
    values: np.ndarray,
    time_grid: np.ndarray,
    n_bootstrap: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Bootstrap-resample a (time, values) window and return per-draw inter-point
    slopes on `time_grid`. This is the only function in this module that fits a
    spline for the bootstrap backend; it is called once per window while building
    the on-disk cache.
    """
    n_points = len(time)
    k_out = len(time_grid) - 1
    slopes = np.full((n_bootstrap, k_out), np.nan)
    if n_points < 4:
        return slopes

    for b in range(n_bootstrap):
        idx = rng.choice(n_points, size=n_points, replace=True)
        bt, bv = time[idx], values[idx]
        order = np.argsort(bt)
        bt, bv = bt[order], bv[order]

        unique_t, inverse = np.unique(bt, return_inverse=True)
        if len(unique_t) < len(bt):
            bv = np.array([bv[inverse == i].mean() for i in range(len(unique_t))])
            bt = unique_t
        if len(bt) < 4:
            continue

        try:
            spline = UnivariateSpline(bt, bv, k=min(3, len(bt) - 1), s=None)
            traj = spline(time_grid)
            slopes[b] = np.diff(traj) / np.diff(time_grid)
        except Exception:
            continue

    return slopes


def _bootstrap_process_window(
    pid, window_id, wdf: pd.DataFrame, time_col: str, value_col: str,
    grid_freq: int, n_bootstrap: int, seed: int,
):
    rng = np.random.RandomState(seed)
    time_arr = wdf[time_col].to_numpy(dtype=float)
    values_arr = wdf[value_col].to_numpy(dtype=float)
    t_min, t_max = time_arr.min(), time_arr.max()
    time_grid = np.linspace(t_min, t_max, grid_freq)
    slopes = _bootstrap_spline_slopes(time_arr, values_arr, time_grid, n_bootstrap, rng)
    return pid, window_id, slopes


def build_or_load_bootstrap_slope_cache(
    ts_df: pd.DataFrame,
    id_col: str,
    time_col: str,
    windowing_col: str,
    value_col: str,
    cache_path: Path,
    window: float,
    n_bootstrap: int,
    grid_freq: int,
    min_points_per_window: int = 5,
    n_jobs: int = 4,
    force_rebuild: bool = False,
) -> dict[str, np.ndarray]:
    """Load a cached array of bootstrap-resampled inter-point slopes, building it
    once if absent. Cache schema (npz): slopes (N, n_bootstrap, grid_freq - 1),
    stay_id (N,), time_day (N,). grid_freq here is a literal grid point count
    (matches `BootstrapConfig`/`process_patient_window` semantics), so K is fixed
    across all windows.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        logger.info(f"   Loading cached bootstrap slopes: {cache_path}")
        with np.load(cache_path) as npz:
            return {"slopes": npz["slopes"], "stay_id": npz["stay_id"], "time_day": npz["time_day"]}

    logger.info(f"   No slope cache at {cache_path}; building it once (bootstrap resample + spline fit)...")
    ts_df = _clean_timeseries(ts_df, time_col, windowing_col, value_col)

    windows = _build_window_list(
        ts_df, id_col, time_col, windowing_col, value_col, window, min_points_per_window
    )
    if not windows:
        raise RuntimeError(
            f"No patient-windows with >= {min_points_per_window} points found for '{value_col}'."
        )
    logger.info(f"   {len(windows):,} patient-windows x {n_bootstrap} bootstrap draws")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        jdelayed(_bootstrap_process_window)(pid, wid, wdf, time_col, value_col, grid_freq, n_bootstrap, SEED + i)
        for i, (pid, wid, wdf) in enumerate(windows)
    )

    stay_id = np.array([r[0] for r in results])
    time_day = np.array([r[1] for r in results])
    slopes = np.stack([r[2] for r in results], axis=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, slopes=slopes, stay_id=stay_id, time_day=time_day)
    logger.info(f"   Cached bootstrap slopes -> {cache_path} (shape={slopes.shape})")

    return {"slopes": slopes, "stay_id": stay_id, "time_day": time_day}


# ============================================================================
# Bayesian backend — real PyMC posterior sampling, needs g++ (udocker container)
# ============================================================================


def _bayes_window_worker(
    wdf: pd.DataFrame,
    df_basis: int,
    n_samples: int,
    tune: int,
    min_points: int,
    grid_freq: int,
    value_col: str,
    time_col: str,
    k_max: int,
    *,
    sampler: str,
    chains: int,
    cores: int,
    progressbar: bool,
    chain_method: str,
    target_accept: float,
) -> np.ndarray:
    """Run real MCMC sampling for one window and return NaN-padded inter-point
    slopes, shape (n_posterior_draws, k_max). Mirrors `_window_worker`
    (traj_features/backends/bayes/pipeline.py) up to the point where it would
    collapse y_samples into archetype probabilities; here the raw slopes are kept
    instead, so `phi_sensitivity` can re-classify them for many threshold
    combinations without resampling. Returns shape (0, k_max) on failure.
    """
    # Imported lazily (pymc/pytensor are heavy and only needed for this backend).
    from traj_features.backends.bayes.pipeline import _apply_worker_process_patches
    from traj_features.backends.bayes.sampling import _sample_post_trajs_scaled

    _apply_worker_process_patches()

    try:
        time_grid, y_samples = _sample_post_trajs_scaled(
            wdf, df_basis=df_basis, n_samples=n_samples, tune=tune, min_points=min_points,
            grid_freq=grid_freq, target_accept=target_accept, values=value_col, time_col=time_col,
            sampler=sampler, chains=chains, cores=cores, progressbar=progressbar,
            chain_method=chain_method,
        )
    except Exception as e:
        print(f"[WARNING] Bayesian window worker failed, skipping: {type(e).__name__}: {e}")
        return np.zeros((0, k_max))

    slopes = np.diff(y_samples, axis=1) / np.diff(time_grid)  # (S, K)
    s, k = slopes.shape
    if k >= k_max:
        return slopes[:, :k_max]
    padded = np.full((s, k_max), np.nan)
    padded[:, :k] = slopes
    return padded


def build_or_load_bayes_slope_cache(
    ts_df: pd.DataFrame,
    id_col: str,
    time_col: str,
    windowing_col: str,
    value_col: str,
    cache_path: Path,
    window: float,
    n_samples: int = 200,
    tune: int = 300,
    df_basis: int = 5,
    grid_freq: int = 2,
    min_points_per_window: int = 4,
    n_jobs: int = -1,
    batch_size: int = 200,
    sampler: str = "pymc",
    chains: int = 4,
    cores: int = 1,
    chain_method: str = "vectorized",
    target_accept: float = 0.99,
    force_rebuild: bool = False,
) -> dict[str, np.ndarray]:
    """Load a cached array of posterior-sampled inter-point slopes, building it
    once (real PyMC MCMC per window) if absent.

    Cache schema (npz): slopes (N, S, K), stay_id (N,), time_day (N,), NaN-padded
    on both the posterior-draw axis (S) and the local-slope axis (K). K is
    ragged in the underlying `_sample_post_trajs_scaled` call — its internal grid
    length is `max(3, grid_freq, round(window_span * grid_freq))`, so a window
    with a longer observed span gets more grid points/slopes than a short one.
    K is padded to the worst case (span == `window`) so the array stays rectangular.

    Without g++, PyTensor compiles pure-Python and this is extremely slow — run
    inside the trajectory-modeling udocker/Docker container (see
    docs/CONTAINER_GUIDE.md and scripts/slurm/eicu/run_eicu_sepsis_bayes_slope_cache_container.slurm).
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        logger.info(f"   Loading cached Bayesian slopes: {cache_path}")
        with np.load(cache_path) as npz:
            return {"slopes": npz["slopes"], "stay_id": npz["stay_id"], "time_day": npz["time_day"]}

    logger.info(f"   No Bayesian slope cache at {cache_path}; building it once (PyMC posterior sampling)...")
    logger.info(
        "   NOTE: without g++ this is extremely slow — build inside the trajectory-modeling "
        "udocker/Docker container for C-compiled PyTensor (docs/CONTAINER_GUIDE.md)."
    )

    ts_df = _clean_timeseries(ts_df, time_col, windowing_col, value_col)
    windows = _build_window_list(ts_df, id_col, time_col, windowing_col, value_col, window, min_points_per_window)
    if not windows:
        raise RuntimeError(
            f"No patient-windows with >= {min_points_per_window} points found for '{value_col}'."
        )

    n_grid_ceiling = max(3, grid_freq, int(round(window * grid_freq)))
    k_max = n_grid_ceiling - 1
    logger.info(
        f"   {len(windows):,} patient-windows; sampler={sampler}, n_samples={n_samples}, "
        f"tune={tune}, chains={chains}, k_max={k_max}"
    )

    all_stay_id: list = []
    all_time_day: list = []
    all_slopes: list[np.ndarray] = []
    n_batches = (len(windows) + batch_size - 1) // batch_size

    for b_idx in range(n_batches):
        batch = windows[b_idx * batch_size: (b_idx + 1) * batch_size]
        logger.info(f"   Batch {b_idx + 1}/{n_batches} ({len(batch)} windows)...")
        batch_results = Parallel(n_jobs=n_jobs, backend="loky")(
            jdelayed(_bayes_window_worker)(
                wdf, df_basis, n_samples, tune, min_points_per_window, grid_freq,
                value_col, time_col, k_max,
                sampler=sampler, chains=chains, cores=cores, progressbar=False,
                chain_method=chain_method, target_accept=target_accept,
            )
            for (_, _, wdf) in batch
        )
        for (pid, wid, _), slopes in zip(batch, batch_results):
            if slopes.shape[0] == 0:
                continue
            all_stay_id.append(pid)
            all_time_day.append(wid)
            all_slopes.append(slopes)
        gc.collect()

    if not all_slopes:
        raise RuntimeError(f"All Bayesian window fits failed for '{value_col}'.")

    s_max = max(a.shape[0] for a in all_slopes)
    slopes_arr = np.full((len(all_slopes), s_max, k_max), np.nan)
    for i, a in enumerate(all_slopes):
        slopes_arr[i, : a.shape[0], :] = a

    stay_id = np.array(all_stay_id)
    time_day = np.array(all_time_day)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, slopes=slopes_arr, stay_id=stay_id, time_day=time_day)
    logger.info(f"   Cached Bayesian slopes -> {cache_path} (shape={slopes_arr.shape})")

    return {"slopes": slopes_arr, "stay_id": stay_id, "time_day": time_day}


def merge_cohort_slope_caches(cache_path: Path) -> None:
    """Merge `{cache_path.stem}_cohortNN.npz` shards (written by parallel SLURM
    array tasks) into the final, unsuffixed cache at `cache_path`."""
    cache_path = Path(cache_path)
    shard_paths = sorted(cache_path.parent.glob(f"{cache_path.stem}_cohort*.npz"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No cohort shard caches found matching {cache_path.parent}/{cache_path.stem}_cohort*.npz"
        )
    logger.info(f"   Merging {len(shard_paths)} cohort shard caches into {cache_path}...")

    stay_ids, time_days, slopes_list = [], [], []
    for p in shard_paths:
        with np.load(p) as npz:
            stay_ids.append(npz["stay_id"])
            time_days.append(npz["time_day"])
            slopes_list.append(npz["slopes"])
        logger.info(f"     {p.name}: {slopes_list[-1].shape[0]:,} windows")

    k_max = max(a.shape[2] for a in slopes_list)
    s_max = max(a.shape[1] for a in slopes_list)
    padded = []
    for a in slopes_list:
        n, s, k = a.shape
        pa = np.full((n, s_max, k_max), np.nan)
        pa[:, :s, :k] = a
        padded.append(pa)

    slopes = np.concatenate(padded, axis=0)
    stay_id = np.concatenate(stay_ids)
    time_day = np.concatenate(time_days)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, slopes=slopes, stay_id=stay_id, time_day=time_day)
    logger.info(f"   Merged cache -> {cache_path} (shape={slopes.shape})")


def build_or_load_slope_cache(backend: str, **kwargs) -> dict[str, np.ndarray]:
    if backend == "bootstrap":
        return build_or_load_bootstrap_slope_cache(**kwargs)
    if backend == "bayes":
        return build_or_load_bayes_slope_cache(**kwargs)
    raise ValueError(f"Unknown backend: {backend!r} (expected 'bootstrap' or 'bayes')")


# ============================================================================
# Vectorized archetype mapping
# ============================================================================


def phi_sensitivity(
    slopes: np.ndarray,
    tau_flat: float,
    tau_decline: float,
    gamma_gap: float,
    direction: str = "increase",
) -> np.ndarray:
    """Map cached inter-point slopes to [P_stable, P_gradual, P_nonlinear] per window.

    Parameters
    ----------
    slopes : ndarray, shape (n_windows, n_draws, n_local_slopes)
        Cached sampled inter-point slopes s_k. A draw is either fully populated
        (fit succeeded) or entirely NaN (fit failed); a window's local-slope axis
        may additionally be right-padded with NaN if some windows have fewer
        valid local slopes than the array width (ragged K, e.g. the Bayesian
        backend's span-adaptive grid).
    tau_flat, tau_decline, gamma_gap : float
        Archetype thresholds.
    direction : {"increase", "decrease"}
        "increase" (default; lactate, wbc): rising values are the bad direction,
        mirrors `pos_flags_from_traj`. "decrease" (platelet): falling values are
        the bad direction, mirrors `flags_from_traj`.

    Returns
    -------
    ndarray, shape (n_windows, 3)
        Per-window [P_stable, P_gradual, P_nonlinear]. Flags are computed
        independently per the thesis rules (not forced mutually exclusive), so
        rows need not sum to 1 — mirrors the existing Bayesian-backend convention
        in `traj_features.backends.bayes.classify`. Failed draws and NaN padding
        are excluded from both the per-draw statistics and the final denominator
        (mirrors `bootstrap_trajectory_probs`, which normalizes by the count of
        successfully-classified draws, not the raw draw count).
    """
    slopes = np.asarray(slopes, dtype=float)
    if slopes.ndim != 3:
        raise ValueError(
            f"slopes must have shape (n_windows, n_draws, n_local_slopes), got {slopes.shape}"
        )
    if direction not in ("increase", "decrease"):
        raise ValueError(f"direction must be 'increase' or 'decrease', got {direction!r}")

    valid_mask = ~np.isnan(slopes)            # (N, B, K)
    kv = valid_mask.sum(axis=-1)               # (N, B) — valid local-slope count per draw
    draw_valid = kv > 0                        # (N, B) — did this draw's fit succeed at all

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # nanmean over an all-NaN slice (a fully-failed draw, or a window with no
        # valid draws at all) is expected here and handled explicitly below via
        # draw_valid/n_valid_draws masking — silence numpy's advisory warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if direction == "increase":
            flat_cond = slopes <= tau_decline
            total_flat = np.abs(np.nanmean(slopes, axis=-1)) <= abs(tau_flat)  # |mean slope| <= |tau_flat|
            decl_cond = slopes > tau_decline
        else:
            flat_cond = slopes >= tau_decline
            total_flat = np.abs(np.nanmean(slopes, axis=-1)) <= abs(tau_flat)  # |mean slope| <= |tau_flat|
            decl_cond = slopes < tau_decline

        def _nan_frac(cond: np.ndarray) -> np.ndarray:
            # Boolean comparisons against NaN are always False, which would
            # silently count padding/failed entries as "not flat"/"not
            # declining" — convert to float and re-mask so nanmean correctly
            # ignores them instead.
            as_float = np.where(valid_mask, cond.astype(float), np.nan)
            return np.nanmean(as_float, axis=-1)

        frac_flat = _nan_frac(flat_cond)
        frac_decline = _nan_frac(decl_cond)

        # NaNs sort to the end, so the first kv entries of each row are the real,
        # ascending-sorted local slopes for that draw.
        sorted_slopes = np.sort(slopes, axis=-1)
        half = np.maximum(kv // 2, 1)
        idx = np.arange(slopes.shape[-1])
        slow_mask = idx < half[..., None]
        fast_mask = (idx >= half[..., None]) & (idx < kv[..., None])
        slow_half = np.nanmean(np.where(slow_mask, sorted_slopes, np.nan), axis=-1)
        fast_half = np.nanmean(np.where(fast_mask, sorted_slopes, np.nan), axis=-1)
        nonlinear = np.abs(fast_half - slow_half) > gamma_gap

    stable_flag = (frac_flat >= 0.8) & total_flat
    gradual_flag = frac_decline >= 0.8

    n_valid_draws = draw_valid.sum(axis=-1)  # (N,)

    def _valid_frac(flag: np.ndarray) -> np.ndarray:
        counted = np.where(draw_valid, flag, False).sum(axis=-1)
        return np.divide(counted, n_valid_draws, out=np.zeros(n_valid_draws.shape), where=n_valid_draws > 0)

    p_stable = _valid_frac(stable_flag)
    p_gradual = _valid_frac(gradual_flag)
    p_nonlinear = _valid_frac(nonlinear)

    return np.stack([p_stable, p_gradual, p_nonlinear], axis=-1)


def total_variation_distance(p: np.ndarray, p_baseline: np.ndarray) -> np.ndarray:
    """Per-window 0.5 * L1 distance between two archetype-probability vectors."""
    return 0.5 * np.abs(p - p_baseline).sum(axis=-1)


def primary_archetype_flip_rate(p: np.ndarray, p_baseline: np.ndarray) -> float:
    """% of windows whose argmax (dominant) archetype differs from baseline."""
    return float(np.mean(np.argmax(p, axis=-1) != np.argmax(p_baseline, axis=-1)) * 100.0)


# ============================================================================
# Grid search + identical downstream pipeline
# ============================================================================


def build_param_grid(
    baseline: dict[str, float],
    perturb_factors: tuple[float, ...] = (0.5, 0.75, 1.25, 1.5),
) -> list[dict]:
    """One-at-a-time perturbations around `baseline`, plus the baseline itself."""
    rows = [{"perturbed_param": "baseline", "perturbation_factor": 1.0, **baseline}]
    for param in baseline:
        for factor in perturb_factors:
            if factor == 1.0:
                continue
            combo = dict(baseline)
            combo[param] = baseline[param] * factor
            rows.append({"perturbed_param": param, "perturbation_factor": factor, **combo})
    return rows


def apply_trajectory_probs(
    dataset: pd.DataFrame,
    id_col: str,
    windowing_col: str,
    value_col: str,
    probs: np.ndarray,
    cache_stay_id: np.ndarray,
    cache_time_day: np.ndarray,
) -> pd.DataFrame:
    """Return a copy of `dataset` with the value_col's (Bayesian/primary) T columns
    replaced by `probs`. Bootstrap-fallback `_boot`-suffixed columns (added by
    `TrajectoryAnalysis.prepare_features`) are a separate feature family and left
    untouched.
    """
    stable_col, gradual_col, nonlinear_col = f"{value_col}_stable", f"{value_col}_gradual", f"{value_col}_rapid"
    probs_df = pd.DataFrame(
        {
            id_col: cache_stay_id,
            windowing_col: cache_time_day,
            stable_col: probs[:, 0],
            gradual_col: probs[:, 1],
            nonlinear_col: probs[:, 2],
        }
    )
    ds = dataset.drop(columns=[c for c in (stable_col, gradual_col, nonlinear_col) if c in dataset.columns])
    ds = ds.merge(probs_df, on=[id_col, windowing_col], how="left", validate="one_to_one")
    return ds


def run_identical_pipeline_cv(
    analysis: TrajectoryAnalysis,
    dataset: pd.DataFrame,
    feature_set_names: list[str],
    n_repeats: int,
    n_splits: int,
    n_jobs: int,
    backend: str,
    parallel_axis: str,
) -> list[dict]:
    """Run the *exact* eICU Sepsis analysis pipeline (`TrajectoryAnalysis.run_cv` —
    same feature-set construction incl. multi-marker single-vs-multi framing,
    imputation, scaling, model training as scripts/analysis/eicu/eicu_sepsis_analysis.py)
    and summarize fold-level metrics for every (model type, feature set) pair in
    `feature_set_names` x {LogisticRegression, RandomForest, HistGradientBoosting,
    XGBoost}.

    run_cv's cost is dominated by the ~27-feature-set x 4-model grid it always
    evaluates regardless of what we read back out (see run_sensitivity_grid.sh) —
    capturing multiple feature sets and all four models here is free: the same
    computation that used to be thrown away down to a single (model, feature_set)
    cell. One dict per (model, feature_set) pair is returned; the caller fans
    these into one CSV row each.
    """
    aggregated = analysis.run_cv(
        dataset, n_repeats=n_repeats, n_splits=n_splits, n_jobs=n_jobs, backend=backend,
        parallel_axis=parallel_axis,
    )

    fold_df = pd.DataFrame(analysis._last_fold_rows)
    out = []
    for model_name in sorted(aggregated.keys()):
        for feature_set_name in feature_set_names:
            if feature_set_name not in aggregated[model_name]:
                available = sorted(aggregated[model_name].keys())
                raise KeyError(
                    f"(model='{model_name}', feature_set='{feature_set_name}') not found in run_cv output. "
                    f"Available feature sets for '{model_name}': {available}"
                )
            sub = fold_df[(fold_df["model"] == model_name) & (fold_df["feature_set"] == feature_set_name)]
            auprs = sub["aupr"].to_numpy()
            aurocs = sub["auroc"].to_numpy()
            out.append({
                "model_name": model_name,
                "feature_set": feature_set_name,
                "median_aupr": float(np.median(auprs)),
                "aupr_ci_low": float(np.percentile(auprs, 2.5)),
                "aupr_ci_high": float(np.percentile(auprs, 97.5)),
                "mean_auroc": float(np.mean(aurocs)),
                "auroc_ci_low": float(np.percentile(aurocs, 2.5)),
                "auroc_ci_high": float(np.percentile(aurocs, 97.5)),
                "n_folds": int(len(sub)),
            })
    return out


def evaluate_sensitivity(
    analysis: TrajectoryAnalysis,
    dataset: pd.DataFrame,
    id_col: str,
    windowing_col: str,
    value_col: str,
    direction: str,
    slopes: np.ndarray,
    cache_stay_id: np.ndarray,
    cache_time_day: np.ndarray,
    param_grid: list[dict],
    feature_set_names: list[str],
    n_splits: int,
    n_repeats: int,
    cv_n_jobs: int,
    cv_backend: str,
    cv_parallel_axis: str,
) -> pd.DataFrame:
    baseline_params = {k: v for k, v in BIOMARKERS[value_col_to_marker(value_col)].items()
                        if k in ("tau_flat", "tau_decline", "gamma_gap")}
    baseline_probs = phi_sensitivity(slopes, direction=direction, **baseline_params)

    rows = []
    for combo in param_grid:
        params = {k: combo[k] for k in ("tau_flat", "tau_decline", "gamma_gap")}
        probs = phi_sensitivity(slopes, direction=direction, **params)

        mean_tvd = float(total_variation_distance(probs, baseline_probs).mean())
        flip_pct = primary_archetype_flip_rate(probs, baseline_probs)

        ds = apply_trajectory_probs(dataset, id_col, windowing_col, value_col, probs, cache_stay_id, cache_time_day)

        per_model_cv = run_identical_pipeline_cv(
            analysis, ds, feature_set_names, n_repeats, n_splits, cv_n_jobs, cv_backend, cv_parallel_axis
        )

        shared = {
            **params,
            "perturbed_param": combo["perturbed_param"],
            "perturbation_factor": combo["perturbation_factor"],
            "mean_tvd_vs_baseline": mean_tvd,
            "pct_primary_archetype_flips": flip_pct,
            "n_windows": int(len(probs)),
        }
        for cv in per_model_cv:
            rows.append({**shared, **cv})

        aupr_summary = ", ".join(
            f"{cv['model_name']}/{cv['feature_set']}={cv['median_aupr']:.4f}" for cv in per_model_cv
        )
        logger.info(
            f"   [{combo['perturbed_param']:>11s} x{combo['perturbation_factor']:.2f}] "
            f"tau_flat={params['tau_flat']:.3f} tau_decline={params['tau_decline']:.3f} "
            f"gamma_gap={params['gamma_gap']:.3f} | TVD={mean_tvd:.4f} flips={flip_pct:.1f}% "
            f"AUPR[{aupr_summary}]"
        )

    return pd.DataFrame(rows)


def value_col_to_marker(value_col: str) -> str:
    for marker, cfg in BIOMARKERS.items():
        if cfg["value_col"] == value_col:
            return marker
    raise ValueError(f"No BIOMARKERS entry with value_col={value_col!r}")


# ============================================================================
# Output
# ============================================================================


def select_model(df: pd.DataFrame, model_name: str, feature_set_name: str | None = None) -> pd.DataFrame:
    """Schema-tolerant model/feature-set filter. Pre-multi-model-capture CSVs
    have no `model_name` column (implicitly single-model, historically
    XGBoost) and pre-multi-feature-set-capture CSVs have no `feature_set`
    column (implicitly the one reported feature set) — pass those through
    unchanged rather than erroring, so old and new CSVs plot the same way.
    `feature_set_name` defaults to DEFAULT_FEATURE_SET_NAME so callers that
    only care about the headline feature set don't need to specify it."""
    out = df
    if "model_name" in out.columns:
        out = out.loc[out["model_name"] == model_name]
    if "feature_set" in out.columns:
        fs = feature_set_name if feature_set_name is not None else DEFAULT_FEATURE_SET_NAME
        out = out.loc[out["feature_set"] == fs]
    return out.reset_index(drop=True)


def draw_marker_panels(
    ax_flip, ax_delta, results_df: pd.DataFrame, model_name: str, feature_set_name: str | None = None,
) -> None:
    """Draw the sensitivity view — primary-archetype flip rate onto `ax_flip`,
    and (if `ax_delta` is not None) downstream AUPR change vs. the reported
    baseline onto `ax_delta` — for one marker onto pre-existing axes. Shared
    by `plot_sensitivity_overview` (one marker, both panels, called
    automatically at the end of each run) and `plot_sensitivity_combined.py`
    (all three markers side by side; the manuscript figure only wants the
    flip-rate panel, so it passes `ax_delta=None`). Full explanation of what
    "flip rate" and "Δ AUPR" mean lives in SENSITIVITY_ANALYSIS_METHODS.md and
    FIGURE_CAPTIONS.md, not in the figure itself."""
    df = select_model(results_df, model_name, feature_set_name)
    baseline_row = df.loc[df["perturbed_param"] == "baseline"].iloc[0]
    base_aupr = float(baseline_row["median_aupr"])
    noise_half = (float(baseline_row["aupr_ci_high"]) - float(baseline_row["aupr_ci_low"])) / 2

    for param, color in PARAM_COLORS.items():
        sub = df.loc[df["perturbed_param"] == param]
        sub = pd.concat([sub, baseline_row.to_frame().T], ignore_index=True)
        sub = sub.astype({"perturbation_factor": float, "pct_primary_archetype_flips": float,
                           "median_aupr": float, "aupr_ci_low": float, "aupr_ci_high": float}
                          ).sort_values("perturbation_factor")

        ax_flip.plot(sub["perturbation_factor"], sub["pct_primary_archetype_flips"], marker="o",
                     markersize=6, linewidth=2, color=color, label=param)

        if ax_delta is not None:
            delta = sub["median_aupr"] - base_aupr
            half_ci = (sub["aupr_ci_high"] - sub["aupr_ci_low"]) / 2
            ax_delta.errorbar(sub["perturbation_factor"], delta, yerr=half_ci, fmt="o-", markersize=6,
                              linewidth=2, capsize=3, color=color, ecolor=color, elinewidth=1)

    ax_flip.axvline(1.0, linestyle="--", color="#8a8a86", linewidth=1, alpha=0.8)
    ax_flip.set_ylim(bottom=0)
    ax_flip.set_ylabel("% windows relabeled\n(primary archetype flip)")
    if ax_delta is None:
        # ax_flip is the sole panel (plot_sensitivity_combined's flip-only
        # manuscript figure) — otherwise (plot_sensitivity_overview's stacked
        # sharex pair) only the bottom panel gets an x-label.
        ax_flip.set_xlabel("Perturbation factor (× baseline threshold)")

    if ax_delta is not None:
        ax_delta.axhspan(-noise_half, noise_half, color="#c9ccd1", alpha=0.35, linewidth=0, zorder=0)
        ax_delta.axhline(0.0, color="#8a8a86", linewidth=1, alpha=0.8)
        ax_delta.axvline(1.0, linestyle="--", color="#8a8a86", linewidth=1, alpha=0.8)
        ax_delta.set_ylabel("Δ AUPR vs. baseline\n(×1.0 threshold)")
        ax_delta.set_xlabel("Perturbation factor (× baseline threshold)")


def plot_sensitivity_overview(
    results_df: pd.DataFrame, out_path: Path, marker: str, feature_set_name: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    """Two-panel, single-marker sensitivity figure. Diagnostic only (not a
    manuscript figure — see plot_sensitivity_combined.py for that), so it
    carries just a short title, axis labels, and a legend; full explanation
    is in SENSITIVITY_ANALYSIS_METHODS.md."""
    import matplotlib.pyplot as plt

    set_poster_style()
    fig, (ax_flip, ax_delta) = plt.subplots(
        2, 1, figsize=(7.5, 6.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.12}
    )

    draw_marker_panels(ax_flip, ax_delta, results_df, model_name)

    handles, labels = ax_flip.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.95),
               ncol=3, frameon=False, fontsize=9, handlelength=1.4, columnspacing=1.1)

    fig.suptitle(f"eICU septic shock ({marker}): archetype-threshold sensitivity",
                 fontsize=14, y=0.995, x=0.02, ha="left")

    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.145, right=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for eICU Septic Shock trajectory-archetype thresholds"
    )
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--marker", type=str, default="lactate", choices=list(BIOMARKERS.keys()) + ["all"],
                         help="Biomarker whose archetype thresholds are perturbed. 'all' builds/merges the "
                              "cache for every marker in one invocation (only valid with --build-cache-only "
                              "or --merge-cohort-caches — the grid-search/CV stage is inherently single-marker).")
    parser.add_argument("--backend", type=str, default="bootstrap", choices=["bootstrap", "bayes"],
                         help="Slope-cache backend. 'bayes' needs g++ (udocker container) for tractable runtime.")
    parser.add_argument("--window", type=float, default=None,
                         help="Lookback window in days (default: eICUSepsisConfig.LOOKBACK_WINDOW).")

    # Bootstrap-backend cache options
    parser.add_argument("--n-bootstrap", type=int, default=200)

    # Bayesian-backend cache options (defaults match scripts/trajectory/eicu/sepsis_all_biomarkers.py)
    parser.add_argument("--bayes-n-samples", type=int, default=200)
    parser.add_argument("--bayes-tune", type=int, default=300)
    parser.add_argument("--bayes-df-basis", type=int, default=5)
    parser.add_argument("--bayes-chains", type=int, default=4)
    parser.add_argument("--bayes-cores", type=int, default=1)
    parser.add_argument("--bayes-chain-method", type=str, default="vectorized", choices=["vectorized", "parallel"])
    parser.add_argument("--bayes-target-accept", type=float, default=0.99)
    parser.add_argument("--bayes-sampler", type=str, default="pymc", choices=["pymc", "nutpie", "numpyro"],
                         help="'pymc' is the g++-accelerated path inside the udocker container.")
    parser.add_argument("--bayes-batch-size", type=int, default=200)

    # Shared cache options
    parser.add_argument("--grid-freq", type=int, default=None,
                         help="Default: 12 for bootstrap (literal grid points), 2 for bayes (points/day rate).")
    parser.add_argument("--min-points-per-window", type=int, default=None,
                         help="Default: 5 for bootstrap, 4 for bayes.")
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--cache-n-jobs", type=int, default=4)

    # Cohort splitting for the (expensive) Bayesian cache build — mirrors
    # sepsis_all_biomarkers.py's --cohort-splits/--cohort-index.
    parser.add_argument("--cohort-splits", type=int, default=1)
    parser.add_argument("--cohort-index", type=int, default=0)
    parser.add_argument("--build-cache-only", action="store_true",
                         help="Build (this cohort shard's) slope cache and exit, without preparing "
                              "features or running any CV. Use for parallel container/SLURM cache builds.")
    parser.add_argument("--merge-cohort-caches", action="store_true",
                         help="Merge existing {cache}_cohortNN.npz shards into the final cache and exit.")

    # Downstream CV (identical eICU Sepsis pipeline)
    parser.add_argument("--feature-set-name", type=str, default=DEFAULT_FEATURE_SET_NAME,
                         help="Headline feature set — used for the auto-generated plot and always captured.")
    parser.add_argument("--extra-feature-set-names", type=str, nargs="*",
                         default=[TRAJECTORY_ABLATED_FEATURE_SET_NAME],
                         help="Additional feature sets to capture alongside --feature-set-name (all free: "
                              "run_cv evaluates every feature set at every grid point regardless). Default "
                              "captures the trajectory-ablated counterpart, to test whether the trajectory "
                              "features' contribution survives the threshold sweep.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                         help="Headline model for the auto-generated plot only — the CSV always captures "
                              "all 4 model types run_cv trains (LogisticRegression, RandomForest, "
                              "HistGradientBoosting, XGBoost), regardless of this flag.")
    parser.add_argument("--cv-splits", type=int, default=None)
    parser.add_argument("--cv-repeats", type=int, default=None)
    parser.add_argument("--train-n-jobs", type=int, default=1)
    parser.add_argument("--train-backend", type=str, default="threading",
                         choices=["threading", "loky", "multiprocessing", "sequential"])
    parser.add_argument("--parallel-axis", type=str, default="repeat", choices=["repeat", "feature-set"])

    parser.add_argument("--perturb-factors", type=float, nargs="+", default=[0.5, 0.75, 1.25, 1.5])
    args = parser.parse_args()

    if args.marker == "all" and not (args.build_cache_only or args.merge_cohort_caches):
        parser.error(
            "--marker all is only valid together with --build-cache-only or --merge-cohort-caches "
            "(the grid-search/CV stage is inherently single-marker)."
        )
    markers_to_process = list(BIOMARKERS.keys()) if args.marker == "all" else [args.marker]

    dataset_config = get_dataset_config(DATASET)
    base_dir = Path(args.base_dir) if args.base_dir else dataset_config["base_path"] / TASK
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / DATASET / TASK
    output_dir.mkdir(parents=True, exist_ok=True)

    config = eICUSepsisConfig()
    if args.cv_repeats is not None:
        config.CV_N_REPEATS = args.cv_repeats
    if args.cv_splits is not None:
        config.CV_N_SPLITS = args.cv_splits
    # Grid-search efficiency only — does not change how any fold is trained/scored.
    config.SAVE_OOF = False
    config.SAVE_NOTEBOOK_STYLE_PLOTS = False

    analysis = TrajectoryAnalysis(config, base_dir, output_dir)

    def _cache_path_for(value_col: str) -> Path:
        # --cache-path only makes sense pinned to a single marker; with --marker all
        # each marker always gets its own default-named cache file.
        if args.cache_path and args.marker != "all":
            return Path(args.cache_path)
        return base_dir / f"{value_col}_{args.backend}_slope_samples_cache.npz"

    if args.merge_cohort_caches:
        for marker in markers_to_process:
            logger.info(f"── {marker} ──")
            merge_cohort_slope_caches(_cache_path_for(BIOMARKERS[marker]["value_col"]))
        return

    logger.info("1. Loading base dataset (load_data only; prepare_features deferred until after cache build)...")
    dataset = analysis.load_data()
    id_col = analysis.id_col
    _, windowing_col = pick_time_cols(dataset)

    window = args.window if args.window is not None else config.LOOKBACK_WINDOW

    caches: dict[str, dict] = {}
    for marker in markers_to_process:
        value_col = BIOMARKERS[marker]["value_col"]
        # BIOMARKERS["file"] (e.g. "platelets_timeseries.csv") is the source of truth for
        # the timeseries filename stem — it does NOT always match f"{value_col}_timeseries"
        # (value_col is singular "platelet", the file is plural "platelets_timeseries.csv").
        ts_stem = Path(BIOMARKERS[marker]["file"]).stem

        logger.info(
            f"2. Loading/building cached inter-point slope samples (marker={marker}, backend={args.backend})..."
        )
        ts_df = load_named_table(base_dir, ts_stem, verbose=True)
        ts_time_col, ts_windowing_col = pick_time_cols(ts_df)

        cache_path = _cache_path_for(value_col)
        effective_cache_path = cache_path
        if args.cohort_splits > 1:
            effective_cache_path = cache_path.with_name(
                f"{cache_path.stem}_cohort{args.cohort_index:02d}{cache_path.suffix}"
            )
            all_patients = dataset[id_col].unique()
            total = len(all_patients)
            cohort_size = total // args.cohort_splits
            start = args.cohort_index * cohort_size
            end = total if args.cohort_index == args.cohort_splits - 1 else start + cohort_size
            cohort_patients = set(all_patients[start:end])
            ts_df = ts_df[ts_df[id_col].isin(cohort_patients)]
            logger.info(
                f"   Cohort {args.cohort_index + 1}/{args.cohort_splits}: {len(cohort_patients):,} patients, "
                f"{len(ts_df):,} timeseries rows"
            )

        if args.backend == "bootstrap":
            grid_freq = args.grid_freq if args.grid_freq is not None else 12
            min_points = args.min_points_per_window if args.min_points_per_window is not None else 5
            cache = build_or_load_slope_cache(
                "bootstrap",
                ts_df=ts_df, id_col=id_col, time_col=ts_time_col, windowing_col=ts_windowing_col,
                value_col=value_col, cache_path=effective_cache_path, window=window,
                n_bootstrap=args.n_bootstrap, grid_freq=grid_freq, min_points_per_window=min_points,
                n_jobs=args.cache_n_jobs, force_rebuild=args.force_rebuild_cache,
            )
        else:
            grid_freq = args.grid_freq if args.grid_freq is not None else 2
            min_points = args.min_points_per_window if args.min_points_per_window is not None else 4
            cache = build_or_load_slope_cache(
                "bayes",
                ts_df=ts_df, id_col=id_col, time_col=ts_time_col, windowing_col=ts_windowing_col,
                value_col=value_col, cache_path=effective_cache_path, window=window,
                n_samples=args.bayes_n_samples, tune=args.bayes_tune, df_basis=args.bayes_df_basis,
                grid_freq=grid_freq, min_points_per_window=min_points, n_jobs=args.cache_n_jobs,
                batch_size=args.bayes_batch_size, sampler=args.bayes_sampler, chains=args.bayes_chains,
                cores=args.bayes_cores, chain_method=args.bayes_chain_method,
                target_accept=args.bayes_target_accept, force_rebuild=args.force_rebuild_cache,
            )
        caches[marker] = cache

    if args.build_cache_only:
        logger.info(
            "--build-cache-only set: exiting after cache build. Once all cohort shards are done, run with "
            "--merge-cohort-caches, then a normal invocation to run the grid search."
        )
        return

    # markers_to_process has exactly one entry past this point (validated above).
    marker = markers_to_process[0]
    marker_cfg = BIOMARKERS[marker]
    value_col = marker_cfg["value_col"]
    direction = marker_cfg["direction"]
    cache = caches[marker]

    logger.info("3. Preparing features (identical to eicu_sepsis_analysis.py: summary stats + bootstrap fallback)...")
    dataset = analysis.prepare_features(dataset)

    baseline_params = {k: v for k, v in marker_cfg.items() if k in ("tau_flat", "tau_decline", "gamma_gap")}
    perturb_factors = tuple(args.perturb_factors)
    param_grid = build_param_grid(baseline_params, perturb_factors=perturb_factors)
    feature_set_names = [args.feature_set_name] + [
        fs for fs in args.extra_feature_set_names if fs != args.feature_set_name
    ]
    logger.info(
        f"4. Evaluating {len(param_grid)} parameter combinations for marker='{marker}' "
        f"(feature sets={feature_set_names}, all 4 model types captured; "
        f"'{args.model_name}' is the headline model for the plot)..."
    )

    results_df = evaluate_sensitivity(
        analysis=analysis,
        dataset=dataset,
        id_col=id_col,
        windowing_col=windowing_col,
        value_col=value_col,
        direction=direction,
        slopes=cache["slopes"],
        cache_stay_id=cache["stay_id"],
        cache_time_day=cache["time_day"],
        param_grid=param_grid,
        feature_set_names=feature_set_names,
        n_splits=config.CV_N_SPLITS,
        n_repeats=config.CV_N_REPEATS,
        cv_n_jobs=args.train_n_jobs,
        cv_backend=args.train_backend,
        cv_parallel_axis=args.parallel_axis,
    )

    csv_path = output_dir / "sensitivity_results_eicu_sepsis.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"5. Wrote {csv_path}")

    plot_path = output_dir / "sensitivity_aupr_stability_eicu_sepsis.png"
    plot_sensitivity_overview(
        results_df, plot_path, marker=marker, feature_set_name=args.feature_set_name, model_name=args.model_name
    )
    logger.info(f"6. Wrote {plot_path}")

    logger.info("\n✓ Sensitivity analysis complete")


if __name__ == "__main__":
    main()
