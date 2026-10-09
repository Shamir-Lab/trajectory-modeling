"""
Smoke tests for traj_features backends.

Fast tests (no marker): import checks, config field checks, sampler path logic,
                        PS-removal checks, backward-compat alias checks.
Slow tests (@pytest.mark.slow): end-to-end MCMC runs on synthetic data.
GPU tests  (@pytest.mark.gpu):  numpyro/JAX path — requires JAX + GPU node.

Run fast only (default CI):
    pytest tests/test_smoke_backends.py

Run everything locally:
    pytest tests/test_smoke_backends.py -m "not gpu" -s

Run GPU tests on cluster:
    pytest tests/test_smoke_backends.py -m "gpu" -s
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


# ============================================================================
# Import checks
# ============================================================================

def test_bayes_imports():
    from traj_features.backends.bayes import BayesianTraj, BayesConfig
    from traj_features.backends.bayes.classify import flags_from_traj, pos_flags_from_traj
    from traj_features.backends.bayes.sampling import _sample_post_trajs_scaled
    from traj_features.backends.bayes.pipeline import compute_time_varying_trajectory_covariates_parallel


def test_bootstrap_imports():
    from traj_features.backends.bootstrap import BootstrapTraj, BootstrapConfig


def test_analysis_imports():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from analysis.analysis_template import AnalysisConfig, TrajectoryAnalysis
    from analysis.analysis_utils import pick_id_col, pick_time_cols


def test_eicu_loader_import_from_new_location():
    """eicu_loader.py was moved from src/ to scripts/eicu_preprocessing/.
    Verify it is importable from its new canonical location."""
    eicu_dir = REPO_ROOT / "scripts" / "eicu_preprocessing"
    assert (eicu_dir / "eicu_loader.py").exists(), \
        "eicu_loader.py not found at scripts/eicu_preprocessing/eicu_loader.py"

    sys.path.insert(0, str(eicu_dir))
    try:
        import importlib
        loader_mod = importlib.import_module("eicu_loader")
        assert hasattr(loader_mod, "EICULoader")
        EICULoader = loader_mod.EICULoader
        assert hasattr(EICULoader, "load_aki_cohort")
        assert hasattr(EICULoader, "load_labs")
        assert hasattr(EICULoader, "load_vitals")
    finally:
        sys.path.pop(0)
        if "eicu_loader" in sys.modules:
            del sys.modules["eicu_loader"]


def test_eicu_loader_not_in_src():
    """Confirm eicu_loader.py was removed from src/ (old location)."""
    assert not (REPO_ROOT / "src" / "eicu_loader.py").exists(), \
        "eicu_loader.py still exists in src/ — old location not cleaned up"


# ============================================================================
# PS removal checks
# ============================================================================

def test_bayesian_traj_has_no_fit_method():
    """fit() was part of the removed propensity score / Cox model workflow."""
    from traj_features.backends.bayes import BayesianTraj
    assert not hasattr(BayesianTraj, "fit"), \
        "BayesianTraj.fit() still exists — PS code not fully removed"


def test_bayesian_traj_has_no_ps_method():
    """ps() was part of the removed propensity score / Cox model workflow."""
    from traj_features.backends.bayes import BayesianTraj
    assert not hasattr(BayesianTraj, "ps"), \
        "BayesianTraj.ps() still exists — PS code not fully removed"


def test_bayesian_traj_has_no_save_method():
    """save() was part of the removed propensity score / Cox model workflow."""
    from traj_features.backends.bayes import BayesianTraj
    assert not hasattr(BayesianTraj, "save"), \
        "BayesianTraj.save() still exists — PS code not fully removed"


def test_bayesian_traj_has_no_load_method():
    """load() was part of the removed propensity score / Cox model workflow."""
    from traj_features.backends.bayes import BayesianTraj
    assert not hasattr(BayesianTraj, "load"), \
        "BayesianTraj.load() still exists — PS code not fully removed"


def test_bootstrap_traj_has_no_fit_method():
    """fit() was part of the removed propensity score / Cox model workflow."""
    from traj_features.backends.bootstrap import BootstrapTraj
    assert not hasattr(BootstrapTraj, "fit"), \
        "BootstrapTraj.fit() still exists — PS code not fully removed"


def test_no_lifelines_import():
    """lifelines (survival analysis) was only used by the removed Cox model code."""
    import traj_features.backends.bayes.model  # noqa: F401
    import traj_features.backends.bootstrap.model  # noqa: F401

    lifelines_modules = [m for m in sys.modules if "lifelines" in m]
    assert not lifelines_modules, \
        f"lifelines was imported as a side-effect of loading traj_features: {lifelines_modules}"


# ============================================================================
# Backward-compat alias checks
# ============================================================================

def test_backward_compat_alias_bayes():
    """BayesianTrajPS must be an alias for BayesianTraj (not a separate class)."""
    from traj_features.backends.bayes import BayesianTraj, BayesianTrajPS
    assert BayesianTrajPS is BayesianTraj, \
        "BayesianTrajPS is not the same object as BayesianTraj"


def test_backward_compat_alias_bootstrap():
    """BootstrapTrajPS must be an alias for BootstrapTraj (not a separate class)."""
    from traj_features.backends.bootstrap import BootstrapTraj, BootstrapTrajPS
    assert BootstrapTrajPS is BootstrapTraj, \
        "BootstrapTrajPS is not the same object as BootstrapTraj"


def test_backward_compat_alias_from_top_level():
    """Both old and new names must be importable from the top-level traj_features package."""
    import traj_features
    BayesianTraj = traj_features.BayesianTraj
    BayesianTrajPS = traj_features.BayesianTrajPS
    BootstrapTraj = traj_features.BootstrapTraj
    BootstrapTrajPS = traj_features.BootstrapTrajPS
    assert BayesianTrajPS is BayesianTraj
    assert BootstrapTrajPS is BootstrapTraj


# ============================================================================
# BayesConfig field checks
# ============================================================================

def test_bayesconfig_has_batch_size():
    from traj_features.backends.bayes import BayesConfig
    cfg = BayesConfig()
    assert hasattr(cfg, "batch_size"), "batch_size missing from BayesConfig"
    assert cfg.batch_size == 500


def test_bayesconfig_has_chain_method():
    from traj_features.backends.bayes import BayesConfig
    cfg = BayesConfig()
    assert hasattr(cfg, "chain_method"), "chain_method missing from BayesConfig"
    assert cfg.chain_method == "vectorized"


def test_bayesconfig_custom_values_roundtrip():
    from traj_features.backends.bayes import BayesConfig
    cfg = BayesConfig(batch_size=100, chain_method="parallel", n_samples=50, tune=50)
    assert cfg.batch_size == 100
    assert cfg.chain_method == "parallel"
    assert cfg.n_samples == 50


def test_bayesconfig_default_sampler_is_pymc():
    """Default sampler must be 'pymc' — ensures container default is correct."""
    from traj_features.backends.bayes import BayesConfig
    cfg = BayesConfig()
    assert cfg.sampler == "pymc", \
        f"Default sampler is '{cfg.sampler}', expected 'pymc'"


def test_bayesconfig_default_use_gpu_is_false():
    """Default use_gpu must be False — container is CPU-only."""
    from traj_features.backends.bayes import BayesConfig
    cfg = BayesConfig()
    assert cfg.use_gpu is False


def test_batch_size_and_chain_method_wired_through_embed(monkeypatch):
    """Verify BayesianTraj.embed() passes batch_size and chain_method to the pipeline."""
    from traj_features.backends.bayes import BayesConfig, BayesianTraj
    import traj_features.backends.bayes.model as model_mod

    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(model_mod, "compute_time_varying_trajectory_covariates_parallel", fake_pipeline)

    cfg = BayesConfig(
        batch_size=123, chain_method="parallel",
        sampler="pymc", use_gpu=False, n_jobs=1,
    )
    model = BayesianTraj(cfg=cfg)
    model.embed(pd.DataFrame({"patientid": [], "time": [], "lab_value": []}))

    assert captured.get("batch_size") == 123, \
        f"batch_size not passed through embed(); got {captured.get('batch_size')}"
    assert captured.get("chain_method") == "parallel", \
        f"chain_method not passed through embed(); got {captured.get('chain_method')}"


# ============================================================================
# Sampler auto-detection logic
# ============================================================================

def test_explicit_sampler_not_overridden_nutpie():
    """If user sets sampler='nutpie' explicitly, auto-detection must not change it."""
    from traj_features.backends.bayes import BayesConfig, BayesianTraj
    cfg = BayesConfig(sampler="nutpie", use_gpu=False)
    model = BayesianTraj(cfg=cfg)
    assert model.cfg.sampler == "nutpie"


def test_explicit_sampler_not_overridden_numpyro():
    """If user sets sampler='numpyro' explicitly, auto-detection must not change it."""
    from traj_features.backends.bayes import BayesConfig, BayesianTraj
    cfg = BayesConfig(sampler="numpyro", use_gpu=True)
    model = BayesianTraj(cfg=cfg)
    assert model.cfg.sampler == "numpyro"


def test_auto_detection_cpu_stays_pymc():
    """With use_gpu=False and sampler='pymc', auto-detection must leave sampler as 'pymc'.

    Previously auto-selected nutpie; that logic was removed. CPU path now always
    stays on pymc (which is accelerated by g++ via PyTensor when g++ is available).
    """
    from traj_features.backends.bayes import BayesConfig, BayesianTraj
    cfg = BayesConfig(sampler="pymc", use_gpu=False)
    model = BayesianTraj(cfg=cfg)
    assert model.cfg.sampler == "pymc", \
        f"CPU auto-detection changed sampler to '{model.cfg.sampler}', expected 'pymc'"


def test_gpu_flag_attempts_numpyro_upgrade(monkeypatch):
    """With use_gpu=True and sampler='pymc', auto-detection should attempt numpyro."""
    import types
    import builtins

    fake_jax = types.SimpleNamespace(
        sample_numpyro_nuts=lambda **kw: None,
    )
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pymc.sampling.jax":
            return fake_jax
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from traj_features.backends.bayes import BayesConfig
    # Re-import to pick up the monkeypatched import
    import importlib
    import traj_features.backends.bayes.model as m
    importlib.reload(m)
    BayesianTraj = m.BayesianTraj

    cfg = BayesConfig(sampler="pymc", use_gpu=True)
    model = BayesianTraj(cfg=cfg)
    assert model.cfg.sampler == "numpyro", \
        f"GPU auto-detection did not select numpyro; got '{model.cfg.sampler}'"


# ============================================================================
# Sampling fix: JAX trace not overwritten by CPU pm.sample()
# ============================================================================

def test_jax_trace_not_overwritten_by_cpu(monkeypatch):
    """
    Core regression test for the JAX bug fix.

    Simulate a successful numpyro call by patching pymc.sampling.jax,
    then verify pm.sample() was NOT called (which would overwrite the trace).
    """
    import types
    import traj_features.backends.bayes.sampling as sampling_mod
    import pymc as pm

    sentinel = object()  # unique object that only numpyro "returns"
    cpu_sample_called = []

    fake_sj = types.SimpleNamespace(
        sample_numpyro_nuts=lambda **kw: sentinel,
        sample_blackjax_nuts=lambda **kw: sentinel,
    )

    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pymc.sampling.jax":
            return fake_sj
        return original_import(name, *args, **kwargs)

    original_pm_sample = pm.sample

    def spy_pm_sample(*args, **kwargs):
        cpu_sample_called.append(True)
        return original_pm_sample(*args, **kwargs)

    monkeypatch.setattr(pm, "sample", spy_pm_sample)
    monkeypatch.setattr(builtins, "__import__", mock_import)

    df = _make_synthetic_window(n_points=8)

    try:
        from traj_features.backends.bayes.sampling import _sample_post_trajs_scaled
        try:
            _sample_post_trajs_scaled(
                df, df_basis=4, n_samples=5, tune=5, min_points=5,
                grid_freq=3, target_accept=0.9,
                values="lab_value", time_col="time_days",
                sampler="numpyro", chains=1, cores=1, progressbar=False,
                chain_method="vectorized",
            )
        except Exception:
            pass  # expected — sentinel is not a real ArviZ trace
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import)

    assert not cpu_sample_called, \
        "pm.sample() was called after a successful JAX sample — JAX trace was overwritten!"


# ============================================================================
# Helpers for end-to-end tests
# ============================================================================

def _make_synthetic_window(n_points: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    times = np.linspace(0, 7, n_points)
    values = 1.0 + 0.05 * times + rng.randn(n_points) * 0.1
    return pd.DataFrame({
        "patientid": [0] * n_points,
        "time_days": times,
        "time_day": times.astype(int),
        "lab_value": np.clip(values, 0.2, None),
    })


def _make_synthetic_cohort(n_patients: int = 4, n_timepoints: int = 8, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for pid in range(n_patients):
        times = np.linspace(0, 7, n_timepoints)
        values = 1.0 + 0.05 * times + rng.randn(n_timepoints) * 0.1
        for t, v in zip(times, values):
            rows.append({
                "patientid": pid,
                "time_days": round(float(t), 4),
                "time_day": int(t),
                "lab_value": max(0.2, float(v)),
            })
    return pd.DataFrame(rows)


# ============================================================================
# End-to-end CPU smoke test
# ============================================================================

@pytest.mark.slow
def test_bayesian_embed_cpu_pymc():
    """Full embed() call on synthetic data, forcing CPU PyMC (sampler='pymc', use_gpu=False)."""
    from traj_features.backends.bayes import BayesConfig, BayesianTraj
    from traj_features.backends.bayes.classify import pos_flags_from_traj

    df = _make_synthetic_cohort(n_patients=3, n_timepoints=8)

    cfg = BayesConfig(
        window_years=7.0,
        df_basis=4,
        n_samples=10,
        tune=10,
        chains=1,
        cores=1,
        min_points_per_window=5,
        grid_freq=3,
        pids="patientid",
        values="lab_value",
        time_col="time_days",
        windowing_col="time_day",
        sampler="pymc",
        use_gpu=False,
        batch_size=50,
        n_jobs=1,
        progressbar=False,
        flat_thr=0.1,
        decline_thr=0.3,
        nonlinear_gap=0.5,
        class_func=pos_flags_from_traj,
        traj_types=("stable", "gradual_increase", "rapid_increase"),
        label_map={
            "nonprogression": "stable",
            "linear": "gradual_increase",
            "nonlinear": "rapid_increase",
        },
    )

    model = BayesianTraj(cfg=cfg)
    model.cfg.sampler = "pymc"  # force pymc even if nutpie is installed

    t0 = time.time()
    result = model.embed(df)
    elapsed = time.time() - t0

    print(f"\n[CPU PyMC smoke] {len(result)} rows in {elapsed:.1f}s")

    assert not result.empty, "embed() returned empty DataFrame"
    assert "patientid" in result.columns
    assert "time_day" in result.columns

    prob_cols = [c for c in result.columns if c.startswith("trajtype_")]
    assert len(prob_cols) == 3, f"Expected 3 probability columns, got: {prob_cols}"

    for col in prob_cols:
        assert result[col].between(0.0, 1.0).all(), f"{col} contains values outside [0, 1]"


@pytest.mark.slow
@pytest.mark.gpu
def test_bayesian_embed_numpyro():
    """
    GPU smoke test: embed() with sampler='numpyro'.

    This is the key regression test for the JAX bug fix:
    - Before fix: numpyro ran, then pm.sample() ran and overwrote it (double compute).
    - After fix:  only numpyro runs.

    Expected on a GPU node: completes faster than an equivalent CPU run.
    Skipped automatically if JAX or pymc.sampling.jax are unavailable.
    """
    pytest.importorskip("jax", reason="JAX not installed — skipping numpyro test")
    try:
        import pymc.sampling.jax  # noqa: F401
    except ImportError:
        pytest.skip("pymc.sampling.jax not available")

    from traj_features.backends.bayes import BayesConfig, BayesianTraj
    from traj_features.backends.bayes.classify import pos_flags_from_traj

    df = _make_synthetic_cohort(n_patients=3, n_timepoints=8)

    cfg = BayesConfig(
        window_years=7.0,
        df_basis=4,
        n_samples=20,
        tune=20,
        chains=2,
        cores=1,
        min_points_per_window=5,
        grid_freq=3,
        pids="patientid",
        values="lab_value",
        time_col="time_days",
        windowing_col="time_day",
        sampler="numpyro",
        use_gpu=True,
        batch_size=10,
        n_jobs=1,
        progressbar=False,
        flat_thr=0.1,
        decline_thr=0.3,
        nonlinear_gap=0.5,
        class_func=pos_flags_from_traj,
        traj_types=("stable", "gradual_increase", "rapid_increase"),
        label_map={
            "nonprogression": "stable",
            "linear": "gradual_increase",
            "nonlinear": "rapid_increase",
        },
    )

    model = BayesianTraj(cfg=cfg)
    assert model.cfg.sampler == "numpyro", \
        f"sampler was changed from 'numpyro' to '{model.cfg.sampler}' — auto-detection override bug"

    t0 = time.time()
    result = model.embed(df)
    elapsed = time.time() - t0

    print(f"\n[numpyro smoke] {len(result)} rows in {elapsed:.1f}s")

    assert not result.empty
    prob_cols = [c for c in result.columns if c.startswith("trajtype_")]
    assert len(prob_cols) == 3
    for col in prob_cols:
        assert result[col].between(0.0, 1.0).all()
