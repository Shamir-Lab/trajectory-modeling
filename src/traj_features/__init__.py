"""
Trajectory Features: Extract temporal biomarker trajectories for clinical prediction.

Backends:
- Bayesian: MCMC-based posterior sampling (precise, g++-accelerated via PyTensor)
- Bootstrap: Fast resampling with splines (10-100x faster)

Usage:
    from traj_features.backends.bootstrap import BootstrapTraj, BootstrapConfig
    from traj_features.backends.bayes import BayesianTraj, BayesConfig
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.3.0"

if TYPE_CHECKING:
    from .backends.bayes import BayesianTraj, BayesConfig
    from .backends.bootstrap import BootstrapTraj, BootstrapConfig

__all__ = [
    "BayesianTraj",
    "BayesConfig",
    "BootstrapTraj",
    "BootstrapConfig",
    # Backward-compat aliases
    "BayesianTrajPS",
    "BootstrapTrajPS",
]


def __getattr__(name: str):
    if name in {"BayesianTraj", "BayesianTrajPS", "BayesConfig"}:
        from .backends.bayes import BayesianTraj, BayesianTrajPS, BayesConfig
        return {"BayesianTraj": BayesianTraj, "BayesianTrajPS": BayesianTrajPS, "BayesConfig": BayesConfig}[name]
    if name in {"BootstrapTraj", "BootstrapTrajPS", "BootstrapConfig"}:
        from .backends.bootstrap import BootstrapTraj, BootstrapTrajPS, BootstrapConfig
        return {"BootstrapTraj": BootstrapTraj, "BootstrapTrajPS": BootstrapTrajPS, "BootstrapConfig": BootstrapConfig}[name]
    raise AttributeError(name)

# Backward compatibility: make traj_ps an alias to traj_features
import sys
sys.modules['traj_ps'] = sys.modules[__name__]
