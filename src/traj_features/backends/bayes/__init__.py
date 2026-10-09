"""Bayesian backend package.

Keep imports lazy so loading sibling modules (e.g. ``bayes.classify``)
does not eagerly import PyMC/PyTensor-heavy code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .model import BayesianTraj, BayesConfig

__all__ = ["BayesianTraj", "BayesConfig", "BayesianTrajPS"]


def __getattr__(name: str):
	if name in {"BayesianTraj", "BayesianTrajPS", "BayesConfig"}:
		from .model import BayesianTraj, BayesianTrajPS, BayesConfig
		return {"BayesianTraj": BayesianTraj, "BayesianTrajPS": BayesianTrajPS, "BayesConfig": BayesConfig}[name]
	raise AttributeError(name)
