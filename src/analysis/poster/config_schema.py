from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CANONICAL_FEATURE_SET_IDS = [
    "sd",
    "sd_sum",
    "sd_traj",
    "sd_sum_traj",
    "sd_traj_boot",
    "sd_sum_traj_boot",
]


@dataclass
class RunConfig:
    run_name: str = "poster_v1"
    seed: int = 920
    repeat_policy: str = "all_repeats"


@dataclass
class InputsConfig:
    results_root: str = "results"
    datasets: list[str] = field(default_factory=lambda: ["mimic", "eicu"])
    tasks: list[str] = field(default_factory=list)


@dataclass
class SelectionConfig:
    metric_for_best_model: str = "aupr"
    feature_sets_canonical: list[str] = field(default_factory=lambda: list(CANONICAL_FEATURE_SET_IDS))
    multi_marker_preference: bool = True
    feature_set_aliases: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class StatisticsConfig:
    test: str = "wilcoxon_paired"
    reference_feature_set: str = "sd"
    pvalue_thresholds: dict[float, str] = field(
        default_factory=lambda: {0.001: "***", 0.01: "**", 0.05: "*"}
    )
    pairwise_comparisons: list[list[str]] = field(default_factory=list)


@dataclass
class PlotConfig:
    formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 300
    anchors: list[str] = field(default_factory=lambda: ["#CCD8E7", "#93540A", "#7A620E"])
    figsize_main: list[float] = field(default_factory=lambda: [15.0, 6.5])
    figsize_best: list[float] = field(default_factory=lambda: [13.0, 5.8])


@dataclass
class OutputsConfig:
    root: str = "results/poster"
    write_tables: bool = True
    write_manifest: bool = True


@dataclass
class PosterConfig:
    version: str = "1.0"
    run: RunConfig = field(default_factory=RunConfig)
    inputs: InputsConfig = field(default_factory=InputsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)


def _merge_dataclass(dc_cls, raw: dict[str, Any]):
    kwargs = {}
    for key in dc_cls.__dataclass_fields__.keys():
        if key in raw:
            kwargs[key] = raw[key]
    return dc_cls(**kwargs)


def load_poster_config(config_path: str | Path) -> PosterConfig:
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = PosterConfig(
        version=raw.get("version", "1.0"),
        run=_merge_dataclass(RunConfig, raw.get("run", {})),
        inputs=_merge_dataclass(InputsConfig, raw.get("inputs", {})),
        selection=_merge_dataclass(SelectionConfig, raw.get("selection", {})),
        statistics=_merge_dataclass(StatisticsConfig, raw.get("statistics", {})),
        plot=_merge_dataclass(PlotConfig, raw.get("plot", {})),
        outputs=_merge_dataclass(OutputsConfig, raw.get("outputs", {})),
    )
    validate_poster_config(cfg)
    return cfg


def validate_poster_config(cfg: PosterConfig) -> None:
    if cfg.run.repeat_policy != "all_repeats":
        raise ValueError("run.repeat_policy must be 'all_repeats'")

    if cfg.selection.metric_for_best_model.lower() != "aupr":
        raise ValueError("selection.metric_for_best_model must be 'aupr'")

    missing = [
        fs for fs in cfg.selection.feature_sets_canonical if fs not in CANONICAL_FEATURE_SET_IDS
    ]
    if missing:
        raise ValueError(f"Unsupported canonical feature set ids: {missing}")

    required_anchors = {"#CCD8E7", "#93540A", "#7A620E"}
    anchor_norm = {str(c).upper() for c in cfg.plot.anchors}
    if not required_anchors.issubset(anchor_norm):
        raise ValueError(
            "plot.anchors must include #CCD8E7, #93540A, #7A620E"
        )

    if len(cfg.inputs.datasets) == 0:
        raise ValueError("inputs.datasets cannot be empty")

    if len(cfg.plot.formats) == 0:
        raise ValueError("plot.formats cannot be empty")
