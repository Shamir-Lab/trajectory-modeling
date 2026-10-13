#!/usr/bin/env python3
"""
Qualitative Validation Figures — Trajectory Modeling Framework

One figure per prediction task. Rows are (dataset x clinical parameter),
grouped and bracketed by dataset; columns are the three trajectory archetypes
(stable / gradual change / non-linear change). Each panel overlays the raw EHR
measurements of two patients whose type score for that archetype is
high, so the reader can judge whether the archetype label matches the observed
measurement pattern.

Design rules
------------
* Every clinical parameter for which trajectory features exist is covered
  (8 parameters across 5 tasks and 3 datasets — see SERIES below).
* A plotted window must cover more than MIN_TIME_SPAN of the task's own time
  unit (> 2 days for the day-windowed tasks, > 2 hours for circulatory
  failure). A handful of measurements taken within a few hours is not a
  trajectory, so such windows are never shown — the panel is left empty
  instead.
* The y-axis is shared per clinical parameter across ALL panels of that
  parameter, including across datasets, so rows are directly comparable.
  Limits are robust (1st–99th percentile of the pooled plotted values, padded),
  and any point falling outside is flagged at the axis edge.
* The x-axis is shared across a whole figure and expressed in the task's own
  window unit (days, or hours for circulatory failure).

Outputs: results/figures/qualitative_validation/
    <task>_qualitative_validation.png / .pdf
    selected_examples_manifest.csv      (provenance for every plotted window)

Usage:
    python scripts/figures/qualitative_validation.py
    python scripts/figures/qualitative_validation.py --tasks sepsis aki
    python scripts/figures/qualitative_validation.py --validate-only
    python scripts/figures/qualitative_validation.py --per-series
"""

from __future__ import annotations

import argparse
import os
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = os.environ.get("PHYSIONET_ROOT", "/home/gaga/data/physionet")
OUTPUT_DIR = _REPO_ROOT / "results" / "figures" / "qualitative_validation"

# Files above this size are streamed in chunks and filtered to the shortlisted
# patients instead of being read whole (eICU circulatory-failure vitals are
# several GB).
CHUNK_THRESHOLD_MB = 400
CHUNK_ROWS = 2_000_000

# ─── Selection quality filters ────────────────────────────────────────────────
#
# For the Stable class, require that the relative range of raw values in the
# window is small (values don't swing more than MAX_STABLE_REL_RANGE x mean).
# For directional classes, require a minimum normalised trend: the linear slope
# x window_length must exceed MIN_TREND_STRENGTH x mean. Both filters have a
# relaxed fallback so we always find examples.

MIN_OBS_IN_WINDOW = 3
HIGH_CONF_THRESHOLD = 0.75
# A panel is only evidence of a trajectory if its measurements actually cover
# time. Expressed in the task's own window unit: > 2 days for the day-windowed
# tasks, > 2 hours for circulatory failure. Enforced at every relaxation stage —
# a panel is left empty rather than filled with a window that spans an afternoon.
MIN_TIME_SPAN = 2.0
# The window ends with the anchor bin (day or hour), aggregated over its whole
# unit, so the anchor time (time of prediction) is the end of that bin; the
# x-axis is drawn relative to it.
ANCHOR_OFFSET = 1.0
N_EXAMPLES_PER_CLASS = 2
MAX_STABLE_REL_RANGE = 0.30
MIN_TREND_STRENGTH = 0.15
SHORTLIST_SIZE = 4000  # candidate windows per archetype before touching raw data
EVAL_CAP = 2000        # candidates actually inspected against the raw data per stage

# ─── Task / dataset / parameter metadata ──────────────────────────────────────

ARCHETYPES = ("stable", "gradual", "nonlinear")

COLUMN_TITLES = {
    "stable": "Stable / non-progression",
    "gradual": "Gradual change",
    "nonlinear": "Non-linear change",
}

CLASS_PALETTE = {
    "stable": "#1E6FB8",
    "gradual": "#E08214",
    "nonlinear": "#C1352B",
}

DATASET_LABELS = {
    "eicu": "eICU-CRD",
    "mimic": "MIMIC-III",
    "hirid": "HiRID",
}

TASK_META = {
    "aki": {"title": "Acute Kidney Injury (stage 3)", "window": 7.0, "unit": "days"},
    "liver": {"title": "Acute-on-Chronic Liver Failure", "window": 5.0, "unit": "days"},
    "sepsis": {"title": "Septic Shock", "window": 3.0, "unit": "days"},
    "ventilator": {"title": "Ventilator Weaning Success", "window": 3.0, "unit": "days"},
    "circulatory_failure": {"title": "Circulatory Failure", "window": 12.0, "unit": "hours"},
}

# direction: +1 archetypes describe a rise, -1 a fall.
# change_word: how the pipeline's label_map names the directional archetypes.
MARKER_META = {
    "creatinine": dict(display="Creatinine", unit="mg/dL", direction=+1, change="Increase",
                       threshold=1.2, threshold_label="Upper normal (1.2 mg/dL)"),
    "bilirubin": dict(display="Bilirubin", unit="mg/dL", direction=+1, change="Increase",
                      threshold=1.2, threshold_label="Upper normal (1.2 mg/dL)"),
    "lactate": dict(display="Lactate", unit="mmol/L", direction=+1, change="Increase",
                    threshold=2.0, threshold_label="Normal limit (2 mmol/L)"),
    "wbc": dict(display="WBC", unit="× 10³/μL", direction=+1, change="Increase",
                threshold=11.0, threshold_label="Upper normal (11 × 10³/μL)"),
    "platelet": dict(display="Platelets", unit="× 10³/μL", direction=-1, change="Decline",
                     threshold=150.0, threshold_label="Thrombocytopenia (<150 × 10³/μL)"),
    "pf_ratio": dict(display="P/F ratio", unit="mmHg", direction=+1, change="Improvement",
                     threshold=300.0, threshold_label="ARDS threshold (<300 mmHg)"),
    "heartrate": dict(display="Heart rate", unit="bpm", direction=+1, change="Increase",
                      threshold=100.0, threshold_label="Tachycardia (>100 bpm)"),
    "systolic": dict(display="Systolic BP", unit="mmHg", direction=-1, change="Decline",
                     threshold=90.0, threshold_label="Hypotension (<90 mmHg)"),
}


@dataclass
class SeriesSpec:
    """One (dataset, task, clinical parameter) series."""

    dataset: str
    task: str
    marker: str
    prob_file: str
    prob_cols: dict          # archetype -> column name in prob_file
    ts_file: str
    id_col: str
    value_col: str
    time_col: str            # continuous time within the record
    window_col: str          # integer windowing column the pipeline used

    # filled from metadata
    display_name: str = field(init=False)
    unit: str = field(init=False)
    direction: int = field(init=False)
    change_word: str = field(init=False)
    threshold: float = field(init=False)
    threshold_label: str = field(init=False)
    window: float = field(init=False)
    time_unit: str = field(init=False)

    def __post_init__(self) -> None:
        meta = MARKER_META[self.marker]
        self.display_name = meta["display"]
        self.unit = meta["unit"]
        self.direction = meta["direction"]
        self.change_word = meta["change"]
        self.threshold = meta["threshold"]
        self.threshold_label = meta["threshold_label"]
        self.window = TASK_META[self.task]["window"]
        self.time_unit = TASK_META[self.task]["unit"]

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.task}/{self.marker}"

    @property
    def shares_prob_and_ts(self) -> bool:
        return os.path.abspath(self.prob_file) == os.path.abspath(self.ts_file)

    def archetype_label(self, archetype: str) -> str:
        if archetype == "stable":
            return "Stable"
        if archetype == "gradual":
            return f"Gradual {self.change_word}"
        return f"Non-linear {self.change_word}"


def _p(*parts: str) -> str:
    return os.path.join(_DATA_ROOT, *parts)


def _pc(prefix: str, gradual: str, rapid: str) -> dict:
    """Prob-column map for files using the prob_<archetype> naming."""
    return {"stable": f"{prefix}stable", "gradual": gradual, "nonlinear": rapid}


# ─── Series registry ─────────────────────────────────────────────────────────
#
# Window lengths, archetype directions and column names mirror
# scripts/trajectory/<dataset>/<task>_trajs.py, which produced these files.

SERIES: list[SeriesSpec] = [
    # ── eICU ────────────────────────────────────────────────────────────────
    SeriesSpec("eicu", "aki", "creatinine",
               prob_file=_p("eicu", "aki", "aki_trajectory_probs.csv"),
               prob_cols=_pc("prob_", "prob_gradual_increase", "prob_rapid_increase"),
               ts_file=_p("eicu", "aki", "creatinine_timeseries.csv"),
               id_col="stay_id", value_col="creatinine",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("eicu", "liver", "bilirubin",
               prob_file=_p("eicu", "liver", "liver_trajectory_probs_bayes.csv"),
               prob_cols=_pc("prob_", "prob_gradual_increase", "prob_rapid_increase"),
               ts_file=_p("eicu", "liver", "bilirubin_timeseries.csv"),
               id_col="stay_id", value_col="bilirubin",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("eicu", "sepsis", "lactate",
               prob_file=_p("eicu", "sepsis", "lactate_trajectory_probs_bayes.csv"),
               prob_cols=_pc("lactate_", "lactate_gradual", "lactate_rapid"),
               ts_file=_p("eicu", "sepsis", "lactate_timeseries.csv"),
               id_col="stay_id", value_col="lactate",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("eicu", "sepsis", "wbc",
               prob_file=_p("eicu", "sepsis", "wbc_trajectory_probs_bayes.csv"),
               prob_cols=_pc("wbc_", "wbc_gradual", "wbc_rapid"),
               ts_file=_p("eicu", "sepsis", "wbc_timeseries.csv"),
               id_col="stay_id", value_col="wbc",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("eicu", "sepsis", "platelet",
               prob_file=_p("eicu", "sepsis", "platelet_trajectory_probs_bayes.csv"),
               prob_cols=_pc("platelet_", "platelet_gradual", "platelet_rapid"),
               ts_file=_p("eicu", "sepsis", "platelets_timeseries.csv"),
               id_col="stay_id", value_col="platelet",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("eicu", "ventilator", "pf_ratio",
               prob_file=_p("eicu", "ventilator", "ventilator_trajectory_probs_bayes.csv"),
               prob_cols=_pc("prob_", "prob_gradual_improvement", "prob_rapid_improvement"),
               ts_file=_p("eicu", "ventilator", "pf_ratio_timeseries.csv"),
               id_col="stay_id", value_col="pf_ratio",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("eicu", "circulatory_failure", "lactate",
               prob_file=_p("eicu", "circulatory_failure", "lactate_trajectory_probs_bayes.csv"),
               prob_cols=_pc("lactate_", "lactate_gradual", "lactate_rapid"),
               ts_file=_p("eicu", "circulatory_failure", "lactate_timeseries.csv"),
               id_col="stay_id", value_col="lactate",
               time_col="time_hours", window_col="time_hour"),
    SeriesSpec("eicu", "circulatory_failure", "heartrate",
               prob_file=_p("eicu", "circulatory_failure", "heartrate_trajectory_probs_bayes.csv"),
               prob_cols=_pc("heartrate_", "heartrate_gradual", "heartrate_rapid"),
               ts_file=_p("eicu", "circulatory_failure", "heartrate_timeseries.csv"),
               id_col="stay_id", value_col="heartrate",
               time_col="time_hours", window_col="time_hour"),
    SeriesSpec("eicu", "circulatory_failure", "systolic",
               prob_file=_p("eicu", "circulatory_failure", "systolic_trajectory_probs_bayes.csv"),
               prob_cols=_pc("systolic_", "systolic_gradual", "systolic_rapid"),
               ts_file=_p("eicu", "circulatory_failure", "systolic_timeseries.csv"),
               id_col="stay_id", value_col="systolic",
               time_col="time_hours", window_col="time_hour"),

    # ── MIMIC-III ───────────────────────────────────────────────────────────
    SeriesSpec("mimic", "aki", "creatinine",
               prob_file=_p("mimic", "aki", "aki_prediction_dataset_with_probs.csv"),
               prob_cols=_pc("prob_", "prob_gradual_increase", "prob_rapid_increase"),
               ts_file=_p("mimic", "aki", "creatinine_timeseries.csv"),
               id_col="hadm_id", value_col="creatinine",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("mimic", "liver", "bilirubin",
               prob_file=_p("mimic", "liver", "liver_prediction_dataset_with_probs.csv"),
               prob_cols=_pc("prob_", "prob_gradual_increase", "prob_rapid_increase"),
               ts_file=_p("mimic", "liver", "bilirubin_timeseries.csv"),
               id_col="hadm_id", value_col="bilirubin",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("mimic", "sepsis", "lactate",
               prob_file=_p("mimic", "sepsis", "sepsis_prediction_dataset_with_probs.csv"),
               prob_cols=_pc("lactate_", "lactate_gradual", "lactate_rapid"),
               ts_file=_p("mimic", "sepsis", "lactate_timeseries.csv"),
               id_col="hadm_id", value_col="lactate",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("mimic", "sepsis", "wbc",
               prob_file=_p("mimic", "sepsis", "sepsis_prediction_dataset_with_probs.csv"),
               prob_cols=_pc("wbc_", "wbc_gradual", "wbc_rapid"),
               ts_file=_p("mimic", "sepsis", "wbc_timeseries.csv"),
               id_col="hadm_id", value_col="wbc",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("mimic", "sepsis", "platelet",
               prob_file=_p("mimic", "sepsis", "sepsis_prediction_dataset_with_probs.csv"),
               prob_cols=_pc("platelet_", "platelet_gradual", "platelet_rapid"),
               ts_file=_p("mimic", "sepsis", "platelet_timeseries.csv"),
               id_col="hadm_id", value_col="platelet",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("mimic", "ventilator", "pf_ratio",
               prob_file=_p("mimic", "ventilator", "ventilator_prediction_dataset_with_probs.csv"),
               prob_cols=_pc("prob_", "prob_gradual_improvement", "prob_rapid_improvement"),
               ts_file=_p("mimic", "ventilator", "pf_ratio_timeseries.csv"),
               id_col="hadm_id", value_col="pf_ratio",
               time_col="time_days", window_col="time_day"),
    SeriesSpec("mimic", "circulatory_failure", "lactate",
               prob_file=_p("mimic", "circulatory_failure", "lactate_trajectory_probs_bayes.csv"),
               prob_cols=_pc("prob_", "prob_gradual_increase", "prob_rapid_increase"),
               ts_file=_p("mimic", "circulatory_failure", "lactate_trajectory_probs_bayes.csv"),
               id_col="hadm_id", value_col="lactate",
               time_col="time_hours", window_col="time_hour"),
    SeriesSpec("mimic", "circulatory_failure", "heartrate",
               prob_file=_p("mimic", "circulatory_failure", "heartrate_trajectory_probs_bayes.csv"),
               prob_cols=_pc("prob_", "prob_gradual_increase", "prob_rapid_increase"),
               ts_file=_p("mimic", "circulatory_failure", "heartrate_trajectory_probs_bayes.csv"),
               id_col="hadm_id", value_col="heartrate",
               time_col="time_hours", window_col="time_hour"),
    SeriesSpec("mimic", "circulatory_failure", "systolic",
               prob_file=_p("mimic", "circulatory_failure", "systolic_trajectory_probs_bayes.csv"),
               prob_cols=_pc("prob_", "prob_gradual_decline", "prob_rapid_decline"),
               ts_file=_p("mimic", "circulatory_failure", "systolic_trajectory_probs_bayes.csv"),
               id_col="hadm_id", value_col="systolic",
               time_col="time_hours", window_col="time_hour"),

    # ── HiRID ───────────────────────────────────────────────────────────────
    SeriesSpec("hirid", "circulatory_failure", "lactate",
               prob_file=_p("hirid", "circulatory_failure", "lactate_trajectory_probs_bayes.csv"),
               prob_cols=_pc("prob_", "prob_gradual_increase", "prob_rapid_increase"),
               ts_file=_p("hirid", "circulatory_failure", "lactate_trajectory_probs_bayes.csv"),
               id_col="patientid", value_col="lactate",
               time_col="time_hours", window_col="time_hour"),
    SeriesSpec("hirid", "circulatory_failure", "heartrate",
               prob_file=_p("hirid", "circulatory_failure", "heartrate_trajectory_probs_bayes.csv"),
               prob_cols=_pc("prob_", "prob_gradual_increase", "prob_rapid_increase"),
               ts_file=_p("hirid", "circulatory_failure", "heartrate_trajectory_probs_bayes.csv"),
               id_col="patientid", value_col="heartrate",
               time_col="time_hours", window_col="time_hour"),
    SeriesSpec("hirid", "circulatory_failure", "systolic",
               prob_file=_p("hirid", "circulatory_failure", "systolic_trajectory_probs_bayes.csv"),
               prob_cols=_pc("prob_", "prob_gradual_decline", "prob_rapid_decline"),
               ts_file=_p("hirid", "circulatory_failure", "systolic_trajectory_probs_bayes.csv"),
               id_col="patientid", value_col="systolic",
               time_col="time_hours", window_col="time_hour"),
]

DATASET_ORDER = ["eicu", "mimic", "hirid"]
MARKER_ORDER = ["lactate", "wbc", "platelet", "creatinine", "bilirubin",
                "pf_ratio", "heartrate", "systolic"]


def series_for_task(task: str, datasets: list[str]) -> list[SeriesSpec]:
    """Rows of a task figure: grouped by dataset, ordered by parameter."""
    rows = [s for s in SERIES if s.task == task and s.dataset in datasets]
    rows.sort(key=lambda s: (DATASET_ORDER.index(s.dataset), MARKER_ORDER.index(s.marker)))
    return rows


# ─── Data loading ────────────────────────────────────────────────────────────

def _file_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


def _read_csv_cols(path: str, cols: list[str], ids: set | None = None,
                   id_col: str | None = None) -> pd.DataFrame:
    """Read selected columns, streaming large files and filtering to `ids`."""
    cols = list(dict.fromkeys(cols))
    if ids is None or _file_mb(path) < CHUNK_THRESHOLD_MB:
        df = pd.read_csv(path, usecols=cols)
        if ids is not None and id_col is not None:
            df = df[df[id_col].isin(ids)]
        return df

    kept = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK_ROWS):
        kept.append(chunk[chunk[id_col].isin(ids)])
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=cols)


def _probs_from_frame(df: pd.DataFrame, spec: SeriesSpec) -> pd.DataFrame:
    prob_cols = [spec.prob_cols[a] for a in ARCHETYPES]
    out = df[[spec.id_col, spec.window_col] + prob_cols].dropna(subset=prob_cols, how="all")
    # Merged Bayesian files repeat the window's probabilities on every
    # measurement row of that window.
    return out.drop_duplicates(subset=[spec.id_col, spec.window_col])


def _ts_from_frame(df: pd.DataFrame, spec: SeriesSpec) -> pd.DataFrame:
    out = df[[spec.id_col, spec.value_col, spec.time_col, spec.window_col]].copy()
    out[spec.value_col] = pd.to_numeric(out[spec.value_col], errors="coerce")
    return out.dropna(subset=[spec.value_col])


def load_probs(spec: SeriesSpec) -> pd.DataFrame:
    prob_cols = [spec.prob_cols[a] for a in ARCHETYPES]
    cols = [spec.id_col, spec.window_col] + prob_cols
    return _probs_from_frame(_read_csv_cols(spec.prob_file, cols), spec)


def load_timeseries(spec: SeriesSpec, ids: set | None = None) -> pd.DataFrame:
    cols = [spec.id_col, spec.value_col, spec.time_col, spec.window_col]
    df = _read_csv_cols(spec.ts_file, cols, ids=ids, id_col=spec.id_col)
    return _ts_from_frame(df, spec)


def build_ts_index(ts_df: pd.DataFrame, spec: SeriesSpec) -> dict:
    """Per-patient slices, so candidate screening doesn't rescan millions of rows."""
    return {pid: g for pid, g in ts_df.groupby(spec.id_col, sort=False)}


def get_window_measurements(ts_df: pd.DataFrame, patient_id, window_time,
                            spec: SeriesSpec, index: dict | None = None) -> pd.DataFrame:
    """
    Reproduce the pipeline's windowing logic:
        window = [window_time - window, window_time] on the integer window column.
    Adds rel_time (0 = window end) in the task's own time unit.
    """
    if index is not None:
        pat = index.get(patient_id)
        if pat is None:
            return pd.DataFrame(columns=list(ts_df.columns) + ["rel_time"])
    else:
        pat = ts_df[ts_df[spec.id_col] == patient_id]
    mask = (pat[spec.window_col] >= window_time - spec.window) & (pat[spec.window_col] <= window_time)
    win = pat[mask].copy()
    win["rel_time"] = win[spec.time_col] - window_time
    return win.sort_values("rel_time")


# ─── Example selection ───────────────────────────────────────────────────────

def _quality_checks(x: np.ndarray, y: np.ndarray, window: float, expected_dir: int,
                    check_direction: bool, check_trend: bool, check_stable: bool,
                    check_typical: bool = False,
                    value_bounds: tuple[float, float] | None = None,
                    min_span: float = MIN_TIME_SPAN) -> bool:
    # Hard requirement, never relaxed: the measurements have to cover enough
    # time to show a trajectory rather than a single burst of sampling.
    if min_span > 0:
        if len(x) < 2 or (float(np.max(x)) - float(np.min(x))) <= min_span:
            return False

    # A representative example should not be a physiological outlier: one
    # patient with WBC 130 or platelets 950 would otherwise dictate the shared
    # y-range for every panel of that parameter.
    if check_typical and value_bounds is not None and len(y):
        if float(y.min()) < value_bounds[0] or float(y.max()) > value_bounds[1]:
            return False

    if len(x) < 2:
        return not (check_direction or check_stable or check_trend)

    slope, *_ = stats.linregress(x, y)
    mean_abs = float(np.mean(np.abs(y))) + 1e-9
    norm_trend = slope * window / mean_abs
    rel_range = (float(y.max()) - float(y.min())) / mean_abs

    if check_direction and expected_dir != 0:
        if expected_dir > 0 and slope <= 0:
            return False
        if expected_dir < 0 and slope >= 0:
            return False

    if check_trend and expected_dir != 0 and abs(norm_trend) < MIN_TREND_STRENGTH:
        return False

    if check_stable and expected_dir == 0 and rel_range > MAX_STABLE_REL_RANGE:
        return False

    return True


def shortlist_candidates(probs: pd.DataFrame, spec: SeriesSpec, threshold: float,
                         verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Rank candidate windows per archetype using the probabilities only."""
    prob_cols = [spec.prob_cols[a] for a in ARCHETYPES]
    dominant = probs[prob_cols].idxmax(axis=1)

    out: dict[str, pd.DataFrame] = {}
    for archetype in ARCHETYPES:
        col = spec.prob_cols[archetype]
        chosen = pd.DataFrame()
        for thr in [threshold, 0.70, 0.65, 0.60]:
            mask = (dominant == col) & (probs[col] > thr)
            cand = probs[mask].sort_values(col, ascending=False)
            cand = cand.drop_duplicates(subset=[spec.id_col], keep="first")
            if not cand.empty:
                if thr < threshold and verbose:
                    print(f"      [{spec.key}] '{archetype}': dominance threshold lowered to {thr:.2f}")
                chosen = cand
                break

        if chosen.empty:
            # Last resort: this archetype is never dominant in this cohort.
            best = (probs[probs[col] > 0.10]
                    .sort_values(col, ascending=False)
                    .drop_duplicates(subset=[spec.id_col], keep="first"))
            if not best.empty:
                if verbose:
                    print(f"      [{spec.key}] '{archetype}': dominance relaxed "
                          f"(best available p={best[col].max():.2f})")
                chosen = best
            elif verbose:
                print(f"      [{spec.key}] '{archetype}': no candidate windows")

        out[archetype] = chosen.head(SHORTLIST_SIZE)
    return out


def select_examples(shortlists: dict[str, pd.DataFrame], ts_df: pd.DataFrame,
                    spec: SeriesSpec, n_per_class: int,
                    rng: random.Random | None = None,
                    verbose: bool = True,
                    min_span: float = MIN_TIME_SPAN,
                    ts_index: dict | None = None) -> dict[str, list[tuple]]:
    """Pick up to n_per_class (patient, window, p) tuples per archetype."""
    filter_stages = [
        dict(min_obs=MIN_OBS_IN_WINDOW, check_direction=True, check_trend=True,
             check_stable=True, check_typical=True),
        dict(min_obs=MIN_OBS_IN_WINDOW, check_direction=True, check_trend=False,
             check_stable=True, check_typical=True),
        dict(min_obs=MIN_OBS_IN_WINDOW, check_direction=True, check_trend=False,
             check_stable=False, check_typical=False),
        dict(min_obs=2, check_direction=False, check_trend=False,
             check_stable=False, check_typical=False),
    ]

    # Cohort-level plausible range, used to keep examples representative.
    vals = ts_df[spec.value_col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    value_bounds = (float(np.percentile(vals, 1)), float(np.percentile(vals, 98))) \
        if vals.size >= 50 else None

    results: dict[str, list[tuple]] = {}
    for archetype in ARCHETYPES:
        candidates = shortlists.get(archetype, pd.DataFrame())
        expected_dir = 0 if archetype == "stable" else spec.direction
        col = spec.prob_cols[archetype]

        selected: list[tuple] = []
        if not candidates.empty:
            seed_val = rng.randint(0, 99999) if rng else 42
            for stage_idx, filt in enumerate(filter_stages):
                shuffled = candidates.sample(frac=1, random_state=seed_val + stage_idx)
                passes: list[tuple] = []
                for _, row in shuffled.head(EVAL_CAP).iterrows():
                    pid = row[spec.id_col]
                    wt = row[spec.window_col]
                    win = get_window_measurements(ts_df, pid, wt, spec, index=ts_index)
                    if len(win) < filt["min_obs"]:
                        continue
                    ok = _quality_checks(
                        win["rel_time"].values, win[spec.value_col].values,
                        spec.window, expected_dir,
                        filt["check_direction"], filt["check_trend"], filt["check_stable"],
                        check_typical=filt["check_typical"], value_bounds=value_bounds,
                        min_span=min_span,
                    )
                    if ok:
                        passes.append((len(win), float(row[col]), pid, wt))

                if passes:
                    # A window with more measurements is more convincing evidence
                    # that the archetype label matches the raw data.
                    passes.sort(key=lambda t: (-t[0], -t[1]))
                    seen: set = set()
                    for _n_obs, prob, pid, wt in passes:
                        if pid in seen:
                            continue
                        seen.add(pid)
                        selected.append((pid, wt, prob))
                        if len(selected) >= n_per_class:
                            break
                    break

        results[archetype] = selected
        if verbose:
            msg = f"      [{spec.key}] '{archetype}': {len(selected)} example(s)"
            if len(selected) < n_per_class:
                msg += (f"  [short of {n_per_class}: no further window spans "
                        f"> {min_span:g} {spec.time_unit}]")
            print(msg)
    return results


def build_series_data(spec: SeriesSpec, threshold: float, n_per_class: int,
                      rng: random.Random, verbose: bool = True,
                      min_span: float = MIN_TIME_SPAN) -> tuple[dict, pd.DataFrame, dict]:
    """Load, shortlist, fetch raw measurements and select examples for one series."""
    report = {"key": spec.key, "prob_file": spec.prob_file, "ts_file": spec.ts_file}

    for path in {spec.prob_file, spec.ts_file}:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    # Several Bayesian outputs carry the raw measurements and the window
    # probabilities in one file; read such a file once, not twice.
    shared_raw = None
    if spec.shares_prob_and_ts:
        prob_cols = [spec.prob_cols[a] for a in ARCHETYPES]
        shared_raw = _read_csv_cols(
            spec.prob_file,
            [spec.id_col, spec.window_col, spec.time_col, spec.value_col] + prob_cols,
        )
        probs = _probs_from_frame(shared_raw, spec)
    else:
        probs = load_probs(spec)

    report["n_prob_windows"] = int(len(probs))
    report["n_prob_patients"] = int(probs[spec.id_col].nunique())

    shortlists = shortlist_candidates(probs, spec, threshold, verbose=verbose)
    ids = set()
    for cand in shortlists.values():
        if not cand.empty:
            ids.update(cand[spec.id_col].tolist())

    if shared_raw is not None:
        sel = shared_raw[shared_raw[spec.id_col].isin(ids)] if ids else shared_raw
        ts_df = _ts_from_frame(sel, spec)
        del shared_raw
    else:
        ts_df = load_timeseries(spec, ids=ids if ids else None)
    report["n_ts_rows_loaded"] = int(len(ts_df))
    report["n_ts_patients_loaded"] = int(ts_df[spec.id_col].nunique()) if len(ts_df) else 0
    if len(ts_df):
        report["value_range"] = (float(ts_df[spec.value_col].min()),
                                 float(ts_df[spec.value_col].max()))

    ts_index = build_ts_index(ts_df, spec)
    examples = select_examples(shortlists, ts_df, spec, n_per_class, rng=rng,
                               verbose=verbose, min_span=min_span, ts_index=ts_index)
    report["n_examples"] = {a: len(examples[a]) for a in ARCHETYPES}
    return examples, ts_df, report


# ─── Shared axis limits ──────────────────────────────────────────────────────

def robust_limits(values: np.ndarray, threshold: float | None,
                  lo_pct: float = 1.0, hi_pct: float = 99.0,
                  pad_lo: float = 0.06, pad_hi: float = 0.16) -> tuple[float, float]:
    """Percentile-clipped limits so one extreme patient cannot flatten a row."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)

    lo = float(np.percentile(v, lo_pct))
    hi = float(np.percentile(v, hi_pct))
    if threshold is not None:
        lo = min(lo, threshold)
        hi = max(hi, threshold)
    if hi <= lo:
        hi = lo + max(abs(lo) * 0.1, 1.0)

    # Extra headroom at the top leaves a clear band for the in-panel
    # probability annotation.
    span = hi - lo
    lo -= pad_lo * span
    hi += pad_hi * span
    if float(v.min()) >= 0:
        lo = max(lo, 0.0)
    return (lo, hi)


def marker_ylimits(rows: list[SeriesSpec], examples: dict, ts_by_key: dict) -> dict:
    """One shared y-range per clinical parameter, pooled across datasets."""
    pooled: dict[str, list[float]] = {}
    for spec in rows:
        ts_df = ts_by_key[spec.key]
        for archetype in ARCHETYPES:
            for pid, wt, _p in examples[spec.key][archetype]:
                win = get_window_measurements(ts_df, pid, wt, spec)
                pooled.setdefault(spec.marker, []).extend(win[spec.value_col].tolist())

    limits = {}
    for marker, vals in pooled.items():
        limits[marker] = robust_limits(np.asarray(vals), MARKER_META[marker]["threshold"])
    return limits


# ─── Plotting ────────────────────────────────────────────────────────────────

PATIENT_STYLES = [
    {"marker": "o", "linestyle": "--", "filled": True, "size": 42},
    {"marker": "^", "linestyle": ":", "filled": False, "size": 44},
]


def _density_style(n_points: int) -> dict:
    """
    Vitals sampled every few minutes put hundreds of points in a 12-hour
    window; drawn at lab-sized markers they collapse into a blob. Shrink the
    markers and let the connecting line carry the shape instead.
    """
    if n_points <= 25:
        return {"scale": 1.0, "linewidth": 1.3, "line_alpha": 0.55,
                "scatter_alpha": 0.95, "edge_lw": 1.0}
    if n_points <= 80:
        return {"scale": 0.38, "linewidth": 1.1, "line_alpha": 0.70,
                "scatter_alpha": 0.85, "edge_lw": 0.7}
    return {"scale": 0.15, "linewidth": 0.9, "line_alpha": 0.85,
            "scatter_alpha": 0.70, "edge_lw": 0.5}


def _draw_panel(ax: plt.Axes, spec: SeriesSpec, archetype: str,
                windows: list[tuple], ts_df: pd.DataFrame,
                ylim: tuple[float, float], show_xticklabels: bool) -> int:
    """Draw one archetype panel. Returns the number of off-scale points."""
    color = CLASS_PALETTE[archetype]

    if not windows:
        ax.set_facecolor("#F7F7F7")
        ax.text(0.5, 0.5, "archetype not observed\nin this cohort",
                ha="center", va="center", transform=ax.transAxes,
                color="#9E9E9E", fontsize=8, style="italic", linespacing=1.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#D0D0D0")
        return 0

    off_scale = 0
    probs_shown = []

    for idx, (pid, wt, prob) in enumerate(windows[:2]):
        style = PATIENT_STYLES[idx]
        win = get_window_measurements(ts_df, pid, wt, spec)
        if win.empty:
            continue
        x = win["rel_time"].to_numpy(dtype=float) - ANCHOR_OFFSET
        y = win[spec.value_col].to_numpy(dtype=float)
        probs_shown.append(prob)

        dens = _density_style(len(x))
        ax.plot(x, y, color=color, alpha=dens["line_alpha"], linestyle=style["linestyle"],
                linewidth=dens["linewidth"], zorder=2)
        ax.scatter(x, y, s=style["size"] * dens["scale"], zorder=4, marker=style["marker"],
                   facecolors=color if style["filled"] else "white",
                   edgecolors=color,
                   linewidths=(0.6 if style["filled"] else 1.3) * dens["edge_lw"],
                   alpha=dens["scatter_alpha"])

        below = y < ylim[0]
        above = y > ylim[1]
        off_scale += int(below.sum() + above.sum())
        if below.any():
            ax.scatter(x[below], np.full(below.sum(), ylim[0]), marker="v", s=26,
                       color=color, alpha=0.75, zorder=5, clip_on=False)
        if above.any():
            ax.scatter(x[above], np.full(above.sum(), ylim[1]), marker="^", s=26,
                       color=color, alpha=0.75, zorder=5, clip_on=False)

    ax.axhline(spec.threshold, color="#B3261E", linestyle="--",
               linewidth=0.9, alpha=0.7, zorder=1)
    ax.axvline(0, color="#424242", linewidth=0.7, alpha=0.25, zorder=1)

    ax.set_xlim(-spec.window - ANCHOR_OFFSET - 0.35, 0.35)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=8)
    if not show_xticklabels:
        ax.tick_params(labelbottom=False)
    sns.despine(ax=ax)

    if probs_shown:
        # Type scores of patient 1 and patient 2 (the thesis notation \hat{T}).
        txt = r"$\hat{T}$ = " + ", ".join(f"{p:.2f}" for p in probs_shown)
        ax.text(0.025, 0.97, txt, transform=ax.transAxes, fontsize=7.2,
                va="top", ha="left", color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor=color, alpha=0.92, linewidth=0.6))

    return off_scale


def _dataset_brackets(fig, axes_grid, rows: list[SeriesSpec], left_x: float) -> None:
    """Vertical bracket + rotated dataset name spanning each dataset's rows."""
    groups: list[tuple[str, int, int]] = []
    for i, spec in enumerate(rows):
        if groups and groups[-1][0] == spec.dataset:
            groups[-1] = (spec.dataset, groups[-1][1], i)
        else:
            groups.append((spec.dataset, i, i))

    for dataset, first, last in groups:
        top = axes_grid[first][0].get_position().y1
        bottom = axes_grid[last][0].get_position().y0
        fig.add_artist(Line2D([left_x, left_x], [bottom, top],
                              transform=fig.transFigure, color="#5A5A5A",
                              linewidth=1.4, solid_capstyle="butt"))
        fig.text(left_x - 0.014, (top + bottom) / 2, DATASET_LABELS.get(dataset, dataset),
                 rotation=90, va="center", ha="center",
                 fontsize=11.5, fontweight="bold", color="#3A3A3A")


def plot_task_figure(task: str, rows: list[SeriesSpec], examples: dict, ts_by_key: dict,
                     output_dir: Path, threshold: float, dpi: int,
                     formats: list[str], min_span: float = MIN_TIME_SPAN) -> tuple[list[Path], int]:
    meta = TASK_META[task]
    n_rows = len(rows)
    ylims = marker_ylimits(rows, examples, ts_by_key)

    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.rcParams.update({
        "axes.edgecolor": "#5A5A5A",
        "axes.linewidth": 0.8,
        "grid.color": "#E4E4E4",
        "grid.linewidth": 0.7,
        "font.family": "DejaVu Sans",
    })

    row_h = 2.35
    fig_h = row_h * n_rows + 3.1
    fig = plt.figure(figsize=(13.6, fig_h))

    # Vertical budget in inches, converted to figure fractions, so that the
    # layout is identical whether a task has 1 row or 9.
    left, right = 0.170, 0.985
    top_in, bottom_in = 1.25, 1.85

    gs = fig.add_gridspec(
        n_rows + 1, 3,
        height_ratios=[0.22] + [1.0] * n_rows,
        hspace=0.30, wspace=0.16,
        left=left, right=right,
        top=1 - (top_in / fig_h), bottom=bottom_in / fig_h,
    )

    for col, archetype in enumerate(ARCHETYPES):
        ax_h = fig.add_subplot(gs[0, col])
        ax_h.set_facecolor("#ECEFF1")
        ax_h.text(0.5, 0.5, COLUMN_TITLES[archetype], ha="center", va="center",
                  transform=ax_h.transAxes, fontsize=11.5, fontweight="bold",
                  color=CLASS_PALETTE[archetype])
        ax_h.set_xticks([])
        ax_h.set_yticks([])
        ax_h.grid(False)
        for sp in ax_h.spines.values():
            sp.set_visible(False)

    axes_grid = [[fig.add_subplot(gs[r + 1, c]) for c in range(3)] for r in range(n_rows)]

    total_off_scale = 0
    for r, spec in enumerate(rows):
        ylim = ylims.get(spec.marker, (0.0, 1.0))
        is_bottom = (r == n_rows - 1)
        for c, archetype in enumerate(ARCHETYPES):
            ax = axes_grid[r][c]
            total_off_scale += _draw_panel(
                ax, spec, archetype,
                windows=examples[spec.key][archetype],
                ts_df=ts_by_key[spec.key],
                ylim=ylim,
                show_xticklabels=is_bottom,
            )
            if c > 0:
                ax.tick_params(labelleft=False)

        arrow = "↑" if spec.direction > 0 else "↓"
        axes_grid[r][0].set_ylabel(
            f"{spec.display_name}\n({spec.unit})  {arrow}",
            fontsize=9.5, labelpad=8,
        )

    unit_word = "days" if meta["unit"] == "days" else "hours"
    for c in range(3):
        axes_grid[-1][c].set_xlabel(f"Time relative to anchor time ({unit_word})", fontsize=9.5)

    fig.suptitle(
        f"Qualitative validation of trajectory archetypes — {meta['title']}",
        fontsize=15, fontweight="bold", y=1 - (0.34 / fig_h),
    )
    span_txt = (f" and measurements spanning > {min_span:g} {unit_word}"
                if min_span > 0 else "")
    fig.text(
        0.5, 1 - (0.76 / fig_h),
        f"{meta['window']:g}-{unit_word[:-1]} lookback windows · "
        f"raw measurements of two patients per panel, selected at type score "
        f"≥ {threshold:.2f}{span_txt} "
        f"· y-axis shared per clinical parameter across datasets",
        ha="center", va="center", fontsize=9.8, color="#4A4A4A",
    )

    _dataset_brackets(fig, axes_grid, rows, left_x=left - 0.090)

    handles = [
        Line2D([], [], marker="o", color="#6E6E6E", linestyle="--", markerfacecolor="#6E6E6E",
               markeredgecolor="#6E6E6E", markersize=6, label="Patient 1"),
        Line2D([], [], marker="^", color="#6E6E6E", linestyle=":", markerfacecolor="white",
               markeredgecolor="#6E6E6E", markersize=6, label="Patient 2"),
        Line2D([], [], color="#B3261E", linestyle="--", linewidth=1.2,
               label="Clinical reference threshold"),
        Line2D([], [], marker="^", color="#6E6E6E", linestyle="none", markersize=6,
               label="Value beyond axis range"),
    ]
    fig.legend(handles=handles, loc="center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.95 / fig_h))

    thr_lines = " · ".join(
        f"{MARKER_META[m]['display']}: {MARKER_META[m]['threshold_label']}"
        for m in dict.fromkeys(s.marker for s in rows)
    )
    fig.text(0.5, 0.52 / fig_h, thr_lines, ha="center", va="center",
             fontsize=8.2, color="#6A6A6A")
    fig.text(0.5, 0.24 / fig_h,
             "Row labels give the clinical parameter; ↑ / ↓ marks the direction of that "
             "parameter's gradual and non-linear archetypes.",
             ha="center", va="center", fontsize=8.2, color="#6A6A6A")

    paths = []
    for ext in formats:
        out = output_dir / f"{task}_qualitative_validation.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.25,
                    facecolor="white")
        paths.append(out)
    plt.close(fig)
    return paths, total_off_scale


def plot_series_figure(spec: SeriesSpec, examples: dict, ts_df: pd.DataFrame,
                       output_dir: Path, threshold: float, dpi: int,
                       formats: list[str]) -> list[Path]:
    """Optional single-series figure (1 row x 3 archetypes)."""
    ylim = marker_ylimits([spec], {spec.key: examples}, {spec.key: ts_df}).get(
        spec.marker, (0.0, 1.0))

    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for c, archetype in enumerate(ARCHETYPES):
        _draw_panel(axes[c], spec, archetype, examples[archetype], ts_df, ylim, True)
        axes[c].set_title(spec.archetype_label(archetype), fontsize=10.5,
                          color=CLASS_PALETTE[archetype], pad=6)
        unit_word = "days" if spec.time_unit == "days" else "hours"
        axes[c].set_xlabel(f"Time relative to anchor time ({unit_word})", fontsize=9.5)
        if c > 0:
            axes[c].tick_params(labelleft=False)
    arrow = "↑" if spec.direction > 0 else "↓"
    axes[0].set_ylabel(f"{spec.display_name}\n({spec.unit})  {arrow}", fontsize=9.5)

    fig.suptitle(
        f"{DATASET_LABELS.get(spec.dataset, spec.dataset)} · "
        f"{TASK_META[spec.task]['title']} — {spec.display_name}",
        fontsize=13, fontweight="bold", y=1.04,
    )
    fig.tight_layout()

    paths = []
    for ext in formats:
        out = output_dir / f"{spec.dataset}_{spec.task}_{spec.marker}_validation.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(out)
    plt.close(fig)
    return paths


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=list(TASK_META.keys()),
                   choices=list(TASK_META.keys()), help="Tasks to render (default: all)")
    p.add_argument("--datasets", nargs="+", default=DATASET_ORDER,
                   choices=DATASET_ORDER, help="Datasets to include (default: all)")
    p.add_argument("--threshold", type=float, default=HIGH_CONF_THRESHOLD,
                   help=f"Min type score for a window (default: {HIGH_CONF_THRESHOLD})")
    p.add_argument("--n-examples", type=int, default=N_EXAMPLES_PER_CLASS,
                   help=f"Patients per panel (default: {N_EXAMPLES_PER_CLASS})")
    p.add_argument("--min-span", type=float, default=MIN_TIME_SPAN,
                   help="Minimum time a window's measurements must cover, in the task's\n"
                        f"own unit: days for AKI/liver/sepsis/ventilator, hours for\n"
                        f"circulatory failure (default: {MIN_TIME_SPAN:g}). 0 disables.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"],
                   help="Output formats (default: png pdf)")
    p.add_argument("--per-series", action="store_true",
                   help="Also write one small figure per dataset/parameter")
    p.add_argument("--validate-only", action="store_true",
                   help="Check inputs and report selections without writing figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data root  : {_DATA_ROOT}")
    print(f"Output dir : {OUTPUT_DIR}\n")

    manifest_rows: list[dict] = []
    saved: list[Path] = []
    reports: list[dict] = []

    for task in args.tasks:
        rows = series_for_task(task, args.datasets)
        if not rows:
            print(f"[skip] {task}: no series for the selected datasets")
            continue

        print("=" * 74)
        print(f"  {TASK_META[task]['title']}  ({len(rows)} rows: "
              f"{', '.join(sorted({DATASET_LABELS[s.dataset] for s in rows}))})")
        print("=" * 74)

        examples: dict[str, dict] = {}
        ts_by_key: dict[str, pd.DataFrame] = {}
        usable: list[SeriesSpec] = []

        for spec in rows:
            print(f"  - {spec.key}")
            try:
                ex, ts_df, report = build_series_data(
                    spec, args.threshold, args.n_examples, rng, min_span=args.min_span)
            except FileNotFoundError as exc:
                print(f"      [SKIP] missing file: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"      [SKIP] {type(exc).__name__}: {exc}")
                continue

            print(f"      probs: {report['n_prob_windows']:,} windows / "
                  f"{report['n_prob_patients']:,} patients | "
                  f"raw rows loaded: {report['n_ts_rows_loaded']:,}")
            examples[spec.key] = ex
            ts_by_key[spec.key] = ts_df
            usable.append(spec)
            reports.append(report)

            for archetype in ARCHETYPES:
                for rank, (pid, wt, prob) in enumerate(ex[archetype], start=1):
                    win = get_window_measurements(ts_df, pid, wt, spec)
                    vals = win[spec.value_col].to_numpy(dtype=float)
                    rel = win["rel_time"].to_numpy(dtype=float)
                    span = float(rel.max() - rel.min()) if rel.size >= 2 else 0.0
                    manifest_rows.append({
                        "dataset": spec.dataset,
                        "task": spec.task,
                        "parameter": spec.marker,
                        "display_name": spec.display_name,
                        "unit": spec.unit,
                        "archetype": archetype,
                        "archetype_label": spec.archetype_label(archetype),
                        "patient_column": spec.id_col,
                        "patient_id": pid,
                        "window_column": spec.window_col,
                        "window_end": wt,
                        "window_length": spec.window,
                        "window_unit": spec.time_unit,
                        "panel_slot": rank,
                        "posterior_prob": prob,
                        "n_measurements": int(len(vals)),
                        "time_span_covered": span,
                        "value_min": float(vals.min()) if vals.size else np.nan,
                        "value_max": float(vals.max()) if vals.size else np.nan,
                        "prob_file": os.path.relpath(spec.prob_file, _DATA_ROOT),
                        "prob_column": spec.prob_cols[archetype],
                        "timeseries_file": os.path.relpath(spec.ts_file, _DATA_ROOT),
                    })

        if not usable:
            print(f"  [skip] {task}: no usable series\n")
            continue

        if args.validate_only:
            print()
            continue

        paths, off_scale = plot_task_figure(
            task, usable, examples, ts_by_key, OUTPUT_DIR,
            threshold=args.threshold, dpi=args.dpi, formats=args.formats,
            min_span=args.min_span)
        saved.extend(paths)
        note = f" ({off_scale} point(s) beyond axis range, flagged at the edge)" if off_scale else ""
        print(f"  -> {paths[0].relative_to(_REPO_ROOT)}{note}\n")

        if args.per_series:
            for spec in usable:
                sp = plot_series_figure(spec, examples[spec.key], ts_by_key[spec.key],
                                        OUTPUT_DIR, args.threshold, args.dpi, args.formats)
                saved.extend(sp)

    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        mpath = OUTPUT_DIR / "selected_examples_manifest.csv"
        manifest.to_csv(mpath, index=False)
        print(f"Manifest: {mpath.relative_to(_REPO_ROOT)} ({len(manifest)} plotted windows)")

    print(f"Done. {len(saved)} file(s) written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
