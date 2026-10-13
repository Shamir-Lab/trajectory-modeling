#!/usr/bin/env python
"""
Manuscript-ready sensitivity-analysis figures for the eICU Septic Shock
trajectory-archetype thresholds, combining all three markers (lactate, wbc,
platelet) into single figures instead of three separate same-looking PNGs.

Reads the three per-marker CSVs that eicu_sepsis_sensitivity_analysis.py
already wrote (results/eicu/sepsis/sensitivity_<marker>/sensitivity_results_eicu_sepsis.csv)
and never re-runs any part of the pipeline — this is presentation only.

Produces four figures — full descriptions, data requirements, and how they
relate to each other are in results/eicu/sepsis/SENSITIVITY_ANALYSIS_METHODS.md
§7 (figures) and §9 (output file inventory); summary here:

1. sensitivity_combined_eicu_sepsis.png — always. Three-column small multiples
   (one column per marker), each column the flip-rate panel from
   eicu_sepsis_sensitivity_analysis.draw_marker_panels, for the headline model
   (default: XGBoost), full ×0.5-×1.5 sweep. Works on the existing
   single-model CSVs.

2. sensitivity_model_robustness_eicu_sepsis.png — needs the full-capture CSVs
   (`model_name` column with >1 distinct value). Signed delta AUPR / delta
   AUROC at each combination's worst-case sweep extreme, one point per model
   type, against that SAME model's own baseline.

3. sensitivity_feature_set_stability_eicu_sepsis.png — needs the full-capture
   CSVs (`model_name` AND `feature_set` columns, >1 distinct feature_set).
   One 9x5 winner-grid per classifier (x2 metrics): does the trajectory-
   inclusive feature set remain the AUPR/AUROC winner over its trajectory-
   ablated counterpart, across the whole sweep, per model.

4. sensitivity_trajectory_value_eicu_sepsis.png — same feature_set
   requirement as #3, but for one model only (default: XGBoost). Continuous
   AUPR(with trajectory) - AUPR(without) vs. perturbation factor.

Any figure whose data requirement isn't met is skipped with an explanatory
log line rather than erroring.

Run from workspace root:
    python scripts/analysis/eicu/plot_sensitivity_combined.py
    python scripts/analysis/eicu/plot_sensitivity_combined.py --model-name XGBoost
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

from analysis.poster.palette import set_poster_style  # noqa: E402
from eicu_sepsis_sensitivity_analysis import (  # noqa: E402
    BIOMARKERS,
    DEFAULT_FEATURE_SET_NAME,
    DEFAULT_MODEL_NAME,
    PARAM_COLORS,
    PARAM_LABELS,
    TRAJECTORY_ABLATED_FEATURE_SET_NAME,
    draw_marker_panels,
    select_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MARKERS = ["lactate", "wbc", "platelet"]
MARKER_LABEL = {"lactate": "Lactate", "wbc": "WBC", "platelet": "Platelet"}

# Distinct marker shapes for model identity, layered on top of the fixed
# param-color coding — keeps the categorical hue count at 3 (all-pairs-safe)
# instead of adding a 4th hue for models.
MODEL_MARKERS = {
    "LogisticRegression": "o",
    "RandomForest": "s",
    "HistGradientBoosting": "^",
    "XGBoost": "D",
}
MODEL_ORDER = ["LogisticRegression", "RandomForest", "HistGradientBoosting", "XGBoost"]

# Two-way categorical identity for "which feature set won" — deliberately NOT
# PARAM_COLORS (that hue triple is reserved for threshold-parameter identity
# and is CVD-validated for that use) and NOT the literal poster metric-scale
# anchors used for per-model AUROC/AUPR boxes elsewhere in the manuscript
# (auroc_scale[:4] / aupr_scale[:4] — reusing one of those hues here would
# make it ambiguously mean "this classifier" in one figure and "this feature
# set" in another). Drawn instead from the same two poster hue families
# (palette.py's auroc_scale / aupr_scale), but from their darkest, unused-at
# n=4 steps, so the figure still reads as part of the same manuscript style
# without colliding with an existing meaning.
FEATURE_SET_COLORS_BY_ROLE = {"with_trajectory": "#7A620E", "without_trajectory": "#334A63"}
FEATURE_SET_ABBREV_BY_ROLE = {"with_trajectory": "Traj", "without_trajectory": "No-traj"}


def load_marker_csv(results_dir: Path, marker: str) -> pd.DataFrame:
    path = results_dir / f"sensitivity_{marker}" / "sensitivity_results_eicu_sepsis.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No sensitivity results for marker='{marker}' at {path}. "
            f"Run eicu_sepsis_sensitivity_analysis.py (or run_sensitivity_grid.sh) for this marker first."
        )
    return pd.read_csv(path)


def plot_combined_overview(
    results_by_marker: dict[str, pd.DataFrame], out_path: Path, feature_set_name: str, model_name: str,
) -> None:
    """Flip-rate-only manuscript figure (one row, one panel per marker) — see
    FIGURE_CAPTIONS.md Figure 1 for the full caption text; the AUPR-vs-sweep
    story it used to carry as a second row now belongs to Figure 2
    (sensitivity_model_robustness) and Figure 4 (sensitivity_trajectory_value)."""
    import matplotlib.pyplot as plt

    set_poster_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"wspace": 0.25})

    for col, marker in enumerate(MARKERS):
        ax_flip = axes[col]
        draw_marker_panels(ax_flip, None, results_by_marker[marker], model_name)
        ax_flip.set_title(MARKER_LABEL[marker], fontsize=13, fontweight="bold")
        if col > 0:
            ax_flip.set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Perturbed threshold", loc="upper right",
               bbox_to_anchor=(0.99, 0.92), frameon=False, ncol=1)

    fig.suptitle("Archetype-threshold sensitivity: label stability", fontsize=15, y=0.99, x=0.01, ha="left")

    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.065, right=0.985, wspace=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def plot_trajectory_value(
    results_by_marker: dict[str, pd.DataFrame], out_path: Path, model_name: str,
    feature_set_with: str, feature_set_without: str,
) -> None:
    """Does the trajectory-archetype features' contribution to AUPR survive
    the threshold perturbation sweep, or does it wash out (or reverse) for
    some threshold choice? Plots AUPR(feature_set_with) -
    AUPR(feature_set_without) — the trajectory features' marginal value,
    holding the model and every other feature fixed — vs. perturbation
    factor, one line per threshold parameter, per marker. This is the
    central robustness question when the paper's contribution *is* the
    trajectory feature type: not "does AUPR move" but "does the
    trajectory-features-help finding survive the threshold choice.\""""
    import matplotlib.pyplot as plt

    def _gain_df(df: pd.DataFrame) -> pd.DataFrame | None:
        if "feature_set" not in df.columns:
            return None
        cols = ["perturbed_param", "perturbation_factor", "median_aupr", "aupr_ci_low", "aupr_ci_high"]
        with_df = select_model(df, model_name, feature_set_with)[cols]
        without_df = select_model(df, model_name, feature_set_without)[cols]
        if with_df.empty or without_df.empty:
            return None
        merged = with_df.merge(without_df, on=["perturbed_param", "perturbation_factor"],
                                suffixes=("_with", "_without"))
        merged["gain_aupr"] = merged["median_aupr_with"] - merged["median_aupr_without"]
        # Conservative (upper-bound) noise estimate for the *difference*: sum
        # of the two feature sets' own CI half-widths in quadrature, treating
        # them as independent. The true paired-difference variance is
        # typically smaller than this (same folds, positively correlated
        # errors), so this errs toward calling a real gain "within noise"
        # rather than the reverse.
        half_with = (merged["aupr_ci_high_with"] - merged["aupr_ci_low_with"]) / 2
        half_without = (merged["aupr_ci_high_without"] - merged["aupr_ci_low_without"]) / 2
        merged["gain_aupr_half_ci"] = np.sqrt(half_with**2 + half_without**2)
        return merged

    gains = {marker: g for marker, df in results_by_marker.items() if (g := _gain_df(df)) is not None}
    if not gains:
        logger.info(
            "Skipping trajectory-value figure: needs the multi-feature-set capture "
            f"('{feature_set_without}' alongside '{feature_set_with}') — see "
            "eicu_sepsis_sensitivity_analysis.TRAJECTORY_ABLATED_FEATURE_SET_NAME."
        )
        return

    set_poster_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), sharey=True, gridspec_kw={"wspace": 0.08})

    for col, marker in enumerate(MARKERS):
        ax = axes[col]
        if marker not in gains:
            ax.axis("off")
            continue
        g = gains[marker]
        baseline_row = g.loc[g["perturbed_param"] == "baseline"].iloc[0]
        baseline_gain = float(baseline_row["gain_aupr"])
        noise_half = float(baseline_row["gain_aupr_half_ci"])

        ax.axhspan(baseline_gain - noise_half, baseline_gain + noise_half,
                   color="#c9ccd1", alpha=0.35, linewidth=0, zorder=0)
        ax.axhline(baseline_gain, color="#8a8a86", linewidth=1, linestyle=(0, (3, 2)), zorder=1)
        ax.axhline(0.0, color="#b03a2e", linewidth=1, alpha=0.7, zorder=1)
        ax.axvline(1.0, linestyle="--", color="#8a8a86", linewidth=1, alpha=0.8, zorder=1)

        for param, color in PARAM_COLORS.items():
            sub = g.loc[g["perturbed_param"] == param]
            sub = pd.concat([sub, baseline_row.to_frame().T], ignore_index=True)
            sub = sub.astype({"perturbation_factor": float, "gain_aupr": float}).sort_values("perturbation_factor")
            ax.plot(sub["perturbation_factor"], sub["gain_aupr"], marker="o", markersize=6,
                    linewidth=2, color=color, label=PARAM_LABELS[param])

        ax.set_title(MARKER_LABEL[marker], fontsize=13, fontweight="bold")
        ax.set_xlabel("Perturbation factor (× baseline threshold)")

    axes[0].set_ylabel("Δ AUPR (with trajectories − without)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Perturbed threshold", loc="upper right",
               bbox_to_anchor=(0.99, 0.92), frameon=False, ncol=1)

    fig.suptitle("Trajectory-feature AUPR gain vs. threshold perturbation", fontsize=15, y=0.98, x=0.01, ha="left")

    fig.subplots_adjust(top=0.85, bottom=0.16, left=0.06, right=0.985, wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def _collect_worst_case_deltas(results_by_marker: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """For every (marker, perturbed threshold, model), find the +/-50% sweep
    extreme with the larger |delta AUPR| vs. that SAME model's own baseline
    (perturbation_factor==1.0) row, and record the SIGNED delta for both AUPR
    and AUROC at that one grid point — AUROC is read at the AUPR-worst-case
    point rather than independently re-optimized, so the two metrics in
    plot_model_robustness's two panels always describe the same grid point."""
    rows = []
    for marker, df in results_by_marker.items():
        if "model_name" not in df.columns:
            continue
        baseline_all = df.loc[df["perturbed_param"] == "baseline"]
        for param in PARAM_COLORS:
            sub = df.loc[df["perturbed_param"] == param]
            for model_name in sorted(df["model_name"].unique()):
                m_sub = sub.loc[sub["model_name"] == model_name]
                m_base = baseline_all.loc[baseline_all["model_name"] == model_name]
                if m_sub.empty or m_base.empty:
                    continue
                base_aupr = float(m_base["median_aupr"].iloc[0])
                base_auroc = float(m_base["mean_auroc"].iloc[0])
                candidates = []
                for factor in (0.5, 1.5):
                    at_factor = m_sub.loc[np.isclose(m_sub["perturbation_factor"], factor)]
                    if at_factor.empty:
                        continue
                    d_aupr = float(at_factor["median_aupr"].iloc[0]) - base_aupr
                    d_auroc = float(at_factor["mean_auroc"].iloc[0]) - base_auroc
                    candidates.append((abs(d_aupr), d_aupr, d_auroc, factor))
                if not candidates:
                    continue
                _, worst_d_aupr, worst_d_auroc, worst_factor = max(candidates, key=lambda c: c[0])
                rows.append({
                    "marker": marker, "param": param, "model_name": model_name,
                    "delta_aupr": worst_d_aupr, "delta_auroc": worst_d_auroc,
                    "worst_factor": worst_factor,
                })
    if not rows:
        return None
    return pd.DataFrame(rows)


def plot_model_robustness(
    results_by_marker: dict[str, pd.DataFrame], out_path: Path, feature_set_name: str,
) -> None:
    """For every (marker, perturbed threshold) combination, show the SIGNED
    change in AUPR and AUROC vs. baseline at whichever +/-50% sweep extreme
    hits that model hardest, once per model type — so a reader can see both
    whether "which threshold matters most downstream" is classifier-dependent
    AND whether the effect is a performance gain or a loss (an unsigned |delta|
    collapses that distinction)."""
    import matplotlib.pyplot as plt

    long_df = _collect_worst_case_deltas(results_by_marker)
    if long_df is None:
        logger.info(
            "Skipping model-robustness figure: none of the loaded CSVs have a `model_name` column with "
            "more than one model. This figure needs the full-capture re-run (all 4 model types kept per "
            "grid point) — see eicu_sepsis_sensitivity_analysis.run_identical_pipeline_cv."
        )
        return
    n_models = long_df["model_name"].nunique()
    if n_models < 2:
        logger.info(
            f"Skipping model-robustness figure: only {n_models} model type found in the CSVs "
            f"(need the full-capture re-run to get all 4)."
        )
        return

    combo_order = (
        long_df.assign(abs_delta_aupr=long_df["delta_aupr"].abs())
        .groupby(["marker", "param"])["abs_delta_aupr"].mean()
        .sort_values(ascending=True).index.tolist()
    )

    set_poster_style()
    n_rows = len(combo_order)
    fig, (ax_aupr, ax_auroc) = plt.subplots(
        1, 2, figsize=(13, 0.55 * n_rows + 2.9), sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    for ax, metric_col, metric_label in (
        (ax_aupr, "delta_aupr", "Δ AUPR"), (ax_auroc, "delta_auroc", "Δ AUROC"),
    ):
        for i, (marker, param) in enumerate(combo_order):
            cell = long_df.loc[(long_df["marker"] == marker) & (long_df["param"] == param)]
            color = PARAM_COLORS[param]
            vals = cell[metric_col].to_numpy()
            if len(vals) > 1:
                ax.plot([vals.min(), vals.max()], [i, i], color=color, linewidth=1.5, alpha=0.5, zorder=1)
            for _, r in cell.iterrows():
                ax.scatter(r[metric_col], i, marker=MODEL_MARKERS.get(r["model_name"], "o"),
                           s=70, color=color, edgecolor="white", linewidth=0.8, zorder=2)
        ax.axvline(0.0, color="#8a8a86", linewidth=1, zorder=0)
        ax.set_xlabel(f"{metric_label} at worst-case extreme\n(← worse    better →)")

    ax_aupr.set_yticks(range(n_rows))
    ax_aupr.set_yticklabels([f"{MARKER_LABEL[m]} · {PARAM_LABELS[p]}" for m, p in combo_order], fontsize=10)

    # Fixed figure-fraction placement throughout this header, not axes-relative
    # bbox_to_anchor / tight_layout: the first version of this figure had the
    # title, both legends, and the caption all competing for the same space
    # because ax.legend(bbox_to_anchor=...) is axes-relative and tight_layout
    # doesn't reserve room for legend/text artists — the same overflow-into-
    # neighbor bug fixed in plot_sensitivity_overview, in a new spot again.
    fig.suptitle("Worst-case sensitivity by classifier", fontsize=13.5, y=0.975, x=0.01, ha="left")

    param_handles = [plt.Line2D([0], [0], marker="o", color=c, linestyle="", markersize=8, label=PARAM_LABELS[p])
                      for p, c in PARAM_COLORS.items()]
    model_handles = [plt.Line2D([0], [0], marker=m, color="#5a5a56", linestyle="", markersize=8, label=name)
                      for name, m in MODEL_MARKERS.items()]
    # Stacked, not side-by-side: two independent legends each anchored at
    # x=0.01/0.99 (figure edges) collided in the middle once the model
    # legend's long labels (HistGradientBoosting, LogisticRegression) were
    # accounted for — each row gets the figure's full width instead.
    fig.legend(handles=param_handles, title="Threshold (color)", loc="upper left",
               bbox_to_anchor=(0.01, 0.925), ncol=3, frameon=False, fontsize=8.5)
    fig.legend(handles=model_handles, title="Model (shape)", loc="upper left",
               bbox_to_anchor=(0.01, 0.87), ncol=4, frameon=False, fontsize=8.5)

    header_frac = 2.05 / fig.get_figheight()   # suptitle + 2 stacked legend rows, fixed inches regardless of n_rows
    fig.subplots_adjust(top=1 - header_frac, bottom=0.14, left=0.20, right=0.97)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def _collect_feature_set_winners(
    results_by_marker: dict[str, pd.DataFrame], feature_set_with: str, feature_set_without: str,
) -> pd.DataFrame | None:
    """For every (marker, threshold param incl. "baseline", perturbation
    factor, model), which of the two captured feature sets — trajectory-
    inclusive vs. trajectory-ablated — has the higher AUPR, and separately the
    higher AUROC. One row per (marker, param, factor, model); "winner" columns
    hold the *feature set name*, not a model name — this is the feature-set
    analogue of the old model-comparison grid, evaluated independently per
    classifier rather than pooled across classifiers."""
    rows = []
    for marker, df in results_by_marker.items():
        if "feature_set" not in df.columns or "model_name" not in df.columns:
            continue
        for model_name in sorted(df["model_name"].unique()):
            m_df = df.loc[df["model_name"] == model_name]
            for param in list(PARAM_COLORS) + ["baseline"]:
                sub = m_df.loc[m_df["perturbed_param"] == param]
                for factor in sorted(sub["perturbation_factor"].unique()):
                    grid = sub.loc[np.isclose(sub["perturbation_factor"], factor)]
                    with_row = grid.loc[grid["feature_set"] == feature_set_with]
                    without_row = grid.loc[grid["feature_set"] == feature_set_without]
                    if with_row.empty or without_row.empty:
                        continue
                    aupr_with = float(with_row["median_aupr"].iloc[0])
                    aupr_without = float(without_row["median_aupr"].iloc[0])
                    auroc_with = float(with_row["mean_auroc"].iloc[0])
                    auroc_without = float(without_row["mean_auroc"].iloc[0])
                    rows.append({
                        "marker": marker, "param": param, "model_name": model_name,
                        "perturbation_factor": float(factor),
                        "aupr_winner": feature_set_with if aupr_with >= aupr_without else feature_set_without,
                        "auroc_winner": feature_set_with if auroc_with >= auroc_without else feature_set_without,
                    })
    if not rows:
        return None
    return pd.DataFrame(rows)


def plot_feature_set_stability(
    results_by_marker: dict[str, pd.DataFrame], out_path: Path,
    feature_set_with: str, feature_set_without: str,
) -> None:
    """For every (marker, threshold, perturbation level), does the
    trajectory-inclusive feature set still beat its trajectory-ablated
    counterpart on AUPR, and separately on AUROC — evaluated separately per
    classifier family (one grid per model), rather than the earlier version of
    this figure, which asked which *model* wins (pooling across models and
    losing the with/without-trajectory distinction entirely). This is the
    sharper question for a paper whose contribution *is* the trajectory
    feature type: not "does AUPR move" (sensitivity_combined /
    sensitivity_model_robustness) but "does the trajectory-feature advantage
    remain the winning feature set," and is that answer classifier-dependent."""
    import matplotlib.pyplot as plt

    long_df = _collect_feature_set_winners(results_by_marker, feature_set_with, feature_set_without)
    if long_df is None:
        logger.info(
            "Skipping feature-set-stability figure: needs both `feature_set` and `model_name` columns (the "
            "full-capture re-run) — see eicu_sepsis_sensitivity_analysis.run_identical_pipeline_cv."
        )
        return
    present_models = [m for m in MODEL_ORDER if m in long_df["model_name"].unique()]
    if not present_models:
        logger.info("Skipping feature-set-stability figure: none of the expected model types found in the CSVs.")
        return

    perts = [0.5, 0.75, 1.0, 1.25, 1.5]
    combo_order = [(m, p) for m in MARKERS if m in results_by_marker for p in PARAM_COLORS]
    n_rows = len(combo_order)

    baseline_winner: dict[tuple, dict] = {}
    for marker in MARKERS:
        for model_name in present_models:
            base = long_df.loc[
                (long_df["marker"] == marker) & (long_df["model_name"] == model_name)
                & (long_df["param"] == "baseline")
            ]
            if base.empty:
                continue
            baseline_winner[(marker, model_name)] = {
                "aupr": base["aupr_winner"].iloc[0], "auroc": base["auroc_winner"].iloc[0],
            }

    def role(feature_set_name: str) -> str:
        return "with_trajectory" if feature_set_name == feature_set_with else "without_trajectory"

    neutral_bg, changed_bg = "#eef1f4", "#f6e4b8"  # changed_bg: a light tint of the poster's own gold family

    def _draw_panel(ax, model_name: str, metric_key: str) -> None:
        for i, (marker, param) in enumerate(combo_order):
            base_winner = baseline_winner.get((marker, model_name), {}).get(metric_key)
            for j, pert in enumerate(perts):
                cell = long_df.loc[
                    (long_df["marker"] == marker) & (long_df["param"] == param)
                    & (long_df["model_name"] == model_name) & (np.isclose(long_df["perturbation_factor"], pert))
                ]
                if cell.empty:
                    continue
                winner = cell[f"{metric_key}_winner"].iloc[0]
                changed = base_winner is not None and pert != 1.0 and winner != base_winner
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                            facecolor=changed_bg if changed else neutral_bg,
                                            edgecolor="white", linewidth=2, zorder=1))
                if pert == 1.0:
                    ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, facecolor="none",
                                                edgecolor="#2a78d6", linewidth=1.6, linestyle=(0, (3, 2)), zorder=3))
                r = role(winner)
                ax.text(j, i, FEATURE_SET_ABBREV_BY_ROLE[r], ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=FEATURE_SET_COLORS_BY_ROLE[r], zorder=2)
        ax.set_xlim(-0.6, len(perts) - 0.4)
        ax.set_ylim(n_rows - 0.4, -0.6)
        ax.set_xticks(range(len(perts)))
        ax.set_xticklabels([f"×{p:g}" for p in perts], fontsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    set_poster_style()
    fig, axes = plt.subplots(
        2, len(present_models), figsize=(3.05 * len(present_models) + 1.7, 0.42 * n_rows + 1.7),
        sharey=True, gridspec_kw={"wspace": 0.10, "hspace": 0.38},
    )
    axes = np.atleast_2d(axes).reshape(2, len(present_models))
    for col, model_name in enumerate(present_models):
        _draw_panel(axes[0, col], model_name, "aupr")
        axes[0, col].set_title(f"{model_name}\nbest by AUPR", fontsize=10.5, loc="left", pad=8)
        _draw_panel(axes[1, col], model_name, "auroc")
        axes[1, col].set_title("best by AUROC", fontsize=10.5, loc="left", pad=8)
    axes[0, 0].set_yticks(range(n_rows))
    axes[0, 0].set_yticklabels([f"{MARKER_LABEL[m]} · {PARAM_LABELS[p]}" for m, p in combo_order], fontsize=9)
    axes[1, 0].set_yticks(range(n_rows))
    axes[1, 0].set_yticklabels([f"{MARKER_LABEL[m]} · {PARAM_LABELS[p]}" for m, p in combo_order], fontsize=9)

    fig.suptitle("Winning feature set by classifier", fontsize=13, y=0.98, x=0.01, ha="left")

    winner_legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=neutral_bg, edgecolor="none", label="same winner as baseline"),
        plt.Rectangle((0, 0), 1, 1, facecolor=changed_bg, edgecolor="none", label="different winner than baseline"),
    ]
    fig.legend(handles=winner_legend_handles, loc="upper left", bbox_to_anchor=(0.01, 0.90),
               ncol=2, frameon=False, fontsize=8.5)

    feature_set_legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
                   color=FEATURE_SET_COLORS_BY_ROLE[role_key], label=FEATURE_SET_ABBREV_BY_ROLE[role_key])
        for role_key in ("with_trajectory", "without_trajectory")
    ]
    fig.legend(handles=feature_set_legend_handles, loc="upper left", bbox_to_anchor=(0.42, 0.90),
               ncol=2, frameon=False, fontsize=8.5)

    # Panel titles (ax.set_title) live INSIDE each axes' own allocated space,
    # not the header_frac band — that band only needs to fit the suptitle and
    # legend row above it. Sized empirically (previous values were inherited
    # from the single-row-of-panels predecessor figure and left ~30% of the
    # figure as dead space once this one became a 2-row-by-4-column grid of
    # much shorter individual panels).
    header_frac = 1.2 / fig.get_figheight()   # suptitle + legend row, PLUS the panel
                                                # titles' own pad, which (no tight_layout/spines here)
                                                # is drawn just above each axes' top edge, encroaching
                                                # into this band rather than the axes' own space.
    caption_frac = 0.5 / fig.get_figheight()  # each axes' xtick labels, which (no tight_layout/spines here)
                                                # render just below the axes' bottom edge into this band.
    fig.subplots_adjust(top=1 - header_frac, bottom=caption_frac, left=0.15, right=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Combined manuscript figures for the eICU sepsis sensitivity grid")
    parser.add_argument("--results-dir", type=str, default="results/eicu/sepsis")
    parser.add_argument("--feature-set-name", type=str, default=DEFAULT_FEATURE_SET_NAME)
    parser.add_argument("--trajectory-ablated-feature-set-name", type=str,
                         default=TRAJECTORY_ABLATED_FEATURE_SET_NAME)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_by_marker = {marker: load_marker_csv(results_dir, marker) for marker in MARKERS}

    plot_combined_overview(
        results_by_marker, results_dir / "sensitivity_combined_eicu_sepsis.png",
        feature_set_name=args.feature_set_name, model_name=args.model_name,
    )
    plot_model_robustness(
        results_by_marker, results_dir / "sensitivity_model_robustness_eicu_sepsis.png",
        feature_set_name=args.feature_set_name,
    )
    plot_feature_set_stability(
        results_by_marker, results_dir / "sensitivity_feature_set_stability_eicu_sepsis.png",
        feature_set_with=args.feature_set_name, feature_set_without=args.trajectory_ablated_feature_set_name,
    )
    plot_trajectory_value(
        results_by_marker, results_dir / "sensitivity_trajectory_value_eicu_sepsis.png",
        model_name=args.model_name, feature_set_with=args.feature_set_name,
        feature_set_without=args.trajectory_ablated_feature_set_name,
    )


if __name__ == "__main__":
    main()
