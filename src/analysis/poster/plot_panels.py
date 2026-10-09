from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .feature_set_mapper import CANONICAL_LABELS
from .palette import get_metric_palettes, set_poster_style
import math


def _save_figure(fig: plt.Figure, out_base: Path, formats: list[str], dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.2)


MODEL_ABBR = {
    "LogisticRegression": "LR",
    "Logistic Regression": "LR",
    "RandomForest": "RF",
    "Random Forest": "RF",
    "HistGradientBoosting": "HGB",
    "Gradient Boosting": "HGB",
    "XGBoost": "XGB",
}

MODEL_ORDER = ["LR", "RF", "HGB", "XGB"]

FEATURE_ABBR = {
    "Static + Dynamic": "SD",
    "Static + Dynamic + Trajectory": "SD+T",
    "Static + Dynamic + Trajectory (Bootstrap)": "SD+T (B)",
    "Static + Dynamic + Summary": "SD+S",
    "Static + Dynamic + Summary + Trajectory": "SD+S+T",
    "Static + Dynamic + Summary + Trajectory (Bootstrap)": "SD+S+T (B)",
}


def _abbr_feature_label(label: str) -> str:
    return FEATURE_ABBR.get(label, label)


def _abbr_model_label(label: str) -> str:
    return MODEL_ABBR.get(label, label)


def plot_main_panel(
    fold_df: pd.DataFrame,
    canonical_order: list[str],
    title_prefix: str,
    out_base: Path,
    formats: list[str],
    dpi: int,
    figsize: tuple[float, float],
    chance_baseline: dict[str, float] | None = None,
) -> None:
    """
    chance_baseline: optional {"auroc": 0.5, "aupr": positive_rate} — draws a
    dashed reference line on the corresponding metric's axis. Omitted (None)
    by default so existing callers are unaffected.
    """
    set_poster_style()
    sns.set_style("whitegrid")
    metric_palettes = get_metric_palettes(len(canonical_order))

    work = fold_df.copy()
    work["feature_label"] = work["canonical_feature_set"].map(CANONICAL_LABELS)
    work["model_label"] = work["model"].map(_abbr_model_label)
    label_order = [CANONICAL_LABELS[c] for c in canonical_order if c in CANONICAL_LABELS]
    model_order = [m for m in MODEL_ORDER if m in work["model_label"].unique()]

    fig_w = max(float(figsize[0]), 18.0)
    fig_h = max(float(figsize[1]), 6.8)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    for ax, metric in zip(axes, ["auroc", "aupr"]):
        pal = dict(zip(label_order, metric_palettes[metric]))
        sns.boxplot(
            data=work,
            x="model_label",
            y=metric,
            order=model_order,
            hue="feature_label",
            hue_order=label_order,
            palette=pal,
            ax=ax,
            fliersize=1.8,
            linewidth=1.0,
        )
        baseline_val = chance_baseline.get(metric) if chance_baseline else None
        if baseline_val is not None:
            ax.axhline(baseline_val, color="black", linestyle="--", linewidth=1.0, alpha=0.6, zorder=0)
        ax.set_title("AUROC" if metric == "auroc" else "AUPR")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.upper())

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        short_labels = [_abbr_feature_label(lbl) for lbl in labels]
        ax.legend(
            handles,
            short_labels,
            title="Feature Set",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=3,
            frameon=False,
        )

    fig.suptitle(f"{title_prefix}: model-stratified CV performance", y=1.03)
    fig.subplots_adjust(bottom=0.28, top=0.88, wspace=0.2)
    _save_figure(fig, out_base, formats, dpi)
    plt.close(fig)


def plot_best_model_panel(
    best_model_df: pd.DataFrame,
    canonical_order: list[str],
    sig_df: pd.DataFrame,
    title_prefix: str,
    best_model: str,
    out_base: Path,
    formats: list[str],
    dpi: int,
    figsize: tuple[float, float],
    chance_baseline: dict[str, float] | None = None,
) -> None:
    """
    chance_baseline: optional {"auroc": 0.5, "aupr": positive_rate} — draws a
    dashed reference line on the corresponding metric's axis. Omitted (None)
    by default so existing callers are unaffected.
    """
    set_poster_style()
    sns.set_style("whitegrid")
    metric_palettes = get_metric_palettes(len(canonical_order))

    work = best_model_df.copy()
    work["feature_label"] = work["canonical_feature_set"].map(CANONICAL_LABELS)
    work["feature_abbr"] = work["feature_label"].map(_abbr_feature_label)
    label_order = [CANONICAL_LABELS[c] for c in canonical_order if c in CANONICAL_LABELS]
    abbr_order = [_abbr_feature_label(lbl) for lbl in label_order]

    fig_w = max(float(figsize[0]), 16.5)
    fig_h = max(float(figsize[1]), 7.2)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    for ax, metric in zip(axes, ["auroc", "aupr"]):
        pal = dict(zip(label_order, metric_palettes[metric]))
        sns.boxplot(
            data=work,
            x="feature_abbr",
            y=metric,
            hue="feature_label",
            order=abbr_order,
            hue_order=label_order,
            palette=pal,
            dodge=False,
            ax=ax,
            fliersize=2.0,
            linewidth=1.1,
        )
        ax.set_title("AUROC" if metric == "auroc" else "AUPR")
        ax.set_xlabel("Feature Set")
        ax.set_ylabel(metric.upper())

        handles, labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.legend(
            handles,
            [_abbr_feature_label(lbl) for lbl in labels],
            title="Feature Set",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=3,
            frameon=False,
        )

        metric_sig = sig_df[sig_df["metric"] == metric].copy()
        if work.empty:
            continue

        y_max = float(work[metric].max())
        y_min = float(work[metric].min())
        baseline_val = chance_baseline.get(metric) if chance_baseline else None
        if baseline_val is not None:
            ax.axhline(baseline_val, color="black", linestyle="--", linewidth=1.0, alpha=0.6, zorder=0)
            # Fold the baseline into the span so it stays visible once ylim
            # is explicitly set below (with or without significance brackets).
            y_max = max(y_max, baseline_val)
            y_min = min(y_min, baseline_val)
        y_span = max(y_max - y_min, 0.02)
        # place brackets closer to the boxplots but increase spacing between stacked brackets
        start_offset = 0.03 * y_span
        step = 0.22 * y_span

        # Draw bracketed pairwise comparisons from metric_sig (expects columns: left,right,stars)
        if not metric_sig.empty:
            # ensure deterministic order
            metric_sig = metric_sig.reset_index(drop=True)
            drawn = 0
            for i, row in metric_sig.iterrows():
                left_id = row.get("left")
                right_id = row.get("right")
                stars = str(row.get("stars", ""))
                if not left_id or not right_id or not stars:
                    continue
                # map canonical ids to abbreviated x positions
                try:
                    left_label = _abbr_feature_label(CANONICAL_LABELS[left_id])
                    right_label = _abbr_feature_label(CANONICAL_LABELS[right_id])
                except Exception:
                    continue
                if left_label not in abbr_order or right_label not in abbr_order:
                    continue
                x_left = abbr_order.index(left_label)
                x_right = abbr_order.index(right_label)
                if x_left == x_right:
                    continue
                # order
                x0, x1 = (x_left, x_right) if x_left < x_right else (x_right, x_left)
                y = y_max + start_offset + (drawn * step)
                # horizontal line
                ax.plot([x0, x1], [y, y], color="black", linewidth=1.0)
                # vertical ticks (shorter since brackets are closer to boxes)
                tick_h = 0.02 * y_span
                ax.plot([x0, x0], [y, y - tick_h], color="black", linewidth=1.0)
                ax.plot([x1, x1], [y, y - tick_h], color="black", linewidth=1.0)
                # stars above horizontal line (keep some gap)
                star_y = y + 0.04 * y_span
                ax.text((x0 + x1) / 2.0, star_y, stars, ha="center", va="bottom", fontsize=12, fontweight="bold")
                drawn += 1

            # expand y limits to accommodate top-most bracket
            y_upper = y_max + start_offset + (step * drawn) + (0.08 * y_span)
            y_lower = y_min - (0.04 * y_span)
            ax.set_ylim(y_lower, y_upper)
        elif baseline_val is not None:
            # No significance brackets to accommodate, but still make sure
            # the baseline line (possibly outside the boxplot's own data
            # range) stays within view.
            ax.set_ylim(y_min - 0.04 * y_span, y_max + 0.08 * y_span)

    fig.suptitle(f"{title_prefix}: best model ({best_model}) with significance", y=1.03)
    fig.subplots_adjust(bottom=0.30, top=0.86, wspace=0.2)
    _save_figure(fig, out_base, formats, dpi)
    plt.close(fig)
