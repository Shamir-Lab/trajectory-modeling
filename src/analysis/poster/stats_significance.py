from __future__ import annotations

from math import comb

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest


def p_to_stars(pvalue: float, thresholds: dict[float, str] = None) -> str:
    if thresholds is None:
        thresholds = {0.001: "***", 0.01: "**", 0.05: "*"}
    for thr in sorted(thresholds.keys()):
        if pvalue < thr:
            return thresholds[thr]
    return ""


def _sign_test_pvalue(deltas: np.ndarray) -> float:
    nonzero = deltas[np.abs(deltas) > 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k_pos = int((nonzero > 0).sum())
    k = min(k_pos, n - k_pos)
    cdf = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * cdf)


def _paired_wilcoxon_or_fallback(x: np.ndarray, y: np.ndarray) -> float:
    deltas = y - x
    try:
        from scipy.stats import wilcoxon  # type: ignore

        stat = wilcoxon(deltas)
        return float(stat.pvalue)
    except Exception:
        return float(_sign_test_pvalue(deltas))


def compare_pairwise(best_model_df: pd.DataFrame,
                     comparisons: List[Tuple[str, str]],
                     metrics: List[str] = ("auroc", "aupr")) -> pd.DataFrame:
    """Compute paired tests for an explicit list of feature-set comparisons.

    comparisons: list of (left, right) canonical feature-set ids, e.g. ("sd","sd_traj").
    Returns DataFrame with columns: metric, left, right, n_pairs, p_value, stars.
    """
    rows = []
    for metric in metrics:
        for left_id, right_id in comparisons:
            left = best_model_df.loc[best_model_df["canonical_feature_set"] == left_id, metric]
            right = best_model_df.loc[best_model_df["canonical_feature_set"] == right_id, metric]
            # align by index order (assumes same ordering of repeats/folds for each feature-set)
            merged = pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1).dropna()
            if merged.shape[0] < 1:
                p = np.nan
                n = 0
            else:
                try:
                    stat, p = wilcoxon(merged.iloc[:, 0], merged.iloc[:, 1])
                    n = merged.shape[0]
                except Exception:
                    # fallback to sign-test-like behavior
                    dif = merged.iloc[:, 1] - merged.iloc[:, 0]
                    n = int(dif.count())
                    # use binomial test as fallback
                    p = float(binomtest(int((dif > 0).sum()), n).pvalue) if n > 0 else np.nan
            rows.append({"metric": metric, "left": left_id, "right": right_id, "n_pairs": n, "p_value": p, "stars": p_to_stars(p)})
    return pd.DataFrame(rows)
