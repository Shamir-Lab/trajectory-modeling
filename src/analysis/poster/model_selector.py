from __future__ import annotations

import pandas as pd


def select_best_model_by_aupr(
    fold_df: pd.DataFrame,
    canonical_feature_ids: list[str],
) -> tuple[str, pd.DataFrame]:
    """
    Select best model type by AUPR across selected canonical feature sets.

    Returns:
        (best_model_name, leaderboard_df)
    """
    work = fold_df[fold_df["canonical_feature_set"].isin(canonical_feature_ids)].copy()
    if work.empty:
        raise ValueError("No rows available for best-model selection")

    grouped = (
        work.groupby("model", as_index=False)
        .agg(
            aupr_mean=("aupr", "mean"),
            auroc_mean=("auroc", "mean"),
            n_rows=("aupr", "size"),
        )
        .sort_values(["aupr_mean", "auroc_mean", "model"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    best_model = str(grouped.iloc[0]["model"])
    return best_model, grouped
