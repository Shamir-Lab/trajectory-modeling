from __future__ import annotations

import matplotlib.pyplot as plt


def get_metric_palettes(n_colors: int) -> dict[str, list[str]]:
    """
    Poster-compatible palette families derived from anchors:
    - blue-gray: #CCD8E7
    - dark orange: #93540A
    - dark gold: #7A620E
    """
    auroc_scale = ["#EAF0F7", "#CCD8E7", "#A8BDD3", "#7C98B3", "#4F6E8C", "#334A63"]
    aupr_scale = ["#F1E5D2", "#D9BE93", "#B88A4A", "#93540A", "#7A620E", "#5D4B10"]

    def _expand(scale: list[str], n: int) -> list[str]:
        if n <= len(scale):
            return scale[:n]
        out = []
        for i in range(n):
            out.append(scale[i % len(scale)])
        return out

    return {
        "auroc": _expand(auroc_scale, n_colors),
        "aupr": _expand(aupr_scale, n_colors),
    }


def set_poster_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
