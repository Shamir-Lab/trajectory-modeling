from __future__ import annotations

import re


CANONICAL_LABELS = {
    "sd": "Static + Dynamic",
    "sd_sum": "Static + Dynamic + Summary",
    "sd_traj": "Static + Dynamic + Trajectory",
    "sd_sum_traj": "Static + Dynamic + Summary + Trajectory",
    "sd_traj_boot": "Static + Dynamic + Trajectory (Bootstrap)",
    "sd_sum_traj_boot": "Static + Dynamic + Summary + Trajectory (Bootstrap)",
}


DEFAULT_ALIASES: dict[str, list[str]] = {
    "sd": [
        r"^Static \+ Dynamic$",
        r"^Baseline$",
    ],
    "sd_sum": [
        r"^Summary Stats \+ Static \+ Dynamic$",
        r"^Static \+ Dynamic \+ Multi-Marker Summary Stats$",
        r"^Static \+ Dynamic \+ .* Summary Stats$",
        r"^Baseline \+ Multi Summary$",
        r"^Baseline \+ Single Summary$",
    ],
    "sd_traj": [
        r"^Trajectory \+ Static \+ Dynamic$",
        r"^Static \+ Dynamic \+ Multi-Marker Trajectories$",
        r"^Static \+ Dynamic \+ .* Trajectory$",
        r"^Baseline \+ Multi Trajectory$",
        r"^Baseline \+ Single Trajectory$",
    ],
    "sd_sum_traj": [
        r"^Trajectory \+ Summary Stats \+ Static \+ Dynamic$",
        r"^Static \+ Dynamic \+ Multi-Marker Trajectories \+ Summary Stats$",
        r"^Static \+ Dynamic \+ .* Trajectory \+ Summary Stats$",
        r"^Baseline \+ Multi Trajectory \+ Summary$",
        r"^Baseline \+ Single Trajectory \+ Summary$",
    ],
    "sd_traj_boot": [
        r"^Trajectory \+ Static \+ Dynamic \(Bootstrap\)$",
        r"^Static \+ Dynamic \+ Multi-Marker Trajectories \(Bootstrap\)$",
        r"^Static \+ Dynamic \+ .* Trajectory \(Bootstrap\)$",
        r"^Baseline \+ Multi Trajectory \(Bootstrap\)$",
        r"^Baseline \+ Single Trajectory \(Bootstrap\)$",
    ],
    "sd_sum_traj_boot": [
        r"^Trajectory \+ Summary Stats \+ Static \+ Dynamic \(Bootstrap\)$",
        r"^Static \+ Dynamic \+ Multi-Marker Trajectories \+ Summary Stats \(Bootstrap\)$",
        r"^Static \+ Dynamic \+ .* Trajectory \+ Summary Stats \(Bootstrap\)$",
        r"^Baseline \+ Multi Trajectory \+ Summary \(Bootstrap\)$",
        r"^Baseline \+ Single Trajectory \+ Summary \(Bootstrap\)$",
    ],
}


def _score_candidate(name: str, multi_marker_preference: bool) -> tuple[int, int, int]:
    lname = name.lower()
    multi_bonus = 1 if (multi_marker_preference and "multi-marker" in lname) else 0
    bootstrap_bonus = 1 if "(bootstrap)" in lname else 0
    # prefer exact-ish concise names for stability
    length_penalty = -len(name)
    return (multi_bonus, bootstrap_bonus, length_penalty)


def _match_aliases(available: list[str], patterns: list[str]) -> list[str]:
    matched: list[str] = []
    for pattern in patterns:
        rx = re.compile(pattern, flags=re.IGNORECASE)
        for name in available:
            if rx.search(name):
                matched.append(name)
    # preserve order but unique
    return list(dict.fromkeys(matched))


def resolve_feature_mapping(
    available_feature_sets: list[str],
    canonical_ids: list[str],
    custom_aliases: dict[str, list[str]] | None = None,
    multi_marker_preference: bool = True,
) -> dict[str, str]:
    aliases = dict(DEFAULT_ALIASES)
    if custom_aliases:
        for key, pats in custom_aliases.items():
            aliases[key] = pats

    mapping: dict[str, str] = {}
    for canonical_id in canonical_ids:
        patterns = aliases.get(canonical_id, [])
        candidates = _match_aliases(available_feature_sets, patterns)
        if not candidates:
            continue
        pick = sorted(
            candidates,
            key=lambda n: _score_candidate(n, multi_marker_preference),
            reverse=True,
        )[0]
        mapping[canonical_id] = pick

    return mapping
