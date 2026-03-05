from __future__ import annotations

from typing import Dict, Mapping


def _min_max_normalize(values: Mapping[str, float], target_scale: float) -> Dict[str, float]:
    if not values:
        return {}
    min_value = min(values.values())
    max_value = max(values.values())
    if max_value == min_value:
        # If all Elo scores are equal, place all at mid-scale.
        return {k: 0.5 * target_scale for k in values}
    span = max_value - min_value
    return {k: ((v - min_value) / span) * target_scale for k, v in values.items()}


def blend_rubric_and_elo(
    rubric_scores: Mapping[str, float],
    elo_ratings: Mapping[str, float],
    rubric_weight: float = 0.6,
    elo_weight: float = 0.4,
    target_scale: float = 100.0,
) -> Dict[str, float]:
    """
    Blend absolute rubric scores with relative Elo rankings.

    rubric_scores should already be on target_scale (e.g., 0-100).
    elo_ratings are min-max normalized to target_scale before blending.
    """

    if rubric_weight < 0 or elo_weight < 0:
        raise ValueError("rubric_weight and elo_weight must be >= 0.")
    if rubric_weight == 0 and elo_weight == 0:
        raise ValueError("At least one weight must be > 0.")
    if target_scale <= 0:
        raise ValueError("target_scale must be > 0.")

    normalized_elo = _min_max_normalize(elo_ratings, target_scale)
    all_ids = set(rubric_scores) | set(normalized_elo)
    if not all_ids:
        return {}

    total_weight = rubric_weight + elo_weight
    blended: Dict[str, float] = {}
    for approach_id in all_ids:
        rubric_value = rubric_scores.get(approach_id, 0.0)
        elo_value = normalized_elo.get(approach_id, 0.0)
        blended_value = (
            rubric_weight * rubric_value + elo_weight * elo_value
        ) / total_weight
        blended[approach_id] = blended_value
    return blended
