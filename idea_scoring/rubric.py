from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional


@dataclass(frozen=True)
class RubricCriterion:
    """Definition for one rubric criterion."""

    name: str
    weight: float
    min_score: float = 1.0
    max_score: float = 5.0
    description: str = ""

    def validate(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"Criterion '{self.name}' must have weight > 0.")
        if self.max_score <= self.min_score:
            raise ValueError(
                f"Criterion '{self.name}' must have max_score > min_score."
            )


@dataclass(frozen=True)
class RubricScore:
    """Final score details for one candidate approach."""

    approach_id: str
    total_score: float
    criterion_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    weighted_contributions: Dict[str, float]
    notes: Optional[str] = None


class RubricScorer:
    """
    Weighted rubric scorer.

    Converts per-criterion raw scores into a normalized weighted score on a
    configurable scale (default 0-100).
    """

    def __init__(self, criteria: List[RubricCriterion], target_scale: float = 100.0):
        if not criteria:
            raise ValueError("At least one criterion is required.")
        if target_scale <= 0:
            raise ValueError("target_scale must be > 0.")
        self.criteria = criteria
        self.target_scale = target_scale
        self._criteria_by_name = {c.name: c for c in criteria}
        if len(self._criteria_by_name) != len(criteria):
            raise ValueError("Criterion names must be unique.")
        for criterion in criteria:
            criterion.validate()
        self._total_weight = sum(c.weight for c in criteria)

    def score(
        self,
        approach_id: str,
        criterion_scores: Mapping[str, float],
        notes: Optional[str] = None,
    ) -> RubricScore:
        missing = [c.name for c in self.criteria if c.name not in criterion_scores]
        if missing:
            raise ValueError(
                f"Missing scores for approach '{approach_id}': {', '.join(missing)}"
            )

        normalized_scores: Dict[str, float] = {}
        weighted_contributions: Dict[str, float] = {}
        weighted_sum = 0.0

        for criterion in self.criteria:
            raw_score = float(criterion_scores[criterion.name])
            if raw_score < criterion.min_score or raw_score > criterion.max_score:
                raise ValueError(
                    f"Score for criterion '{criterion.name}' on approach "
                    f"'{approach_id}' must be in [{criterion.min_score}, "
                    f"{criterion.max_score}]."
                )
            normalized = (raw_score - criterion.min_score) / (
                criterion.max_score - criterion.min_score
            )
            contribution = normalized * criterion.weight
            normalized_scores[criterion.name] = normalized
            weighted_contributions[criterion.name] = contribution
            weighted_sum += contribution

        total_score = (weighted_sum / self._total_weight) * self.target_scale
        return RubricScore(
            approach_id=approach_id,
            total_score=total_score,
            criterion_scores={k: float(v) for k, v in criterion_scores.items()},
            normalized_scores=normalized_scores,
            weighted_contributions=weighted_contributions,
            notes=notes,
        )

    def score_many(
        self,
        evaluations: Mapping[str, Mapping[str, float]],
    ) -> Dict[str, RubricScore]:
        return {
            approach_id: self.score(approach_id, criterion_scores)
            for approach_id, criterion_scores in evaluations.items()
        }
