from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class PairwiseResult:
    """
    One pairwise comparison result.

    score_a is expected to be:
    - 1.0 when approach_a wins
    - 0.0 when approach_b wins
    - 0.5 for a tie
    """

    approach_a: str
    approach_b: str
    score_a: float
    weight: float = 1.0
    notes: Optional[str] = None


class BradleyTerryElo:
    """
    Bradley-Terry style Elo ranking for pairwise comparisons.

    Uses a logistic expected-score model and updates ratings after each match.
    """

    def __init__(
        self,
        base_rating: float = 1500.0,
        k_factor: float = 24.0,
        scale: float = 400.0,
    ) -> None:
        if k_factor <= 0:
            raise ValueError("k_factor must be > 0.")
        if scale <= 0:
            raise ValueError("scale must be > 0.")
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.scale = scale
        self.ratings: Dict[str, float] = {}

    def get_rating(self, approach_id: str) -> float:
        return self.ratings.get(approach_id, self.base_rating)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / self.scale))

    def record_result(self, result: PairwiseResult) -> Tuple[float, float]:
        if result.score_a < 0.0 or result.score_a > 1.0:
            raise ValueError("score_a must be in [0, 1].")
        if result.weight <= 0:
            raise ValueError("weight must be > 0.")
        if result.approach_a == result.approach_b:
            raise ValueError("approach_a and approach_b must be different.")

        rating_a = self.get_rating(result.approach_a)
        rating_b = self.get_rating(result.approach_b)
        expected_a = self._expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        score_b = 1.0 - result.score_a
        delta_a = self.k_factor * result.weight * (result.score_a - expected_a)
        delta_b = self.k_factor * result.weight * (score_b - expected_b)

        new_rating_a = rating_a + delta_a
        new_rating_b = rating_b + delta_b
        self.ratings[result.approach_a] = new_rating_a
        self.ratings[result.approach_b] = new_rating_b
        return new_rating_a, new_rating_b

    def record_win(self, winner: str, loser: str, weight: float = 1.0) -> None:
        self.record_result(
            PairwiseResult(
                approach_a=winner,
                approach_b=loser,
                score_a=1.0,
                weight=weight,
            )
        )

    def record_tie(self, approach_a: str, approach_b: str, weight: float = 1.0) -> None:
        self.record_result(
            PairwiseResult(
                approach_a=approach_a,
                approach_b=approach_b,
                score_a=0.5,
                weight=weight,
            )
        )

    def run_tournament(self, results: Iterable[PairwiseResult]) -> Dict[str, float]:
        for result in results:
            self.record_result(result)
        return dict(self.ratings)

    def ranking(self) -> List[Tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
