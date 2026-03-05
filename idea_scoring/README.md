# Idea Scoring Module

Reusable scoring utilities for evaluating idea-generation approaches with:

1. **Rubric scoring** (absolute quality evaluation)
2. **Bradley-Terry Elo scoring** (pairwise preference ranking)
3. **Optional blended score** combining both views

## Why this module exists

Idea-generation systems often need two complementary lenses:

- **Rubric scores** tell you how good an approach is against explicit criteria.
- **Elo scores** tell you which approach tends to win in direct comparisons.

Using both improves stability versus relying on only one signal.

## Module structure

- `idea_scoring/rubric.py` - weighted rubric scorer
- `idea_scoring/elo.py` - Bradley-Terry style Elo updates
- `idea_scoring/blend.py` - helper to blend rubric + Elo
- `idea_scoring/__init__.py` - public exports

## 1) Rubric scoring

Define a rubric once, then score each candidate approach.

```python
from idea_scoring import RubricCriterion, RubricScorer

criteria = [
    RubricCriterion(name="novelty", weight=0.30, min_score=1, max_score=5),
    RubricCriterion(name="feasibility", weight=0.25, min_score=1, max_score=5),
    RubricCriterion(name="expected_impact", weight=0.30, min_score=1, max_score=5),
    RubricCriterion(name="evaluation_clarity", weight=0.15, min_score=1, max_score=5),
]

scorer = RubricScorer(criteria, target_scale=100.0)

score_a = scorer.score(
    "approach_a",
    {
        "novelty": 4.5,
        "feasibility": 3.8,
        "expected_impact": 4.2,
        "evaluation_clarity": 4.0,
    },
)

print(score_a.total_score)
```

## 2) Bradley-Terry Elo scoring

Track pairwise wins/losses/ties between approaches.

```python
from idea_scoring import BradleyTerryElo, PairwiseResult

elo = BradleyTerryElo(base_rating=1500, k_factor=24, scale=400)

results = [
    PairwiseResult("approach_a", "approach_b", score_a=1.0),  # A beats B
    PairwiseResult("approach_a", "approach_c", score_a=0.0),  # C beats A
    PairwiseResult("approach_b", "approach_c", score_a=0.5),  # tie
]

elo.run_tournament(results)
print(elo.ranking())
```

## 3) Blend rubric + Elo

Use this when you want one final ranking signal.

```python
from idea_scoring import blend_rubric_and_elo

rubric_scores = {
    "approach_a": 82.4,
    "approach_b": 76.1,
    "approach_c": 79.0,
}

elo_ratings = {
    "approach_a": 1525.4,
    "approach_b": 1490.0,
    "approach_c": 1512.7,
}

final_scores = blend_rubric_and_elo(
    rubric_scores=rubric_scores,
    elo_ratings=elo_ratings,
    rubric_weight=0.65,
    elo_weight=0.35,
    target_scale=100.0,
)
```

## Recommended workflow for experiments

1. Score each idea with the rubric.
2. Run pairwise head-to-head comparisons and update Elo.
3. Blend both scores for final ranking.
4. Log all raw rubric ratings and pairwise outcomes for reproducibility.

## Notes

- Elo is **relative** and depends on who is compared and in what sequence.
- Rubric is **absolute** and depends on criterion calibration.
- If all Elo ratings are equal, the blend helper assigns all approaches mid-scale Elo.
