"""What monitoring without labels can and cannot see.

Three worlds, one model, one set of observable signals. The article's argument
is that everything a monitoring system can measure before labels arrive — the
share of positive predictions, the mean confidence, the calibration of the
scores against themselves — can be identical in a world where the model is
right and a world where it is exactly wrong.

The three worlds are joint distributions over a binary input ``X`` and a binary
outcome ``Y``. The model scores 0.9 when ``X = 1`` and 0.1 otherwise, and
predicts the majority. Then:

* **Reference** — ``Y`` follows ``X``, and the model is 90% accurate.
* **Hidden reversal** — ``Y`` is the opposite of ``X``, and the model is 10%
  accurate. The positive-prediction share and the mean confidence do not move.
* **Input shift only** — ``X`` shifts towards one, the model is still 90%
  accurate, and the positive-prediction share moves a long way.

So a change in the observable signal does not imply a change in accuracy, and an
unchanged signal does not imply an unchanged accuracy. Both directions fail.

The distributions are enumerated exactly. Nothing is sampled.
"""

from dataclasses import dataclass
from typing import Final

from blog_reproducibility.common.validation import probability

__all__ = [
    "OUTCOMES",
    "SCORES",
    "WORLDS",
    "WorldSummary",
    "example_payload",
    "summarise_world",
]

# The four (X, Y) cells, in order.
INPUTS: Final[tuple[int, ...]] = (0, 0, 1, 1)
OUTCOMES: Final[tuple[int, ...]] = (0, 1, 0, 1)
# The model scores on the input alone and predicts the majority.
SCORES: Final[tuple[float, ...]] = (0.1, 0.1, 0.9, 0.9)
PREDICTIONS: Final[tuple[int, ...]] = (0, 0, 1, 1)

WORLDS: Final[dict[str, tuple[float, ...]]] = {
    "Reference": (0.45, 0.05, 0.05, 0.45),
    "Hidden reversal": (0.05, 0.45, 0.45, 0.05),
    "Input shift only": (0.09, 0.01, 0.09, 0.81),
}


@dataclass(frozen=True, slots=True)
class WorldSummary:
    """What one world looks like, before and after labels arrive."""

    name: str
    positive_share: float
    mean_confidence: float
    accuracy: float
    brier_score: float


def summarise_world(name: str, distribution: tuple[float, ...]) -> WorldSummary:
    """Return the observable signals and the label-dependent measures."""
    if len(distribution) != len(INPUTS):
        raise ValueError(f"A world needs {len(INPUTS)} cells")

    mass = tuple(probability(value, name="cell") for value in distribution)
    total = sum(mass)
    if abs(total - 1.0) > 1e-12:
        raise ValueError("A world's cells must sum to one")

    confidence = tuple(max(score, 1 - score) for score in SCORES)
    return WorldSummary(
        name=name,
        # Observable before any label arrives.
        positive_share=sum(p * x for p, x in zip(mass, INPUTS, strict=True)),
        mean_confidence=sum(p * c for p, c in zip(mass, confidence, strict=True)),
        # Requires outcomes.
        accuracy=sum(
            p * (prediction == outcome)
            for p, prediction, outcome in zip(mass, PREDICTIONS, OUTCOMES, strict=True)
        ),
        brier_score=sum(
            p * (score - outcome) ** 2
            for p, score, outcome in zip(mass, SCORES, OUTCOMES, strict=True)
        ),
    )


def example_payload() -> tuple[WorldSummary, ...]:
    """Return the three worlds the article compares."""
    return tuple(summarise_world(name, mass) for name, mass in WORLDS.items())
