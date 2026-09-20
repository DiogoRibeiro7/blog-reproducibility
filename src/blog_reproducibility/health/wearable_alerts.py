"""What a wearable alert is worth, for the heart-alert article.

One test, one sensitivity, one specificity, three populations. The article's
point is that the same alert means three different things depending on how
common the condition is among the people wearing the device: at 1% prevalence
fewer than one alert in six is real, and at 20% more than four in five are.

Nothing here is measured. The sensitivity and specificity are stated, each
person contributes exactly one evaluable opportunity, and the counts are
expectations per ten thousand people rather than a simulation.
"""

from dataclasses import dataclass
from typing import Final

from blog_reproducibility.common.validation import positive, probability

__all__ = [
    "COHORT",
    "SENSITIVITY",
    "SPECIFICITY",
    "AlertCounts",
    "alert_counts",
    "example_payload",
]

SENSITIVITY: Final[float] = 0.90
SPECIFICITY: Final[float] = 0.95
COHORT: Final[float] = 10_000.0


@dataclass(frozen=True, slots=True)
class AlertCounts:
    """Expected counts per cohort at one prevalence."""

    prevalence: float
    true_positive: float
    false_positive: float
    false_negative: float
    true_negative: float
    positive_predictive_value: float

    @property
    def alerts(self) -> float:
        """Every alert the device raises, real or not."""
        return self.true_positive + self.false_positive

    @property
    def negative_predictive_value(self) -> float:
        """Share of people told nothing who really have nothing."""
        return self.true_negative / (self.true_negative + self.false_negative)


def alert_counts(
    prevalence: float,
    *,
    sensitivity: float = SENSITIVITY,
    specificity: float = SPECIFICITY,
    cohort: float = COHORT,
) -> AlertCounts:
    """Return the expected two-by-two table at one prevalence."""
    share = probability(prevalence, name="prevalence", inclusive=False)
    detects = probability(sensitivity, name="sensitivity")
    clears = probability(specificity, name="specificity")
    people = positive(cohort, name="cohort")

    true_positive = people * share * detects
    false_positive = people * (1 - share) * (1 - clears)

    return AlertCounts(
        prevalence=share,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=people * share * (1 - detects),
        true_negative=people * (1 - share) * clears,
        positive_predictive_value=true_positive / (true_positive + false_positive),
    )


def example_payload() -> tuple[AlertCounts, ...]:
    """Return the three prevalences the article compares."""
    return tuple(alert_counts(prevalence) for prevalence in (0.01, 0.05, 0.20))
