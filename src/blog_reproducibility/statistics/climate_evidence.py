"""Distribution shift and accumulated evidence, for the cold-days article.

The article's argument is that a cold day is not evidence against a warming
climate, because a warmer distribution still puts mass below freezing. Shifting
a normal distribution with standard deviation 5 by two degrees leaves the chance
of a sub-zero reading at 8% rather than 16%: halved, not abolished.

What a single reading is worth is a likelihood ratio, and what a season is worth
is a sum of them. Both are computed here, together with the information cost of
reducing a temperature to "did it freeze": the Kullback-Leibler divergence
between the two full distributions against the divergence between the two
Bernoulli indicators.

Everything is standard library. The KL divergences are closed forms, and the
tests check them against direct integration of the density ratio.
"""

from dataclasses import dataclass
from math import log, sqrt
from statistics import NormalDist
from typing import Final

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "COOLER_MEAN",
    "FREEZING",
    "HOT_DAY",
    "SPREAD",
    "WARMER_MEAN",
    "ClimateSummary",
    "EvidenceRow",
    "TailPair",
    "binary_kl",
    "climate_log_ratio",
    "climate_tails",
    "evidence_after",
    "example_payload",
    "normal_kl",
]

COOLER_MEAN: Final[float] = 5.0
WARMER_MEAN: Final[float] = 7.0
SPREAD: Final[float] = 5.0
FREEZING: Final[float] = 0.0
HOT_DAY: Final[float] = 15.0


@dataclass(frozen=True, slots=True)
class TailPair:
    """How often one climate produces a freezing day and a hot day."""

    mean: float
    below_freezing: float
    above_hot_day: float


@dataclass(frozen=True, slots=True)
class EvidenceRow:
    """Accumulated log likelihood ratio after a number of independent readings."""

    observations: int
    mean: float
    standard_deviation: float
    probability_of_pointing_the_wrong_way: float


@dataclass(frozen=True, slots=True)
class ClimateSummary:
    """Every number the article reports."""

    tails: tuple[TailPair, ...]
    full_reading_kl: float
    freezing_indicator_kl: float
    evidence: tuple[EvidenceRow, ...]


def climate_tails(mean: float, sd: float = SPREAD) -> TailPair:
    """Return the chance of a freezing day and of a hot day under one climate."""
    centre = real(mean, name="mean")
    spread = positive(sd, name="sd")

    distribution = NormalDist(centre, spread)
    return TailPair(
        mean=centre,
        below_freezing=distribution.cdf(FREEZING),
        above_hot_day=1 - distribution.cdf(HOT_DAY),
    )


def normal_kl(mean_p: float, sd_p: float, mean_q: float, sd_q: float) -> float:
    """``D(P || Q)`` in nats for two univariate normal distributions."""
    first_mean = real(mean_p, name="mean_p")
    first_sd = positive(sd_p, name="sd_p")
    second_mean = real(mean_q, name="mean_q")
    second_sd = positive(sd_q, name="sd_q")

    return (
        log(second_sd / first_sd)
        + (first_sd**2 + (first_mean - second_mean) ** 2) / (2 * second_sd**2)
        - 0.5
    )


def binary_kl(p: float, q: float) -> float:
    """``D(Bernoulli(p) || Bernoulli(q))`` in nats, for interior probabilities."""
    first = probability(p, name="p", inclusive=False)
    second = probability(q, name="q", inclusive=False)

    return first * log(first / second) + (1 - first) * log((1 - first) / (1 - second))


def climate_log_ratio(temperature: float, sd: float = SPREAD) -> float:
    """Log density ratio of the warmer climate against the cooler one.

    With equal spreads this is linear in the reading, which is why a single
    observation can point either way and a run of them cannot.
    """
    reading = real(temperature, name="temperature")
    spread = positive(sd, name="sd")

    return ((reading - COOLER_MEAN) ** 2 - (reading - WARMER_MEAN) ** 2) / (2 * spread**2)


def evidence_after(observations: int, sd: float = SPREAD) -> EvidenceRow:
    """Distribution of the total log likelihood ratio after independent readings.

    Under the warmer climate each reading contributes a mean of
    ``(m1 - m0)^2 / (2 s^2)`` nats with standard deviation ``(m1 - m0) / s``, so
    the total grows like ``n`` while its spread grows like ``sqrt(n)``. The last
    column is how often the accumulated evidence still points the wrong way.
    """
    number = count(observations, name="observations", minimum=1)
    spread = positive(sd, name="sd")

    gap = WARMER_MEAN - COOLER_MEAN
    per_observation_mean = gap**2 / (2 * spread**2)
    per_observation_sd = gap / spread

    total = NormalDist(number * per_observation_mean, per_observation_sd * sqrt(number))
    return EvidenceRow(
        observations=number,
        mean=total.mean,
        standard_deviation=total.stdev,
        probability_of_pointing_the_wrong_way=total.cdf(0.0),
    )


def example_payload() -> ClimateSummary:
    """Return the numbers the article reports."""
    cooler = climate_tails(COOLER_MEAN)
    warmer = climate_tails(WARMER_MEAN)

    return ClimateSummary(
        tails=(cooler, warmer),
        full_reading_kl=normal_kl(WARMER_MEAN, SPREAD, COOLER_MEAN, SPREAD),
        freezing_indicator_kl=binary_kl(warmer.below_freezing, cooler.below_freezing),
        evidence=tuple(evidence_after(number) for number in (1, 10, 25, 100)),
    )
