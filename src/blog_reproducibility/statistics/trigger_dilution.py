"""What dilution costs in sample size, for the article on triggered analysis.

The population splits into users who reach a feature (a share ``p``, outcome
mean 30 and standard deviation 14) and users who never do (mean 18, standard
deviation 9). The feature lifts triggered users' outcomes by 6 percent, an
effect of ``delta = 1.8`` on each of them and nothing on anyone else, so the
effect on the whole population is ``p delta``.

An analysis of all users sees that diluted effect against the variance of the
whole mixture,

    sigma_all^2 = p (sigma_t^2 + mu_t^2) + (1 - p) (sigma_n^2 + mu_n^2) - mu_all^2,

and needs ``2 sigma_all^2 (z_0.975 + z_0.8)^2 / (p delta)^2`` users per arm for
80 percent power. An analysis restricted to triggered users sees the full
effect against the triggered variance, and needs ``2 sigma_t^2 (z_0.975 +
z_0.8)^2 / delta^2`` triggered users, or that divided by ``p`` users per arm.
The first grows roughly as ``1 / p^2``, the second exactly as ``1 / p``.

The figure draws both closed forms on 40 trigger rates from 1 to 60 percent
and labels their ratio at the grid points nearest 2, 8 and 20 percent. The
article's sample-size table and its visibility table (the diluted effect
against the smallest effect detectable with 200,000 users per arm) are the
same closed forms and are reproduced exactly. Its simulated tables (generators
seeded at 3, 101 and 5) are not; the closed forms are checked against smaller
simulations of the same population in the tests.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, non_negative, probability

__all__ = [
    "ANNOTATED_RATES",
    "CURVE_POINTS",
    "CURVE_START",
    "CURVE_STOP",
    "LIFT",
    "MEAN_NON_TRIGGERED",
    "MEAN_TRIGGERED",
    "POWER",
    "SD_NON_TRIGGERED",
    "SD_TRIGGERED",
    "SIGNIFICANCE",
    "TABLE_RATES",
    "TRUE_EFFECT",
    "VISIBILITY_USERS",
    "GapLabel",
    "SampleSizeRow",
    "TriggerDilutionSummary",
    "VisibilityRow",
    "annotated_gaps",
    "draw_arm",
    "example_payload",
    "population_mean",
    "population_variance",
    "sample_size_curve",
    "sample_size_row",
    "smallest_detectable_effect",
    "users_all",
    "users_triggered",
    "visibility_row",
]

MEAN_NON_TRIGGERED: Final[float] = 18.0
SD_NON_TRIGGERED: Final[float] = 9.0
MEAN_TRIGGERED: Final[float] = 30.0
SD_TRIGGERED: Final[float] = 14.0
LIFT: Final[float] = 0.06
TRUE_EFFECT: Final[float] = MEAN_TRIGGERED * LIFT
SIGNIFICANCE: Final[float] = 0.05
POWER: Final[float] = 0.80
TABLE_RATES: Final[tuple[float, ...]] = (0.50, 0.20, 0.08, 0.02)
ANNOTATED_RATES: Final[tuple[float, ...]] = (0.02, 0.08, 0.20)
VISIBILITY_USERS: Final[int] = 200_000
# The figure's curve: 40 trigger rates spaced evenly on a log scale.
CURVE_START: Final[float] = 0.01
CURVE_STOP: Final[float] = 0.6
CURVE_POINTS: Final[int] = 40


@dataclass(frozen=True, slots=True)
class SampleSizeRow:
    """Users per arm for 80 percent power under each analysis at one trigger rate."""

    trigger_rate: float
    users_all: float
    users_triggered: float
    ratio: float


@dataclass(frozen=True, slots=True)
class GapLabel:
    """One of the figure's ratio labels, at the grid point nearest the requested rate."""

    requested_rate: float
    trigger_rate: float
    users_all: float
    users_triggered: float
    ratio: float


@dataclass(frozen=True, slots=True)
class VisibilityRow:
    """The diluted effect and the smallest detectable effect, as shares of the metric."""

    trigger_rate: float
    diluted_effect: float
    detectable_effect: float
    visible: bool


@dataclass(frozen=True, slots=True)
class TriggerDilutionSummary:
    """The article's sample-size and visibility tables and the figure's labels."""

    sample_sizes: tuple[SampleSizeRow, ...]
    gap_labels: tuple[GapLabel, ...]
    visibility: tuple[VisibilityRow, ...]


def _rate(value: float) -> float:
    return probability(value, name="trigger_rate", inclusive=False)


def _z_sum(significance: float, power: float) -> float:
    alpha = probability(significance, name="significance", inclusive=False)
    beta = probability(power, name="power", inclusive=False)
    return float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(beta))


def population_mean(trigger_rate: float) -> float:
    """Mean outcome over the whole population at trigger rate ``p``."""
    p = _rate(trigger_rate)
    return p * MEAN_TRIGGERED + (1 - p) * MEAN_NON_TRIGGERED


def population_variance(trigger_rate: float) -> float:
    """Variance of the outcome over the whole population: second moment less squared mean."""
    p = _rate(trigger_rate)
    return (
        p * (SD_TRIGGERED**2 + MEAN_TRIGGERED**2)
        + (1 - p) * (SD_NON_TRIGGERED**2 + MEAN_NON_TRIGGERED**2)
        - population_mean(p) ** 2
    )


def users_all(
    trigger_rate: float, *, significance: float = SIGNIFICANCE, power: float = POWER
) -> float:
    """Users per arm for the all-user analysis to detect the diluted effect ``p delta``."""
    p = _rate(trigger_rate)
    z_squared = _z_sum(significance, power) ** 2
    return 2 * population_variance(p) * z_squared / (p * TRUE_EFFECT) ** 2


def users_triggered(
    trigger_rate: float, *, significance: float = SIGNIFICANCE, power: float = POWER
) -> float:
    """Users per arm for the triggered analysis: enough triggered users, divided by ``p``."""
    p = _rate(trigger_rate)
    z_squared = _z_sum(significance, power) ** 2
    return 2 * SD_TRIGGERED**2 * z_squared / TRUE_EFFECT**2 / p


def smallest_detectable_effect(
    trigger_rate: float,
    users_per_arm: int = VISIBILITY_USERS,
    *,
    significance: float = SIGNIFICANCE,
    power: float = POWER,
) -> float:
    """Smallest all-user effect detectable with the given users per arm, in outcome units."""
    n = count(users_per_arm, name="users_per_arm", minimum=2)
    spread = sqrt(2 * population_variance(trigger_rate) / n)
    return spread * _z_sum(significance, power)


def sample_size_curve(
    start: float = CURVE_START, stop: float = CURVE_STOP, points: int = CURVE_POINTS
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the figure's curves: trigger rates and users per arm under each analysis."""
    rates = np.geomspace(_rate(start), _rate(stop), count(points, name="points", minimum=2))
    z_squared = _z_sum(SIGNIFICANCE, POWER) ** 2
    mean_all = rates * MEAN_TRIGGERED + (1 - rates) * MEAN_NON_TRIGGERED
    variance_all = (
        rates * (SD_TRIGGERED**2 + MEAN_TRIGGERED**2)
        + (1 - rates) * (SD_NON_TRIGGERED**2 + MEAN_NON_TRIGGERED**2)
        - mean_all**2
    )
    needed_all = 2 * variance_all * z_squared / (rates * TRUE_EFFECT) ** 2
    needed_triggered = 2 * SD_TRIGGERED**2 * z_squared / TRUE_EFFECT**2 / rates
    return rates, needed_all, needed_triggered


def annotated_gaps(rates: tuple[float, ...] = ANNOTATED_RATES) -> tuple[GapLabel, ...]:
    """Label the ratio of the two curves at the grid points nearest each rate."""
    grid, needed_all, needed_triggered = sample_size_curve()
    labels = []
    for rate in rates:
        i = int(np.argmin(np.abs(grid - _rate(rate))))
        labels.append(
            GapLabel(
                requested_rate=rate,
                trigger_rate=float(grid[i]),
                users_all=float(needed_all[i]),
                users_triggered=float(needed_triggered[i]),
                ratio=float(needed_all[i] / needed_triggered[i]),
            )
        )
    return tuple(labels)


def sample_size_row(trigger_rate: float) -> SampleSizeRow:
    """One row of the article's sample-size table."""
    everyone = users_all(trigger_rate)
    triggered = users_triggered(trigger_rate)
    return SampleSizeRow(
        trigger_rate=trigger_rate,
        users_all=everyone,
        users_triggered=triggered,
        ratio=everyone / triggered,
    )


def visibility_row(trigger_rate: float, users_per_arm: int = VISIBILITY_USERS) -> VisibilityRow:
    """Compare the diluted effect with the smallest detectable one, both relative to the metric."""
    mean = population_mean(trigger_rate)
    diluted = trigger_rate * TRUE_EFFECT
    detectable = smallest_detectable_effect(trigger_rate, users_per_arm)
    return VisibilityRow(
        trigger_rate=trigger_rate,
        diluted_effect=diluted / mean,
        detectable_effect=detectable / mean,
        visible=diluted > detectable,
    )


def draw_arm(
    users: int,
    trigger_rate: float,
    rng: np.random.Generator,
    *,
    treated: bool,
    lift: float = LIFT,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Draw one arm of the article's population: outcomes and trigger flags.

    The draws are the article's: trigger flags, then triggered-user outcomes
    and non-triggered outcomes for every user, of which each keeps one.
    Treatment multiplies triggered users' outcomes by ``1 + lift``.
    """
    n = count(users, name="users", minimum=1)
    p = probability(trigger_rate, name="trigger_rate")
    effect = non_negative(lift, name="lift")
    triggered = rng.random(n) < p
    outcome = np.where(
        triggered,
        rng.normal(MEAN_TRIGGERED, SD_TRIGGERED, n),
        rng.normal(MEAN_NON_TRIGGERED, SD_NON_TRIGGERED, n),
    )
    if treated:
        outcome = np.where(triggered, outcome * (1 + effect), outcome)
    return outcome, triggered


def example_payload() -> TriggerDilutionSummary:
    """Return the article's two closed-form tables and the figure's ratio labels."""
    return TriggerDilutionSummary(
        sample_sizes=tuple(sample_size_row(rate) for rate in TABLE_RATES),
        gap_labels=annotated_gaps(),
        visibility=tuple(visibility_row(rate) for rate in TABLE_RATES),
    )
