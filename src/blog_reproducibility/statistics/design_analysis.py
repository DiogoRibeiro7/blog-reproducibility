"""Type S and Type M errors, for the article on what a significant result means in a small study.

Measure the true effect in standard errors, ``z``, and let the estimate be
normal around it with unit variance. A two-sided test at level ``alpha`` with
critical value ``c = z_{1 - alpha / 2}`` is significant with probability

    power(z) = Phi(z - c) + Phi(-c - z),

and the second term is the chance of a significant estimate with the wrong
sign, so the Type S error is ``Phi(-c - z) / power(z)``. The Type M error, or
exaggeration ratio, is the mean absolute size of a significant estimate over
the truth. The article finds it by numerical integration; the integral has the
closed form

    E[|X|; |X| >= c] = z [Phi(z - c) - Phi(-c - z)] + phi(c - z) + phi(c + z),

which is used here, and the tests check it against the article's integral.

The article applies these to a design table (the effect giving each power,
found by bisection), a concrete experiment on an outcome with standard
deviation 0.45 and a true effect of 0.02, power computed from the observed
effect (a monotone function of the p-value), replication of a significant
result by an identical study, and sizing for the width of the interval. All of
those have closed forms, reproduced here; the share of significant results that
shrink on replication is exactly ``1 - power / 2``. Its simulated columns come from a
generator seeded at 37 that the figure does not use, and are not reproduced;
the tests check them against the closed forms instead.

The figure is deterministic: exaggeration against power for significance
thresholds of 0.10, 0.05 and 0.01, over 70 true effects from 0.3 to 4.2
standard errors, with the 0.05 curve labelled at the grid points nearest 10,
20, 50 and 80 percent power.
"""

from dataclasses import dataclass
from math import exp, pi, sqrt
from typing import Final

import numpy as np
from scipy import stats

from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "ANNOTATED_POWERS",
    "BISECTION_STEPS",
    "BISECTION_UPPER",
    "CURVE_POINTS",
    "CURVE_START",
    "CURVE_STOP",
    "FIGURE_SIGNIFICANCES",
    "INTERVAL_WIDTHS",
    "OUTCOME_SD",
    "P_VALUES",
    "REPLICATION_POWERS",
    "SIGNIFICANCE",
    "TABLE_POWERS",
    "TRUE_EFFECT",
    "USERS_PER_ARM",
    "AnnotatedPoint",
    "DesignAnalysisSummary",
    "DesignRow",
    "ExaggerationCurve",
    "ExperimentRow",
    "PostHocRow",
    "PrecisionRow",
    "ReplicationRow",
    "annotated_points",
    "critical_value",
    "design_row",
    "effect_for_power",
    "exaggeration_curve",
    "example_payload",
    "experiment_row",
    "post_hoc_power",
    "power",
    "precision_row",
    "replication_rate",
    "replication_row",
    "shrinkage_rate",
    "type_m",
    "type_s",
]

SIGNIFICANCE: Final[float] = 0.05
# The figure: three thresholds, 70 true effects from 0.3 to 4.2 standard errors.
FIGURE_SIGNIFICANCES: Final[tuple[float, ...]] = (0.10, 0.05, 0.01)
CURVE_START: Final[float] = 0.3
CURVE_STOP: Final[float] = 4.2
CURVE_POINTS: Final[int] = 70
ANNOTATED_POWERS: Final[tuple[float, ...]] = (0.10, 0.20, 0.50, 0.80)
# The article's bisection for the effect that gives a target power.
BISECTION_UPPER: Final[float] = 8.0
BISECTION_STEPS: Final[int] = 60
TABLE_POWERS: Final[tuple[float, ...]] = (0.10, 0.15, 0.20, 0.35, 0.50, 0.80, 0.95)
# The concrete experiment: a 2 percent effect on an outcome with standard deviation 0.45.
TRUE_EFFECT: Final[float] = 0.02
OUTCOME_SD: Final[float] = 0.45
USERS_PER_ARM: Final[tuple[int, ...]] = (400, 2000, 10_000, 50_000)
P_VALUES: Final[tuple[float, ...]] = (0.001, 0.01, 0.05, 0.20, 0.50)
REPLICATION_POWERS: Final[tuple[float, ...]] = (0.20, 0.50, 0.80)
# Full widths of the 95 percent interval: plus or minus 2, 1, 0.5 and 0.25 points.
INTERVAL_WIDTHS: Final[tuple[float, ...]] = (0.04, 0.02, 0.01, 0.005)


@dataclass(frozen=True, slots=True)
class ExaggerationCurve:
    """Power and exaggeration ratio along the figure's grid of true effects, at one threshold."""

    significance: float
    effects: tuple[float, ...]
    powers: tuple[float, ...]
    exaggerations: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AnnotatedPoint:
    """The grid point of a curve nearest a target power, as labelled in the figure."""

    target_power: float
    effect: float
    power: float
    exaggeration: float


@dataclass(frozen=True, slots=True)
class DesignRow:
    """The true effect giving a target power, and what significance is then worth."""

    target_power: float
    effect: float
    power: float
    wrong_sign: float
    exaggeration: float


@dataclass(frozen=True, slots=True)
class ExperimentRow:
    """The concrete experiment at one sample size per arm."""

    users_per_arm: int
    standard_error: float
    power: float
    wrong_sign: float
    exaggeration: float
    significant_estimate: float
    smallest_significant_estimate: float


@dataclass(frozen=True, slots=True)
class PostHocRow:
    """Power computed from the observed effect, as a function of the observed p-value."""

    p_value: float
    post_hoc_power: float


@dataclass(frozen=True, slots=True)
class ReplicationRow:
    """What an identical second study does to a significant first result."""

    target_power: float
    power: float
    replicates: float
    shrinks: float


@dataclass(frozen=True, slots=True)
class PrecisionRow:
    """Users per arm for a target interval width, and the exaggeration left at a 2% effect."""

    width: float
    standard_error: float
    users_per_arm: float
    exaggeration: float


@dataclass(frozen=True, slots=True)
class DesignAnalysisSummary:
    """The figure's curves and labels and the article's closed-form tables."""

    curves: tuple[ExaggerationCurve, ...]
    annotated: tuple[AnnotatedPoint, ...]
    design: tuple[DesignRow, ...]
    experiments: tuple[ExperimentRow, ...]
    post_hoc: tuple[PostHocRow, ...]
    replication: tuple[ReplicationRow, ...]
    precision: tuple[PrecisionRow, ...]


def _phi(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def critical_value(significance: float = SIGNIFICANCE) -> float:
    """Two-sided critical value ``z_{1 - alpha / 2}`` of the normal test."""
    alpha = probability(significance, name="significance", inclusive=False)
    return float(stats.norm.ppf(1 - alpha / 2))


def power(effect: float, *, significance: float = SIGNIFICANCE) -> float:
    """Chance of a significant result of either sign, the true effect in standard errors."""
    z = positive(effect, name="effect")
    c = critical_value(significance)
    return float(stats.norm.cdf(-c - z) + stats.norm.cdf(z - c))


def type_s(effect: float, *, significance: float = SIGNIFICANCE) -> float:
    """Chance that a significant estimate points the wrong way."""
    z = positive(effect, name="effect")
    c = critical_value(significance)
    return float(stats.norm.cdf(-c - z)) / power(z, significance=significance)


def type_m(effect: float, *, significance: float = SIGNIFICANCE) -> float:
    """Mean absolute size of a significant estimate over the true effect, in closed form."""
    z = positive(effect, name="effect")
    c = critical_value(significance)
    upper, lower = float(stats.norm.cdf(z - c)), float(stats.norm.cdf(-c - z))
    tail_mass = z * (upper - lower) + _phi(c - z) + _phi(c + z)
    return tail_mass / (upper + lower) / z


def effect_for_power(target: float, *, significance: float = SIGNIFICANCE) -> float:
    """The true effect giving a target power, by the article's 60-step bisection on [0, 8].

    The result is the last midpoint the bisection visits, as in the article.
    """
    goal = probability(target, name="target", inclusive=False)
    alpha = probability(significance, name="significance", inclusive=False)
    if goal <= alpha:
        raise ValueError("target power must exceed the significance level")
    z, lo, hi = 0.0, 0.0, BISECTION_UPPER
    for _ in range(BISECTION_STEPS):
        z = (lo + hi) / 2
        if power(z, significance=alpha) < goal:
            lo = z
        else:
            hi = z
    return z


def exaggeration_curve(
    significance: float,
    *,
    start: float = CURVE_START,
    stop: float = CURVE_STOP,
    points: int = CURVE_POINTS,
) -> ExaggerationCurve:
    """Power and exaggeration over evenly spaced true effects, as the figure draws them."""
    first = positive(start, name="start")
    last = positive(stop, name="stop")
    if last <= first:
        raise ValueError("stop must exceed start")
    grid = np.linspace(first, last, count(points, name="points", minimum=2))
    effects = tuple(float(z) for z in grid)
    return ExaggerationCurve(
        significance=probability(significance, name="significance", inclusive=False),
        effects=effects,
        powers=tuple(power(z, significance=significance) for z in effects),
        exaggerations=tuple(type_m(z, significance=significance) for z in effects),
    )


def annotated_points(
    curve: ExaggerationCurve, targets: tuple[float, ...] = ANNOTATED_POWERS
) -> tuple[AnnotatedPoint, ...]:
    """The grid points nearest each target power, the first on a tie, as the figure labels."""
    powers = np.asarray(curve.powers)
    points = []
    for target in targets:
        i = int(np.argmin(np.abs(powers - probability(target, name="target"))))
        points.append(
            AnnotatedPoint(target, curve.effects[i], curve.powers[i], curve.exaggerations[i])
        )
    return tuple(points)


def design_row(target_power: float, *, significance: float = SIGNIFICANCE) -> DesignRow:
    """One row of the article's first table, without its simulated columns."""
    z = effect_for_power(target_power, significance=significance)
    return DesignRow(
        target_power=target_power,
        effect=z,
        power=power(z, significance=significance),
        wrong_sign=type_s(z, significance=significance),
        exaggeration=type_m(z, significance=significance),
    )


def experiment_row(
    users_per_arm: int, *, effect: float = TRUE_EFFECT, sd: float = OUTCOME_SD
) -> ExperimentRow:
    """The concrete experiment: standard error ``sd sqrt(2 / n)`` and the effect in its units."""
    n = count(users_per_arm, name="users_per_arm", minimum=1)
    delta = positive(effect, name="effect")
    se = positive(sd, name="sd") * sqrt(2 / n)
    z = delta / se
    return ExperimentRow(
        users_per_arm=n,
        standard_error=se,
        power=power(z),
        wrong_sign=type_s(z),
        exaggeration=type_m(z),
        significant_estimate=delta * type_m(z),
        smallest_significant_estimate=critical_value() * se,
    )


def post_hoc_power(p_value: float, *, significance: float = SIGNIFICANCE) -> float:
    """Power at the observed effect, which is a function of the two-sided p-value alone."""
    p = probability(p_value, name="p_value", inclusive=False)
    observed = float(stats.norm.isf(p / 2))
    return power(observed, significance=significance)


def replication_rate(effect: float, *, significance: float = SIGNIFICANCE) -> float:
    """Chance an identical second study is significant with the same sign as a significant first.

    With ``a`` and ``b`` the chances of a significant positive and negative
    estimate, independent studies give ``(a^2 + b^2) / (a + b)``.
    """
    z = positive(effect, name="effect")
    c = critical_value(significance)
    a, b = float(stats.norm.cdf(z - c)), float(stats.norm.cdf(-c - z))
    return (a * a + b * b) / (a + b)


def shrinkage_rate(effect: float, *, significance: float = SIGNIFICANCE) -> float:
    """Chance an identical second estimate is smaller in size than a significant first one.

    The second is larger only if it is significant too, which happens with
    probability ``power^2``, and then each of two exchangeable estimates is
    the larger half the time. So a significant first result shrinks with
    probability ``(power - power^2 / 2) / power = 1 - power / 2``.
    """
    return 1 - power(effect, significance=significance) / 2


def replication_row(target_power: float) -> ReplicationRow:
    """Replication and shrinkage for a pair of studies with the target power."""
    z = effect_for_power(target_power)
    return ReplicationRow(
        target_power=target_power,
        power=power(z),
        replicates=replication_rate(z),
        shrinks=shrinkage_rate(z),
    )


def precision_row(
    width: float, *, effect: float = TRUE_EFFECT, sd: float = OUTCOME_SD
) -> PrecisionRow:
    """Size for a 95 percent interval of the given full width: ``se = width / (2 z)``."""
    se = positive(width, name="width") / (2 * critical_value())
    return PrecisionRow(
        width=width,
        standard_error=se,
        users_per_arm=2 * positive(sd, name="sd") ** 2 / se**2,
        exaggeration=type_m(positive(effect, name="effect") / se),
    )


def example_payload() -> DesignAnalysisSummary:
    """Return the figure's curves and labels and the article's closed-form tables."""
    curves = tuple(exaggeration_curve(alpha) for alpha in FIGURE_SIGNIFICANCES)
    labelled = next(curve for curve in curves if curve.significance == SIGNIFICANCE)
    return DesignAnalysisSummary(
        curves=curves,
        annotated=annotated_points(labelled),
        design=tuple(design_row(target) for target in TABLE_POWERS),
        experiments=tuple(experiment_row(n) for n in USERS_PER_ARM),
        post_hoc=tuple(PostHocRow(p, post_hoc_power(p)) for p in P_VALUES),
        replication=tuple(replication_row(target) for target in REPLICATION_POWERS),
        precision=tuple(precision_row(width) for width in INTERVAL_WIDTHS),
    )
