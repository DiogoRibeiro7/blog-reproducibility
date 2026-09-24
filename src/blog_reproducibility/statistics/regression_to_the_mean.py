"""Regression to the mean in a fleet of machines, for the article on operational analytics.

Each machine has a stable monthly failure rate ``T``, gamma with shape 4 and
scale 1 (mean 4, variance 4), and its monthly counts are Poisson with that
rate. Gamma and Poisson are conjugate, so the expected rate of a machine that
logged an average of ``y`` failures over ``b`` months is exactly linear,

    E[T | y] = (alpha + b y) / (beta + b),

with ``alpha = 4`` and ``beta = 1 / scale = 1``: the weight on the machine's
own data is ``b / (beta + b)``, a half for a single month. Consecutive months
correlate at ``Var(T) / (Var(T) + E[T]) = 0.5``, and a ``b``-month average
correlates with the next month at
``Var(T) / sqrt((Var(T) + E[T] / b) (Var(T) + E[T]))``. Selecting the worst decile
on one month therefore selects on noise as much as on rate, and with nothing
done the group's next month falls back toward ``mu + rho (y - mu)``.

The figure and the article's first blocks are one computation: 500 machines
from a generator seeded at 42, two months of counts, the worst tenth on month
one, and then jitter for the scatter drawn from the same generator. The
placebo split draws from a second generator seeded at 1, and the rerun with a
real effect of -1.5 failures a month restarts the first seed. All of that is
reproduced draw for draw.

One thing is not portable. The website ranks the counts with NumPy's default
``argsort``, which does not keep ties in order, and many machines tie at the
decile boundaries (20 of the 27 with eight failures make the worst decile, 26
of the 71 with one the best). Which of them it picks depends on the sort
kernel NumPy dispatches for the CPU: the article's numbers came from its AVX2
kernel, and the plain kernel picks others. Here ties are broken by machine
index (a stable sort), so the deciles are the same everywhere. The numbers that
depend only on the counts (the fleet means, 9.44 and 0.52 in month one, the
correlation and the prediction, the placebo halves' month one) do not change;
those that depend on which tied machines are picked do, and every function
that selects accepts the website's deciles instead.

The article's other simulations (the baseline-length table, the shrinkage
ranking from seed 3, the winner's curse from seed 11 and the 500 replications
of the intervention) use other designs, and are checked against the closed
forms here instead: the baseline correlations, the conjugate posterior mean
and the exact expected maximum of ten binomial conversion rates.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "BASELINE_MONTHS",
    "CONVERSION_RATE",
    "DECILE",
    "INTERVENTION_EFFECT",
    "JITTER",
    "MACHINES",
    "PLACEBO_SEED",
    "RATE_FLOOR",
    "RATE_SCALE",
    "RATE_SHAPE",
    "SEED",
    "VARIANTS",
    "VARIANT_USERS",
    "BaselineRow",
    "DecileComparison",
    "Fleet",
    "InterventionEstimates",
    "PlaceboSplit",
    "RegressionToTheMeanSummary",
    "ScatterData",
    "WinnersCurse",
    "baseline_row",
    "compare_deciles",
    "example_payload",
    "expected_rate",
    "extreme_deciles",
    "intervention_run",
    "month_to_month_correlation",
    "placebo_split",
    "regression_prediction",
    "scatter_data",
    "simulate_fleet",
    "winners_curse",
]

SEED: Final[int] = 42
MACHINES: Final[int] = 500
RATE_SHAPE: Final[float] = 4.0
RATE_SCALE: Final[float] = 1.0
# The worst and best tenth: 50 of 500 machines.
DECILE: Final[int] = 10
JITTER: Final[float] = 0.3
PLACEBO_SEED: Final[int] = 1
INTERVENTION_EFFECT: Final[float] = -1.5
# The rerun keeps every treated machine's rate at least this high.
RATE_FLOOR: Final[float] = 0.05
BASELINE_MONTHS: Final[tuple[int, ...]] = (1, 3, 6, 12)
# The winner's curse: ten identical variants, 2,000 users each, converting at 5 percent.
VARIANTS: Final[int] = 10
VARIANT_USERS: Final[int] = 2000
CONVERSION_RATE: Final[float] = 0.05


@dataclass(frozen=True, slots=True)
class Fleet:
    """Each machine's failure rate and its counts in two consecutive months."""

    rates: NDArray[np.float64]
    month_one: NDArray[np.int64]
    month_two: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DecileComparison:
    """The fleet, its worst and its best tenth on month one, in both months."""

    fleet_month_one: float
    fleet_month_two: float
    worst_month_one: float
    worst_month_two: float
    worst_true_rate: float
    best_month_one: float
    best_month_two: float
    correlation: float
    predicted_worst_month_two: float
    worst_also_worst_by_rate: int


@dataclass(frozen=True, slots=True)
class PlaceboSplit:
    """A random half of the worst decile 'treated' with nothing, against the other half."""

    treated_month_one: float
    treated_month_two: float
    control_month_one: float
    control_month_two: float


@dataclass(frozen=True, slots=True)
class InterventionEstimates:
    """Three estimates of a real effect given to the worst decile."""

    true_effect: float
    naive: float
    difference_in_differences: float
    baseline_adjusted: float


@dataclass(frozen=True, slots=True)
class ScatterData:
    """The figure: jittered counts for the rest and the worst decile, and the lines drawn."""

    rest_month_one: NDArray[np.float64]
    rest_month_two: NDArray[np.float64]
    worst_month_one: NDArray[np.float64]
    worst_month_two: NDArray[np.float64]
    correlation: float
    mean: float
    worst_mean_month_one: float
    worst_mean_month_two: float
    limits: tuple[float, float]


@dataclass(frozen=True, slots=True)
class BaselineRow:
    """How a longer baseline changes the correlation and the weight on a machine's own data."""

    months: int
    correlation: float
    own_data_weight: float


@dataclass(frozen=True, slots=True)
class WinnersCurse:
    """The best of identical variants: its expected lift in the test and when measured again."""

    variants: int
    users: int
    rate: float
    expected_lift: float
    relative_lift: float
    lift_when_measured_again: float


@dataclass(frozen=True, slots=True)
class RegressionToTheMeanSummary:
    """The figure's fleet, the placebo and intervention runs, and the article's closed forms."""

    deciles: DecileComparison
    placebo: PlaceboSplit
    intervention: InterventionEstimates
    baselines: tuple[BaselineRow, ...]
    winners_curse: WinnersCurse


def _counts(values: ArrayLike, name: str) -> NDArray[np.int64]:
    data = np.asarray(values)
    if data.ndim != 1 or data.size < 2 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional array of two or more counts")
    if np.any(data < 0):
        raise ValueError(f"{name} must not be negative")
    return data.astype(np.int64)


def _group_size(machines: int, decile: int) -> int:
    k = machines // count(decile, name="decile", minimum=2)
    if k < 1:
        raise ValueError("the fleet is too small to have a machine in each decile")
    return k


def simulate_fleet(
    rng: np.random.Generator,
    machines: int = MACHINES,
    *,
    shape: float = RATE_SHAPE,
    scale: float = RATE_SCALE,
) -> Fleet:
    """Draw the rates, then month one's counts, then month two's, as the website."""
    n = count(machines, name="machines", minimum=2)
    rates = rng.gamma(positive(shape, name="shape"), positive(scale, name="scale"), n)
    month_one = rng.poisson(rates)
    month_two = rng.poisson(rates)
    return Fleet(rates=rates, month_one=month_one, month_two=month_two)


def extreme_deciles(
    counts: ArrayLike, decile: int = DECILE
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """The worst and best tenth by count, each in ascending order, ties broken by index."""
    values = _counts(counts, "counts")
    k = _group_size(values.size, decile)
    order = np.argsort(values, kind="stable")
    return order[-k:], order[:k]


def _checked_group(
    counts: NDArray[np.int64], group: ArrayLike, size: int, *, top: bool
) -> NDArray[np.intp]:
    """Validate a decile given explicitly: distinct machines holding the extreme counts."""
    indices = np.asarray(group)
    if indices.ndim != 1 or indices.size != size or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"a decile must list {size} machines")
    if np.unique(indices).size != size or indices.min() < 0 or indices.max() >= counts.size:
        raise ValueError("a decile must list distinct machines of the fleet")
    inside = np.zeros(counts.size, dtype=bool)
    inside[indices] = True
    if inside.all():
        return indices.astype(np.intp)
    if top and counts[inside].min() < counts[~inside].max():
        raise ValueError("the worst decile must hold the highest counts")
    if not top and counts[inside].max() > counts[~inside].min():
        raise ValueError("the best decile must hold the lowest counts")
    return indices.astype(np.intp)


def regression_prediction(value: float, mean: float, correlation: float) -> float:
    """``mu + rho (y - mu)``: the expected next measurement of a unit measured at ``y``."""
    rho = real(correlation, name="correlation")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    mu = real(mean, name="mean")
    return mu + rho * (real(value, name="value") - mu)


def compare_deciles(
    fleet: Fleet,
    *,
    worst: ArrayLike | None = None,
    best: ArrayLike | None = None,
    decile: int = DECILE,
) -> DecileComparison:
    """Both months for the fleet and its extremes, and the regression prediction for the worst."""
    m1, m2 = _counts(fleet.month_one, "month_one"), _counts(fleet.month_two, "month_two")
    if m2.shape != m1.shape or fleet.rates.shape != m1.shape:
        raise ValueError("rates and both months must describe the same machines")
    k = _group_size(m1.size, decile)
    stable_worst, stable_best = extreme_deciles(m1, decile)
    top = _checked_group(m1, stable_worst if worst is None else worst, k, top=True)
    bottom = _checked_group(m1, stable_best if best is None else best, k, top=False)
    rho = float(np.corrcoef(m1, m2)[0, 1])
    mu = float(m1.mean())
    worst_one = float(m1[top].mean())
    by_rate = np.argsort(fleet.rates, kind="stable")[-k:]
    return DecileComparison(
        fleet_month_one=mu,
        fleet_month_two=float(m2.mean()),
        worst_month_one=worst_one,
        worst_month_two=float(m2[top].mean()),
        worst_true_rate=float(fleet.rates[top].mean()),
        best_month_one=float(m1[bottom].mean()),
        best_month_two=float(m2[bottom].mean()),
        correlation=rho,
        predicted_worst_month_two=regression_prediction(worst_one, mu, rho),
        worst_also_worst_by_rate=int(np.intersect1d(top, by_rate).size),
    )


def placebo_split(
    rng: np.random.Generator, fleet: Fleet, worst: ArrayLike
) -> tuple[PlaceboSplit, NDArray[np.intp], NDArray[np.intp]]:
    """Choose half the worst decile at random, as the website, and compare the halves."""
    m1, m2 = _counts(fleet.month_one, "month_one"), _counts(fleet.month_two, "month_two")
    group = np.asarray(worst)
    k = group.size
    _checked_group(m1, group, k, top=True)
    treated = rng.choice(group, size=k // 2, replace=False)
    control = np.setdiff1d(group, treated)
    split = PlaceboSplit(
        treated_month_one=float(m1[treated].mean()),
        treated_month_two=float(m2[treated].mean()),
        control_month_one=float(m1[control].mean()),
        control_month_two=float(m2[control].mean()),
    )
    return split, treated.astype(np.intp), control.astype(np.intp)


def intervention_run(
    rng: np.random.Generator,
    machines: int = MACHINES,
    *,
    effect: float = INTERVENTION_EFFECT,
    floor: float = RATE_FLOOR,
    worst: ArrayLike | None = None,
    decile: int = DECILE,
) -> InterventionEstimates:
    """Treat the worst decile of month one, then estimate the effect three ways.

    The draws are the website's: rates, month one, then month two with the
    treated machines' rates lowered by the effect and floored. The naive
    estimate is the treated machines' change; difference-in-differences
    subtracts the other machines' change; the baseline-adjusted estimate is the
    treatment coefficient in a regression of month two on month one.
    """
    n = count(machines, name="machines", minimum=2)
    delta = real(effect, name="effect")
    lowest = positive(floor, name="floor")
    rates = rng.gamma(RATE_SHAPE, RATE_SCALE, n)
    month_one = rng.poisson(rates)
    k = _group_size(n, decile)
    chosen = extreme_deciles(month_one, decile)[0] if worst is None else worst
    treated = np.zeros(n, dtype=bool)
    treated[_checked_group(month_one, chosen, k, top=True)] = True
    month_two = rng.poisson(np.clip(rates + delta * treated, lowest, None))
    naive = month_two[treated].mean() - month_one[treated].mean()
    did = naive - (month_two[~treated].mean() - month_one[~treated].mean())
    design = np.column_stack([np.ones(n), month_one, treated.astype(float)])
    adjusted = np.linalg.lstsq(design, month_two, rcond=None)[0][2]
    return InterventionEstimates(
        true_effect=delta,
        naive=float(naive),
        difference_in_differences=float(did),
        baseline_adjusted=float(adjusted),
    )


def scatter_data(seed: int = SEED, *, worst: ArrayLike | None = None) -> ScatterData:
    """The figure: the fleet, then the jitter for the rest and the worst decile, in that order."""
    rng = np.random.default_rng(count(seed, name="seed"))
    fleet = simulate_fleet(rng)
    m1, m2 = fleet.month_one, fleet.month_two
    k = _group_size(m1.size, DECILE)
    top = _checked_group(m1, extreme_deciles(m1)[0] if worst is None else worst, k, top=True)
    rest = np.setdiff1d(np.arange(m1.size), top)

    def jitter(values: NDArray[np.int64]) -> NDArray[np.float64]:
        jittered: NDArray[np.float64] = values + rng.uniform(-JITTER, JITTER, values.size)
        return jittered

    rest_one, rest_two = jitter(m1[rest]), jitter(m2[rest])
    worst_one, worst_two = jitter(m1[top]), jitter(m2[top])
    return ScatterData(
        rest_month_one=rest_one,
        rest_month_two=rest_two,
        worst_month_one=worst_one,
        worst_month_two=worst_two,
        correlation=float(np.corrcoef(m1, m2)[0, 1]),
        mean=float(m1.mean()),
        worst_mean_month_one=float(m1[top].mean()),
        worst_mean_month_two=float(m2[top].mean()),
        limits=(-0.5, float(max(m1.max(), m2.max())) + 1.0),
    )


def month_to_month_correlation(
    months: int = 1, *, shape: float = RATE_SHAPE, scale: float = RATE_SCALE
) -> float:
    """Correlation of a machine's ``b``-month average with its next month."""
    b = count(months, name="months", minimum=1)
    a, s = positive(shape, name="shape"), positive(scale, name="scale")
    mean, variance = a * s, a * s * s
    return variance / sqrt((variance + mean / b) * (variance + mean))


def expected_rate(
    average: float, months: int = 1, *, shape: float = RATE_SHAPE, scale: float = RATE_SCALE
) -> float:
    """Posterior mean rate after an average of ``y`` failures over ``b`` months, linear in ``y``."""
    b = count(months, name="months", minimum=1)
    y = real(average, name="average")
    if y < 0:
        raise ValueError("average must not be negative")
    beta = 1 / positive(scale, name="scale")
    return (positive(shape, name="shape") + b * y) / (beta + b)


def baseline_row(months: int) -> BaselineRow:
    """The correlation a ``b``-month baseline has with the next month, and its own-data weight."""
    b = count(months, name="months", minimum=1)
    return BaselineRow(
        months=b,
        correlation=month_to_month_correlation(b),
        own_data_weight=b / (1 / RATE_SCALE + b),
    )


def winners_curse(
    variants: int = VARIANTS, users: int = VARIANT_USERS, rate: float = CONVERSION_RATE
) -> WinnersCurse:
    """Expected lift of the best of ``K`` identical variants: ``E[max] = sum_j 1 - F(j)^K``."""
    k = count(variants, name="variants", minimum=1)
    n = count(users, name="users", minimum=1)
    p = probability(rate, name="rate", inclusive=False)
    below = stats.binom.cdf(np.arange(n), n, p)
    expected_max = float(np.sum(1 - below**k))
    lift = expected_max / n - p
    return WinnersCurse(
        variants=k,
        users=n,
        rate=p,
        expected_lift=lift,
        relative_lift=lift / p,
        lift_when_measured_again=0.0,
    )


def example_payload() -> RegressionToTheMeanSummary:
    """Return the figure's fleet, the placebo and intervention runs, and the closed forms."""
    fleet = simulate_fleet(np.random.default_rng(SEED))
    worst, _ = extreme_deciles(fleet.month_one)
    placebo, _, _ = placebo_split(np.random.default_rng(PLACEBO_SEED), fleet, worst)
    return RegressionToTheMeanSummary(
        deciles=compare_deciles(fleet),
        placebo=placebo,
        intervention=intervention_run(np.random.default_rng(SEED)),
        baselines=tuple(baseline_row(months) for months in BASELINE_MONTHS),
        winners_curse=winners_curse(),
    )
