"""The noise in period-over-period comparisons, for the article on week-over-week numbers.

A daily metric is ``y_t = 1000 w_t (1 + z_t)`` with a weekday factor ``w_t``
(1.05, 1.08, 1.06, 1.04, 1.10, 0.82 and 0.85, Monday to Sunday) and noise
``z_t`` that follows an AR(1) with persistence ``rho = 0.55`` and stationary
standard deviation ``sigma = 0.035``, started at zero. Nothing ever changes, so
every movement a comparison reports is noise.

Comparing a day with the same weekday a week earlier cancels the weekday factor
and gives ``R = (1 + z_t) / (1 + z_{t-7}) - 1``; comparing the last seven days
with the seven before gives ``R = (A - B) / (W + B)``, with ``W`` the sum of the
week's factors and ``A``, ``B`` the factor-weighted sums of the noise in each
window. Both are ``(U - V) / (W + V)`` with ``(U, V)`` jointly normal: equal
variances ``s^2`` and covariance ``c`` from the autocovariance
``sigma^2 rho^|h|``. Since ``W + V > 0`` (it would take a 28-sigma draw to fail),

    P(R > x)  = P(U - (1 + x) V > x W),
    P(R < -x) = P(U - (1 - x) V < -x W),

which are normal probabilities, so the chance a quiet day moves by more than
five percent, and the threshold that a given false-alarm rate needs, have
closed forms. The mean and spread of ``R`` are one-dimensional integrals over
``V``, given which ``U - V`` is normal, computed with Gauss-Hermite quadrature.
The seven-day windows hold each weekday once, but which weekday sits where
depends on the day of the week of the comparison, so their laws are averaged
over the days compared.

The figure is its own simulation, not the article's: 250 series of 400 days
from one generator seeded at 83 (each series' 400 innovations in turn, the
first unused), each compared on days 60 to 399, and histograms of the 85,000
comparisons with their spreads. Drawing all innovations at once and running the
recursion across series gives the website's series bit for bit. The article's
tables come from other seeds (3 for 400 series of 560 days compared on days 400
to 559, and 11 for its 2,000 comparisons on day 100) and are not reproduced;
the tests compare them with the closed forms. Its detection study (seeds 7 and
5, thresholds tuned by bisection) is not reproduced either.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy import optimize, stats

from blog_reproducibility.common.validation import count, non_negative, positive, probability

__all__ = [
    "BIN_EDGES",
    "FALSE_ALARM_RATES",
    "DAYS",
    "FIRST_DAY",
    "LARGE_MOVE",
    "LEVEL",
    "NOISE_SD",
    "PERSISTENCE",
    "SEED",
    "SERIES",
    "SMALL_MOVE",
    "WEEK",
    "WEEKDAY_FACTORS",
    "YEAR",
    "ComparisonRow",
    "ComparisonStats",
    "RatioLaw",
    "WeekOverWeekSummary",
    "example_payload",
    "exceedance",
    "expected_stats",
    "false_alarm_threshold",
    "phase_weights",
    "ratio_moments",
    "same_weekday_changes",
    "same_weekday_law",
    "seven_day_changes",
    "seven_day_law",
    "simulate_metric",
    "simulated_stats",
]

SEED: Final[int] = 83
SERIES: Final[int] = 250
DAYS: Final[int] = 400
# The first day compared, well after the noise has forgotten its start at zero.
FIRST_DAY: Final[int] = 60
LEVEL: Final[float] = 1000.0
WEEKDAY_FACTORS: Final[tuple[float, ...]] = (1.05, 1.08, 1.06, 1.04, 1.10, 0.82, 0.85)
PERSISTENCE: Final[float] = 0.55
NOISE_SD: Final[float] = 0.035
WEEK: Final[int] = 7
# The nearest same weekday a year earlier.
YEAR: Final[int] = 364
SMALL_MOVE: Final[float] = 0.05
LARGE_MOVE: Final[float] = 0.10
FALSE_ALARM_RATES: Final[tuple[float, float]] = (0.05, 0.01)
BIN_EDGES: Final[tuple[float, ...]] = tuple(float(x) for x in np.linspace(-0.20, 0.20, 81))
_HERMITE_NODES: Final[int] = 80


@dataclass(frozen=True, slots=True)
class RatioLaw:
    """``R = (U - V) / (W + V)``: ``W`` is ``scale``, ``Var U = Var V`` and ``Cov(U, V)``."""

    scale: float
    variance: float
    covariance: float


@dataclass(frozen=True, slots=True)
class ComparisonStats:
    """Mean and spread of a comparison, how often it moves by five or ten percent, thresholds."""

    mean: float
    spread: float
    exceeds_small: float
    exceeds_large: float
    threshold_five_percent: float
    threshold_one_percent: float


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """A comparison on the figure's quiet series, and its closed form."""

    name: str
    simulated: ComparisonStats
    expected: ComparisonStats


@dataclass(frozen=True, slots=True)
class WeekOverWeekSummary:
    """The figure's two histograms and the numbers behind them."""

    comparisons: int
    same_weekday: ComparisonRow
    seven_day: ComparisonRow
    same_weekday_last_year: ComparisonStats
    bin_edges: tuple[float, ...]
    same_weekday_density: tuple[float, ...]
    seven_day_density: tuple[float, ...]


def _factors(factors: Sequence[float]) -> NDArray[np.float64]:
    if len(factors) != WEEK:
        raise ValueError("there must be one weekday factor for each day of the week")
    return np.array([positive(f, name="weekday factor") for f in factors])


def _noise(persistence: float, noise_sd: float) -> tuple[float, float]:
    rho = probability(persistence, name="persistence")
    if rho >= 1.0:
        raise ValueError("persistence must be below one for the noise to be stationary")
    return rho, positive(noise_sd, name="noise_sd")


def simulate_metric(
    rng: np.random.Generator,
    series: int = SERIES,
    days: int = DAYS,
    *,
    factors: Sequence[float] = WEEKDAY_FACTORS,
    persistence: float = PERSISTENCE,
    noise_sd: float = NOISE_SD,
) -> NDArray[np.float64]:
    """Quiet daily series, one per row: each row's innovations are drawn in turn.

    The noise starts at zero and follows ``z_i = rho z_{i-1} + e_i`` with
    innovations of standard deviation ``sigma sqrt(1 - rho^2)``; the first
    innovation of each series is drawn and not used.
    """
    rows = count(series, name="series", minimum=1)
    n = count(days, name="days", minimum=2)
    weights = _factors(factors)
    rho, sd = _noise(persistence, noise_sd)
    eps = rng.normal(0, sd * np.sqrt(1 - rho**2), (rows, n))
    z = np.zeros((rows, n))
    for i in range(1, n):
        z[:, i] = rho * z[:, i - 1] + eps[:, i]
    metric: NDArray[np.float64] = LEVEL * weights[np.arange(n) % WEEK] * (1 + z)
    return metric


def _metric(metric: NDArray[np.float64]) -> NDArray[np.float64]:
    y = np.asarray(metric, dtype=np.float64)
    if y.ndim != 2 or not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("metric must be a two-dimensional array of positive values")
    return y


def same_weekday_changes(
    metric: NDArray[np.float64], first_day: int = FIRST_DAY, *, lag: int = WEEK
) -> NDArray[np.float64]:
    """Each day from ``first_day`` against the day ``lag`` earlier, one row per series."""
    y = _metric(metric)
    gap = count(lag, name="lag", minimum=1)
    start = count(first_day, name="first_day", minimum=gap)
    if start >= y.shape[1]:
        raise ValueError("first_day must fall inside the series")
    changes: NDArray[np.float64] = y[:, start:] / y[:, start - gap : y.shape[1] - gap] - 1
    return changes


def seven_day_changes(
    metric: NDArray[np.float64], first_day: int = FIRST_DAY
) -> NDArray[np.float64]:
    """The seven days to each day from ``first_day`` against the seven before them."""
    y = _metric(metric)
    start = count(first_day, name="first_day", minimum=2 * WEEK - 1)
    n = y.shape[1]
    if start >= n:
        raise ValueError("first_day must fall inside the series")
    # means[:, k] is the mean of days k to k + 6.
    means = sliding_window_view(y, WEEK, axis=1).mean(axis=-1)
    current = means[:, start - WEEK + 1 : n - WEEK + 1]
    previous = means[:, start - 2 * WEEK + 1 : n - 2 * WEEK + 1]
    changes: NDArray[np.float64] = current / previous - 1
    return changes


def same_weekday_law(
    lag: int = WEEK, *, persistence: float = PERSISTENCE, noise_sd: float = NOISE_SD
) -> RatioLaw:
    """A day against the day ``lag`` earlier: ``(1 + z_t) / (1 + z_{t-lag}) - 1``."""
    gap = count(lag, name="lag", minimum=1)
    rho, sd = _noise(persistence, noise_sd)
    return RatioLaw(scale=1.0, variance=sd**2, covariance=sd**2 * rho**gap)


def seven_day_law(
    phase: int,
    *,
    factors: Sequence[float] = WEEKDAY_FACTORS,
    persistence: float = PERSISTENCE,
    noise_sd: float = NOISE_SD,
) -> RatioLaw:
    """Seven days to a day of weekday index ``phase`` (Monday 0) against the seven before."""
    day = count(phase, name="phase")
    weights = _factors(factors)[(np.arange(-WEEK + 1, 1) + day) % WEEK]
    rho, sd = _noise(persistence, noise_sd)
    offsets = np.subtract.outer(np.arange(WEEK), np.arange(WEEK))
    products = np.outer(weights, weights)
    return RatioLaw(
        scale=float(weights.sum()),
        variance=float(sd**2 * np.sum(products * rho ** np.abs(offsets))),
        covariance=float(sd**2 * np.sum(products * rho ** np.abs(offsets + WEEK))),
    )


def phase_weights(first_day: int = FIRST_DAY, days: int = DAYS) -> tuple[float, ...]:
    """Share of the compared days falling on each weekday, Monday first."""
    start = count(first_day, name="first_day")
    n = count(days, name="days", minimum=start + 1)
    compared = np.arange(start, n) % WEEK
    return tuple(float(x) for x in np.bincount(compared, minlength=WEEK) / compared.size)


def _mixture(
    laws: RatioLaw | Sequence[RatioLaw], weights: Sequence[float] | None
) -> tuple[tuple[RatioLaw, ...], tuple[float, ...]]:
    group = (laws,) if isinstance(laws, RatioLaw) else tuple(laws)
    if not group:
        raise ValueError("at least one law is needed")
    shares = (
        tuple(1 / len(group) for _ in group)
        if weights is None
        else tuple(non_negative(w, name="weight") for w in weights)
    )
    if len(shares) != len(group) or abs(sum(shares) - 1) > 1e-12:
        raise ValueError("weights must match the laws and sum to one")
    return group, shares


def exceedance(
    threshold: float,
    laws: RatioLaw | Sequence[RatioLaw],
    weights: Sequence[float] | None = None,
) -> float:
    """Chance a quiet comparison moves by more than ``threshold`` either way."""
    x = non_negative(threshold, name="threshold")
    group, shares = _mixture(laws, weights)
    total = 0.0
    for law, share in zip(group, shares, strict=True):
        up = sqrt(law.variance * (1 + (1 + x) ** 2) - 2 * (1 + x) * law.covariance)
        down = sqrt(law.variance * (1 + (1 - x) ** 2) - 2 * (1 - x) * law.covariance)
        total += share * float(
            stats.norm.sf(x * law.scale / up) + stats.norm.cdf(-x * law.scale / down)
        )
    return total


def ratio_moments(
    laws: RatioLaw | Sequence[RatioLaw], weights: Sequence[float] | None = None
) -> tuple[float, float]:
    """Mean and standard deviation of the comparison, integrating over ``V``."""
    group, shares = _mixture(laws, weights)
    nodes, node_weights = np.polynomial.hermite_e.hermegauss(_HERMITE_NODES)
    node_weights = node_weights / sqrt(2 * np.pi)
    first = second = 0.0
    for law, share in zip(group, shares, strict=True):
        v = sqrt(law.variance) * nodes
        slope = law.covariance / law.variance
        centre = (slope - 1) * v
        spread = law.variance * (1 - slope**2)
        first += share * float(np.sum(node_weights * centre / (law.scale + v)))
        second += share * float(np.sum(node_weights * (centre**2 + spread) / (law.scale + v) ** 2))
    return first, sqrt(second - first**2)


def false_alarm_threshold(
    rate: float,
    laws: RatioLaw | Sequence[RatioLaw],
    weights: Sequence[float] | None = None,
) -> float:
    """The movement a quiet comparison exceeds with probability ``rate``."""
    alpha = probability(rate, name="rate", inclusive=False)
    group, shares = _mixture(laws, weights)
    return float(
        optimize.brentq(
            lambda x: exceedance(x, group, shares) - alpha, 1e-9, 0.99, xtol=1e-14, rtol=1e-14
        )
    )


def expected_stats(
    laws: RatioLaw | Sequence[RatioLaw], weights: Sequence[float] | None = None
) -> ComparisonStats:
    """The closed-form statistics of a comparison, or of a mixture over weekdays."""
    group, shares = _mixture(laws, weights)
    mean, spread = ratio_moments(group, shares)
    five, one = FALSE_ALARM_RATES
    return ComparisonStats(
        mean=mean,
        spread=spread,
        exceeds_small=exceedance(SMALL_MOVE, group, shares),
        exceeds_large=exceedance(LARGE_MOVE, group, shares),
        threshold_five_percent=false_alarm_threshold(five, group, shares),
        threshold_one_percent=false_alarm_threshold(one, group, shares),
    )


def simulated_stats(changes: NDArray[np.float64]) -> ComparisonStats:
    """The same statistics over every simulated comparison, series by series as the website."""
    values = np.asarray(changes, dtype=np.float64).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("changes must be a non-empty, finite array")
    moves = np.abs(values)
    five, one = FALSE_ALARM_RATES
    return ComparisonStats(
        mean=float(np.mean(values)),
        spread=float(np.std(values)),
        exceeds_small=float(np.mean(moves > SMALL_MOVE)),
        exceeds_large=float(np.mean(moves > LARGE_MOVE)),
        threshold_five_percent=float(np.quantile(moves, 1 - five)),
        threshold_one_percent=float(np.quantile(moves, 1 - one)),
    )


def example_payload() -> WeekOverWeekSummary:
    """Simulate the figure's quiet series and put both comparisons beside their closed forms."""
    metric = simulate_metric(np.random.default_rng(SEED))
    single = same_weekday_changes(metric)
    weekly = seven_day_changes(metric)
    edges = np.array(BIN_EDGES)
    phases = phase_weights()
    weekly_laws = [seven_day_law(phase) for phase in range(WEEK)]

    def density(changes: NDArray[np.float64]) -> tuple[float, ...]:
        heights, _ = np.histogram(changes.ravel(), bins=edges, density=True)
        return tuple(float(h) for h in heights)

    return WeekOverWeekSummary(
        comparisons=int(single.size),
        same_weekday=ComparisonRow(
            "same weekday last week",
            simulated_stats(single),
            expected_stats(same_weekday_law()),
        ),
        seven_day=ComparisonRow(
            "seven-day averages",
            simulated_stats(weekly),
            expected_stats(weekly_laws, phases),
        ),
        same_weekday_last_year=expected_stats(same_weekday_law(YEAR)),
        bin_edges=BIN_EDGES,
        same_weekday_density=density(single),
        seven_day_density=density(weekly),
    )
