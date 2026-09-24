"""Digit heaping and thresholds, for the article on rounded entries and SLA breaches.

Handling times are lognormal with a median of 18 minutes and a log standard
deviation of 0.55. A share ``q`` of the entries is typed rounded to the nearest
five minutes and the rest to the nearest minute, independently of the time
itself. Every recorded value is an integer, and each recording rule is a
deterministic function of the true time ``X``, so the recorded distribution has
closed forms in the lognormal distribution function ``F``:

    P(recorded <= v) = (1 - q) F(floor(v) + 1/2) + q F(5 floor(v / 5) + 5/2).

The share recorded strictly above a threshold ``t`` is one minus that, and the
share at or above it is ``1 - (1 - q) F(ceil(t) - 1/2) - q F(5 ceil(t / 5) -
5/2)``. At a multiple of five every heaped entry below the next half-way point
drops to the threshold or under it, so the recorded share steps down across the
true one there; on the figure's grid the gap within every five minutes is widest
exactly at the multiple of five. Halfway between two multiples of five, at
``5k + 5/2``, both rules put their boundary on the threshold itself, and the
recorded share equals the true one sample for sample.

Whipple's index is 500 times the share of the values in a band that sit on a
multiple of five. Its closed form, the share of values on a multiple of five,
the quantiles of the recorded values and their mean all follow from the same
distribution function. Rounding to the nearest five minutes leaves the mean
within 0.002 of the truth, while a quantile moves to the round number nearest it.

The figure is one computation from a generator seeded at 23: 400,000 true times,
then which of them are heaped, with half of them heaped, and the share above
each threshold from 15 to 45.5 minutes in steps of half a minute. It is
reproduced draw for draw. The article's tables come from generators seeded at 2,
3 and 5 over 50,000 records, so they are compared with the closed forms rather
than reproduced; the monitoring runs and the interval-censored fit are not
reproduced here.
"""

from dataclasses import dataclass
from math import ceil, floor, log
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, probability, real

__all__ = [
    "ARTICLE_RECORDS",
    "GRID_START",
    "GRID_STEP",
    "GRID_STOP",
    "HEAPED_SHARE",
    "HEAPING_UNIT",
    "LOG_SD",
    "MEDIAN_MINUTES",
    "OFF_ROUND_THRESHOLDS",
    "QUANTILE_LEVELS",
    "QUANTILE_SHARES",
    "RECORDS",
    "ROUND_THRESHOLDS",
    "SEED",
    "WHIPPLE_BAND",
    "WHIPPLE_SHARES",
    "HeapingSummary",
    "QuantileRow",
    "Records",
    "ThresholdCurve",
    "ThresholdRow",
    "WhippleRow",
    "example_payload",
    "expected_whipple_index",
    "multiple_of_five_share",
    "record",
    "recorded_cdf",
    "recorded_mean",
    "recorded_quantile",
    "recorded_share_above",
    "share_above",
    "threshold_curve",
    "threshold_row",
    "true_mean",
    "true_quantile",
    "true_share_above",
    "whipple_index",
]

SEED: Final[int] = 23
RECORDS: Final[int] = 400_000
# True handling time: lognormal with a median of 18 minutes.
MEDIAN_MINUTES: Final[float] = 18.0
LOG_SD: Final[float] = 0.55
# The figure: half the entries rounded to the nearest five minutes, thresholds 15 to 45.5.
HEAPED_SHARE: Final[float] = 0.5
HEAPING_UNIT: Final[int] = 5
GRID_START: Final[float] = 15.0
GRID_STOP: Final[float] = 46.0
GRID_STEP: Final[float] = 0.5
ROUND_THRESHOLDS: Final[tuple[int, ...]] = (20, 25, 30, 35, 40, 45)
# The article's tables: 50,000 records, thresholds on and off round numbers.
ARTICLE_RECORDS: Final[int] = 50_000
OFF_ROUND_THRESHOLDS: Final[tuple[int, ...]] = (27, 28, 32, 33)
WHIPPLE_BAND: Final[tuple[int, int]] = (10, 60)
WHIPPLE_SHARES: Final[tuple[float, ...]] = (0.0, 0.2, 0.5, 0.8, 1.0)
QUANTILE_SHARES: Final[tuple[float, ...]] = (0.0, 0.2, 0.5, 0.8)
QUANTILE_LEVELS: Final[tuple[float, ...]] = (0.5, 0.9, 0.99)


@dataclass(frozen=True, slots=True)
class Records:
    """True times, whether each entry was heaped, and what was recorded."""

    true: NDArray[np.float64]
    heaped: NDArray[np.bool_]
    recorded: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ThresholdCurve:
    """The figure: share above each threshold, true and recorded, simulated and exact."""

    heaped_share: float
    thresholds: tuple[float, ...]
    true_shares: tuple[float, ...]
    recorded_shares: tuple[float, ...]
    expected_true: tuple[float, ...]
    expected_recorded: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ThresholdRow:
    """A threshold in closed form: truly over, recorded over, recorded at least, exactly on."""

    threshold: float
    truly_over: float
    strictly_over: float
    at_least: float
    exactly_on: float


@dataclass(frozen=True, slots=True)
class WhippleRow:
    """Share of recorded values on a multiple of five, and Whipple's index, in closed form."""

    heaped_share: float
    on_multiple_of_five: float
    whipple: float


@dataclass(frozen=True, slots=True)
class QuantileRow:
    """Mean and quantiles of the recorded values in closed form, beside the true ones."""

    heaped_share: float
    mean: float
    quantiles: tuple[int, ...]
    below_quantiles: tuple[float, ...]
    at_quantiles: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HeapingSummary:
    """The figure's curves and the closed forms behind the article's tables."""

    curve: ThresholdCurve
    on_round: tuple[ThresholdRow, ...]
    off_round: tuple[ThresholdRow, ...]
    whipple: tuple[WhippleRow, ...]
    true_mean: float
    true_quantiles: tuple[float, ...]
    quantiles: tuple[QuantileRow, ...]


def _share(value: float) -> float:
    return probability(value, name="heaped_share")


def _standardised(x: ArrayLike) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    points = np.asarray(x, dtype=np.float64)
    positive_time = points > 0
    safe = np.where(positive_time, points, 1.0)
    return positive_time, (np.log(safe) - log(MEDIAN_MINUTES)) / LOG_SD


def _cdf(x: ArrayLike) -> NDArray[np.float64]:
    """The true distribution function at ``x``, zero at or below zero."""
    positive_time, z = _standardised(x)
    values: NDArray[np.float64] = np.where(positive_time, stats.norm.cdf(z), 0.0)
    return values


def _sf(x: ArrayLike) -> NDArray[np.float64]:
    """The true survival function at ``x``, one at or below zero."""
    positive_time, z = _standardised(x)
    values: NDArray[np.float64] = np.where(positive_time, stats.norm.sf(z), 1.0)
    return values


def _upper_minutes() -> float:
    """A time beyond which the true distribution holds less than ``1e-19``."""
    return MEDIAN_MINUTES * float(np.exp(9.0 * LOG_SD))


def record(
    rng: np.random.Generator, records: int = RECORDS, heaped_share: float = HEAPED_SHARE
) -> Records:
    """Draw the true times, then which entries are heaped, then round each, as the article."""
    n = count(records, name="records", minimum=1)
    q = _share(heaped_share)
    true = rng.lognormal(np.log(MEDIAN_MINUTES), LOG_SD, n)
    heaped = rng.random(n) < q
    recorded = np.where(heaped, np.round(true / HEAPING_UNIT) * HEAPING_UNIT, np.round(true))
    return Records(true=true, heaped=heaped, recorded=recorded)


def share_above(values: ArrayLike, thresholds: ArrayLike) -> tuple[float, ...]:
    """Share of the values strictly above each threshold."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    return tuple(float(np.mean(data > t)) for t in np.asarray(thresholds, dtype=np.float64))


def true_share_above(threshold: float) -> float:
    """Share of true times above ``threshold``: the lognormal survival function."""
    return float(_sf(real(threshold, name="threshold")))


def _recorded_cdf(values: ArrayLike, q: float) -> NDArray[np.float64]:
    v = np.asarray(values, dtype=np.float64)
    unit = HEAPING_UNIT
    shares: NDArray[np.float64] = (1 - q) * _cdf(np.floor(v) + 0.5) + q * _cdf(
        unit * np.floor(v / unit) + unit / 2
    )
    return shares


def recorded_cdf(value: float, heaped_share: float = HEAPED_SHARE) -> float:
    """Share of recorded values at or below ``value``."""
    return float(_recorded_cdf(real(value, name="value"), _share(heaped_share)))


def recorded_share_above(
    threshold: float, heaped_share: float = HEAPED_SHARE, *, inclusive: bool = False
) -> float:
    """Share of recorded values above ``threshold``, or at or above it when ``inclusive``."""
    t, q = real(threshold, name="threshold"), _share(heaped_share)
    unit = HEAPING_UNIT
    if inclusive:
        fine, coarse = ceil(t) - 0.5, unit * ceil(t / unit) - unit / 2
    else:
        fine, coarse = floor(t) + 0.5, unit * floor(t / unit) + unit / 2
    return (1 - q) * float(_sf(fine)) + q * float(_sf(coarse))


def threshold_row(threshold: float, heaped_share: float = HEAPED_SHARE) -> ThresholdRow:
    """The article's threshold table in closed form, at one threshold."""
    strictly = recorded_share_above(threshold, heaped_share)
    at_least = recorded_share_above(threshold, heaped_share, inclusive=True)
    return ThresholdRow(
        threshold=threshold,
        truly_over=true_share_above(threshold),
        strictly_over=strictly,
        at_least=at_least,
        exactly_on=at_least - strictly,
    )


def whipple_index(values: ArrayLike, band: tuple[int, int] = WHIPPLE_BAND) -> float:
    """Whipple's index as the article computes it: 500 times the band's share on a five."""
    data = np.asarray(values, dtype=np.float64)
    low, high = band
    if data.ndim != 1 or high <= low:
        raise ValueError("values must be one-dimensional and the band must be increasing")
    in_band = data[(data >= low) & (data <= high)]
    if in_band.size == 0:
        raise ValueError("no values fall in the band")
    on_five = np.isclose(in_band % 5, 0) | np.isclose(in_band % 5, 5)
    return float(500 * on_five.sum() / in_band.size)


def _recorded_mass(values: NDArray[np.int64], q: float) -> NDArray[np.float64]:
    """Probability that the recorded value is each of the given integers."""
    mass: NDArray[np.float64] = _recorded_cdf(values, q) - _recorded_cdf(values - 1, q)
    return mass


def multiple_of_five_share(heaped_share: float) -> float:
    """Share of all recorded values that are a multiple of five, zero included."""
    q = _share(heaped_share)
    fives = np.arange(0, ceil(_upper_minutes() / HEAPING_UNIT) + 2) * HEAPING_UNIT
    unheaped = float(np.sum(_cdf(fives + 0.5) - _cdf(fives - 0.5)))
    return q + (1 - q) * unheaped


def expected_whipple_index(heaped_share: float, band: tuple[int, int] = WHIPPLE_BAND) -> float:
    """Whipple's index of the recorded distribution over the integers in ``band``."""
    low, high = band
    if high <= low:
        raise ValueError("the band must be increasing")
    values = np.arange(low, high + 1)
    mass = _recorded_mass(values, _share(heaped_share))
    return float(500 * mass[values % HEAPING_UNIT == 0].sum() / mass.sum())


def true_mean() -> float:
    """Mean true time: ``median exp(sigma^2 / 2)``."""
    return MEDIAN_MINUTES * float(np.exp(LOG_SD**2 / 2))


def recorded_mean(heaped_share: float) -> float:
    """Mean recorded value: the sum of the chances of reaching each rounded value.

    ``E[round X] = sum_k P(X >= k - 1/2)`` and ``E[5 round(X / 5)] = 5 sum_k
    P(X >= 5 k - 5/2)`` over ``k >= 1``.
    """
    q = _share(heaped_share)
    top = ceil(_upper_minutes()) + 2
    fine = float(np.sum(_sf(np.arange(1, top) - 0.5)))
    steps = np.arange(1, top // HEAPING_UNIT + 2)
    coarse = HEAPING_UNIT * float(np.sum(_sf(HEAPING_UNIT * steps - HEAPING_UNIT / 2)))
    return (1 - q) * fine + q * coarse


def recorded_quantile(level: float, heaped_share: float) -> int:
    """The smallest recorded value whose distribution function reaches ``level``."""
    p = probability(level, name="level", inclusive=False)
    values = np.arange(ceil(_upper_minutes()) + 2)
    reached = np.flatnonzero(_recorded_cdf(values, _share(heaped_share)) >= p)
    return int(values[reached[0]])


def true_quantile(level: float) -> float:
    """Quantile of the true times: ``median exp(sigma z_p)``."""
    p = probability(level, name="level", inclusive=False)
    return MEDIAN_MINUTES * float(np.exp(LOG_SD * stats.norm.ppf(p)))


def _quantile_row(heaped_share: float) -> QuantileRow:
    quantiles = tuple(recorded_quantile(p, heaped_share) for p in QUANTILE_LEVELS)
    return QuantileRow(
        heaped_share=heaped_share,
        mean=recorded_mean(heaped_share),
        quantiles=quantiles,
        below_quantiles=tuple(recorded_cdf(v - 1, heaped_share) for v in quantiles),
        at_quantiles=tuple(recorded_cdf(v, heaped_share) for v in quantiles),
    )


def threshold_curve(
    seed: int = SEED, *, records: int = RECORDS, heaped_share: float = HEAPED_SHARE
) -> ThresholdCurve:
    """The figure: one draw of records, and the share above each threshold on the grid."""
    q = _share(heaped_share)
    data = record(np.random.default_rng(count(seed, name="seed")), records, q)
    grid = np.arange(GRID_START, GRID_STOP, GRID_STEP)
    return ThresholdCurve(
        heaped_share=q,
        thresholds=tuple(float(t) for t in grid),
        true_shares=share_above(data.true, grid),
        recorded_shares=share_above(data.recorded, grid),
        expected_true=tuple(true_share_above(float(t)) for t in grid),
        expected_recorded=tuple(recorded_share_above(float(t), q) for t in grid),
    )


def example_payload() -> HeapingSummary:
    """Return the figure's curves and the closed forms behind the article's tables."""
    return HeapingSummary(
        curve=threshold_curve(),
        on_round=tuple(threshold_row(t) for t in ROUND_THRESHOLDS[1:4]),
        off_round=tuple(threshold_row(t) for t in OFF_ROUND_THRESHOLDS),
        whipple=tuple(
            WhippleRow(q, multiple_of_five_share(q), expected_whipple_index(q))
            for q in WHIPPLE_SHARES
        ),
        true_mean=true_mean(),
        true_quantiles=tuple(true_quantile(p) for p in QUANTILE_LEVELS),
        quantiles=tuple(_quantile_row(q) for q in QUANTILE_SHARES),
    )
