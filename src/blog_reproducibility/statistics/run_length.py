"""Run length of monitoring charts, for the article on how long a monitor takes to notice.

A standardised daily metric is normal with unit variance, shifted by ``delta``
from the first day. Three two-sided charts watch it:

* the three-sigma (Shewhart) rule signals on a day with ``|x_t| > 3``;
* the exponentially weighted average ``z_t = lambda x_t + (1 - lambda) z_{t-1}``,
  with ``z_0 = 0`` and ``lambda = 0.2``, signals when ``|z_t| > L s_t``, where
  ``s_t^2 = lambda / (2 - lambda) [1 - (1 - lambda)^(2 t)]`` is its exact variance
  in control and ``L = 2.86``;
* the cumulative sums ``C+_t = max(0, C+_{t-1} + x_t - k)`` and
  ``C-_t = max(0, C-_{t-1} - x_t - k)``, with ``k = 0.5``, signal when either
  exceeds ``h = 4.72``.

The run length is the day of the first signal, or the horizon if there is none.
The Shewhart rule signals each day independently with probability
``p = Phi(-3 - delta) + Phi(-3 + delta)``, so its run length is geometric: mean
``1 / p``, ``q`` quantile ``log(1 - q) / log(1 - p)``, and mean
``[1 - (1 - p)^M] / p`` when cut at a horizon of ``M`` days. The article's
limits for the other two charts come from a bisection on simulated in-control
run lengths near 370; the tests check them with Markov-chain approximations.

The figure simulates the three charts in that order, and within each the shifts
0.25 to 3 in turn, 1,500 runs of 2,000 days each, every run one call
``normal(delta, 1, 2000)`` on a generator seeded at 23. Here each shift's runs are
drawn as one ``(1500, 2000)`` array, which is the same stream, and each chart's
statistics are updated for all runs at once, day by day, stopping once every run
has signalled; the tests check both against a transcription of the website's
loops. The article's own tables use other generators (seeds 5, 9, 11 and 13),
more runs and a 4,000-day horizon, so they are not reproduced; the Shewhart
columns and quantiles, the tightened limits and the moving-range standard
deviation have closed forms, computed here, and the tests compare the article's
simulated values with them.
"""

from dataclasses import dataclass
from math import ceil, log, pi, sqrt
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import (
    count,
    non_negative,
    positive,
    probability,
    real,
)

__all__ = [
    "ARTICLE_MAX_DAYS",
    "ARTICLE_SHIFTS",
    "AUTOCORRELATIONS",
    "CHARTS",
    "CUSUM",
    "DELAY_QUANTILES",
    "EWMA",
    "FALSE_ALARM_QUANTILES",
    "MAX_DAYS",
    "MOVING_RANGE_CONSTANT",
    "REPLICATIONS",
    "SEED",
    "SHEWHART",
    "SHIFTS",
    "TIGHTER_LIMIT",
    "Chart",
    "ChartKind",
    "CorrelationRow",
    "DelayQuantiles",
    "DetectionCurve",
    "FalseAlarms",
    "RunLengthSummary",
    "ShewhartRow",
    "correlation_row",
    "delay_quantiles",
    "detection_curve",
    "example_payload",
    "false_alarms",
    "first_signal_days",
    "geometric_quantile",
    "moving_range_sigma",
    "run_lengths",
    "shewhart_average_run_length",
    "shewhart_row",
    "signal_probability",
]

ChartKind = Literal["shewhart", "ewma", "cusum"]


@dataclass(frozen=True, slots=True)
class Chart:
    """A two-sided chart: its kind, limit, and weight (EWMA) or allowance (CUSUM)."""

    name: str
    kind: ChartKind
    limit: float
    tuning: float = 0.0


SEED: Final[int] = 23
REPLICATIONS: Final[int] = 1500
MAX_DAYS: Final[int] = 2000
SHIFTS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
SHEWHART: Final[Chart] = Chart("Three-sigma rule", "shewhart", 3.0)
EWMA: Final[Chart] = Chart("Exponentially weighted", "ewma", 2.86, 0.2)
CUSUM: Final[Chart] = Chart("Cumulative sum", "cusum", 4.72, 0.5)
# The figure's order, which is also the order the charts consume the generator.
CHARTS: Final[tuple[Chart, ...]] = (SHEWHART, EWMA, CUSUM)
# The article's closed-form quantities.
FALSE_ALARM_QUANTILES: Final[tuple[float, ...]] = (0.10, 0.25, 0.50)
ARTICLE_SHIFTS: Final[tuple[float, ...]] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
ARTICLE_MAX_DAYS: Final[int] = 4000
DELAY_QUANTILES: Final[tuple[float, ...]] = (0.10, 0.25, 0.50, 0.75, 0.90)
TIGHTER_LIMIT: Final[float] = 2.0
AUTOCORRELATIONS: Final[tuple[float, ...]] = (0.0, 0.3, 0.6, 0.8)
# d2 for ranges of two, as the article rounds it.
MOVING_RANGE_CONSTANT: Final[float] = 1.128


@dataclass(frozen=True, slots=True)
class DetectionCurve:
    """The figure: one chart's simulated days to detection at each shift."""

    chart: str
    shifts: tuple[float, ...]
    mean_days: tuple[float, ...]
    median_days: tuple[float, ...]
    sd_days: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FalseAlarms:
    """The three-sigma rule in control: daily signal probability and the geometric run length."""

    signal_probability: float
    mean_days: float
    median_days: float
    quantiles: tuple[float, ...]
    quantile_days: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ShewhartRow:
    """The three-sigma rule's exact mean and median run length at one shift."""

    shift: float
    mean_days: float
    median_days: int


@dataclass(frozen=True, slots=True)
class DelayQuantiles:
    """Days within which a share of shifts is caught: the smallest day with that probability."""

    shift: float
    mean_days: float
    quantiles: tuple[float, ...]
    days: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CorrelationRow:
    """Limits from the moving range on an autocorrelated metric, with plugged-in means."""

    autocorrelation: float
    sigma_estimate: float
    daily_false_alarm: float
    false_alarm_interval: float


@dataclass(frozen=True, slots=True)
class RunLengthSummary:
    """The figure's curves and the article's closed forms."""

    replications: int
    max_days: int
    curves: tuple[DetectionCurve, ...]
    shewhart_exact: tuple[float, ...]
    false_alarms: FalseAlarms
    shewhart_table: tuple[ShewhartRow, ...]
    one_sigma_delay: DelayQuantiles
    tighter_in_control: float
    tighter_one_sigma: float
    correlations: tuple[CorrelationRow, ...]


def signal_probability(shift: float, limit: float = SHEWHART.limit) -> float:
    """Chance a single day falls outside ``+-limit``: ``Phi(-limit - d) + Phi(-limit + d)``."""
    d = real(shift, name="shift")
    c = positive(limit, name="limit")
    return float(stats.norm.cdf(-c - d) + stats.norm.sf(c - d))


def shewhart_average_run_length(
    shift: float, limit: float = SHEWHART.limit, *, max_days: int | None = None
) -> float:
    """Mean days to the first signal, ``1 / p``, or ``[1 - (1 - p)^M] / p`` cut at ``M``."""
    p = signal_probability(shift, limit)
    if max_days is None:
        return 1 / p
    horizon = count(max_days, name="max_days", minimum=1)
    return float(-np.expm1(horizon * np.log1p(-p)) / p)


def geometric_quantile(q: float, p: float) -> float:
    """The article's continuous quantile ``log(1 - q) / log(1 - p)`` of a geometric run length."""
    share = probability(q, name="q", inclusive=False)
    daily = probability(p, name="p", inclusive=False)
    return log(1 - share) / log(1 - daily)


def _first_day(q: float, p: float) -> int:
    """The smallest day by which a share ``q`` of geometric run lengths have ended."""
    day = ceil(geometric_quantile(q, p))
    # Guard the ceiling against rounding in the logarithms.
    while day > 1 and 1 - (1 - p) ** (day - 1) >= q:
        day -= 1
    while 1 - (1 - p) ** day < q:
        day += 1
    return day


def _chart(chart: Chart) -> Chart:
    if chart.kind == "cusum":
        # A zero limit is allowed: with the allowance at L it is the L-sigma rule.
        non_negative(chart.limit, name="limit")
    else:
        positive(chart.limit, name="limit")
    if chart.kind == "ewma":
        weight = probability(chart.tuning, name="weight", inclusive=True)
        if weight == 0:
            raise ValueError("the EWMA weight must be positive")
    elif chart.kind == "cusum":
        real(chart.tuning, name="allowance")
    elif chart.kind != "shewhart":
        raise ValueError(f"unknown chart kind: {chart.kind}")
    return chart


def first_signal_days(chart: Chart, observations: ArrayLike) -> NDArray[np.int64]:
    """Day of each row's first signal, or the number of days if it never signals.

    Rows are runs and columns days. The recursive charts are updated for all
    runs at once, day by day, and stop once every run has signalled, since later
    days cannot change a first signal.
    """
    spec = _chart(chart)
    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0 or not np.all(np.isfinite(x)):
        raise ValueError("observations must be a finite 2-D array with at least one day")
    runs, days = x.shape
    if spec.kind == "shewhart":
        signals = np.abs(x) > spec.limit
        first = np.argmax(signals, axis=1)
        hit = signals[np.arange(runs), first]
        shewhart: NDArray[np.int64] = np.where(hit, first + 1, days).astype(np.int64)
        return shewhart

    by_day = np.ascontiguousarray(x.T)
    result = np.full(runs, days, dtype=np.int64)
    waiting = np.ones(runs, dtype=bool)
    if spec.kind == "ewma":
        lam = spec.tuning
        sd = np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * (np.arange(days) + 1))))
        bounds = spec.limit * sd
        acc = np.zeros(runs)
        for day in range(days):
            acc = lam * by_day[day] + (1 - lam) * acc
            fired = waiting & (np.abs(acc) > bounds[day])
            result[fired] = day + 1
            waiting &= ~fired
            if not waiting.any():
                break
        return result

    k, h = spec.tuning, spec.limit
    hi = np.zeros(runs)
    lo = np.zeros(runs)
    for day in range(days):
        v = by_day[day]
        hi = np.maximum(0.0, hi + v - k)
        lo = np.maximum(0.0, lo - v - k)
        fired = waiting & ((hi > h) | (lo > h))
        result[fired] = day + 1
        waiting &= ~fired
        if not waiting.any():
            break
    return result


def run_lengths(
    rng: np.random.Generator,
    shift: float,
    chart: Chart,
    *,
    replications: int = REPLICATIONS,
    max_days: int = MAX_DAYS,
) -> NDArray[np.int64]:
    """Days to the first signal in runs shifted from day one, one row of draws per run."""
    reps = count(replications, name="replications", minimum=1)
    days = count(max_days, name="max_days", minimum=1)
    x = rng.normal(real(shift, name="shift"), 1.0, (reps, days))
    return first_signal_days(chart, x)


def detection_curve(
    rng: np.random.Generator,
    chart: Chart,
    shifts: tuple[float, ...] = SHIFTS,
    *,
    replications: int = REPLICATIONS,
    max_days: int = MAX_DAYS,
) -> DetectionCurve:
    """Mean, median and standard deviation of the days to detection at each shift in turn."""
    if not shifts:
        raise ValueError("shifts must not be empty")
    lengths = [
        run_lengths(rng, s, chart, replications=replications, max_days=max_days) for s in shifts
    ]
    return DetectionCurve(
        chart=chart.name,
        shifts=shifts,
        mean_days=tuple(float(np.mean(days)) for days in lengths),
        median_days=tuple(float(np.median(days)) for days in lengths),
        sd_days=tuple(float(np.std(days)) for days in lengths),
    )


def false_alarms(
    limit: float = SHEWHART.limit, quantiles: tuple[float, ...] = FALSE_ALARM_QUANTILES
) -> FalseAlarms:
    """The article's first table: the geometric run length of a threshold in control."""
    p = signal_probability(0.0, limit)
    return FalseAlarms(
        signal_probability=p,
        mean_days=1 / p,
        median_days=geometric_quantile(0.5, p),
        quantiles=quantiles,
        quantile_days=tuple(geometric_quantile(q, p) for q in quantiles),
    )


def shewhart_row(
    shift: float, limit: float = SHEWHART.limit, *, max_days: int = ARTICLE_MAX_DAYS
) -> ShewhartRow:
    """Exact mean (cut at the article's horizon) and median run length at one shift."""
    return ShewhartRow(
        shift=shift,
        mean_days=shewhart_average_run_length(shift, limit, max_days=max_days),
        median_days=_first_day(0.5, signal_probability(shift, limit)),
    )


def delay_quantiles(
    shift: float = 1.0,
    limit: float = SHEWHART.limit,
    quantiles: tuple[float, ...] = DELAY_QUANTILES,
) -> DelayQuantiles:
    """Days within which each share of shifts is caught by the threshold."""
    p = signal_probability(shift, limit)
    return DelayQuantiles(
        shift=shift,
        mean_days=1 / p,
        quantiles=quantiles,
        days=tuple(_first_day(q, p) for q in quantiles),
    )


def moving_range_sigma(autocorrelation: float, *, constant: float = MOVING_RANGE_CONSTANT) -> float:
    """Mean moving-range estimate of a unit standard deviation on a stationary AR(1) metric.

    Consecutive differences are normal with variance ``2 (1 - rho)``, so their
    mean absolute value is ``2 sqrt((1 - rho) / pi)``; divided by ``d2`` this is
    ``sqrt(1 - rho)`` up to the rounding of ``d2 = 2 / sqrt(pi) = 1.1284``.
    """
    rho = real(autocorrelation, name="autocorrelation")
    if not -1 < rho < 1:
        raise ValueError("autocorrelation must lie strictly between -1 and 1")
    return 2 * sqrt((1 - rho) / pi) / positive(constant, name="constant")


def correlation_row(autocorrelation: float, limit: float = SHEWHART.limit) -> CorrelationRow:
    """False alarms at the mean moving-range estimate, ignoring the noise in the estimate."""
    sigma = moving_range_sigma(autocorrelation)
    daily = signal_probability(0.0, limit * sigma)
    return CorrelationRow(
        autocorrelation=autocorrelation,
        sigma_estimate=sigma,
        daily_false_alarm=daily,
        false_alarm_interval=1 / daily,
    )


def example_payload() -> RunLengthSummary:
    """Return the figure's detection curves (seed 23) and the article's closed forms."""
    rng = np.random.default_rng(SEED)
    curves = tuple(detection_curve(rng, chart) for chart in CHARTS)
    return RunLengthSummary(
        replications=REPLICATIONS,
        max_days=MAX_DAYS,
        curves=curves,
        shewhart_exact=tuple(shewhart_average_run_length(s, max_days=MAX_DAYS) for s in SHIFTS),
        false_alarms=false_alarms(),
        shewhart_table=tuple(shewhart_row(s) for s in ARTICLE_SHIFTS),
        one_sigma_delay=delay_quantiles(),
        tighter_in_control=shewhart_average_run_length(0.0, TIGHTER_LIMIT),
        tighter_one_sigma=shewhart_average_run_length(1.0, TIGHTER_LIMIT),
        correlations=tuple(correlation_row(rho) for rho in AUTOCORRELATIONS),
    )
