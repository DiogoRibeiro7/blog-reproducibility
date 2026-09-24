"""Forecasts of a series that is mostly zeros, for the article on intermittent demand.

Each week has demand with probability ``p = 0.2``, and a demand is ``1 + N``
units with ``N`` Poisson of mean three, so four on average. The true rate is
``p m = 0.8`` units a week and four weeks in five have no demand at all, so the
median week is zero. With ``E[Y^2] = p (m - 1 + m^2) = 3.8`` the weekly variance
is ``3.16``.

Five one-step forecasts are compared: zero, the mean of the last 52 weeks,
exponential smoothing with ``alpha = 0.1`` started from the mean of the first
104 weeks, Croston's method, which smooths the size of a demand and the gap
between demands separately and forecasts their ratio, and Croston multiplied by
``1 - alpha / 2``. A constant forecast ``c`` between zero and one has mean
absolute error ``(1 - p) c + p (m - c)``: 0.8 at zero and 1.28 at the true
rate, which is why the absolute error picks the forecast of zero. Squared error
is minimised by the mean; a forecast unbiased for the rate with its own variance
``v`` has mean squared error ``Var(Y) + v``, with ``v = Var(Y) / 52`` for the
rolling mean and ``alpha / (2 - alpha) Var(Y)`` for smoothing. Croston's ratio
is biased upward because ``E[1 / x] > 1 / E[x]``: to second order in the
smoothed interval's variance ``alpha / (2 - alpha) (1 - p) / p^2``,

    E[z / x] = m p (1 + alpha (1 - p) / (2 - alpha)) = 0.834,

and the correction brings it to 0.792 (Syntetos and Boylan's approximation).

The figure is its own simulation, not the article's: 40 series of 1,040 weeks,
series ``r`` from a generator seeded at ``900 + r`` (weeks with demand, then
the sizes), each method forecasting every fourth week from week 104. The bars
are the mean over series of each series' average forecast, with 1.96 standard
errors. The website recomputes smoothing and Croston from the start of the
series for each forecast week; here each series is passed once and the state
read off at every forecast week, the same arithmetic in the same order, so the
bars are the website's bit for bit. The errors against each forecast week's
demand are computed as the article's evaluation computes them, to test the
title. The article's tables come from other seeds (1 for its sample series,
``100 + r`` for 200 series forecast every week, ``500 + r`` for 60 series in
its stock simulation) and are not reproduced; the tests compare them with the
closed forms.
"""

from dataclasses import dataclass
from math import floor, sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, non_negative, positive, probability

__all__ = [
    "DEMAND_PROBABILITY",
    "FORECAST_STEP",
    "MEAN_SIZE",
    "METHODS",
    "SEED_BASE",
    "SERIES",
    "SMOOTHING",
    "WARMUP",
    "WEEKS",
    "WINDOW",
    "ClosedForms",
    "IntermittentSummary",
    "MethodRow",
    "closed_forms",
    "constant_forecast_mae",
    "croston_expectation",
    "croston_forecasts",
    "demand_series",
    "example_payload",
    "forecast_weeks",
    "method_forecasts",
    "rolling_mean_forecasts",
    "smoothing_forecasts",
]

SEED_BASE: Final[int] = 900
SERIES: Final[int] = 40
WEEKS: Final[int] = 1040
WARMUP: Final[int] = 104
# The figure forecasts every fourth week from the end of the warm-up.
FORECAST_STEP: Final[int] = 4
DEMAND_PROBABILITY: Final[float] = 0.20
MEAN_SIZE: Final[float] = 4.0
WINDOW: Final[int] = 52
SMOOTHING: Final[float] = 0.1
METHODS: Final[tuple[str, ...]] = (
    "Always zero",
    "52-week mean",
    "Exponential smoothing",
    "Croston",
    "Croston, debiased",
)


@dataclass(frozen=True, slots=True)
class MethodRow:
    """A method's average forecast over series with its standard error, and its errors."""

    method: str
    average_forecast: float
    standard_error: float
    series_averages: tuple[float, ...]
    bias: float
    mean_absolute_error: float
    root_mean_squared_error: float


@dataclass(frozen=True, slots=True)
class ClosedForms:
    """Population values for the demand process and the forecasts."""

    true_rate: float
    zero_share: float
    weekly_variance: float
    zero_forecast_mae: float
    rate_forecast_mae: float
    zero_forecast_rmse: float
    rolling_mean_rmse: float
    smoothing_rmse: float
    croston: float
    croston_debiased: float


@dataclass(frozen=True, slots=True)
class IntermittentSummary:
    """The figure's bars, the errors behind its title, and the closed forms."""

    series: int
    forecasts_per_series: int
    rows: tuple[MethodRow, ...]
    closed_forms: ClosedForms


def _probability(value: float) -> float:
    return probability(value, name="probability", inclusive=False)


def _mean_size(value: float) -> float:
    size = positive(value, name="mean_size")
    if size < 1.0:
        raise ValueError("mean_size must be at least one: every demand is one unit or more")
    return size


def _alpha(value: float) -> float:
    return probability(value, name="alpha", inclusive=False)


def _series(values: ArrayLike) -> NDArray[np.float64]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("the series must be a non-empty, finite 1-D array")
    return y


def _weeks(weeks: ArrayLike, length: int, first: int = 1) -> NDArray[np.int64]:
    t = np.asarray(weeks, dtype=np.int64)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("weeks must be a non-empty 1-D array")
    if np.any(t < first) or np.any(t > length):
        raise ValueError(f"weeks must lie between {first} and the series length")
    return t


def demand_series(
    rng: np.random.Generator,
    weeks: int = WEEKS,
    *,
    demand_probability: float = DEMAND_PROBABILITY,
    mean_size: float = MEAN_SIZE,
) -> NDArray[np.float64]:
    """Draw which weeks have demand, then every week's size, and keep the sizes that occur."""
    n = count(weeks, name="weeks", minimum=1)
    p = _probability(demand_probability)
    m = _mean_size(mean_size)
    occurs = rng.random(n) < p
    size = 1 + rng.poisson(m - 1, n)
    demand: NDArray[np.float64] = np.where(occurs, size, 0).astype(float)
    return demand


def forecast_weeks(
    weeks: int = WEEKS, *, warmup: int = WARMUP, step: int = FORECAST_STEP
) -> NDArray[np.int64]:
    """The weeks the figure forecasts: every ``step``-th from ``warmup``."""
    n = count(weeks, name="weeks", minimum=1)
    start = count(warmup, name="warmup", minimum=1)
    if start >= n:
        raise ValueError("warmup must be shorter than the series")
    return np.arange(start, n, count(step, name="step", minimum=1), dtype=np.int64)


def rolling_mean_forecasts(
    series: ArrayLike, weeks: ArrayLike, *, window: int = WINDOW
) -> NDArray[np.float64]:
    """Mean of the last ``window`` weeks before each forecast week, or of all if fewer."""
    y = _series(series)
    t = _weeks(weeks, y.size)
    w = count(window, name="window", minimum=1)
    return np.array([y[max(0, week - w) : week].mean() for week in t])


def smoothing_forecasts(
    series: ArrayLike, weeks: ArrayLike, *, alpha: float = SMOOTHING, warmup: int = WARMUP
) -> NDArray[np.float64]:
    """Exponential smoothing started from the warm-up mean, read before each forecast week."""
    y = _series(series)
    start = count(warmup, name="warmup", minimum=1)
    t = _weeks(weeks, y.size, first=start)
    a = _alpha(alpha)
    level = y[:start].mean()
    levels = [level]
    for value in y[start : int(t.max())]:
        level = a * value + (1 - a) * level
        levels.append(level)
    return np.array(levels)[t - start]


def croston_forecasts(
    series: ArrayLike, weeks: ArrayLike, *, alpha: float = SMOOTHING, debias: bool = False
) -> NDArray[np.float64]:
    """Croston's rate before each forecast week: smoothed size over smoothed interval.

    Both start from the first demand, the interval counting from the start of
    the series, and update only when a demand occurs. Before the second demand
    the forecast is the mean so far. ``debias`` multiplies by ``1 - alpha / 2``.
    """
    y = _series(series)
    t = _weeks(weeks, y.size)
    a = _alpha(alpha)
    demands = np.flatnonzero(y)
    sizes: list[float] = []
    intervals: list[float] = []
    if demands.size:
        z, x, last = y[demands[0]], float(demands[0] + 1), demands[0]
        sizes.append(z)
        intervals.append(x)
        for i in demands[1:]:
            z = a * y[i] + (1 - a) * z
            x = a * (i - last) + (1 - a) * x
            last = i
            sizes.append(z)
            intervals.append(x)
    seen = np.searchsorted(demands, t, side="left")
    forecasts = []
    for week, k in zip(t, seen, strict=True):
        if k < 2:
            forecasts.append(y[:week].mean())
            continue
        rate = sizes[k - 1] / intervals[k - 1]
        forecasts.append(rate * (1 - a / 2) if debias else rate)
    return np.array(forecasts)


def method_forecasts(
    series: ArrayLike, weeks: ArrayLike, *, alpha: float = SMOOTHING
) -> tuple[NDArray[np.float64], ...]:
    """Each method's forecast for every forecast week, in the order of ``METHODS``."""
    y = _series(series)
    t = _weeks(weeks, y.size)
    return (
        np.zeros(t.size),
        rolling_mean_forecasts(y, t),
        smoothing_forecasts(y, t, alpha=alpha),
        croston_forecasts(y, t, alpha=alpha),
        croston_forecasts(y, t, alpha=alpha, debias=True),
    )


def constant_forecast_mae(
    forecast: float,
    *,
    demand_probability: float = DEMAND_PROBABILITY,
    mean_size: float = MEAN_SIZE,
) -> float:
    """Mean absolute error of a constant forecast ``c``.

    ``E|Y - c| = (1 - p) c + p E|S - c|`` with ``E|S - c| = m - c + 2 E[(c - S)^+]``,
    and since every demand is at least one unit the last term vanishes for ``c <= 1``,
    leaving ``(1 - p) c + p (m - c)``.
    """
    c = non_negative(forecast, name="forecast")
    p = _probability(demand_probability)
    m = _mean_size(mean_size)
    sizes = np.arange(1, floor(c) + 1)
    shortfall = float(np.sum((c - sizes) * stats.poisson.pmf(sizes - 1, m - 1)))
    return (1 - p) * c + p * (m - c + 2 * shortfall)


def croston_expectation(
    *,
    alpha: float = SMOOTHING,
    demand_probability: float = DEMAND_PROBABILITY,
    mean_size: float = MEAN_SIZE,
) -> float:
    """Croston's expected forecast to second order: ``m p (1 + alpha (1 - p) / (2 - alpha))``."""
    a = _alpha(alpha)
    p = _probability(demand_probability)
    m = _mean_size(mean_size)
    return m * p * (1 + a * (1 - p) / (2 - a))


def closed_forms(
    *,
    alpha: float = SMOOTHING,
    window: int = WINDOW,
    demand_probability: float = DEMAND_PROBABILITY,
    mean_size: float = MEAN_SIZE,
) -> ClosedForms:
    """Population values of the demand and of each forecast's accuracy."""
    a = _alpha(alpha)
    w = count(window, name="window", minimum=1)
    p = _probability(demand_probability)
    m = _mean_size(mean_size)
    rate = p * m
    second_moment = p * ((m - 1) + m * m)
    variance = second_moment - rate * rate
    croston = croston_expectation(alpha=a, demand_probability=p, mean_size=m)
    return ClosedForms(
        true_rate=rate,
        zero_share=1 - p,
        weekly_variance=variance,
        zero_forecast_mae=constant_forecast_mae(0.0, demand_probability=p, mean_size=m),
        rate_forecast_mae=constant_forecast_mae(rate, demand_probability=p, mean_size=m),
        zero_forecast_rmse=sqrt(second_moment),
        rolling_mean_rmse=sqrt(variance * (1 + 1 / w)),
        smoothing_rmse=sqrt(variance * (1 + a / (2 - a))),
        croston=croston,
        croston_debiased=croston * (1 - a / 2),
    )


def example_payload() -> IntermittentSummary:
    """Forecast the figure's 40 series with every method and summarise the bars and errors."""
    weeks = forecast_weeks()
    averages: list[list[float]] = [[] for _ in METHODS]
    bias: list[list[float]] = [[] for _ in METHODS]
    mae: list[list[float]] = [[] for _ in METHODS]
    rmse: list[list[float]] = [[] for _ in METHODS]
    for r in range(SERIES):
        y = demand_series(np.random.default_rng(SEED_BASE + r))
        actual = y[weeks]
        for index, predictions in enumerate(method_forecasts(y, weeks)):
            error = predictions - actual
            averages[index].append(float(np.mean(predictions)))
            bias[index].append(float(np.mean(error)))
            mae[index].append(float(np.mean(np.abs(error))))
            rmse[index].append(float(np.sqrt(np.mean(error**2))))

    rows = tuple(
        MethodRow(
            method=name,
            average_forecast=float(np.mean(averages[index])),
            standard_error=float(np.std(averages[index]) / np.sqrt(SERIES)),
            series_averages=tuple(averages[index]),
            bias=float(np.mean(bias[index])),
            mean_absolute_error=float(np.mean(mae[index])),
            root_mean_squared_error=float(np.mean(rmse[index])),
        )
        for index, name in enumerate(METHODS)
    )
    return IntermittentSummary(
        series=SERIES,
        forecasts_per_series=int(weeks.size),
        rows=rows,
        closed_forms=closed_forms(),
    )
