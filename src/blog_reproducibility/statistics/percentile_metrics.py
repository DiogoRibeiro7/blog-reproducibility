"""Precision of the mean, median and tail percentiles, for the article on latency metrics.

Latency is modelled as a two-part mixture: a lognormal body with log-mean
``log(80) - 0.18`` and log-sd 0.6 (mean 80 ms, median about 67 ms) and, for 3
percent of requests, a slow path that is lognormal around 900 ms with log-sd
0.7. The mixture has a heavy right tail, so its mean (112 ms) sits far above
its median (68 ms).

A sample quantile at probability ``p`` has asymptotic standard error

    se(x_p) = sqrt(p (1 - p) / n) / f(x_p),

with ``f`` the density at the quantile. The numerator barely changes between
the median and p99, but the density falls by orders of magnitude, so the
99th percentile is several times less precise than the median at every sample
size. The mean's relative standard error is ``sd / (mean sqrt(n))``. Both are
computed exactly here from the mixture's CDF and density.

The figure measures the relative standard error (standard deviation over mean,
across 200 repeats) of all four statistics at six sample sizes from 500 to
100,000 requests, from one generator seeded at 0, and is reproduced draw for
draw. The article's tables come from a generator that first draws two million
requests per scenario, so their streams differ from the figure's and they are
not reproduced; the simulation is checked against the closed forms instead.
"""

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "BODY_LOG_MEAN",
    "BODY_LOG_SD",
    "QUANTILES",
    "REPLICATIONS",
    "REPORTED_QUANTILES",
    "SAMPLE_SIZES",
    "SEED",
    "SLOW_LOG_MEAN",
    "SLOW_LOG_SD",
    "SLOW_SHARE",
    "PercentileSummary",
    "PopulationSummary",
    "PrecisionRow",
    "QuantilePoint",
    "draw_latency",
    "example_payload",
    "mean_relative_se",
    "population_cdf",
    "population_density",
    "population_mean",
    "population_quantile",
    "population_sd",
    "population_summary",
    "precision_rows",
    "predicted_rows",
    "quantile_relative_se",
]

SEED: Final[int] = 0
REPLICATIONS: Final[int] = 200
SAMPLE_SIZES: Final[tuple[int, ...]] = (500, 1000, 5000, 10_000, 50_000, 100_000)
QUANTILES: Final[tuple[float, ...]] = (0.5, 0.95, 0.99)
# NumPy's logarithms, as the figure computes them.
BODY_LOG_MEAN: Final[float] = float(np.log(80.0)) - 0.18
BODY_LOG_SD: Final[float] = 0.6
SLOW_SHARE: Final[float] = 0.03
SLOW_LOG_MEAN: Final[float] = float(np.log(900.0))
SLOW_LOG_SD: Final[float] = 0.7
REPORTED_QUANTILES: Final[tuple[float, ...]] = (0.5, 0.9, 0.95, 0.99)


@dataclass(frozen=True, slots=True)
class PrecisionRow:
    """Relative standard error of each statistic at one sample size."""

    requests: int
    mean: float
    median: float
    p95: float
    p99: float


@dataclass(frozen=True, slots=True)
class QuantilePoint:
    """One quantile of the latency mixture and the density there."""

    probability: float
    latency: float
    density: float


@dataclass(frozen=True, slots=True)
class PopulationSummary:
    """Exact mean, standard deviation and quantiles of the latency mixture, in ms."""

    mean: float
    sd: float
    quantiles: tuple[QuantilePoint, ...]


@dataclass(frozen=True, slots=True)
class PercentileSummary:
    """The figure's simulated precision, its closed-form prediction and the population."""

    simulated: tuple[PrecisionRow, ...]
    predicted: tuple[PrecisionRow, ...]
    population: PopulationSummary


def draw_latency(requests: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Draw request latencies in ms, in the figure's order: body, slow flags, slow path.

    Every request gets a body draw and a slow-path draw, and keeps one of them.
    """
    n = count(requests, name="requests", minimum=1)
    latency = rng.lognormal(BODY_LOG_MEAN, BODY_LOG_SD, n)
    slow = rng.random(n) < SLOW_SHARE
    np.copyto(latency, rng.lognormal(SLOW_LOG_MEAN, SLOW_LOG_SD, n), where=slow)
    return latency


def _relative_errors(values: NDArray[np.float64]) -> tuple[float, float, float, float]:
    mean, median, p95, p99 = (float(np.std(row) / np.mean(row)) for row in values)
    return mean, median, p95, p99


def precision_rows(
    seed: int = SEED,
    *,
    sample_sizes: tuple[int, ...] = SAMPLE_SIZES,
    replications: int = REPLICATIONS,
) -> tuple[PrecisionRow, ...]:
    """Simulate the figure: relative standard errors over repeated samples, in draw order."""
    rng = np.random.default_rng(count(seed, name="seed"))
    reps = count(replications, name="replications", minimum=2)
    sizes = tuple(count(n, name="requests", minimum=1) for n in sample_sizes)
    rows = []
    for n in sizes:
        values = np.empty((1 + len(QUANTILES), reps))
        for rep in range(reps):
            latency = draw_latency(n, rng)
            values[0, rep] = latency.mean()
            values[1:, rep] = np.quantile(latency, QUANTILES)
        rows.append(PrecisionRow(n, *_relative_errors(values)))
    return tuple(rows)


def population_cdf(latency: float) -> float:
    """Share of requests at or below ``latency`` ms."""
    log_x = log(positive(latency, name="latency"))
    body = stats.norm.cdf((log_x - BODY_LOG_MEAN) / BODY_LOG_SD)
    slow = stats.norm.cdf((log_x - SLOW_LOG_MEAN) / SLOW_LOG_SD)
    return float((1 - SLOW_SHARE) * body + SLOW_SHARE * slow)


def population_density(latency: float) -> float:
    """Density of the latency mixture at ``latency`` ms."""
    x = positive(latency, name="latency")
    body = stats.norm.pdf((log(x) - BODY_LOG_MEAN) / BODY_LOG_SD) / BODY_LOG_SD
    slow = stats.norm.pdf((log(x) - SLOW_LOG_MEAN) / SLOW_LOG_SD) / SLOW_LOG_SD
    return float(((1 - SLOW_SHARE) * body + SLOW_SHARE * slow) / x)


def population_quantile(p: float) -> float:
    """Latency below which a share ``p`` of requests fall, by root-finding on the CDF."""
    target = probability(p, name="p", inclusive=False)
    low = BODY_LOG_MEAN - 12 * BODY_LOG_SD
    high = SLOW_LOG_MEAN + 12 * SLOW_LOG_SD
    root = optimize.brentq(lambda log_x: population_cdf(exp(log_x)) - target, low, high)
    return exp(float(root))


def population_mean() -> float:
    """Mean latency of the mixture: each lognormal contributes ``exp(mu + sigma^2 / 2)``."""
    body = exp(BODY_LOG_MEAN + BODY_LOG_SD**2 / 2)
    slow = exp(SLOW_LOG_MEAN + SLOW_LOG_SD**2 / 2)
    return (1 - SLOW_SHARE) * body + SLOW_SHARE * slow


def population_sd() -> float:
    """Return the standard deviation of the mixture, from its second moment."""
    body = exp(2 * BODY_LOG_MEAN + 2 * BODY_LOG_SD**2)
    slow = exp(2 * SLOW_LOG_MEAN + 2 * SLOW_LOG_SD**2)
    return sqrt((1 - SLOW_SHARE) * body + SLOW_SHARE * slow - population_mean() ** 2)


def quantile_relative_se(p: float, requests: int) -> float:
    """Asymptotic standard error of the sample quantile, relative to the quantile."""
    n = count(requests, name="requests", minimum=1)
    x = population_quantile(p)
    return sqrt(p * (1 - p) / n) / population_density(x) / x


def mean_relative_se(requests: int) -> float:
    """Return the standard error of the sample mean, relative to the mean."""
    n = count(requests, name="requests", minimum=1)
    return population_sd() / population_mean() / sqrt(n)


def predicted_rows(sample_sizes: tuple[int, ...] = SAMPLE_SIZES) -> tuple[PrecisionRow, ...]:
    """Closed-form relative standard errors at each of the figure's sample sizes."""
    return tuple(
        PrecisionRow(
            n,
            mean_relative_se(n),
            *(quantile_relative_se(p, n) for p in QUANTILES),
        )
        for n in sample_sizes
    )


def population_summary() -> PopulationSummary:
    """Exact mean, spread, and the quantiles the article reports, with their densities."""
    points = []
    for p in REPORTED_QUANTILES:
        x = population_quantile(p)
        points.append(QuantilePoint(p, x, population_density(x)))
    return PopulationSummary(mean=population_mean(), sd=population_sd(), quantiles=tuple(points))


def example_payload() -> PercentileSummary:
    """Return the figure's simulated precision, the closed forms and the population."""
    return PercentileSummary(
        simulated=precision_rows(),
        predicted=predicted_rows(),
        population=population_summary(),
    )
