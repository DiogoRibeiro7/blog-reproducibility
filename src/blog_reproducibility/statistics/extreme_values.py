"""Peaks over threshold, for the article on estimating the tail of operational load.

Three years of hourly peak load (``n = 3 H`` hours, ``H = 8760``) are drawn as
``100 + 10 T`` with ``T`` a Student t with four degrees of freedom, a heavy tail
of shape ``xi = 1 / 4`` whose quantiles are known: the level exceeded on average
once in ``T`` years has hourly exceedance probability ``1 / (T H)``. The sample
maximum sits near the three-year level, and a normal distribution fitted to all
hours reads the ten-year level off a tail it does not have.

Above a high threshold ``u`` the excess approaches a generalised Pareto
distribution with shape ``xi`` and scale ``sigma``. With ``zeta`` the share of
hours above ``u`` the fitted exceedance probability of a level ``x`` is
``zeta (1 + xi (x - u) / sigma)^(-1 / xi)``, and the ``T``-year level is

    x_T = u + sigma / xi [(T H zeta)^xi - 1].

The fit is SciPy's maximum likelihood with the location fixed at zero, as in the
article. The largest of ``n`` independent hours is below ``x`` with probability
``F(x)^n``, which gives the distribution of the sample maximum across records in
closed form.

The figure and the article are one computation from a generator seeded at 0: the
hourly loads, the normal fit, and the Pareto fit above the 98th percentile, drawn
as exceedance curves on a grid of 400 levels up to 620. The article's threshold
sweep refits the same hours, and its dependent series continues the same
generator: a first latent value and then ``n`` innovations drive an
autoregression with coefficient 0.8, mapped to the same t marginal. Exceedances
of its 98th percentile start a new cluster after 24 hours below it. Every number
these print is reproduced here. The bootstrap interval resamples the same hours
from a generator seeded at 1, 500 times; that takes several seconds, so the
number of resamples is a parameter. The article's replication study over 200
records gives no code or seeds and is not reproduced; its rows for the sample
maximum and the normal fit are compared with closed forms.
"""

from dataclasses import dataclass
from math import expm1, log, log1p, sqrt
from typing import Final

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
    "AUTOCORRELATION",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CLUSTER_GAP",
    "DEGREES_OF_FREEDOM",
    "GRID_POINTS",
    "GRID_STOP",
    "HOURS_PER_YEAR",
    "INTERVAL_PERCENTILES",
    "LOCATION",
    "MARGIN",
    "RECORD_YEARS",
    "RETURN_PERIODS",
    "SCALE",
    "SEED",
    "THRESHOLD_QUANTILE",
    "THRESHOLD_QUANTILES",
    "BootstrapIntervals",
    "DependentRecord",
    "ExtremeValueSummary",
    "MaximumDistribution",
    "PotFit",
    "RecordEstimates",
    "TailCurves",
    "bootstrap_return_levels",
    "count_clusters",
    "example_payload",
    "fit_peaks_over_threshold",
    "maximum_distribution",
    "normal_return_level",
    "pareto_exceedance",
    "return_level",
    "simulate_dependent_load",
    "simulate_load",
    "true_exceedance",
    "true_return_level",
]

SEED: Final[int] = 0
BOOTSTRAP_SEED: Final[int] = 1
# Hourly peak load: 100 + 10 T with T a Student t on four degrees of freedom.
DEGREES_OF_FREEDOM: Final[int] = 4
LOCATION: Final[float] = 100.0
SCALE: Final[float] = 10.0
HOURS_PER_YEAR: Final[int] = 8760
RECORD_YEARS: Final[int] = 3
RETURN_PERIODS: Final[tuple[int, ...]] = (10, 100)
THRESHOLD_QUANTILE: Final[float] = 0.98
THRESHOLD_QUANTILES: Final[tuple[float, ...]] = (0.90, 0.95, 0.98, 0.99, 0.995)
# "Sample maximum plus 20 percent."
MARGIN: Final[float] = 0.2
BOOTSTRAP_REPLICATES: Final[int] = 500
INTERVAL_PERCENTILES: Final[tuple[float, float]] = (5.0, 95.0)
# The dependent series: lag-one correlation of the latent process, and the run rule.
AUTOCORRELATION: Final[float] = 0.8
CLUSTER_GAP: Final[int] = 24
# The figure's grid of levels, from the threshold to 620.
GRID_STOP: Final[float] = 620.0
GRID_POINTS: Final[int] = 400


@dataclass(frozen=True, slots=True)
class PotFit:
    """A generalised Pareto fit above one threshold, and the return levels it implies."""

    quantile: float
    threshold: float
    exceedances: int
    shape: float
    scale: float
    rate: float
    ten_year: float
    hundred_year: float


@dataclass(frozen=True, slots=True)
class TailCurves:
    """The figure: exceedance probability per hour on a grid of levels, and the observed tail."""

    levels: tuple[float, ...]
    true: tuple[float, ...]
    normal: tuple[float, ...]
    pareto: tuple[float, ...]
    observed_levels: tuple[float, ...]
    observed_probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RecordEstimates:
    """The three-year record: the true levels and the shortcuts' estimates of them."""

    hours: int
    true_record_level: float
    true_ten_year: float
    true_hundred_year: float
    sample_maximum: float
    maximum_with_margin: float
    normal_mean: float
    normal_sd: float
    normal_ten_year: float
    normal_hundred_year: float
    hours_above_normal_ten_year: int
    normal_understatement: float


@dataclass(frozen=True, slots=True)
class BootstrapIntervals:
    """Percentile intervals for the ten- and hundred-year levels from resampled hours."""

    replicates: int
    ten_year: tuple[float, float]
    hundred_year: tuple[float, float]


@dataclass(frozen=True, slots=True)
class DependentRecord:
    """The autocorrelated series with the same marginal: clusters and the naive fit."""

    autocorrelation: float
    exceedances: int
    clusters: int
    extremal_index: float
    fit: PotFit


@dataclass(frozen=True, slots=True)
class MaximumDistribution:
    """Closed forms for the sample maximum times ``1 + margin`` across three-year records."""

    margin: float
    median: float
    lower_quartile: float
    upper_quartile: float
    share_below_ten_year: float


@dataclass(frozen=True, slots=True)
class ExtremeValueSummary:
    """The figure's curves and every number the article prints from its generators."""

    curves: TailCurves
    record: RecordEstimates
    fit: PotFit
    bootstrap: BootstrapIntervals
    thresholds: tuple[PotFit, ...]
    dependent: DependentRecord
    maxima: tuple[MaximumDistribution, ...]
    normal_limit_ten_year: float


def _autocorrelation(value: float) -> float:
    phi = real(value, name="autocorrelation")
    if not -1 < phi < 1:
        raise ValueError("autocorrelation must lie strictly between -1 and 1")
    return phi


def _hours(years: float, hours_per_year: int) -> float:
    return positive(years, name="years") * count(hours_per_year, name="hours_per_year", minimum=1)


def true_return_level(
    years: float,
    *,
    df: float = DEGREES_OF_FREEDOM,
    loc: float = LOCATION,
    scale: float = SCALE,
    hours_per_year: int = HOURS_PER_YEAR,
) -> float:
    """The level an hour exceeds with probability ``1 / (T H)``, from the t quantile."""
    tail = 1 / _hours(years, hours_per_year)
    if tail >= 1:
        raise ValueError("the return period must exceed one hour")
    nu = positive(df, name="df")
    return loc + positive(scale, name="scale") * float(stats.t.ppf(1 - tail, nu))


def true_exceedance(
    levels: ArrayLike,
    *,
    df: float = DEGREES_OF_FREEDOM,
    loc: float = LOCATION,
    scale: float = SCALE,
) -> NDArray[np.float64]:
    """The chance an hour exceeds each level, from the t survival function."""
    x = np.asarray(levels, dtype=np.float64)
    sf: NDArray[np.float64] = stats.t.sf((x - loc) / positive(scale, name="scale"), df)
    return sf


def simulate_load(
    rng: np.random.Generator,
    hours: int = RECORD_YEARS * HOURS_PER_YEAR,
    *,
    df: float = DEGREES_OF_FREEDOM,
    loc: float = LOCATION,
    scale: float = SCALE,
) -> NDArray[np.float64]:
    """Independent hourly peak loads, ``loc + scale T`` with ``T`` a Student t."""
    n = count(hours, name="hours", minimum=1)
    load: NDArray[np.float64] = loc + positive(scale, name="scale") * rng.standard_t(
        positive(df, name="df"), n
    )
    return load


def simulate_dependent_load(
    rng: np.random.Generator,
    hours: int = RECORD_YEARS * HOURS_PER_YEAR,
    *,
    autocorrelation: float = AUTOCORRELATION,
    df: float = DEGREES_OF_FREEDOM,
    loc: float = LOCATION,
    scale: float = SCALE,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A Gaussian autoregression mapped to the same t marginal: the loads and the latent series.

    One normal draw starts the latent series and ``hours`` more are the
    innovations, the first of them unused, as in the article's loop.
    """
    n = count(hours, name="hours", minimum=2)
    phi = _autocorrelation(autocorrelation)
    z = np.empty(n)
    z[0] = rng.normal()
    eps = rng.normal(size=n)
    step = np.sqrt(1 - phi**2)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + step * eps[t]
    load: NDArray[np.float64] = loc + positive(scale, name="scale") * stats.t.ppf(
        stats.norm.cdf(z), positive(df, name="df")
    )
    return load, z


def return_level(
    threshold: float,
    shape: float,
    scale: float,
    rate: float,
    years: float,
    *,
    hours_per_year: int = HOURS_PER_YEAR,
) -> float:
    """The ``T``-year level ``u + sigma / xi [(T H zeta)^xi - 1]``, or its limit at ``xi = 0``."""
    u = real(threshold, name="threshold")
    xi = real(shape, name="shape")
    sigma = positive(scale, name="scale")
    zeta = probability(rate, name="rate", inclusive=False)
    periods = _hours(years, hours_per_year) * zeta
    if xi == 0:
        return u + sigma * log(periods)
    return u + sigma / xi * (float(periods**xi) - 1)


def fit_peaks_over_threshold(
    sample: ArrayLike,
    quantile: float = THRESHOLD_QUANTILE,
    *,
    hours_per_year: int = HOURS_PER_YEAR,
) -> PotFit:
    """Fit a generalised Pareto distribution to the excesses over a sample quantile."""
    x = np.asarray(sample, dtype=np.float64)
    if x.ndim != 1 or x.size < 3 or not np.all(np.isfinite(x)):
        raise ValueError("sample must be a finite 1-D array of at least three values")
    q = probability(quantile, name="quantile", inclusive=False)
    u = float(np.quantile(x, q))
    excess = x[x > u] - u
    if excess.size < 2:
        raise ValueError("the threshold leaves fewer than two exceedances")
    shape, _, scale = stats.genpareto.fit(excess, floc=0)
    rate = float(np.mean(x > u))
    xi, sigma = float(shape), float(scale)
    return PotFit(
        quantile=q,
        threshold=u,
        exceedances=int(excess.size),
        shape=xi,
        scale=sigma,
        rate=rate,
        ten_year=return_level(u, xi, sigma, rate, 10, hours_per_year=hours_per_year),
        hundred_year=return_level(u, xi, sigma, rate, 100, hours_per_year=hours_per_year),
    )


def pareto_exceedance(fit: PotFit, levels: ArrayLike) -> NDArray[np.float64]:
    """The fitted chance an hour exceeds each level at or above the threshold."""
    x = np.asarray(levels, dtype=np.float64)
    if fit.shape == 0:
        exponential: NDArray[np.float64] = fit.rate * np.exp(-(x - fit.threshold) / fit.scale)
        return exponential
    base = 1 + fit.shape * (x - fit.threshold) / fit.scale
    inside = base > 0
    curve: NDArray[np.float64] = np.zeros_like(x)
    curve[inside] = fit.rate * base[inside] ** (-1 / fit.shape)
    return curve


def normal_return_level(
    mean: float, sd: float, years: float, *, hours_per_year: int = HOURS_PER_YEAR
) -> float:
    """The ``T``-year level of a normal distribution: ``mean + sd z_{1 - 1 / (T H)}``."""
    tail = 1 / _hours(years, hours_per_year)
    return mean + positive(sd, name="sd") * float(stats.norm.ppf(1 - tail))


def bootstrap_return_levels(
    sample: ArrayLike,
    rng: np.random.Generator,
    replicates: int = BOOTSTRAP_REPLICATES,
    *,
    quantile: float = THRESHOLD_QUANTILE,
) -> NDArray[np.float64]:
    """Refit resampled hours: one row per resample, the ten- and hundred-year levels."""
    x = np.asarray(sample, dtype=np.float64)
    n = x.size
    levels = np.empty((count(replicates, name="replicates", minimum=1), 2))
    for i in range(levels.shape[0]):
        fit = fit_peaks_over_threshold(x[rng.integers(0, n, n)], quantile)
        levels[i] = fit.ten_year, fit.hundred_year
    return levels


def _interval(values: NDArray[np.float64]) -> tuple[float, float]:
    low, high = np.percentile(values, INTERVAL_PERCENTILES)
    return float(low), float(high)


def count_clusters(exceeds: ArrayLike, gap: int = CLUSTER_GAP) -> int:
    """Clusters of exceedances: a new one starts after ``gap`` or more hours below.

    The first exceedance always starts a cluster, and between exceedances at
    hours ``i < j`` there are ``j - i - 1`` hours below.
    """
    over = np.asarray(exceeds)
    if over.dtype != np.bool_ or over.ndim != 1:
        raise ValueError("exceeds must be a 1-D boolean array")
    hours = np.flatnonzero(over)
    if hours.size == 0:
        return 0
    run = count(gap, name="gap", minimum=1)
    return 1 + int(np.count_nonzero(np.diff(hours) - 1 >= run))


def maximum_distribution(
    margin: float = 0.0,
    *,
    hours: int = RECORD_YEARS * HOURS_PER_YEAR,
    target: float | None = None,
    df: float = DEGREES_OF_FREEDOM,
    loc: float = LOCATION,
    scale: float = SCALE,
) -> MaximumDistribution:
    """Quartiles of ``(1 + margin)`` times the largest of ``hours`` independent loads.

    The maximum is below ``x`` with probability ``F(x)^n``, so its ``p`` quantile
    is the level with exceedance probability ``1 - p^(1 / n)``. The share below
    ``target`` (by default the true ten-year level) is ``F(target / (1 + m))^n``.
    """
    factor = 1 + non_negative(margin, name="margin")
    n = count(hours, name="hours", minimum=1)
    sd = positive(scale, name="scale")

    def quantile(p: float) -> float:
        return factor * (loc + sd * float(stats.t.isf(-expm1(log(p) / n), df)))

    level = (
        true_return_level(10, df=df, loc=loc, scale=scale)
        if target is None
        else positive(target, name="target")
    )
    tail = float(stats.t.sf((level / factor - loc) / sd, df))
    return MaximumDistribution(
        margin=factor - 1,
        median=quantile(0.5),
        lower_quartile=quantile(0.25),
        upper_quartile=quantile(0.75),
        share_below_ten_year=float(np.exp(n * log1p(-tail))),
    )


def example_payload(*, bootstrap_replicates: int = BOOTSTRAP_REPLICATES) -> ExtremeValueSummary:
    """Return the figure's curves and the article's numbers, with the bootstrap as asked."""
    rng = np.random.default_rng(SEED)
    x = simulate_load(rng)
    n = x.size
    fit = fit_peaks_over_threshold(x)
    mean, sd = float(x.mean()), float(x.std())

    grid = np.linspace(fit.threshold, GRID_STOP, GRID_POINTS)
    top = np.sort(x[x > fit.threshold])[::-1]
    curves = TailCurves(
        levels=tuple(float(v) for v in grid),
        true=tuple(float(v) for v in true_exceedance(grid)),
        normal=tuple(float(v) for v in stats.norm.sf(grid, x.mean(), x.std())),
        pareto=tuple(float(v) for v in pareto_exceedance(fit, grid)),
        observed_levels=tuple(float(v) for v in top),
        observed_probabilities=tuple(float(v) for v in (np.arange(len(top)) + 1) / n),
    )

    normal_ten_year = normal_return_level(mean, sd, 10)
    true_ten_year = true_return_level(10)
    record = RecordEstimates(
        hours=n,
        true_record_level=true_return_level(RECORD_YEARS),
        true_ten_year=true_ten_year,
        true_hundred_year=true_return_level(100),
        sample_maximum=float(x.max()),
        maximum_with_margin=(1 + MARGIN) * float(x.max()),
        normal_mean=mean,
        normal_sd=sd,
        normal_ten_year=normal_ten_year,
        normal_hundred_year=normal_return_level(mean, sd, 100),
        hours_above_normal_ten_year=int(np.count_nonzero(x > normal_ten_year)),
        normal_understatement=float(true_exceedance(normal_ten_year)) * 10 * HOURS_PER_YEAR,
    )

    # The dependent series continues the figure's generator.
    dependent, _ = simulate_dependent_load(rng)
    over = dependent > np.quantile(dependent, THRESHOLD_QUANTILE)
    clusters = count_clusters(over)
    exceedances = int(over.sum())

    levels = bootstrap_return_levels(x, np.random.default_rng(BOOTSTRAP_SEED), bootstrap_replicates)
    df = DEGREES_OF_FREEDOM
    return ExtremeValueSummary(
        curves=curves,
        record=record,
        fit=fit,
        bootstrap=BootstrapIntervals(
            replicates=bootstrap_replicates,
            ten_year=_interval(levels[:, 0]),
            hundred_year=_interval(levels[:, 1]),
        ),
        thresholds=tuple(fit_peaks_over_threshold(x, q) for q in THRESHOLD_QUANTILES),
        dependent=DependentRecord(
            autocorrelation=AUTOCORRELATION,
            exceedances=exceedances,
            clusters=clusters,
            extremal_index=clusters / exceedances,
            fit=fit_peaks_over_threshold(dependent),
        ),
        maxima=(maximum_distribution(0.0), maximum_distribution(MARGIN)),
        normal_limit_ten_year=normal_return_level(LOCATION, SCALE * sqrt(df / (df - 2)), 10),
    )
