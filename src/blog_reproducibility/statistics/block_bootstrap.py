"""Bootstrap intervals for the mean of a dependent series, for the article on bootstrap failures.

The ordinary bootstrap resamples observations independently, so on a series
whose neighbours repeat each other's information it builds a bootstrap world
with more independent information than the data hold. For a stationary AR(1)
series ``x_t = phi x_{t-1} + e_t`` with unit-variance innovations, the
observations have variance ``gamma_0 = 1 / (1 - phi^2)`` and autocorrelation
``phi^k`` at lag ``k``, so the mean of ``n`` of them has variance

    Var(xbar) = gamma_0 / n [(1 + phi) / (1 - phi) - 2 phi (1 - phi^n) / (n (1 - phi)^2)],

about ``(1 + phi) / (1 - phi)`` times the value for independent data. The
series carries the information of ``gamma_0 / Var(xbar)`` independent
observations, roughly ``n (1 - phi) / (1 + phi)``. The ordinary bootstrap
instead estimates the variance of the mean by the sample variance over ``n``,
whose expectation is ``(gamma_0 - Var(xbar)) / n``. Were its interval normal
with that expected width, it would cover the mean with probability
``2 Phi(1.96 / r) - 1``, with ``r`` the ratio of the true standard error to the
bootstrap one; the sampling noise in the width takes a few points more.

The moving-block bootstrap resamples ``ceil(n / l)`` blocks of ``l``
consecutive observations, their starts drawn uniformly among the ``n - l + 1``
possible, and truncates the joined blocks to ``n``, so that dependence within a
block survives. A block length of one is the ordinary bootstrap. The percentile
interval reads the 2.5th and 97.5th percentiles of 400 resampled means.

The figure and the article's coverage table are the same computation: 400
series of 200 points at each of four autocorrelations, series ``s`` drawn from a
generator seeded at ``s`` that then drives the resampling for each block length
in turn. It is reproduced draw for draw and the table is pinned exactly. Each
block length's starts are drawn here in one call rather than one call per
resample; PCG64 hands out bounded integers from one continuous stream, so the
draws and the intervals are identical, in a fraction of the time. The article's
other simulations (the standard error at 0.7 from seeds 1 and 10,000 up,
clustered data, small skewed samples and the sample maximum) use other draws
and are not reproduced; its closed-form arithmetic is checked in the tests.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, probability, real

__all__ = [
    "AUTOCORRELATIONS",
    "BLOCK_LENGTHS",
    "FIRST_SEED",
    "INTERVAL_PERCENTILES",
    "LENGTH",
    "NOMINAL_COVERAGE",
    "REPLICATIONS",
    "RESAMPLES",
    "TRUE_MEAN",
    "BlockBootstrapSummary",
    "CoverageRow",
    "DependenceRow",
    "ar1_series",
    "block_bootstrap_interval",
    "block_bootstrap_means",
    "coverage_row",
    "dependence_row",
    "effective_sample_size",
    "example_payload",
    "mean_variance",
    "normal_coverage",
    "ordinary_bootstrap_variance",
    "suggested_block_length",
]

# Replication s draws from np.random.default_rng(FIRST_SEED + s).
FIRST_SEED: Final[int] = 0
LENGTH: Final[int] = 200
REPLICATIONS: Final[int] = 400
RESAMPLES: Final[int] = 400
AUTOCORRELATIONS: Final[tuple[float, ...]] = (0.0, 0.3, 0.7, 0.9)
BLOCK_LENGTHS: Final[tuple[int, ...]] = (1, 2, 5, 10, 20, 40)
NOMINAL_COVERAGE: Final[float] = 0.95
INTERVAL_PERCENTILES: Final[tuple[float, float]] = (2.5, 97.5)
TRUE_MEAN: Final[float] = 0.0


@dataclass(frozen=True, slots=True)
class CoverageRow:
    """Series whose percentile interval covered the true mean, at each block length."""

    autocorrelation: float
    block_lengths: tuple[int, ...]
    replications: int
    covered: tuple[int, ...]
    coverage: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DependenceRow:
    """What autocorrelation does to the mean of a series, in closed form."""

    autocorrelation: float
    mean_standard_error: float
    ordinary_bootstrap_standard_error: float
    standard_error_ratio: float
    effective_sample_size: float
    approximate_effective_sample_size: float
    normal_coverage: float


@dataclass(frozen=True, slots=True)
class BlockBootstrapSummary:
    """The figure's coverage table and the closed forms that explain it."""

    length: int
    resamples: int
    rows: tuple[CoverageRow, ...]
    dependence: tuple[DependenceRow, ...]
    suggested_block_length: float


def _autocorrelation(value: float) -> float:
    phi = real(value, name="autocorrelation")
    if not -1.0 < phi < 1.0:
        raise ValueError("autocorrelation must lie in (-1, 1) for a stationary series")
    return phi


def _series(series: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("series must be a 1-D array of two or more observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("series must be finite")
    return values


def _block_length(block_length: int, length: int) -> int:
    block = count(block_length, name="block_length", minimum=1)
    if block > length:
        raise ValueError("block_length cannot exceed the length of the series")
    return block


def ar1_series(
    rng: np.random.Generator, length: int = LENGTH, autocorrelation: float = 0.0
) -> NDArray[np.float64]:
    """Draw a stationary AR(1) series with unit-variance innovations.

    The draws are the article's: the first observation from the stationary
    distribution, then ``length`` innovations of which the first is unused.
    """
    n = count(length, name="length", minimum=2)
    phi = _autocorrelation(autocorrelation)
    series = np.empty(n)
    series[0] = rng.normal(0, 1 / sqrt(1 - phi**2))
    innovations = rng.normal(size=n)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + innovations[t]
    return series


def block_bootstrap_means(
    rng: np.random.Generator,
    series: ArrayLike,
    block_length: int,
    resamples: int = RESAMPLES,
) -> NDArray[np.float64]:
    """Means of moving-block resamples: random blocks joined and cut to the series length."""
    values = _series(series)
    n = values.size
    block = _block_length(block_length, n)
    draws = count(resamples, name="resamples", minimum=1)
    blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(draws, blocks))
    # Row r of the window view is the block starting at r; the gather is C-ordered,
    # so each resample is summed in the same order as the article's loop.
    resampled = sliding_window_view(values, block)[starts].reshape(draws, blocks * block)[:, :n]
    means: NDArray[np.float64] = resampled.mean(axis=1)
    return means


def block_bootstrap_interval(
    rng: np.random.Generator,
    series: ArrayLike,
    block_length: int,
    resamples: int = RESAMPLES,
) -> tuple[float, float]:
    """Percentile interval for the mean from moving-block resamples."""
    means = block_bootstrap_means(rng, series, block_length, resamples)
    low, high = np.percentile(means, INTERVAL_PERCENTILES)
    return float(low), float(high)


def coverage_row(
    autocorrelation: float,
    *,
    length: int = LENGTH,
    replications: int = REPLICATIONS,
    resamples: int = RESAMPLES,
    block_lengths: tuple[int, ...] = BLOCK_LENGTHS,
    first_seed: int = FIRST_SEED,
) -> CoverageRow:
    """Coverage of the true mean by the percentile interval, one fresh generator per series.

    Each series is drawn and then resampled at every block length in turn from
    its own generator. The percentiles are taken together at the end, which
    gives the same intervals as :func:`block_bootstrap_interval`.
    """
    phi = _autocorrelation(autocorrelation)
    n = count(length, name="length", minimum=2)
    reps = count(replications, name="replications", minimum=1)
    seed = count(first_seed, name="first_seed")
    if not block_lengths:
        raise ValueError("block_lengths must not be empty")
    blocks = tuple(_block_length(block, n) for block in block_lengths)

    means = np.empty((reps, len(blocks), count(resamples, name="resamples", minimum=1)))
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        series = ar1_series(rng, n, phi)
        for j, block in enumerate(blocks):
            means[rep, j] = block_bootstrap_means(rng, series, block, resamples)
    low, high = np.percentile(means, INTERVAL_PERCENTILES, axis=-1)
    covered = np.count_nonzero((low <= TRUE_MEAN) & (high >= TRUE_MEAN), axis=0)
    return CoverageRow(
        autocorrelation=phi,
        block_lengths=blocks,
        replications=reps,
        covered=tuple(int(hits) for hits in covered),
        coverage=tuple(int(hits) / reps for hits in covered),
    )


def mean_variance(length: int, autocorrelation: float) -> float:
    """Variance of the mean of a stationary AR(1) series with unit-variance innovations."""
    n = count(length, name="length", minimum=1)
    phi = _autocorrelation(autocorrelation)
    gamma_0 = 1 / (1 - phi**2)
    inflation = (1 + phi) / (1 - phi) - 2 * phi * (1 - phi**n) / (n * (1 - phi) ** 2)
    return gamma_0 / n * inflation


def ordinary_bootstrap_variance(length: int, autocorrelation: float) -> float:
    """Expected variance the ordinary bootstrap gives the mean: the sample variance over ``n``."""
    n = count(length, name="length", minimum=2)
    gamma_0 = 1 / (1 - _autocorrelation(autocorrelation) ** 2)
    return (gamma_0 - mean_variance(n, autocorrelation)) / n


def effective_sample_size(length: int, autocorrelation: float) -> float:
    """Independent observations whose mean is as precise as that of the series."""
    gamma_0 = 1 / (1 - _autocorrelation(autocorrelation) ** 2)
    return gamma_0 / mean_variance(length, autocorrelation)


def normal_coverage(
    length: int, autocorrelation: float, *, nominal: float = NOMINAL_COVERAGE
) -> float:
    """Coverage of a normal interval with the ordinary bootstrap's expected standard error."""
    level = probability(nominal, name="nominal", inclusive=False)
    ratio = sqrt(mean_variance(length, autocorrelation)) / sqrt(
        ordinary_bootstrap_variance(length, autocorrelation)
    )
    critical = float(stats.norm.ppf((1 + level) / 2))
    return float(2 * stats.norm.cdf(critical / ratio) - 1)


def suggested_block_length(length: int = LENGTH) -> float:
    """Hall, Horowitz and Jing's rate for the block length of a mean: ``n^(1/3)``."""
    return float(count(length, name="length", minimum=1) ** (1 / 3))


def dependence_row(autocorrelation: float, length: int = LENGTH) -> DependenceRow:
    """Standard errors, effective sample sizes and the normal coverage at one autocorrelation."""
    phi = _autocorrelation(autocorrelation)
    n = count(length, name="length", minimum=2)
    truth = sqrt(mean_variance(n, phi))
    ordinary = sqrt(ordinary_bootstrap_variance(n, phi))
    return DependenceRow(
        autocorrelation=phi,
        mean_standard_error=truth,
        ordinary_bootstrap_standard_error=ordinary,
        standard_error_ratio=truth / ordinary,
        effective_sample_size=effective_sample_size(n, phi),
        approximate_effective_sample_size=n * (1 - phi) / (1 + phi),
        normal_coverage=normal_coverage(n, phi),
    )


def example_payload() -> BlockBootstrapSummary:
    """Return the figure's coverage table (400 series per autocorrelation) and the closed forms."""
    return BlockBootstrapSummary(
        length=LENGTH,
        resamples=RESAMPLES,
        rows=tuple(coverage_row(phi) for phi in AUTOCORRELATIONS),
        dependence=tuple(dependence_row(phi) for phi in AUTOCORRELATIONS),
        suggested_block_length=suggested_block_length(),
    )
