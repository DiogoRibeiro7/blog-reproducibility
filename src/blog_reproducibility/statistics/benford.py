"""Benford's law and the spread of the data, for the article on Benford's law as a screen.

Benford's law gives the leading digit ``d`` the share ``log10(1 + 1 / d)``,
from 30.10 percent for a one to 4.58 percent for a nine. Auditors measure how
far a column's leading digits sit from it by the mean absolute deviation (MAD)
over the nine digits, and read below 0.006 as close conformity and below 0.012
as acceptable.

Whether a legitimate column conforms is decided by how many orders of magnitude
it spans. For a lognormal column, ``log10 X`` is normal with mean
``m = mu / ln 10`` and standard deviation ``s = sigma / ln 10``, and the leading
digit is ``d`` when the fractional part of ``log10 X`` lies in
``[log10 d, log10(d + 1))``. Summing the normal probabilities of those intervals
over every decade gives the population digit shares in closed form. The
deviation from Benford is the deviation of the wrapped normal from uniform,
which falls like ``exp(-2 pi^2 s^2)``: fast once ``s`` passes a quarter of a
decade, and irrelevant for a column confined to a factor of a few.

A finite column adds sampling noise. Each share is approximately normal around
its population value ``p_d`` with variance ``p_d (1 - p_d) / n``, so the expected
MAD is the mean over the digits of a folded normal mean,
``tau sqrt(2 / pi) exp(-delta^2 / (2 tau^2)) + delta (1 - 2 Phi(-delta / tau))``
with ``delta = p_d - log10(1 + 1 / d)`` and ``tau^2 = p_d (1 - p_d) / n``. For a
conforming column of 50,000 values this floor is about 0.001, so the MAD stops
improving once the population deviation falls below it.

The figure draws 50,000 lognormal values of location 6 at 22 spreads from 0.15
to 2.6, each from a generator seeded at 3 afresh, and plots the MAD against the
ratio of the 97.5th to the 2.5th percentile, ``exp(2 x 1.96 sigma)``. It is
reproduced draw for draw. The article's spread table uses a generator seeded at
2 and other spreads, so its MADs are compared with the closed forms rather than
reproduced; its percentile ratios and the Benford shares are exact.
"""

from dataclasses import dataclass
from math import log, pi, sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from blog_reproducibility.common.validation import count, positive, real

__all__ = [
    "ACCEPTABLE_CONFORMITY",
    "ARTICLE_SIGMAS",
    "CLOSE_CONFORMITY",
    "COVERAGE_Z",
    "LOCATION",
    "SEED",
    "SIGMA_POINTS",
    "SIGMA_START",
    "SIGMA_STOP",
    "VALUES",
    "BenfordSummary",
    "SpreadCurve",
    "SpreadRow",
    "benford_shares",
    "conformity_span",
    "digit_shares",
    "example_payload",
    "expected_mad",
    "first_digits",
    "lognormal_digit_shares",
    "mean_absolute_deviation",
    "percentile_span",
    "population_mad",
    "spread_curve",
    "spread_row",
]

SEED: Final[int] = 3
# The figure's columns: 50,000 lognormal values of location 6, at 22 spreads.
LOCATION: Final[float] = 6.0
VALUES: Final[int] = 50_000
SIGMA_START: Final[float] = 0.15
SIGMA_STOP: Final[float] = 2.6
SIGMA_POINTS: Final[int] = 22
# The range is the ratio of the 97.5th to the 2.5th percentile.
COVERAGE_Z: Final[float] = 1.96
# Nigrini's thresholds for the first-digit mean absolute deviation.
CLOSE_CONFORMITY: Final[float] = 0.006
ACCEPTABLE_CONFORMITY: Final[float] = 0.012
# The spreads in the article's table.
ARTICLE_SIGMAS: Final[tuple[float, ...]] = (0.2, 0.4, 0.8, 1.2, 1.6, 2.4)


@dataclass(frozen=True, slots=True)
class SpreadCurve:
    """The figure: MAD from Benford against the range of the column, simulated and expected."""

    sigmas: tuple[float, ...]
    spans: tuple[float, ...]
    mads: tuple[float, ...]
    population_mads: tuple[float, ...]
    expected_mads: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SpreadRow:
    """One spread of a lognormal column in closed form: its range and its deviation."""

    sigma: float
    span: float
    population_mad: float
    expected_mad: float


@dataclass(frozen=True, slots=True)
class BenfordSummary:
    """The figure's curve, the law's shares, the threshold crossings and the article's table."""

    benford: tuple[float, ...]
    curve: SpreadCurve
    close_conformity_span: float
    acceptable_conformity_span: float
    article_rows: tuple[SpreadRow, ...]


def benford_shares() -> NDArray[np.float64]:
    """Benford's shares of the leading digits one to nine: ``log10(1 + 1 / d)``."""
    shares: NDArray[np.float64] = np.log10(1 + 1 / np.arange(1, 10))
    return shares


def first_digits(values: ArrayLike) -> NDArray[np.int64]:
    """Leading digit of each non-zero value, sign ignored, as the article computes it."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not np.all(np.isfinite(data)):
        raise ValueError("values must be a finite one-dimensional array")
    magnitudes = np.abs(data)
    magnitudes = magnitudes[magnitudes > 0]
    if magnitudes.size == 0:
        raise ValueError("values must include at least one non-zero value")
    digits: NDArray[np.int64] = (magnitudes / 10 ** np.floor(np.log10(magnitudes))).astype(np.int64)
    return digits


def digit_shares(values: ArrayLike) -> NDArray[np.float64]:
    """Share of the non-zero values with each leading digit, one to nine."""
    digits = first_digits(values)
    shares: NDArray[np.float64] = np.bincount(digits, minlength=10)[1:10] / digits.size
    return shares


def mean_absolute_deviation(values: ArrayLike) -> float:
    """Mean absolute deviation of the leading-digit shares from Benford's, over nine digits."""
    return float(np.abs(digit_shares(values) - benford_shares()).mean())


def percentile_span(sigma: float, *, z: float = COVERAGE_Z) -> float:
    """Ratio of the upper to the lower ``z`` percentile of a lognormal: ``exp(2 z sigma)``."""
    width = 2 * positive(z, name="z") * positive(sigma, name="sigma")
    return float(np.exp(width))


def lognormal_digit_shares(location: float, sigma: float) -> NDArray[np.float64]:
    """Population shares of the leading digits of a lognormal, summed over every decade.

    ``log10 X`` is normal with mean ``location / ln 10`` and standard deviation
    ``sigma / ln 10``; the digit is ``d`` when it lies in ``[k + log10 d,
    k + log10(d + 1))`` for some integer ``k``.
    """
    m = real(location, name="location") / log(10)
    s = positive(sigma, name="sigma") / log(10)
    decades = np.arange(np.floor(m - 12 * s) - 1, np.ceil(m + 12 * s) + 2)
    edges = decades[:, None] + np.log10(np.arange(1, 11))[None, :]
    cdf = stats.norm.cdf(edges, loc=m, scale=s)
    shares: NDArray[np.float64] = np.diff(cdf, axis=1).sum(axis=0)
    return shares


def population_mad(location: float, sigma: float) -> float:
    """Mean absolute deviation from Benford of a lognormal population's digit shares."""
    return float(np.abs(lognormal_digit_shares(location, sigma) - benford_shares()).mean())


def expected_mad(location: float, sigma: float, values: int) -> float:
    """Expected MAD of a column of ``values`` draws, each share taken as normal.

    Each digit contributes the mean of a folded normal around its population
    deviation ``delta`` with standard deviation ``tau = sqrt(p (1 - p) / n)``.
    """
    n = count(values, name="values", minimum=1)
    shares = lognormal_digit_shares(location, sigma)
    delta = shares - benford_shares()
    tau = np.sqrt(shares * (1 - shares) / n)
    folded = np.abs(delta)
    noisy = tau > 0
    t, d = tau[noisy], delta[noisy]
    folded[noisy] = t * sqrt(2 / pi) * np.exp(-(d**2) / (2 * t**2)) + d * (
        1 - 2 * stats.norm.cdf(-d / t)
    )
    return float(folded.mean())


def conformity_span(
    threshold: float,
    *,
    location: float = LOCATION,
    values: int = VALUES,
    lower: float = SIGMA_START,
    upper: float = SIGMA_STOP,
) -> float:
    """The range at which the expected MAD of a lognormal column falls to ``threshold``."""
    level = positive(threshold, name="threshold")
    low, high = positive(lower, name="lower"), positive(upper, name="upper")
    if high <= low:
        raise ValueError("upper must exceed lower")

    def excess(sigma: float) -> float:
        return expected_mad(location, sigma, values) - level

    if excess(low) <= 0 or excess(high) >= 0:
        raise ValueError("threshold is not crossed between lower and upper")
    sigma = float(optimize.brentq(excess, low, high, xtol=1e-14))
    return percentile_span(sigma)


def spread_row(sigma: float, *, location: float = LOCATION, values: int = VALUES) -> SpreadRow:
    """Range, population MAD and expected MAD of a lognormal column at one spread."""
    return SpreadRow(
        sigma=sigma,
        span=percentile_span(sigma),
        population_mad=population_mad(location, sigma),
        expected_mad=expected_mad(location, sigma, values),
    )


def spread_curve(
    seed: int = SEED,
    *,
    location: float = LOCATION,
    values: int = VALUES,
    sigmas: tuple[float, ...] | None = None,
) -> SpreadCurve:
    """The figure: one column per spread, each from a generator seeded afresh."""
    rng_seed = count(seed, name="seed")
    n = count(values, name="values", minimum=1)
    mu = real(location, name="location")
    grid = (
        tuple(float(s) for s in np.linspace(SIGMA_START, SIGMA_STOP, SIGMA_POINTS))
        if sigmas is None
        else tuple(positive(s, name="sigma") for s in sigmas)
    )
    if not grid:
        raise ValueError("sigmas must not be empty")
    mads = []
    for sigma in grid:
        column = np.random.default_rng(rng_seed).lognormal(mu, sigma, n)
        mads.append(mean_absolute_deviation(column))
    return SpreadCurve(
        sigmas=grid,
        spans=tuple(percentile_span(sigma) for sigma in grid),
        mads=tuple(mads),
        population_mads=tuple(population_mad(mu, sigma) for sigma in grid),
        expected_mads=tuple(expected_mad(mu, sigma, n) for sigma in grid),
    )


def example_payload() -> BenfordSummary:
    """Return the figure's curve, the law, where the thresholds are crossed and the table."""
    return BenfordSummary(
        benford=tuple(float(share) for share in benford_shares()),
        curve=spread_curve(),
        close_conformity_span=conformity_span(CLOSE_CONFORMITY),
        acceptable_conformity_span=conformity_span(ACCEPTABLE_CONFORMITY),
        article_rows=tuple(spread_row(sigma) for sigma in ARTICLE_SIGMAS),
    )
