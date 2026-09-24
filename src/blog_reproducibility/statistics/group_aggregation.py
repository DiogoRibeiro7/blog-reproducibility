"""Correlations between group averages, for the article on the ecological fallacy.

Each of two variables measured on people inside groups is a group part plus a
person part, ``x = sqrt(lambda_x) g_x + sqrt(1 - lambda_x) w_x`` and likewise for
``y``, with the group parts ``(g_x, g_y)`` standard bivariate normal with
correlation ``rho_B`` across groups and the person parts ``(w_x, w_y)`` with
correlation ``rho_W`` across people. The intraclass correlation ``lambda`` is the
share of a variable's unit variance that is group. The individual correlation is

    rho_ind = rho_B sqrt(lambda_x lambda_y) + rho_W sqrt((1 - lambda_x)(1 - lambda_y)),

whatever the group size, because grouping does not change people. Averaging
``m`` people keeps the group part and divides the person part's variance by
``m``, so the correlation between group means is

    rho_means = [rho_B sqrt(lambda_x lambda_y) + rho_W sqrt((1 - lambda_x)(1 - lambda_y)) / m]
                / sqrt((lambda_x + (1 - lambda_x) / m) (lambda_y + (1 - lambda_y) / m)),

which is the individual correlation at ``m = 1`` and tends to ``rho_B`` as the
groups grow. Both variables have unit variance, so the individual slope of ``y``
on ``x`` equals the individual correlation; the slope between group means is the
covariance of the means over the variance of the ``x`` means.

The figure draws 600 groups whose group parts correlate at 0.9 and whose people
do not correlate at all, at intraclass correlations of 5, 10 and 25 percent and
22 group sizes spaced geometrically from 2 to 2,000. Every point starts a fresh
generator seeded at 31 and draws the 600 group pairs, then 600 times ``m`` person
pairs. The draws do not depend on the intraclass correlation, so each group size
is drawn once here and combined with each intraclass correlation in turn, which
gives the website loop's correlations exactly; the figure is reproduced draw for
draw.

The article's tables are simulated on 1,000 groups from generators seeded at 2,
4, 6, 8 and 10, which the figure does not use, and are not reproduced. Their
"predicted" columns are the closed forms above and are reproduced here, with
the closed forms for its other tables; the tests check the simulated columns and
the Fisher interval against them.
"""

from dataclasses import dataclass
from math import atanh, sqrt, tanh
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "BETWEEN_CORRELATION",
    "FISHER_CRITICAL_VALUE",
    "GROUPS",
    "GROUP_SIZE_POINTS",
    "INTRACLASS_CORRELATIONS",
    "LARGEST_GROUP",
    "LEVELS_GROUP_SIZE",
    "LEVELS_INTRACLASS_CORRELATION",
    "SEED",
    "SIGN_POPULATIONS",
    "SIZE_TABLE_GROUP_SIZES",
    "SIZE_TABLE_INTRACLASS_CORRELATION",
    "SLOPE_POPULATION",
    "SLOPE_TABLE_INTRACLASS_CORRELATIONS",
    "SMALLEST_GROUP",
    "WITHIN_CORRELATION",
    "AggregationCurve",
    "GroupAggregationSummary",
    "LevelsRow",
    "aggregation_curves",
    "combine_parts",
    "draw_parts",
    "example_payload",
    "fisher_interval",
    "group_mean_correlation",
    "group_size_for_correlation",
    "group_sizes",
    "individual_correlation",
    "levels_row",
    "population",
    "predicted_group_mean_correlation",
    "predicted_group_mean_slope",
    "predicted_individual_correlation",
]

# Each point of the figure draws from np.random.default_rng(SEED).
SEED: Final[int] = 31
GROUPS: Final[int] = 600
BETWEEN_CORRELATION: Final[float] = 0.90
WITHIN_CORRELATION: Final[float] = 0.0
INTRACLASS_CORRELATIONS: Final[tuple[float, ...]] = (0.05, 0.10, 0.25)
SMALLEST_GROUP: Final[int] = 2
LARGEST_GROUP: Final[int] = 2000
GROUP_SIZE_POINTS: Final[int] = 22
# The article's tables. Group sizes at a tenth group share, between 0.9 and within 0:
SIZE_TABLE_INTRACLASS_CORRELATION: Final[float] = 0.10
SIZE_TABLE_GROUP_SIZES: Final[tuple[int, ...]] = (5, 20, 100, 1000)
# Populations of 200 people per group with 15 percent group share, as (between, within):
LEVELS_INTRACLASS_CORRELATION: Final[float] = 0.15
LEVELS_GROUP_SIZE: Final[int] = 200
SIGN_POPULATIONS: Final[tuple[tuple[float, float], ...]] = (
    (0.80, -0.30),
    (-0.60, 0.40),
    (0.00, 0.50),
    (0.70, 0.70),
)
# Slopes at between 0.8 and within -0.2, for four group shares:
SLOPE_POPULATION: Final[tuple[float, float]] = (0.80, -0.20)
SLOPE_TABLE_INTRACLASS_CORRELATIONS: Final[tuple[float, ...]] = (0.05, 0.10, 0.25, 0.50)
# The article's Fisher interval uses 1.96 rather than the exact normal quantile.
FISHER_CRITICAL_VALUE: Final[float] = 1.96


@dataclass(frozen=True, slots=True)
class AggregationCurve:
    """Correlation between group means along the figure's group sizes, at one group share."""

    intraclass_correlation: float
    group_sizes: tuple[int, ...]
    measured: tuple[float, ...]
    predicted: tuple[float, ...]
    individual: float


@dataclass(frozen=True, slots=True)
class LevelsRow:
    """Closed forms for one population with equal group shares, at one group size."""

    between_correlation: float
    within_correlation: float
    intraclass_correlation: float
    group_size: int
    individual_correlation: float
    group_mean_correlation: float
    group_mean_slope: float


@dataclass(frozen=True, slots=True)
class GroupAggregationSummary:
    """The figure's curves and the closed forms behind the article's tables."""

    groups: int
    between_correlation: float
    within_correlation: float
    curves: tuple[AggregationCurve, ...]
    size_rows: tuple[LevelsRow, ...]
    sign_rows: tuple[LevelsRow, ...]
    slope_rows: tuple[LevelsRow, ...]


def _correlation(value: float, *, name: str) -> float:
    rho = real(value, name=name)
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"{name} must lie in [-1, 1]")
    return rho


def _share(value: float, *, name: str = "intraclass_correlation") -> float:
    return probability(value, name=name)


def _size(value: int) -> int:
    return count(value, name="group_size", minimum=1)


def group_sizes(
    smallest: int = SMALLEST_GROUP, largest: int = LARGEST_GROUP, points: int = GROUP_SIZE_POINTS
) -> tuple[int, ...]:
    """Distinct rounded group sizes spaced geometrically from ``smallest`` to ``largest``."""
    low = count(smallest, name="smallest", minimum=1)
    high = count(largest, name="largest", minimum=low)
    steps = count(points, name="points", minimum=1)
    sizes = np.unique(np.round(np.geomspace(low, high, steps)).astype(int))
    return tuple(int(size) for size in sizes)


def draw_parts(
    rng: np.random.Generator,
    groups: int = GROUPS,
    group_size: int = 1,
    *,
    between: float = BETWEEN_CORRELATION,
    within: float = WITHIN_CORRELATION,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw the group parts, shape ``(groups, 2)``, then the person parts, ``(groups, m, 2)``."""
    g = count(groups, name="groups", minimum=2)
    m = _size(group_size)
    rho_b = _correlation(between, name="between")
    rho_w = _correlation(within, name="within")
    group_parts = rng.multivariate_normal([0, 0], [[1, rho_b], [rho_b, 1]], g)
    person_parts = rng.multivariate_normal([0, 0], [[1, rho_w], [rho_w, 1]], (g, m))
    return group_parts, person_parts


def combine_parts(
    group_parts: NDArray[np.float64],
    person_parts: NDArray[np.float64],
    intraclass_x: float,
    intraclass_y: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Weight the parts into the two variables, one row per group and one column per person."""
    share_x = _share(intraclass_x, name="intraclass_x")
    share_y = _share(intraclass_y, name="intraclass_y")
    if group_parts.ndim != 2 or person_parts.ndim != 3:
        raise ValueError("group parts must be (groups, 2) and person parts (groups, m, 2)")
    if group_parts.shape[0] != person_parts.shape[0]:
        raise ValueError("group and person parts must cover the same groups")
    x = np.sqrt(share_x) * group_parts[:, None, 0] + np.sqrt(1 - share_x) * person_parts[:, :, 0]
    y = np.sqrt(share_y) * group_parts[:, None, 1] + np.sqrt(1 - share_y) * person_parts[:, :, 1]
    return x, y


def population(
    rng: np.random.Generator,
    groups: int,
    group_size: int,
    intraclass_x: float,
    intraclass_y: float,
    *,
    between: float = BETWEEN_CORRELATION,
    within: float = WITHIN_CORRELATION,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The article's population: two variables, one row per group and one column per person."""
    group_parts, person_parts = draw_parts(rng, groups, group_size, between=between, within=within)
    return combine_parts(group_parts, person_parts, intraclass_x, intraclass_y)


def _pair(x: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] < 2:
        raise ValueError("x and y must be matching (groups, people) arrays with two or more groups")
    return a, b


def individual_correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Correlation across every person, ignoring the groups."""
    a, b = _pair(x, y)
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def group_mean_correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Correlation between the group averages of the two variables."""
    a, b = _pair(x, y)
    return float(np.corrcoef(a.mean(axis=1), b.mean(axis=1))[0, 1])


def predicted_individual_correlation(
    intraclass_x: float, intraclass_y: float, between: float, within: float
) -> float:
    """Individual correlation implied by the components; also the individual slope."""
    share_x = _share(intraclass_x, name="intraclass_x")
    share_y = _share(intraclass_y, name="intraclass_y")
    rho_b = _correlation(between, name="between")
    rho_w = _correlation(within, name="within")
    return rho_b * sqrt(share_x * share_y) + rho_w * sqrt((1 - share_x) * (1 - share_y))


def _mean_moments(
    intraclass_x: float, intraclass_y: float, between: float, within: float, group_size: int
) -> tuple[float, float, float]:
    """Covariance of the group means and the variances of the ``x`` and ``y`` means."""
    share_x = _share(intraclass_x, name="intraclass_x")
    share_y = _share(intraclass_y, name="intraclass_y")
    rho_b = _correlation(between, name="between")
    rho_w = _correlation(within, name="within")
    m = _size(group_size)
    covariance = rho_b * sqrt(share_x * share_y) + rho_w * sqrt((1 - share_x) * (1 - share_y)) / m
    return covariance, share_x + (1 - share_x) / m, share_y + (1 - share_y) / m


def predicted_group_mean_correlation(
    intraclass_x: float, intraclass_y: float, between: float, within: float, group_size: int
) -> float:
    """Correlation of group means: the group parts stay, the person parts average away."""
    covariance, var_x, var_y = _mean_moments(
        intraclass_x, intraclass_y, between, within, group_size
    )
    return covariance / sqrt(var_x * var_y)


def predicted_group_mean_slope(
    intraclass_x: float, intraclass_y: float, between: float, within: float, group_size: int
) -> float:
    """Slope of the ``y`` means on the ``x`` means, which a two-level model calls between-group."""
    covariance, var_x, _ = _mean_moments(intraclass_x, intraclass_y, between, within, group_size)
    return covariance / var_x


def group_size_for_correlation(
    target: float, intraclass_correlation: float, between: float, within: float
) -> float:
    """Group size at which the correlation of means reaches ``target``, for equal group shares.

    Solving ``(rho_B lambda + rho_W (1 - lambda) / m) / (lambda + (1 - lambda) / m) = r``
    gives ``m = (1 - lambda)(r - rho_W) / (lambda (rho_B - r))``, which needs ``r``
    strictly between the individual correlation and ``rho_B``.
    """
    goal = _correlation(target, name="target")
    share = probability(intraclass_correlation, name="intraclass_correlation", inclusive=False)
    rho_b = _correlation(between, name="between")
    rho_w = _correlation(within, name="within")
    individual = rho_b * share + rho_w * (1 - share)
    if not min(individual, rho_b) < goal < max(individual, rho_b):
        raise ValueError("target must lie strictly between the individual correlation and between")
    return (1 - share) * (goal - rho_w) / (share * (rho_b - goal))


def fisher_interval(
    correlation: float, groups: int, *, critical_value: float = FISHER_CRITICAL_VALUE
) -> tuple[float, float]:
    """Interval for a correlation from ``groups`` pairs, on Fisher's ``z`` scale as the article."""
    r = real(correlation, name="correlation")
    if not -1.0 < r < 1.0:
        raise ValueError("correlation must lie in (-1, 1)")
    g = count(groups, name="groups", minimum=4)
    half_width = positive(critical_value, name="critical_value") / sqrt(g - 3)
    z = atanh(r)
    return tanh(z - half_width), tanh(z + half_width)


def levels_row(
    between: float, within: float, intraclass_correlation: float, group_size: int
) -> LevelsRow:
    """Individual and group-mean correlation and the group-mean slope, for equal group shares."""
    share = _share(intraclass_correlation)
    m = _size(group_size)
    return LevelsRow(
        between_correlation=_correlation(between, name="between"),
        within_correlation=_correlation(within, name="within"),
        intraclass_correlation=share,
        group_size=m,
        individual_correlation=predicted_individual_correlation(share, share, between, within),
        group_mean_correlation=predicted_group_mean_correlation(share, share, between, within, m),
        group_mean_slope=predicted_group_mean_slope(share, share, between, within, m),
    )


def aggregation_curves(
    seed: int = SEED,
    *,
    groups: int = GROUPS,
    intraclass_correlations: tuple[float, ...] = INTRACLASS_CORRELATIONS,
    sizes: tuple[int, ...] | None = None,
    between: float = BETWEEN_CORRELATION,
    within: float = WITHIN_CORRELATION,
) -> tuple[AggregationCurve, ...]:
    """Measured and predicted correlation of group means, a fresh generator at every group size.

    The website loops over intraclass correlations and then group sizes, seeding
    a generator at every point. The draws depend only on the group size, so each
    size is drawn once and combined with every intraclass correlation.
    """
    start = count(seed, name="seed")
    shares = tuple(_share(value) for value in intraclass_correlations)
    if not shares:
        raise ValueError("intraclass_correlations must not be empty")
    grid = group_sizes() if sizes is None else tuple(_size(size) for size in sizes)
    if not grid:
        raise ValueError("sizes must not be empty")

    measured = np.empty((len(shares), len(grid)))
    for j, size in enumerate(grid):
        parts = draw_parts(
            np.random.default_rng(start), groups, size, between=between, within=within
        )
        for i, share in enumerate(shares):
            measured[i, j] = group_mean_correlation(*combine_parts(*parts, share, share))
    return tuple(
        AggregationCurve(
            intraclass_correlation=share,
            group_sizes=grid,
            measured=tuple(float(value) for value in measured[i]),
            predicted=tuple(
                predicted_group_mean_correlation(share, share, between, within, size)
                for size in grid
            ),
            individual=predicted_individual_correlation(share, share, between, within),
        )
        for i, share in enumerate(shares)
    )


def example_payload() -> GroupAggregationSummary:
    """Return the figure's curves and the closed forms for the article's three tables."""
    slope_between, slope_within = SLOPE_POPULATION
    return GroupAggregationSummary(
        groups=GROUPS,
        between_correlation=BETWEEN_CORRELATION,
        within_correlation=WITHIN_CORRELATION,
        curves=aggregation_curves(),
        size_rows=tuple(
            levels_row(
                BETWEEN_CORRELATION, WITHIN_CORRELATION, SIZE_TABLE_INTRACLASS_CORRELATION, size
            )
            for size in SIZE_TABLE_GROUP_SIZES
        ),
        sign_rows=tuple(
            levels_row(rho_b, rho_w, LEVELS_INTRACLASS_CORRELATION, LEVELS_GROUP_SIZE)
            for rho_b, rho_w in SIGN_POPULATIONS
        ),
        slope_rows=tuple(
            levels_row(slope_between, slope_within, share, LEVELS_GROUP_SIZE)
            for share in SLOPE_TABLE_INTRACLASS_CORRELATIONS
        ),
    )
