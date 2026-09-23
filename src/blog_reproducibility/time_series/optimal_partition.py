"""Penalised optimal partitioning for the article on offline change-point detection.

Given a whole record, the segmentation minimises the total within-segment sum
of squares plus a penalty per change point. The dynamic programme

    F(b) = min over a < b of F(a) + cost(a, b) + penalty,   F(0) = -penalty,

is exact and runs in O(n^2). Cumulative sums make each segment cost O(1), and
the candidates for one end point are evaluated together as an array, which
keeps the published arithmetic and tie-breaking while avoiding a Python loop
over every start point.

The penalty is expressed as a multiple of ``sigma^2 log n``. The noise level
comes from the median absolute deviation of first differences, which the
shifts in mean barely touch. The article's examples use the same four-shift
series with independent noise (seed 0) and with AR(1) noise at 0.6 (seed 1).
"""

from dataclasses import dataclass
from math import log, sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count, non_negative, positive

__all__ = [
    "AUTOCORRELATED_MULTIPLES",
    "AUTOCORRELATION",
    "BOUNDS",
    "FIGURE_MULTIPLES",
    "INDEPENDENT_MULTIPLES",
    "SEGMENT_MEANS",
    "TRUE_CHANGE_POINTS",
    "PartitionRow",
    "PartitionSummary",
    "example_payload",
    "make_series",
    "optimal_partition",
    "partition_sweep",
    "robust_sigma",
    "score",
]

BOUNDS: Final[tuple[int, ...]] = (0, 120, 200, 350, 420, 600)
SEGMENT_MEANS: Final[tuple[float, ...]] = (0.0, 1.5, 0.0, -1.2, 1.0)
TRUE_CHANGE_POINTS: Final[tuple[int, ...]] = BOUNDS[1:-1]
AUTOCORRELATION: Final[float] = 0.6
INDEPENDENT_MULTIPLES: Final[tuple[float, ...]] = (0.5, 1, 2, 4, 8)
AUTOCORRELATED_MULTIPLES: Final[tuple[float, ...]] = (2, 4, 8, 16)
# The figure's penalty sweep covers both tables.
FIGURE_MULTIPLES: Final[tuple[float, ...]] = (0.5, 1, 2, 4, 8, 16)


@dataclass(frozen=True, slots=True)
class PartitionRow:
    """Detections at one penalty and their agreement with the true change points."""

    penalty_multiple: float
    change_points: tuple[int, ...]
    precision: float
    recall: float


@dataclass(frozen=True, slots=True)
class PartitionSummary:
    """Every number the article reports."""

    robust_sigma: float
    sample_sd: float
    independent: tuple[PartitionRow, ...]
    autocorrelated_robust_sigma: float
    autocorrelated: tuple[PartitionRow, ...]


def make_series(
    rng: np.random.Generator, *, phi: float = 0.0
) -> tuple[NDArray[np.float64], tuple[int, ...]]:
    """Four shifts in mean over 600 points, with unit-variance AR(1) noise.

    ``phi = 0`` gives independent noise. Otherwise each innovation is scaled by
    ``sqrt(1 - phi^2)`` so the marginal variance stays one.
    """
    if not -1 < phi < 1:
        raise ValueError("phi must lie in (-1, 1)")
    size = BOUNDS[-1]
    mean = np.zeros(size)
    for start, end, level in zip(BOUNDS[:-1], BOUNDS[1:], SEGMENT_MEANS, strict=True):
        mean[start:end] = level
    noise = rng.normal(size=size)
    if phi:
        scale = np.sqrt(1 - phi**2)
        for index in range(1, size):
            noise[index] = phi * noise[index - 1] + scale * noise[index]
    return mean + noise, TRUE_CHANGE_POINTS


def optimal_partition(series: ArrayLike, penalty: float) -> tuple[int, ...]:
    """Change points minimising within-segment squares plus ``penalty`` per change."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("series must be a non-empty one-dimensional finite sequence")
    price = non_negative(penalty, name="penalty")

    size = values.size
    sums = np.concatenate(([0.0], np.cumsum(values)))
    squares = np.concatenate(([0.0], np.cumsum(values**2)))
    best = np.full(size + 1, np.inf)
    best[0] = -price
    last = np.zeros(size + 1, dtype=np.intp)
    for end in range(1, size + 1):
        starts = np.arange(end)
        total = sums[end] - sums[:end]
        cost = (squares[end] - squares[:end]) - total * total / (end - starts)
        candidates = best[:end] + cost + price
        start = int(np.argmin(candidates))
        best[end] = candidates[start]
        last[end] = start

    change_points = []
    end = size
    while last[end] > 0:
        change_points.append(int(last[end]))
        end = int(last[end])
    return tuple(sorted(change_points))


def robust_sigma(series: ArrayLike) -> float:
    """Noise level from the median absolute deviation of first differences."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("series must hold at least two observations")
    differences = np.diff(values)
    deviation = np.median(np.abs(differences - np.median(differences)))
    return float(deviation / 0.6745 / sqrt(2))


def score(
    found: tuple[int, ...], true: tuple[int, ...], *, tolerance: int = 5
) -> tuple[float, float]:
    """Precision and recall, counting a detection within ``tolerance`` as a match."""
    window = count(tolerance, name="tolerance")
    if not true:
        raise ValueError("true change points must not be empty")
    hits = sum(any(abs(point - actual) <= window for actual in true) for point in found)
    precision = hits / len(found) if found else 1.0
    recall = sum(any(abs(point - actual) <= window for point in found) for actual in true)
    return precision, recall / len(true)


def partition_sweep(
    series: NDArray[np.float64], multiples: tuple[float, ...], sigma: float
) -> tuple[PartitionRow, ...]:
    """Partition at each penalty multiple of ``sigma^2 log n`` and score the result."""
    scale = log(series.size) * positive(sigma, name="sigma") ** 2
    rows = []
    for multiple in multiples:
        found = optimal_partition(series, multiple * scale)
        precision, recall = score(found, TRUE_CHANGE_POINTS)
        rows.append(PartitionRow(float(multiple), found, precision, recall))
    return tuple(rows)


def example_payload() -> PartitionSummary:
    """Return the detections behind the article's two tables."""
    independent, _ = make_series(np.random.default_rng(0))
    autocorrelated, _ = make_series(np.random.default_rng(1), phi=AUTOCORRELATION)
    sigma = robust_sigma(independent)
    sigma_ar = robust_sigma(autocorrelated)
    return PartitionSummary(
        robust_sigma=sigma,
        sample_sd=float(independent.std()),
        independent=partition_sweep(independent, INDEPENDENT_MULTIPLES, sigma),
        autocorrelated_robust_sigma=sigma_ar,
        autocorrelated=partition_sweep(autocorrelated, AUTOCORRELATED_MULTIPLES, sigma_ar),
    )
