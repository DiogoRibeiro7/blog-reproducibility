"""Kernel density bandwidth, for the article on kernel density estimation.

A Gaussian kernel density estimate places a normal curve of standard deviation
``h`` on every observation and averages them. The bandwidth ``h`` decides what
the estimate shows: too small and every cluster of a few points becomes a mode,
too large and genuine modes merge.

The sample mixes 220 draws from N(-1.6, 0.5^2) with 280 from N(1.7, 0.7^2), a
distribution with two well-separated modes. The article's three bandwidths are
0.12, 0.45, and 1.40.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count, positive

__all__ = [
    "BANDWIDTHS",
    "GRID",
    "SEED",
    "BandwidthSummary",
    "count_modes",
    "example_payload",
    "gaussian_kde",
    "mixture_density",
    "mixture_sample",
]

SEED: Final[int] = 20260816
# Component mean, standard deviation, and number of draws.
COMPONENTS: Final[tuple[tuple[float, float, int], ...]] = ((-1.6, 0.5, 220), (1.7, 0.7, 280))
BANDWIDTHS: Final[tuple[float, ...]] = (0.12, 0.45, 1.40)
GRID: Final[NDArray[np.float64]] = np.linspace(-4.5, 5, 600)


@dataclass(frozen=True, slots=True)
class BandwidthSummary:
    """Modes found, and distance from the true density, at one bandwidth."""

    bandwidth: float
    modes: int
    integrated_squared_error: float


def mixture_sample(*, seed: int = SEED) -> NDArray[np.float64]:
    """Draw the two-component sample."""
    rng = np.random.default_rng(count(seed, name="seed"))
    return np.concatenate([rng.normal(mean, sd, size) for mean, sd, size in COMPONENTS])


def mixture_density(grid: ArrayLike) -> NDArray[np.float64]:
    """True density of the mixture the sample is drawn from."""
    points = np.asarray(grid, dtype=np.float64)
    total = sum(size for _, _, size in COMPONENTS)
    density = np.zeros_like(points)
    for mean, sd, size in COMPONENTS:
        density += (
            size / total * np.exp(-0.5 * ((points - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        )
    return density


def gaussian_kde(
    sample: ArrayLike, bandwidth: float, grid: ArrayLike = GRID
) -> NDArray[np.float64]:
    """Gaussian kernel density estimate evaluated on ``grid``."""
    data = np.asarray(sample, dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("sample must be a non-empty one-dimensional sequence")
    width = positive(bandwidth, name="bandwidth")
    scaled = (np.asarray(grid, dtype=np.float64)[:, None] - data[None, :]) / width
    return np.asarray(
        np.exp(-0.5 * scaled**2).sum(axis=1) / (data.size * width * np.sqrt(2 * np.pi))
    )


def count_modes(density: ArrayLike) -> int:
    """Number of strict local maxima of a density evaluated on a grid."""
    values = np.asarray(density, dtype=np.float64)
    interior = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    return int(interior.sum())


def example_payload() -> tuple[BandwidthSummary, ...]:
    """Modes and error against the true density at each of the article's bandwidths."""
    sample = mixture_sample()
    truth = mixture_density(GRID)
    step = float(GRID[1] - GRID[0])
    rows = []
    for bandwidth in BANDWIDTHS:
        estimate = gaussian_kde(sample, bandwidth)
        error = float(np.sum((estimate - truth) ** 2) * step)
        rows.append(BandwidthSummary(bandwidth, count_modes(estimate), error))
    return tuple(rows)
