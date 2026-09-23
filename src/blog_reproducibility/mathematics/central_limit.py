"""Sampling distributions of the mean, for the article on the central limit theorem.

The population is exponential with mean one, as skewed as a common distribution
gets. The mean of ``n`` draws is Gamma distributed with shape ``n`` and scale
``1 / n``: mean 1, variance ``1 / n``, and skewness ``2 / sqrt(n)``. The skewness
falls with ``sqrt(n)``, so the shape becomes symmetric well before the data does,
and the normal reference ``N(1, 1 / n)`` fits at ``n = 30``.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count

__all__ = [
    "REPLICATIONS",
    "SAMPLE_SIZES",
    "SEED",
    "MeanDistribution",
    "example_payload",
    "normal_reference",
    "sample_means",
    "summarise",
]

SEED: Final[int] = 20260816
SAMPLE_SIZES: Final[tuple[int, ...]] = (2, 5, 30)
REPLICATIONS: Final[int] = 40_000


@dataclass(frozen=True, slots=True)
class MeanDistribution:
    """Simulated and exact moments of the sample mean at one sample size."""

    sample_size: int
    mean: float
    variance: float
    skewness: float
    exact_variance: float
    exact_skewness: float


def sample_means(
    sample_sizes: tuple[int, ...] = SAMPLE_SIZES,
    *,
    replications: int = REPLICATIONS,
    seed: int = SEED,
) -> tuple[NDArray[np.float64], ...]:
    """Means of exponential samples, one array of ``replications`` per sample size."""
    rng = np.random.default_rng(count(seed, name="seed"))
    draws = count(replications, name="replications", minimum=2)
    return tuple(
        rng.exponential(1.0, size=(draws, count(size, name="sample_size", minimum=1))).mean(axis=1)
        for size in sample_sizes
    )


def normal_reference(grid: ArrayLike, sample_size: int) -> NDArray[np.float64]:
    """Density of ``N(1, 1 / n)``, the limit the sample mean approaches."""
    size = count(sample_size, name="sample_size", minimum=1)
    points = np.asarray(grid, dtype=np.float64)
    return np.asarray(np.exp(-((points - 1) ** 2) * size / 2) / sqrt(2 * np.pi / size))


def summarise(means: NDArray[np.float64], sample_size: int) -> MeanDistribution:
    """Moments of simulated sample means next to their exact Gamma values."""
    centred = means - means.mean()
    variance = float(np.mean(centred**2))
    return MeanDistribution(
        sample_size=sample_size,
        mean=float(means.mean()),
        variance=variance,
        skewness=float(np.mean(centred**3) / variance**1.5),
        exact_variance=1 / sample_size,
        exact_skewness=2 / sqrt(sample_size),
    )


def example_payload() -> tuple[MeanDistribution, ...]:
    """Return the moments behind the article's three panels."""
    return tuple(
        summarise(means, size) for means, size in zip(sample_means(), SAMPLE_SIZES, strict=True)
    )
