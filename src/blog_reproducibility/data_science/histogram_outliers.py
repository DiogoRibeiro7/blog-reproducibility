"""Two-dimensional histogram outlier flags, for the article on outlier algorithms.

Four thousand points come from a correlated bivariate normal, and 45 strays are
scattered uniformly over the square [-5, 5]^2. A 40-by-40 histogram over that
square counts points per cell; a point is flagged when its cell holds at most
one point, which means it is alone there.

The rule needs no distribution, only the grid, and it catches most of the strays
because a uniform scatter over a large area rarely shares a cell. Strays that
land inside the dense core are indistinguishable from it and are missed.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from blog_reproducibility.common.validation import count

__all__ = [
    "BINS",
    "EXTENT",
    "SEED",
    "FlaggedSample",
    "OutlierSummary",
    "example_payload",
    "flag_sparse_cells",
    "flagged_sample",
    "simulate_points",
]

SEED: Final[int] = 20260816
CORE_POINTS: Final[int] = 4000
STRAYS: Final[int] = 45
CORRELATION: Final[float] = 0.65
BINS: Final[int] = 40
EXTENT: Final[tuple[float, float]] = (-5.0, 5.0)
SPARSE_COUNT: Final[int] = 1


@dataclass(frozen=True, slots=True)
class FlaggedSample:
    """Simulated points, which of them are strays, and which are flagged."""

    points: NDArray[np.float64]
    is_stray: NDArray[np.bool_]
    flagged: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class OutlierSummary:
    """How the sparse-cell rule sorts strays from the core."""

    flagged: int
    strays_flagged: int
    strays: int
    core_flagged: int


def simulate_points(*, seed: int = SEED) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Correlated normal core followed by uniform strays."""
    rng = np.random.default_rng(count(seed, name="seed"))
    covariance = [[1.0, CORRELATION], [CORRELATION, 1.0]]
    core = rng.multivariate_normal([0.0, 0.0], covariance, CORE_POINTS)
    strays = rng.uniform(*EXTENT, size=(STRAYS, 2))
    is_stray = np.r_[np.zeros(CORE_POINTS, dtype=bool), np.ones(STRAYS, dtype=bool)]
    return np.vstack([core, strays]), is_stray


def flag_sparse_cells(
    points: NDArray[np.float64], *, bins: int = BINS, threshold: int = SPARSE_COUNT
) -> NDArray[np.bool_]:
    """Flag points whose histogram cell holds at most ``threshold`` points."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an (n, 2) array")
    cells = count(bins, name="bins", minimum=1)
    limits = [list(EXTENT), list(EXTENT)]
    counts, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=cells, range=limits)
    column = np.clip(np.digitize(points[:, 0], x_edges) - 1, 0, cells - 1)
    row = np.clip(np.digitize(points[:, 1], y_edges) - 1, 0, cells - 1)
    return np.asarray(counts[column, row] <= count(threshold, name="threshold"))


def flagged_sample(*, seed: int = SEED) -> FlaggedSample:
    """Simulate the points and apply the sparse-cell rule."""
    points, is_stray = simulate_points(seed=seed)
    return FlaggedSample(points, is_stray, flag_sparse_cells(points))


def example_payload() -> OutlierSummary:
    """Count what the rule flags among strays and core points."""
    sample = flagged_sample()
    return OutlierSummary(
        flagged=int(sample.flagged.sum()),
        strays_flagged=int((sample.flagged & sample.is_stray).sum()),
        strays=int(sample.is_stray.sum()),
        core_flagged=int((sample.flagged & ~sample.is_stray).sum()),
    )
