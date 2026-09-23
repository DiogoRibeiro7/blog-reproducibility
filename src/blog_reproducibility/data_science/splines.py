"""Smoothing spline against global polynomials, for the article on splines.

Ninety noisy observations of ``sin(x) + 0.15 x`` on [0, 10] are fitted three
ways: a straight line, a degree-10 polynomial, and a cubic smoothing spline. The
spline is piecewise cubic, so each piece bends to local data; the polynomial is
one global function, so fitting the interior forces oscillation at the edges.

The smoothing factor bounds the spline's residual sum of squares at 0.09 per
point, a little above the noise variance of 0.28^2, so the spline follows the
signal without chasing the noise.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import UnivariateSpline

from blog_reproducibility.common.validation import count

__all__ = [
    "POLYNOMIAL_DEGREE",
    "SEED",
    "SMOOTHING_PER_POINT",
    "CurveFits",
    "FitErrors",
    "example_payload",
    "fit_curves",
    "true_curve",
]

SEED: Final[int] = 20260816
POINTS: Final[int] = 90
NOISE_SD: Final[float] = 0.28
POLYNOMIAL_DEGREE: Final[int] = 10
SMOOTHING_PER_POINT: Final[float] = 0.09
# The outer tenth of the range at each end, where a global polynomial wobbles.
EDGE_WIDTH: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class CurveFits:
    """Observations and each fit evaluated at the observation points."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    truth: NDArray[np.float64]
    line: NDArray[np.float64]
    polynomial: NDArray[np.float64]
    spline: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FitErrors:
    """Root mean squared error of one fit against the true curve."""

    fit: str
    overall: float
    edges: float
    interior: float


def true_curve(x: ArrayLike) -> NDArray[np.float64]:
    """The signal the observations are drawn around."""
    points = np.asarray(x, dtype=np.float64)
    return np.asarray(np.sin(points) + 0.15 * points)


def fit_curves(*, seed: int = SEED) -> CurveFits:
    """Draw the observations and fit the line, polynomial, and spline."""
    rng = np.random.default_rng(count(seed, name="seed"))
    x = np.linspace(0, 10, POINTS)
    truth = true_curve(x)
    y = truth + rng.normal(0, NOISE_SD, x.size)
    spline = UnivariateSpline(x, y, s=x.size * SMOOTHING_PER_POINT)
    return CurveFits(
        x=x,
        y=y,
        truth=truth,
        line=np.asarray(np.polyval(np.polyfit(x, y, 1), x), dtype=np.float64),
        polynomial=np.asarray(np.polyval(np.polyfit(x, y, POLYNOMIAL_DEGREE), x), dtype=np.float64),
        spline=np.asarray(spline(x)),
    )


def example_payload() -> tuple[FitErrors, ...]:
    """Error of each fit against the truth, overall and split into edges and interior."""
    fits = fit_curves()
    edges = (fits.x < fits.x.min() + EDGE_WIDTH) | (fits.x > fits.x.max() - EDGE_WIDTH)

    def rmse(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
        return float(np.sqrt(np.mean((values[mask] - fits.truth[mask]) ** 2)))

    everywhere = np.ones_like(edges)
    return tuple(
        FitErrors(name, rmse(values, everywhere), rmse(values, edges), rmse(values, ~edges))
        for name, values in (
            ("line", fits.line),
            ("polynomial", fits.polynomial),
            ("spline", fits.spline),
        )
    )
