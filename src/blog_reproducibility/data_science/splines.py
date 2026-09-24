"""Smoothing spline against global polynomials, for the article on splines.

Ninety noisy observations on [0, 10] of a sharp Runge-type peak on a gentle
trend, ``2.5 / (1 + ((x - 5) / 0.5)^2) + 0.15 x``, are fitted three ways: a
straight line, a degree-10 polynomial, and a cubic smoothing spline. The spline
is piecewise cubic and places its knots where the curve bends, so it follows the
peak and stays flat elsewhere. The polynomial is one global function: it cannot
bend sharply at the peak without bending everywhere, so it falls short of the
peak and oscillates across the flat stretches, most visibly at the edges.

The smoothing factor bounds the spline's residual sum of squares at 0.025 per
point, a little above the noise variance of 0.15^2, so the spline follows the
signal without chasing the noise. With this design the spline is closer to the
true curve than the polynomial, both overall and in the outer tenth of the range
at each end, for 998 of the first 1,000 seeds.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import UnivariateSpline

from blog_reproducibility.common.validation import count

__all__ = [
    "EDGE_WIDTH",
    "NOISE_SD",
    "POLYNOMIAL_DEGREE",
    "SEED",
    "SMOOTHING_PER_POINT",
    "CurveFits",
    "FitErrors",
    "example_payload",
    "fit_curves",
    "fit_errors",
    "true_curve",
]

SEED: Final[int] = 20260816
POINTS: Final[int] = 90
NOISE_SD: Final[float] = 0.15
POLYNOMIAL_DEGREE: Final[int] = 10
SMOOTHING_PER_POINT: Final[float] = 0.025
PEAK_CENTRE: Final[float] = 5.0
PEAK_HEIGHT: Final[float] = 2.5
# Half-width at half height: the peak falls to half its height 0.5 either side.
PEAK_HALF_WIDTH: Final[float] = 0.5
TREND_SLOPE: Final[float] = 0.15
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
    """Return the signal the observations are drawn around: a sharp peak on a trend."""
    points = np.asarray(x, dtype=np.float64)
    peak = PEAK_HEIGHT / (1 + ((points - PEAK_CENTRE) / PEAK_HALF_WIDTH) ** 2)
    return np.asarray(peak + TREND_SLOPE * points)


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


def fit_errors(fits: CurveFits) -> tuple[FitErrors, ...]:
    """Error of each fit against the truth, overall and split into edges and interior."""
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


def example_payload() -> tuple[FitErrors, ...]:
    """Error of each fit in the article's figure, overall, at the edges, and inside."""
    return fit_errors(fit_curves())
