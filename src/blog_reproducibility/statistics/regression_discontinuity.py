"""Local linear estimates at a cutoff, for the article on regression discontinuity designs.

Units have a running variable ``x`` uniform on ``[-50, 50]`` and are treated when
``x >= 0``. The outcome is

    y = 10 + 0.08 x + k x^2 + tau d + e,    k = 0.002 above the cutoff, -0.001 below,

with ``tau = 2`` and normal noise of standard deviation 4. The local linear
estimate fits a line to the outcome on each side within a bandwidth ``h`` and
takes the difference of the intercepts at the cutoff.

A line fitted to ``k x^2`` over a uniform window ``[0, h]`` has intercept
``-k h^2 / 6``, and likewise over ``[-h, 0]``, so the estimate converges to
``tau - (k_above - k_below) h^2 / 6 = tau - 0.0005 h^2``: unbiased near the
cutoff, off by 1.25 when the window takes every unit. The intercept of a line
fitted to ``m`` uniform points has variance ``4 sigma^2 / m``, and each side
holds ``m = N h / 100`` units, so the estimate has standard deviation
``sqrt(8 sigma^2 / m) = 1.6 / sqrt(h)`` for the article's 5,000 units. The
root mean squared error ``sqrt(0.0005^2 h^4 + 2.56 / h)`` is smallest at
``h = (2.56 / (4 x 0.0005^2))^(1/5)``, about 19. A global polynomial of degree
``p`` on each side has endpoint variance ``(p + 1)^2 sigma^2 / m``; it is
unbiased from degree two on, because the trend is exactly quadratic.

When a share ``s`` of the units within five of the cutoff below it move to a
point uniform on ``[0, 3)`` and gain three points of outcome, the density of the
running variable just above over just below is ``(1 + s) / (1 - s)``, and the
estimate's limit is the difference of two population least-squares lines,
computed here from exact polynomial moments.

The figure runs 600 samples of 5,000 units at each of ten bandwidths from 2 to
50, from one generator seeded at 0, and is reproduced draw for draw. The
article's tables run 2,000 samples per row (500 for manipulation and for the
sample sizes) from a generator seeded at 0 in a different order (naive
comparison first, other bandwidths, polynomials, placebos, manipulation), so
they are not reproduced; the tests check them against the closed forms above
instead.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "BANDWIDTHS",
    "CURVATURE_ABOVE",
    "CURVATURE_BELOW",
    "INTERCEPT",
    "LANDING_WIDTH",
    "MANIPULATION_SHARES",
    "MANIPULATION_WINDOW",
    "NOISE_SD",
    "PLACEBO_BANDWIDTH",
    "PLACEBO_CUTOFFS",
    "POLYNOMIAL_DEGREES",
    "PUSHER_GAIN",
    "RANGE",
    "REPLICATIONS",
    "SAMPLE_SIZES",
    "SEED",
    "SLOPE",
    "TRUE_EFFECT",
    "UNITS",
    "BandwidthRow",
    "ManipulationRow",
    "PlaceboRow",
    "PolynomialRow",
    "RegressionDiscontinuitySummary",
    "SampleSizeRow",
    "bandwidth_row",
    "density_ratio",
    "draw_sample",
    "example_payload",
    "local_linear",
    "local_linear_bias",
    "local_linear_limit",
    "local_linear_sd",
    "naive_difference",
    "naive_difference_limit",
    "optimal_bandwidth",
    "polynomial_sd",
    "units_in_window",
]

SEED: Final[int] = 0
TRUE_EFFECT: Final[float] = 2.0
UNITS: Final[int] = 5000
REPLICATIONS: Final[int] = 600
BANDWIDTHS: Final[tuple[int, ...]] = (2, 3, 5, 7, 10, 15, 20, 30, 40, 50)
RANGE: Final[float] = 50.0
INTERCEPT: Final[float] = 10.0
SLOPE: Final[float] = 0.08
CURVATURE_ABOVE: Final[float] = 0.002
CURVATURE_BELOW: Final[float] = -0.001
NOISE_SD: Final[float] = 4.0
# The article's manipulation: near-miss units within 5 below move to [0, 3) and gain 3.
MANIPULATION_WINDOW: Final[float] = 5.0
LANDING_WIDTH: Final[float] = 3.0
PUSHER_GAIN: Final[float] = 3.0
# The article's other tables, checked against the closed forms.
POLYNOMIAL_DEGREES: Final[tuple[int, ...]] = (1, 2, 4, 6)
PLACEBO_CUTOFFS: Final[tuple[float, ...]] = (-25.0, -10.0, 10.0, 25.0)
PLACEBO_BANDWIDTH: Final[float] = 10.0
MANIPULATION_SHARES: Final[tuple[float, ...]] = (0.0, 0.3, 0.6)
SAMPLE_SIZES: Final[tuple[int, ...]] = (1000, 5000, 20000)


@dataclass(frozen=True, slots=True)
class BandwidthRow:
    """The figure's error decomposition at one bandwidth, simulated and in closed form."""

    bandwidth: float
    replications: int
    mean_estimate: float
    absolute_bias: float
    standard_deviation: float
    root_mean_squared_error: float
    expected_bias: float
    expected_standard_deviation: float
    expected_root_mean_squared_error: float
    units_in_window: float


@dataclass(frozen=True, slots=True)
class PolynomialRow:
    """Bias and spread of a global polynomial of one degree on each side."""

    degree: int
    expected_bias: float
    expected_standard_deviation: float


@dataclass(frozen=True, slots=True)
class PlaceboRow:
    """Limit and spread of the local linear estimate at a cutoff where nothing changes."""

    cutoff: float
    expected_estimate: float
    expected_standard_deviation: float


@dataclass(frozen=True, slots=True)
class ManipulationRow:
    """The estimate's limit and the density ratio when near-miss units cross the cutoff."""

    share: float
    expected_estimate: float
    density_ratio: float


@dataclass(frozen=True, slots=True)
class SampleSizeRow:
    """Units inside a bandwidth of 10 and the spread of the estimate, by total units."""

    units: int
    units_in_window: float
    expected_standard_deviation: float


@dataclass(frozen=True, slots=True)
class RegressionDiscontinuitySummary:
    """The figure's bias, noise and error by bandwidth, and the article's closed forms."""

    effect: float
    units: int
    rows: tuple[BandwidthRow, ...]
    naive_difference: float
    optimal_bandwidth: float
    polynomials: tuple[PolynomialRow, ...]
    placebos: tuple[PlaceboRow, ...]
    manipulation: tuple[ManipulationRow, ...]
    sample_sizes: tuple[SampleSizeRow, ...]


def _bandwidth(bandwidth: float) -> float:
    h = positive(bandwidth, name="bandwidth")
    if h > 2 * RANGE:
        raise ValueError("bandwidth cannot exceed the range of the running variable")
    return h


def draw_sample(
    rng: np.random.Generator, units: int = UNITS, *, manipulation: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Draw running variable, treatment and outcome in the article's order.

    Without manipulation this is the figure's draw: the running variable, then
    the noise. With it, the article's: a uniform per unit decides who of the
    near-miss units moves, and another where it lands, before the noise.
    """
    n = count(units, name="units", minimum=1)
    share = probability(manipulation, name="manipulation")
    x = rng.uniform(-RANGE, RANGE, n)
    push = np.zeros(n, dtype=bool)
    if share > 0:
        near = (x < 0) & (x > -MANIPULATION_WINDOW)
        push = near & (rng.random(n) < share)
        x = np.where(push, rng.uniform(0, LANDING_WIDTH, n), x)
    d = (x >= 0).astype(float)
    curve = np.where(x >= 0, CURVATURE_ABOVE, CURVATURE_BELOW) * x**2
    y: NDArray[np.float64] = (
        INTERCEPT + SLOPE * x + curve + TRUE_EFFECT * d + rng.normal(0, NOISE_SD, n)
    )
    if share > 0:
        y = y + PUSHER_GAIN * push
    return x, d, y


def _local_linear(x: NDArray[np.float64], y: NDArray[np.float64], h: float, cutoff: float) -> float:
    estimates = []
    for side in (x >= cutoff, x < cutoff):
        mask = side & (np.abs(x - cutoff) <= h)
        if np.count_nonzero(mask) < 2:
            raise ValueError("each side of the cutoff needs two units within the bandwidth")
        design = np.column_stack([np.ones(mask.sum()), x[mask] - cutoff])
        coefficients, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
        estimates.append(coefficients[0])
    return float(estimates[0] - estimates[1])


def _sample(running: ArrayLike, outcome: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    x = np.asarray(running, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("running and outcome must be 1-D arrays of one length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("running and outcome must be finite")
    return x, y


def local_linear(
    running: ArrayLike, outcome: ArrayLike, bandwidth: float, *, cutoff: float = 0.0
) -> float:
    """Difference of the intercepts of lines fitted within ``bandwidth`` on each side."""
    x, y = _sample(running, outcome)
    return _local_linear(x, y, positive(bandwidth, name="bandwidth"), real(cutoff, name="cutoff"))


def naive_difference(running: ArrayLike, outcome: ArrayLike, *, cutoff: float = 0.0) -> float:
    """Mean outcome of every unit above the cutoff minus that of every unit below."""
    x, y = _sample(running, outcome)
    above = x >= real(cutoff, name="cutoff")
    if not (np.any(above) and np.any(~above)):
        raise ValueError("both sides of the cutoff need units")
    return float(y[above].mean() - y[~above].mean())


def bandwidth_row(
    rng: np.random.Generator,
    bandwidth: float,
    *,
    replications: int = REPLICATIONS,
    units: int = UNITS,
) -> BandwidthRow:
    """Estimate the effect on ``replications`` fresh samples from ``rng`` at one bandwidth."""
    h = _bandwidth(bandwidth)
    reps = count(replications, name="replications", minimum=1)
    n = count(units, name="units", minimum=2)
    estimates = np.array([_local_linear(*draw_sample(rng, n)[::2], h, 0.0) for _ in range(reps)])
    bias = local_linear_bias(h)
    sd = local_linear_sd(h, n)
    return BandwidthRow(
        bandwidth=h,
        replications=reps,
        mean_estimate=float(estimates.mean()),
        absolute_bias=float(abs(estimates.mean() - TRUE_EFFECT)),
        standard_deviation=float(estimates.std()),
        root_mean_squared_error=float(np.sqrt(np.mean((estimates - TRUE_EFFECT) ** 2))),
        expected_bias=bias,
        expected_standard_deviation=sd,
        expected_root_mean_squared_error=sqrt(bias**2 + sd**2),
        units_in_window=units_in_window(h, n),
    )


def local_linear_bias(bandwidth: float) -> float:
    """Limit of the estimate minus the effect: ``-(k_above - k_below) h^2 / 6``."""
    h = _bandwidth(bandwidth)
    if h > RANGE:
        raise ValueError("the closed form needs the window inside the running variable's range")
    return -(CURVATURE_ABOVE - CURVATURE_BELOW) * h**2 / 6


def units_in_window(bandwidth: float, units: int = UNITS) -> float:
    """Expected units within ``bandwidth`` of the cutoff on either side."""
    h = _bandwidth(bandwidth)
    return count(units, name="units", minimum=1) * min(2 * h, 2 * RANGE) / (2 * RANGE)


def local_linear_sd(bandwidth: float, units: int = UNITS) -> float:
    """Large-sample spread: ``sqrt(2 x 4 sigma^2 / m)`` with ``m`` units on each side."""
    per_side = units_in_window(bandwidth, units) / 2
    return sqrt(8 * NOISE_SD**2 / per_side)


def optimal_bandwidth(units: int = UNITS) -> float:
    """Bandwidth minimising ``b^2 h^4 + V / h``: ``(V / (4 b^2))^(1/5)``."""
    b = (CURVATURE_ABOVE - CURVATURE_BELOW) / 6
    variance_times_h = local_linear_sd(1.0, units) ** 2
    return float((variance_times_h / (4 * b**2)) ** 0.2)


def polynomial_sd(degree: int, units: int = UNITS) -> float:
    """Spread of a global polynomial on each side, from the endpoint variance ``(p + 1)^2 / m``."""
    p = count(degree, name="degree", minimum=0)
    per_side = count(units, name="units", minimum=2) / 2
    return sqrt(2 * (p + 1) ** 2 * NOISE_SD**2 / per_side)


def naive_difference_limit() -> float:
    """Mean above minus mean below, over the whole uniform range: 8.5 for the article's trend."""
    # E[x] = +-R/2 and E[x^2] = R^2/3 on either half.
    half = RANGE / 2
    square = RANGE**2 / 3
    return TRUE_EFFECT + SLOPE * 2 * half + (CURVATURE_ABOVE - CURVATURE_BELOW) * square


def density_ratio(manipulation: float) -> float:
    """Expected units in ``[0, 5)`` over units in ``(-5, 0)`` when a share ``s`` moves up."""
    share = probability(manipulation, name="manipulation")
    if share >= 1.0:
        raise ValueError("manipulation must be below one for units to remain below the cutoff")
    return (1 + share) / (1 - share)


# A population piece: (low, high, density, (c0, c1, c2) of the mean outcome in x).
_Piece = tuple[float, float, float, tuple[float, float, float]]


def _pieces(low: float, high: float, share: float) -> list[_Piece]:
    """Density and mean outcome of the population on ``[low, high]`` under manipulation."""
    base = 1 / (2 * RANGE)
    below = (INTERCEPT, SLOPE, CURVATURE_BELOW)
    above = (INTERCEPT + TRUE_EFFECT, SLOPE, CURVATURE_ABOVE)
    pushed = (INTERCEPT + TRUE_EFFECT + PUSHER_GAIN, SLOPE, CURVATURE_ABOVE)
    moved = share * MANIPULATION_WINDOW * base / LANDING_WIDTH
    segments: list[_Piece] = [
        (-RANGE, -MANIPULATION_WINDOW, base, below),
        (-MANIPULATION_WINDOW, 0.0, base * (1 - share), below),
        (0.0, RANGE, base, above),
        (0.0, LANDING_WIDTH, moved, pushed),
    ]
    clipped = []
    for lo, hi, density, mean in segments:
        a, b = max(lo, low), min(hi, high)
        if b > a and density > 0:
            clipped.append((a, b, density, mean))
    return clipped


def _intercept(pieces: list[_Piece], at: float) -> float:
    """Population least-squares line through the pieces, evaluated at ``at``."""

    def moment(k: int) -> float:
        return sum(
            density * (hi ** (k + 1) - lo ** (k + 1)) / (k + 1) for lo, hi, density, _ in pieces
        )

    def cross(k: int) -> float:
        return sum(
            density * c * (hi ** (k + j + 1) - lo ** (k + j + 1)) / (k + j + 1)
            for lo, hi, density, mean in pieces
            for j, c in enumerate(mean)
        )

    s0, s1, s2 = moment(0), moment(1), moment(2)
    t0, t1 = cross(0), cross(1)
    determinant = s0 * s2 - s1**2
    if determinant <= 0:
        raise ValueError("the window must contain a spread of units on each side")
    slope = (s0 * t1 - s1 * t0) / determinant
    return (t0 - slope * s1) / s0 + slope * at


def local_linear_limit(
    bandwidth: float, *, cutoff: float = 0.0, manipulation: float = 0.0
) -> float:
    """Large-sample limit of the local linear estimate, from exact population moments."""
    h = _bandwidth(bandwidth)
    c = real(cutoff, name="cutoff")
    share = probability(manipulation, name="manipulation")
    right = _pieces(c, c + h, share)
    left = _pieces(c - h, c, share)
    return _intercept(right, c) - _intercept(left, c)


def example_payload() -> RegressionDiscontinuitySummary:
    """Return the figure's error decomposition (600 samples per bandwidth) and the closed forms."""
    rng = np.random.default_rng(SEED)
    return RegressionDiscontinuitySummary(
        effect=TRUE_EFFECT,
        units=UNITS,
        rows=tuple(bandwidth_row(rng, h) for h in BANDWIDTHS),
        naive_difference=naive_difference_limit(),
        optimal_bandwidth=optimal_bandwidth(),
        polynomials=tuple(
            PolynomialRow(
                degree=p,
                expected_bias=local_linear_bias(RANGE) if p == 1 else 0.0,
                expected_standard_deviation=polynomial_sd(p),
            )
            for p in POLYNOMIAL_DEGREES
        ),
        placebos=tuple(
            PlaceboRow(
                cutoff=c,
                expected_estimate=local_linear_limit(PLACEBO_BANDWIDTH, cutoff=c),
                expected_standard_deviation=local_linear_sd(PLACEBO_BANDWIDTH),
            )
            for c in PLACEBO_CUTOFFS
        ),
        manipulation=tuple(
            ManipulationRow(
                share=s,
                expected_estimate=local_linear_limit(PLACEBO_BANDWIDTH, manipulation=s),
                density_ratio=density_ratio(s),
            )
            for s in MANIPULATION_SHARES
        ),
        sample_sizes=tuple(
            SampleSizeRow(
                units=n,
                units_in_window=units_in_window(PLACEBO_BANDWIDTH, n),
                expected_standard_deviation=local_linear_sd(PLACEBO_BANDWIDTH, n),
            )
            for n in SAMPLE_SIZES
        ),
    )
