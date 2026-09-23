"""Sequential test boundaries, for the article on optional stopping.

Checking an experiment after every day and stopping at the first p-value
below 0.05 is a different procedure from testing once, with a larger error
rate. With equal information between looks, the running z statistic at look
``k`` is ``S_k / sqrt(k)`` for a random walk ``S`` with unit-variance normal
steps, so the probability that it ever leaves boundaries ``b_1, ..., b_K`` can
be computed without simulation. Armitage, McPherson and Rowe's recursion
carries the density of ``S_k`` forward on a grid, deletes the part beyond
``b_k sqrt(k)``, and convolves with the step density between looks.

The recursion calibrates two group-sequential boundaries over seven looks by
bisection on a common scale: Pocock's flat boundary and O'Brien and Fleming's,
which scales as ``sqrt(K / k)``. A mixture sequential rule instead rejects when
the likelihood ratio averaged over a normal prior of scale ``tau`` on the effect,

    Lambda_n = sqrt(V_n / (V_n + tau^2)) exp(tau^2 delta_n^2 / (2 V_n (V_n + tau^2))),

passes ``1 / alpha``. It is a non-negative martingale with mean one under the
null, so Ville's inequality bounds its error rate by ``alpha`` under any
stopping rule. Solving ``Lambda_n = 1 / alpha`` for ``z = delta_n / sqrt(V_n)``
gives a boundary that moves with the sample size.

The figure is deterministic. It runs the recursion on a fixed grid of
half-width 16 with 50 bisection steps; the article's code sizes the grid from
the widest boundary (``max b sqrt(K) + 8``) and bisects 60 times. Both grids
are available here: the figure's boundaries use the first, and the article's
printed recursion results (the peeking and monitoring tables and the crossing
probabilities of its calibrated boundaries) use the second and are reproduced
exactly. The two calibrations agree to the two decimals the article prints.
The article's simulated columns come from seeded generators the figure does
not use, and are not reproduced.
"""

from dataclasses import dataclass
from math import isnan, sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.signal import fftconvolve

from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "ALPHA",
    "ARTICLE_ITERATIONS",
    "DAYS",
    "FIGURE_ITERATIONS",
    "FIGURE_SPAN",
    "LOOKS",
    "MONITORING",
    "PEEKING_LOOKS",
    "PER_DAY",
    "SIGMA",
    "TAU",
    "FigureBoundaries",
    "GroupSequentialBoundary",
    "MonitoringRow",
    "PeekingRow",
    "SequentialSummary",
    "calibrate",
    "crossing_probability",
    "example_payload",
    "figure_boundaries",
    "group_sequential_boundary",
    "mixture_boundary",
    "mixture_likelihood_ratio",
    "monitoring_rate",
    "naive_critical_value",
    "obrien_fleming_shape",
    "peeking_table",
    "pocock_shape",
]

ALPHA: Final[float] = 0.05
LOOKS: Final[int] = 7
SIGMA: Final[float] = 10.0
PER_DAY: Final[int] = 225
DAYS: Final[int] = 28
TAU: Final[float] = 0.5
GRID_STEP: Final[float] = 0.01
GRID_PAD: Final[float] = 8.0
FIGURE_SPAN: Final[float] = 16.0
FIGURE_ITERATIONS: Final[int] = 50
ARTICLE_ITERATIONS: Final[int] = 60
BISECTION_START: Final[tuple[float, float]] = (1.0, 8.0)
PEEKING_LOOKS: Final[tuple[int, ...]] = (1, 2, 3, 4, 7, 14, 28)
MONITORING: Final[tuple[tuple[str, int, float], ...]] = (
    ("Daily", 1, 0.01),
    ("Every 6 hours", 4, 0.01),
    ("Hourly", 24, 0.02),
    ("Every 10 minutes", 144, 0.05),
)


@dataclass(frozen=True, slots=True)
class GroupSequentialBoundary:
    """Critical z at each look, its nominal two-sided level, and the overall crossing chance."""

    critical_values: tuple[float, ...]
    nominal_levels: tuple[float, ...]
    crossing_probability: float


@dataclass(frozen=True, slots=True)
class FigureBoundaries:
    """The figure's four rules: seven-look boundaries and the daily mixture boundary."""

    naive: float
    pocock: GroupSequentialBoundary
    obrien_fleming: GroupSequentialBoundary
    mixture: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PeekingRow:
    """Chance that a 0.05 test applied at equally spaced looks ever rejects under the null."""

    looks: int
    crossing_probability: float


@dataclass(frozen=True, slots=True)
class MonitoringRow:
    """Chance of a false positive when a 28-day test is checked at a finer cadence."""

    cadence: str
    looks: int
    crossing_probability: float


@dataclass(frozen=True, slots=True)
class SequentialSummary:
    """The figure's boundaries and the article's recursion results on its own grid."""

    figure: FigureBoundaries
    article_pocock: GroupSequentialBoundary
    article_obrien_fleming: GroupSequentialBoundary
    peeking: tuple[PeekingRow, ...]
    monitoring: tuple[MonitoringRow, ...]


def naive_critical_value(alpha: float = ALPHA) -> float:
    """Two-sided normal critical value for a single test at level ``alpha``."""
    return float(stats.norm.ppf(1 - probability(alpha, name="alpha", inclusive=False) / 2))


def crossing_probability(
    bounds: ArrayLike,
    *,
    step: float = GRID_STEP,
    span: float | None = None,
    pad: float = GRID_PAD,
) -> float:
    """Chance that a unit-step random walk's z statistic ever reaches ``bounds``, look by look.

    ``bounds`` are critical z values, one per look, and may be infinite for looks
    that cannot stop. The density lives on a grid of spacing ``step`` over
    ``[-span, span]``; without ``span`` the grid is sized as the article's code
    sizes it, from the widest finite boundary in sum units plus ``pad``.
    """
    values = [float(value) for value in np.asarray(bounds, dtype=np.float64).ravel()]
    if not values or any(isnan(value) or value < 0 for value in values):
        raise ValueError("bounds must be a non-empty sequence of non-negative critical values")
    width = positive(step, name="step")
    if span is None:
        finite = [value for value in values if np.isfinite(value)]
        if not finite:
            raise ValueError("at least one bound must be finite to size the grid")
        half = max(finite) * sqrt(len(values)) + positive(pad, name="pad")
    else:
        half = positive(span, name="span")

    x = np.arange(-half, half + width, width)
    kernel = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi) * width
    density = kernel.copy()
    total = 0.0
    for look, bound in enumerate(values, start=1):
        inside = np.abs(x) < bound * np.sqrt(look)
        total += float(density[~inside].sum())
        density = density * inside
        if look < len(values):
            density = np.asarray(fftconvolve(density, kernel, mode="same"), dtype=np.float64)
    return total


def pocock_shape(looks: int = LOOKS) -> NDArray[np.float64]:
    """Pocock's boundary shape: the same critical value at every look."""
    return np.ones(count(looks, name="looks", minimum=1))


def obrien_fleming_shape(looks: int = LOOKS) -> NDArray[np.float64]:
    """O'Brien-Fleming's shape, ``sqrt(K / k)``: severe early, near the unadjusted value last."""
    total = count(looks, name="looks", minimum=1)
    return np.sqrt(total / np.arange(1, total + 1))


def calibrate(
    shape: ArrayLike,
    *,
    target: float = ALPHA,
    iterations: int = FIGURE_ITERATIONS,
    span: float | None = FIGURE_SPAN,
) -> NDArray[np.float64]:
    """Scale a boundary shape by bisection until its crossing probability is ``target``."""
    form = np.asarray(shape, dtype=np.float64)
    goal = probability(target, name="target", inclusive=False)
    lo, hi = BISECTION_START
    for _ in range(count(iterations, name="iterations", minimum=1)):
        mid = (lo + hi) / 2
        if crossing_probability(mid * form, span=span) > goal:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 * form


def group_sequential_boundary(
    shape: ArrayLike, *, iterations: int, span: float | None
) -> GroupSequentialBoundary:
    """Calibrate a shape and report its critical values, nominal levels and crossing chance."""
    bounds = calibrate(shape, iterations=iterations, span=span)
    return GroupSequentialBoundary(
        critical_values=tuple(float(bound) for bound in bounds),
        nominal_levels=tuple(float(2 * stats.norm.sf(bound)) for bound in bounds),
        crossing_probability=crossing_probability(bounds, span=span),
    )


def mixture_likelihood_ratio(
    difference: ArrayLike, standard_error: ArrayLike, tau: float = TAU
) -> NDArray[np.float64]:
    """Likelihood ratio of the running estimate against the null, mixed over ``N(0, tau^2)``."""
    scale = positive(tau, name="tau")
    diff = np.asarray(difference, dtype=np.float64)
    v = np.asarray(standard_error, dtype=np.float64) ** 2
    if np.any(v <= 0):
        raise ValueError("standard_error must be positive")
    return np.asarray(
        np.sqrt(v / (v + scale**2)) * np.exp(scale**2 * diff**2 / (2 * v * (v + scale**2))),
        dtype=np.float64,
    )


def mixture_boundary(
    days: int = DAYS,
    *,
    per_day: int = PER_DAY,
    sigma: float = SIGMA,
    tau: float = TAU,
    alpha: float = ALPHA,
) -> NDArray[np.float64]:
    """Critical z on each day at which the mixture likelihood ratio reaches ``1 / alpha``."""
    n = count(per_day, name="per_day", minimum=1) * np.arange(1, count(days, name="days") + 1)
    v = 2 * positive(sigma, name="sigma") ** 2 / n
    scale = positive(tau, name="tau")
    level = probability(alpha, name="alpha", inclusive=False)
    return np.asarray(
        np.sqrt(
            2 * (v + scale**2) / scale**2 * (np.log(1 / level) + 0.5 * np.log((v + scale**2) / v))
        ),
        dtype=np.float64,
    )


def figure_boundaries(looks: int = LOOKS) -> FigureBoundaries:
    """The figure's naive, Pocock, O'Brien-Fleming and mixture boundaries."""
    return FigureBoundaries(
        naive=naive_critical_value(),
        pocock=group_sequential_boundary(
            pocock_shape(looks), iterations=FIGURE_ITERATIONS, span=FIGURE_SPAN
        ),
        obrien_fleming=group_sequential_boundary(
            obrien_fleming_shape(looks), iterations=FIGURE_ITERATIONS, span=FIGURE_SPAN
        ),
        mixture=tuple(float(value) for value in mixture_boundary()),
    )


def peeking_table(looks: tuple[int, ...] = PEEKING_LOOKS) -> tuple[PeekingRow, ...]:
    """The article's recursion column: a 0.05 test repeated at each of ``k`` looks."""
    z = naive_critical_value()
    return tuple(
        PeekingRow(k, crossing_probability(np.full(count(k, name="looks", minimum=1), z)))
        for k in looks
    )


def monitoring_rate(subdivisions: int, *, step: float, days: int = DAYS) -> float:
    """False positive rate of a 0.05 test checked ``subdivisions`` times a day for ``days``.

    Checks start at the end of the first day, so earlier looks cannot stop.
    """
    per_day = count(subdivisions, name="subdivisions", minimum=1)
    looks = count(days, name="days", minimum=1) * per_day
    bounds = np.where(np.arange(1, looks + 1) >= per_day, naive_critical_value(), np.inf)
    return crossing_probability(bounds, step=step)


def example_payload() -> SequentialSummary:
    """Return the figure's boundaries and the article's recursion results."""
    return SequentialSummary(
        figure=figure_boundaries(),
        article_pocock=group_sequential_boundary(
            pocock_shape(), iterations=ARTICLE_ITERATIONS, span=None
        ),
        article_obrien_fleming=group_sequential_boundary(
            obrien_fleming_shape(), iterations=ARTICLE_ITERATIONS, span=None
        ),
        peeking=peeking_table(),
        monitoring=tuple(
            MonitoringRow(cadence, DAYS * subdivisions, monitoring_rate(subdivisions, step=step))
            for cadence, subdivisions, step in MONITORING
        ),
    )
