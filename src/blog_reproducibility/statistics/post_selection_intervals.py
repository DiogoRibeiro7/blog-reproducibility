"""Exact intervals after selection, for the article on why they can be enormous.

One observation ``X ~ N(mu, 1)`` is reported only when it exceeds ``c = 2``.
Given that event its distribution function is

    F_mu(x) = P(X <= x | X > c) = 1 - sf(x - mu) / sf(c - mu),

computed from log survival functions so that it stays exact for means far
below the threshold. ``F_mu(X)`` is uniform given selection and ``F_mu(x)``
falls as ``mu`` rises, so the equal-tailed 95% interval is the pair of means at
which the observation sits at the 97.5% and 2.5% conditional quantiles. The
bracket for each root grows until it contains it, because the lower limit
recedes to -367 at ``x = 2.01``.

For a mean far below the threshold the excess over it is nearly exponential
with rate ``c - mu``, which gives the lower limit ``c - ln(2 / alpha) / d`` at a
distance ``d`` above the threshold: the width grows as 3.69 over the distance.
Given selection, ``d`` has a density that does not vanish at zero (the Gaussian
hazard at the threshold), so the width has a ``1 / w`` tail and no mean. Its
quantiles are exact all the same: the width falls as the observation rises, so
its ``q``-quantile is the width at the ``1 - q`` quantile of the truncated
normal.

Two procedures avoid the hard edge. Data splitting infers from an independent
half with variance 2, an ordinary interval of width ``2 * 1.96 * sqrt(2)``.
Randomised selection keeps ``X`` when ``X + omega > c`` with
``omega ~ N(0, gamma^2)``; ``X`` then has density proportional to
``phi(x - mu) Phi((x - c) / gamma)``, whose distribution function is summed on a
fixed grid of 14,001 points from -45 to 25. Its interval is tabulated at 113
observations from -4 to 10, and its width given selection is simulated: for
each of 17 means from 0 to 4, a generator seeded at 20260918 draws 400,000
observations and then 400,000 randomisation draws.

Everything the article prints comes from these computations: the table of
exact intervals and their tail approximations, the width quantiles and the
share wider than 20, the coverage of the ordinary interval, the selection
rates, and the randomised columns of the procedure table, which follow the
figure's seeded simulation. The article's coverage simulations (20,000
selected draws per row), the run whose mean width was 88, and the lasso stress
test use generators and designs it does not give, and are not reproduced.
"""

from dataclasses import dataclass
from math import log, sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import norm

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "ALPHA",
    "COVERAGE_MEANS",
    "DENSITY_MEANS",
    "DENSITY_UPPER",
    "DISTANCE_POINTS",
    "DISTANCE_RANGE",
    "DRAWS",
    "FIGURE_MEANS",
    "INTEGRATION_POINTS",
    "INTEGRATION_RANGE",
    "OBSERVED",
    "ORDINARY_Z",
    "RANDOMISATION_SD",
    "SEED",
    "TABLE_MEANS",
    "TABLE_OBSERVATIONS",
    "TABULATED_OBSERVATIONS",
    "THRESHOLD",
    "WIDE",
    "ConditionalPosition",
    "CoverageRow",
    "IntervalLimits",
    "ObservationRow",
    "PostSelectionSummary",
    "ProcedureCurves",
    "ProcedureRow",
    "RandomisedTable",
    "WidthCurve",
    "WidthRow",
    "conditional_cdf",
    "conditional_density",
    "example_payload",
    "log10_selection_probability",
    "ordinary_coverage",
    "ordinary_interval",
    "procedure_curves",
    "randomised_conditional_cdf",
    "randomised_interval",
    "randomised_selection_probability",
    "randomised_width_table",
    "selection_probability",
    "selective_interval",
    "split_width",
    "tail_constant",
    "tail_limit",
    "threshold_hazard",
    "width_curve",
    "width_exceedance",
    "width_quantile",
]

THRESHOLD: Final[float] = 2.0
ALPHA: Final[float] = 0.05
# The article's ordinary interval, and the figures' reference width 2 * 1.96.
ORDINARY_Z: Final[float] = 1.96
OBSERVED: Final[float] = 2.05
TABLE_OBSERVATIONS: Final[tuple[float, ...]] = (2.01, 2.05, 2.20, 2.50, 3.00, 3.50, 5.00)
TABLE_MEANS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 3.0, 4.0)
COVERAGE_MEANS: Final[tuple[float, ...]] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
# The width quantile table reports the share of intervals wider than this.
WIDE: Final[float] = 20.0

# First figure: conditional densities from the threshold to 2.6 at three means,
# and the exact width at 60 distances from 0.01 to 4 above the threshold.
DENSITY_MEANS: Final[tuple[float, ...]] = (1.0, -5.0, -20.0)
DENSITY_UPPER: Final[float] = 2.6
DISTANCE_RANGE: Final[tuple[float, float]] = (0.01, 4.0)
DISTANCE_POINTS: Final[int] = 60

# Second figure: 17 means from 0 to 4 (the same values as np.linspace(0, 4, 17)).
FIGURE_MEANS: Final[tuple[float, ...]] = tuple(0.25 * i for i in range(17))
RANDOMISATION_SD: Final[float] = 1.0
SEED: Final[int] = 20260918
DRAWS: Final[int] = 400_000
INTEGRATION_RANGE: Final[tuple[float, float]] = (-45.0, 25.0)
INTEGRATION_POINTS: Final[int] = 14_001
TABULATED_OBSERVATIONS: Final[tuple[float, float, int]] = (-4.0, 10.0, 113)
# The randomised limits are searched from -40 to the observation plus 8.
_RANDOMISED_FLOOR: Final[float] = -40.0
_RANDOMISED_CEILING: Final[float] = 8.0
_SELECTIVE_XTOL: Final[float] = 1e-12
_RANDOMISED_XTOL: Final[float] = 1e-7

_GRID: Final[NDArray[np.float64]] = np.linspace(*INTEGRATION_RANGE, INTEGRATION_POINTS)
_GRID.setflags(write=False)


@dataclass(frozen=True, slots=True)
class IntervalLimits:
    """Lower and upper limits of an interval for the mean."""

    lower: float
    upper: float

    @property
    def width(self) -> float:
        """Upper limit minus lower limit."""
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class ObservationRow:
    """The ordinary and exact selective intervals at one observed value."""

    observed: float
    distance: float
    ordinary: IntervalLimits
    selective: IntervalLimits
    width: float
    tail_limit: float


@dataclass(frozen=True, slots=True)
class ConditionalPosition:
    """Where the observation 2.05 sits in the conditional distribution at one mean."""

    mean: float
    exponential_mean: float
    percentile: float
    density_at_threshold: float


@dataclass(frozen=True, slots=True)
class WidthRow:
    """Selection probability and quantiles of the exact width given selection."""

    mean: float
    selected: float
    median: float
    p90: float
    p99: float
    wider_than_20: float


@dataclass(frozen=True, slots=True)
class CoverageRow:
    """Coverage of the ordinary interval among reported observations."""

    mean: float
    coverage: float


@dataclass(frozen=True, slots=True)
class ProcedureRow:
    """Selection rates and widths given selection under the three valid procedures."""

    mean: float
    hard_selected: float
    randomised_selected: float
    hard_median: float
    hard_p90: float
    randomised_median: float
    randomised_p90: float
    split: float


@dataclass(frozen=True, slots=True)
class WidthCurve:
    """The first figure's right panel: exact width against the distance above the threshold."""

    distances: tuple[float, ...]
    widths: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ProcedureCurves:
    """The second figure: median and 90th percentile width given selection, by mean."""

    means: tuple[float, ...]
    hard_median: tuple[float, ...]
    hard_p90: tuple[float, ...]
    randomised_median: tuple[float, ...]
    randomised_p90: tuple[float, ...]
    randomised_selected: tuple[float, ...]
    split: float


@dataclass(frozen=True, slots=True)
class RandomisedTable:
    """Randomised-selection interval limits tabulated over the observation."""

    observations: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    @property
    def widths(self) -> NDArray[np.float64]:
        """Width of the interval at each tabulated observation."""
        return np.asarray(self.upper - self.lower, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PostSelectionSummary:
    """Every number the article prints from the model, and the data behind both figures."""

    tail_constant: float
    observations: tuple[ObservationRow, ...]
    log10_selection_at_lower_limit: float
    positions: tuple[ConditionalPosition, ...]
    null_hazard: float
    widths: tuple[WidthRow, ...]
    coverage: tuple[CoverageRow, ...]
    zero_coverage_below: float
    procedures: tuple[ProcedureRow, ...]
    randomised_at_observed: IntervalLimits
    widest_randomised: float
    width_curve: WidthCurve
    procedure_curves: ProcedureCurves


def _threshold(value: float) -> float:
    return real(value, name="threshold")


def _alpha(value: float) -> float:
    return probability(value, name="alpha", inclusive=False)


def tail_constant(alpha: float = ALPHA) -> float:
    """``ln(2 / alpha)``, 3.69 at the 95% level."""
    return log(2 / _alpha(alpha))


def selection_probability(mean: float, *, threshold: float = THRESHOLD) -> float:
    """``P(X > c)`` for ``X ~ N(mean, 1)``."""
    return float(norm.sf(_threshold(threshold) - real(mean, name="mean")))


def log10_selection_probability(mean: float, *, threshold: float = THRESHOLD) -> float:
    """Base-10 logarithm of ``P(X > c)``, finite far below the threshold."""
    log_sf = norm.logsf(_threshold(threshold) - real(mean, name="mean"))
    return float(log_sf) / log(10.0)


def threshold_hazard(mean: float, *, threshold: float = THRESHOLD) -> float:
    """Conditional density at the threshold: the Gaussian hazard ``phi(t) / sf(t)``."""
    t = _threshold(threshold) - real(mean, name="mean")
    return float(np.exp(norm.logpdf(t) - norm.logsf(t)))


def _conditional_cdf(mean: float, x: float, threshold: float) -> float:
    return float(-np.expm1(norm.logsf(x - mean) - norm.logsf(threshold - mean)))


def conditional_cdf(
    mean: float, x: ArrayLike, *, threshold: float = THRESHOLD
) -> NDArray[np.float64]:
    """``P(X <= x | X > c)`` from log survival functions; zero at and below the threshold."""
    c = _threshold(threshold)
    mu = real(mean, name="mean")
    points = np.asarray(x, dtype=np.float64)
    cdf = -np.expm1(norm.logsf(points - mu) - norm.logsf(c - mu))
    return np.asarray(np.where(points > c, cdf, 0.0), dtype=np.float64)


def conditional_density(
    x: ArrayLike, mean: float, *, threshold: float = THRESHOLD
) -> NDArray[np.float64]:
    """Density of ``X`` given ``X > c``: ``phi(x - mean) / sf(c - mean)`` above the threshold."""
    c = _threshold(threshold)
    mu = real(mean, name="mean")
    points = np.asarray(x, dtype=np.float64)
    density = np.exp(norm.logpdf(points - mu) - norm.logsf(c - mu))
    return np.asarray(np.where(points >= c, density, 0.0), dtype=np.float64)


def selective_interval(
    x: float, *, threshold: float = THRESHOLD, alpha: float = ALPHA
) -> IntervalLimits:
    """Exact equal-tailed interval given ``X > c``, by inverting the truncated-normal pivot."""
    c = _threshold(threshold)
    level = _alpha(alpha)
    observed = real(x, name="x")
    if observed <= c:
        raise ValueError("x must exceed the threshold")

    def limit(target: float) -> float:
        def gap(mu: float) -> float:
            return _conditional_cdf(mu, observed, c) - target

        low, high = observed - 1.0, observed + 1.0
        while gap(low) < 0:  # the conditional cdf falls as the mean rises
            low -= 2 * (observed - low)
        while gap(high) > 0:
            high += 2 * (high - observed)
        return float(brentq(gap, low, high, xtol=_SELECTIVE_XTOL))

    return IntervalLimits(lower=limit(1 - level / 2), upper=limit(level / 2))


def tail_limit(x: float, *, threshold: float = THRESHOLD, alpha: float = ALPHA) -> float:
    """Exponential-tail approximation to the lower limit, ``c - ln(2 / alpha) / (x - c)``."""
    c = _threshold(threshold)
    distance = real(x, name="x") - c
    if distance <= 0:
        raise ValueError("x must exceed the threshold")
    return c - tail_constant(alpha) / distance


def width_quantile(
    mean: float, level: float, *, threshold: float = THRESHOLD, alpha: float = ALPHA
) -> float:
    """``level``-quantile of the exact width given selection.

    The width falls as the observation rises, so it is the width at the
    observation exceeded, given selection, with probability ``level``.
    """
    c = _threshold(threshold)
    mu = real(mean, name="mean")
    q = probability(level, name="level", inclusive=False)
    x = mu + float(norm.isf(q * norm.sf(c - mu)))
    return selective_interval(x, threshold=c, alpha=alpha).width


def width_exceedance(
    mean: float, width: float, *, threshold: float = THRESHOLD, alpha: float = ALPHA
) -> float:
    """``P(width > w | X > c)``: the chance of landing closer to the threshold than ``w`` allows."""
    c = _threshold(threshold)
    target = positive(width, name="width")

    def gap(distance: float) -> float:
        return selective_interval(c + distance, threshold=c, alpha=alpha).width - target

    near, far = 1e-4, 10.0
    if not gap(far) < 0 < gap(near):
        raise ValueError("width must lie between the widths at distances 1e-4 and 10")
    crossing = float(brentq(gap, near, far, xtol=1e-12))
    return float(conditional_cdf(mean, c + crossing, threshold=c))


def ordinary_interval(x: float, *, z: float = ORDINARY_Z) -> IntervalLimits:
    """``x +/- z``, ignoring the selection."""
    observed = real(x, name="x")
    half = positive(z, name="z")
    return IntervalLimits(lower=observed - half, upper=observed + half)


def ordinary_coverage(mean: float, *, threshold: float = THRESHOLD, z: float = ORDINARY_Z) -> float:
    """``P(|X - mean| <= z | X > c)``: zero whenever ``mean + z`` does not clear the threshold."""
    c = _threshold(threshold)
    mu = real(mean, name="mean")
    half = positive(z, name="z")
    inside = float(norm.sf(max(-half, c - mu)) - norm.sf(half))
    if inside <= 0.0:
        return 0.0
    return inside / float(norm.sf(c - mu))


def split_width(*, z: float = ORDINARY_Z) -> float:
    """Ordinary width from an independent half: each half has variance 2."""
    return 2 * positive(z, name="z") * sqrt(2.0)


def randomised_selection_probability(
    mean: float, *, threshold: float = THRESHOLD, sd: float = RANDOMISATION_SD
) -> float:
    """``P(X + omega > c)`` with ``omega ~ N(0, sd^2)`` independent of ``X``."""
    spread = sqrt(1.0 + positive(sd, name="sd") ** 2)
    return float(norm.sf((_threshold(threshold) - real(mean, name="mean")) / spread))


def _selection_log_weights(threshold: float, sd: float) -> NDArray[np.float64]:
    c = _threshold(threshold)
    return np.asarray(norm.logcdf((_GRID - c) / positive(sd, name="sd")), dtype=np.float64)


def _randomised_cdf(mean: float, x: float, log_select: NDArray[np.float64]) -> float:
    grid = _GRID
    # The normal constant of phi(t - mean) cancels once the maximum is removed.
    log_weight = -0.5 * (grid - mean) ** 2 + log_select
    weight = np.exp(log_weight - log_weight.max())
    return float(weight[grid <= x].sum() / weight.sum())


def randomised_conditional_cdf(
    mean: float, x: float, *, threshold: float = THRESHOLD, sd: float = RANDOMISATION_SD
) -> float:
    """``P(X <= x | X + omega > c)``, summed on the fixed integration grid."""
    log_select = _selection_log_weights(threshold, sd)
    return _randomised_cdf(real(mean, name="mean"), real(x, name="x"), log_select)


def _randomised_limits(x: float, log_select: NDArray[np.float64], alpha: float) -> IntervalLimits:
    def limit(target: float) -> float:
        def gap(mu: float) -> float:
            return _randomised_cdf(mu, x, log_select) - target

        return float(brentq(gap, _RANDOMISED_FLOOR, x + _RANDOMISED_CEILING, xtol=_RANDOMISED_XTOL))

    return IntervalLimits(lower=limit(1 - alpha / 2), upper=limit(alpha / 2))


def randomised_interval(
    x: float,
    *,
    threshold: float = THRESHOLD,
    sd: float = RANDOMISATION_SD,
    alpha: float = ALPHA,
) -> IntervalLimits:
    """Equal-tailed interval for the mean given ``X + omega > c``."""
    log_select = _selection_log_weights(threshold, sd)
    return _randomised_limits(real(x, name="x"), log_select, _alpha(alpha))


def randomised_width_table(
    *, threshold: float = THRESHOLD, sd: float = RANDOMISATION_SD, alpha: float = ALPHA
) -> RandomisedTable:
    """The randomised interval at 113 observations from -4 to 10."""
    log_select = _selection_log_weights(threshold, sd)
    level = _alpha(alpha)
    start, stop, points = TABULATED_OBSERVATIONS
    observations = np.linspace(start, stop, points)
    limits = [_randomised_limits(float(x), log_select, level) for x in observations]
    return RandomisedTable(
        observations=observations,
        lower=np.array([interval.lower for interval in limits]),
        upper=np.array([interval.upper for interval in limits]),
    )


def width_curve(*, threshold: float = THRESHOLD, alpha: float = ALPHA) -> WidthCurve:
    """Exact width at 60 geometrically spaced distances above the threshold."""
    distances = np.geomspace(*DISTANCE_RANGE, DISTANCE_POINTS)
    widths = [
        selective_interval(threshold + float(d), threshold=threshold, alpha=alpha).width
        for d in distances
    ]
    return WidthCurve(distances=tuple(float(d) for d in distances), widths=tuple(widths))


def procedure_curves(
    table: RandomisedTable | None = None,
    *,
    means: tuple[float, ...] = FIGURE_MEANS,
    seed: int = SEED,
    draws: int = DRAWS,
    threshold: float = THRESHOLD,
    sd: float = RANDOMISATION_SD,
    alpha: float = ALPHA,
) -> ProcedureCurves:
    """Width quantiles given selection: exact for the hard threshold, simulated when randomised.

    For each mean in turn the generator draws the observations and then the
    randomisation noise; the randomised width of each selected draw is
    interpolated from the table.
    """
    c = _threshold(threshold)
    spread = positive(sd, name="sd")
    grid = (
        table if table is not None else randomised_width_table(threshold=c, sd=spread, alpha=alpha)
    )
    rng = np.random.default_rng(count(seed, name="seed"))
    size = count(draws, name="draws", minimum=1)

    hard: list[tuple[float, float]] = []
    soft: list[tuple[float, float]] = []
    selected: list[float] = []
    for mean in means:
        mu = real(mean, name="mean")
        hard.append(
            (
                width_quantile(mu, 0.5, threshold=c, alpha=alpha),
                width_quantile(mu, 0.9, threshold=c, alpha=alpha),
            )
        )
        x = mu + rng.standard_normal(size)
        keep = x + spread * rng.standard_normal(x.size) > c
        widths = np.interp(x[keep], grid.observations, grid.widths)
        median, p90 = np.quantile(widths, [0.5, 0.9])
        soft.append((float(median), float(p90)))
        selected.append(float(np.mean(keep)))

    return ProcedureCurves(
        means=tuple(float(m) for m in means),
        hard_median=tuple(row[0] for row in hard),
        hard_p90=tuple(row[1] for row in hard),
        randomised_median=tuple(row[0] for row in soft),
        randomised_p90=tuple(row[1] for row in soft),
        randomised_selected=tuple(selected),
        split=split_width(),
    )


def _observation_row(x: float) -> ObservationRow:
    selective = selective_interval(x)
    return ObservationRow(
        observed=x,
        distance=x - THRESHOLD,
        ordinary=ordinary_interval(x),
        selective=selective,
        width=selective.width,
        tail_limit=tail_limit(x),
    )


def _position(mean: float) -> ConditionalPosition:
    return ConditionalPosition(
        mean=mean,
        exponential_mean=1 / (THRESHOLD - mean),
        percentile=float(conditional_cdf(mean, OBSERVED)),
        density_at_threshold=threshold_hazard(mean),
    )


def _width_row(mean: float) -> WidthRow:
    return WidthRow(
        mean=mean,
        selected=selection_probability(mean),
        median=width_quantile(mean, 0.5),
        p90=width_quantile(mean, 0.9),
        p99=width_quantile(mean, 0.99),
        wider_than_20=width_exceedance(mean, WIDE),
    )


def _procedure_rows(curves: ProcedureCurves) -> tuple[ProcedureRow, ...]:
    rows = []
    for mean in TABLE_MEANS:
        i = curves.means.index(mean)
        rows.append(
            ProcedureRow(
                mean=mean,
                hard_selected=selection_probability(mean),
                randomised_selected=randomised_selection_probability(mean),
                hard_median=curves.hard_median[i],
                hard_p90=curves.hard_p90[i],
                randomised_median=curves.randomised_median[i],
                randomised_p90=curves.randomised_p90[i],
                split=curves.split,
            )
        )
    return tuple(rows)


def example_payload() -> PostSelectionSummary:
    """Return the article's tables and prose numbers, and the data behind both figures."""
    observations = tuple(_observation_row(x) for x in TABLE_OBSERVATIONS)
    lower_at_observed = observations[TABLE_OBSERVATIONS.index(OBSERVED)].selective.lower
    table = randomised_width_table()
    curves = procedure_curves(table)
    return PostSelectionSummary(
        tail_constant=tail_constant(),
        observations=observations,
        log10_selection_at_lower_limit=log10_selection_probability(lower_at_observed),
        positions=tuple(_position(mean) for mean in DENSITY_MEANS),
        null_hazard=threshold_hazard(0.0),
        widths=tuple(_width_row(mean) for mean in TABLE_MEANS),
        coverage=tuple(CoverageRow(mean, ordinary_coverage(mean)) for mean in COVERAGE_MEANS),
        zero_coverage_below=THRESHOLD - ORDINARY_Z,
        procedures=_procedure_rows(curves),
        randomised_at_observed=randomised_interval(OBSERVED),
        widest_randomised=float(np.max(table.widths)),
        width_curve=width_curve(),
        procedure_curves=curves,
    )
