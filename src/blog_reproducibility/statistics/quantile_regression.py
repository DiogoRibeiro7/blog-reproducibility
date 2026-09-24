"""Linear quantile regression against a mean plus a margin, for the article on predicting the range.

Deliveries have a distance ``d`` uniform on 1 to 10 km and a network load ``L``
uniform on 0 to 1, and take

    y = 20 + 3 d + (2 + 12 L) Z,    Z = (G - 2) / sqrt(2),  G ~ Gamma(2, 1),

minutes: distance moves the centre, load the spread, and the standardised gamma
shock ``Z`` (mean zero, unit variance) has a long right tail. Its quantiles are
``z_tau = (g_tau - 2) / sqrt(2)`` with ``g_tau`` the gamma quantile, so the
conditional ``tau``-quantile is linear in both inputs,

    Q_tau(y | d, L) = (20 + 2 z_tau) + 3 d + 12 z_tau L,

which is what linear quantile regression estimates and what the tests compare
it with. Load has no effect on the mean, and every delivery exceeds its mean
with probability ``P(G > 2) = 3 exp(-2)``, about 0.406, whatever its load.

Quantile regression minimises the pinball loss ``sum rho_tau(y_i - x_i b)`` with
``rho_tau(u) = u (tau - 1[u < 0])``, a linear program. The article fits it with
statsmodels' ``QuantReg``, which iterates reweighted least squares to a
tolerance. Here the program's dual,

    maximise y'a  subject to  X'a = (1 - tau) X'1,  0 <= a <= 1,

is solved with HiGHS through ``scipy.optimize.linprog``: three equality
constraints and one bounded weight per delivery, a fiftieth of a second for the
article's 4,000 training rows. The coefficients are the constraints' multipliers,
the exact vertex that scikit-learn's ``QuantileRegressor`` also finds and that
passes through three training deliveries; statsmodels stops within about
``2e-6`` minutes of it, with a pinball loss higher by about ``1e-6``. Every
coefficient, coverage, width, miss rate and pinball loss the article prints is
the same either way, and the figure's curves move by less than ``2e-6`` minutes.

The figure and the article are one computation from a generator seeded at 0:
6,000 deliveries (distances, loads, then the gamma draws), the first 4,000 to
fit least squares with a normal 90 percent margin (``1.645`` residual standard
deviations) and quantile regression at 0.05, 0.5 and 0.95, the last 2,000 to
test them by load segment. The figure then draws 2,500 fresh deliveries at 5 km
(loads, then gamma draws) from the same generator and plots both bands against
load. The article's boosting columns are not reproduced: they come from
scikit-learn's ``HistGradientBoostingRegressor``, which the figure does not use
and whose results depend on the library version.

The expected coverage of a band, for a new delivery in a load segment, is the
double integral over distance and load of the gamma probability between the
band's edges; the tests compare the segment coverages with it.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, optimize, special, stats

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "BASE_MINUTES",
    "BASE_SPREAD",
    "COST_RATIOS",
    "DELIVERIES",
    "DISTANCE_RANGE",
    "FIGURE_DELIVERIES",
    "FIGURE_DISTANCE",
    "GAMMA_SHAPE",
    "GRID_POINTS",
    "LOAD_SEGMENTS",
    "MINUTES_PER_KM",
    "NORMAL_MARGIN",
    "QUANTILES",
    "SEED",
    "SPREAD_PER_LOAD",
    "TRAINING",
    "BandFigure",
    "Deliveries",
    "FitRow",
    "Misses",
    "PinballRow",
    "PromiseRow",
    "QuantileRegressionSummary",
    "SegmentRow",
    "band_coverage",
    "conditional_quantile",
    "coverage",
    "example_payload",
    "late_share_at_mean",
    "least_squares",
    "pinball_loss",
    "population_coefficients",
    "promise_quantile",
    "quantile_regression",
    "shock_quantile",
    "simulate_deliveries",
    "simulate_fixed_distance",
]

SEED: Final[int] = 0
DELIVERIES: Final[int] = 6000
# The first 4,000 deliveries train the models; the other 2,000 test them.
TRAINING: Final[int] = 4000
DISTANCE_RANGE: Final[tuple[float, float]] = (1.0, 10.0)
BASE_MINUTES: Final[float] = 20.0
MINUTES_PER_KM: Final[float] = 3.0
BASE_SPREAD: Final[float] = 2.0
SPREAD_PER_LOAD: Final[float] = 12.0
GAMMA_SHAPE: Final[float] = 2.0
QUANTILES: Final[tuple[float, ...]] = (0.05, 0.5, 0.95)
# The article's normal 90 percent margin, in residual standard deviations.
NORMAL_MARGIN: Final[float] = 1.645
LOAD_SEGMENTS: Final[tuple[tuple[float, float], ...]] = (
    (0.0, 0.25),
    (0.25, 0.5),
    (0.5, 0.75),
    (0.75, 1.0),
)
# Costs of a late and an early delivery.
COST_RATIOS: Final[tuple[tuple[float, float], ...]] = ((1.0, 1.0), (4.0, 1.0), (9.0, 1.0))
# The figure's fresh deliveries, all at one distance, and its load grid.
FIGURE_DISTANCE: Final[float] = 5.0
FIGURE_DELIVERIES: Final[int] = 2500
GRID_POINTS: Final[int] = 100


@dataclass(frozen=True, slots=True)
class Deliveries:
    """Distance in km, network load and delivery time in minutes."""

    distance: NDArray[np.float64]
    load: NDArray[np.float64]
    minutes: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FitRow:
    """Fitted intercept and slopes, and the population values they estimate."""

    model: str
    quantile: float | None
    intercept: float
    distance: float
    load: float
    population: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SegmentRow:
    """Test coverage and mean width of both 90 percent intervals in one load segment."""

    lower: float
    upper: float
    deliveries: int
    ols_coverage: float
    quantile_coverage: float
    ols_width: float
    quantile_width: float
    expected_ols_coverage: float
    expected_quantile_coverage: float


@dataclass(frozen=True, slots=True)
class Misses:
    """Share of test deliveries above and below each interval, and overall coverage."""

    ols_coverage: float
    quantile_coverage: float
    ols_above: float
    ols_below: float
    quantile_above: float
    quantile_below: float


@dataclass(frozen=True, slots=True)
class PinballRow:
    """Mean pinball loss on the test deliveries at one quantile."""

    quantile: float
    ols: float
    quantile_regression: float


@dataclass(frozen=True, slots=True)
class PromiseRow:
    """The quantile that minimises expected cost when late and early deliveries cost these."""

    late_cost: float
    early_cost: float
    quantile: float


@dataclass(frozen=True, slots=True)
class BandFigure:
    """The figure: fresh deliveries at 5 km and both bands on a load grid."""

    loads: tuple[float, ...]
    minutes: tuple[float, ...]
    grid: tuple[float, ...]
    ols_lower: tuple[float, ...]
    ols_upper: tuple[float, ...]
    lower: tuple[float, ...]
    median: tuple[float, ...]
    upper: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class QuantileRegressionSummary:
    """The figure's bands and every linear-model number the article prints."""

    residual_sd: float
    fits: tuple[FitRow, ...]
    segments: tuple[SegmentRow, ...]
    misses: Misses
    pinball: tuple[PinballRow, ...]
    mean_minutes: float
    median_minutes: float
    population_mean_minutes: float
    late_share: float
    population_late_share: float
    promises: tuple[PromiseRow, ...]
    figure: BandFigure


def _quantile(value: float) -> float:
    return probability(value, name="quantile", inclusive=False)


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} must be finite")
    return data


def _design(predictors: ArrayLike, rows: int) -> NDArray[np.float64]:
    x = np.asarray(predictors, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != rows:
        raise ValueError("predictors must be a two-dimensional array with one row per outcome")
    if not np.all(np.isfinite(x)):
        raise ValueError("predictors must be finite")
    return np.column_stack([np.ones(rows), x])


def simulate_deliveries(rng: np.random.Generator, deliveries: int = DELIVERIES) -> Deliveries:
    """Draw distances, then loads, then the gamma shocks, and build the delivery times."""
    n = count(deliveries, name="deliveries", minimum=1)
    distance = rng.uniform(*DISTANCE_RANGE, n)
    load = rng.uniform(0, 1, n)
    mu = BASE_MINUTES + MINUTES_PER_KM * distance
    sigma = BASE_SPREAD + SPREAD_PER_LOAD * load
    minutes = mu + sigma * (rng.gamma(GAMMA_SHAPE, 1.0, n) - GAMMA_SHAPE) / np.sqrt(GAMMA_SHAPE)
    return Deliveries(distance=distance, load=load, minutes=minutes)


def simulate_fixed_distance(
    rng: np.random.Generator,
    deliveries: int = FIGURE_DELIVERIES,
    *,
    distance: float = FIGURE_DISTANCE,
) -> Deliveries:
    """The figure's fresh deliveries: loads, then gamma shocks, all at one distance."""
    n = count(deliveries, name="deliveries", minimum=1)
    km = positive(distance, name="distance")
    load = rng.uniform(0, 1, n)
    centre = BASE_MINUTES + MINUTES_PER_KM * km
    sigma = BASE_SPREAD + SPREAD_PER_LOAD * load
    minutes = centre + sigma * (rng.gamma(GAMMA_SHAPE, 1.0, n) - GAMMA_SHAPE) / np.sqrt(GAMMA_SHAPE)
    return Deliveries(distance=np.full(n, km), load=load, minutes=minutes)


def shock_quantile(quantile: float) -> float:
    """``z_tau``: the ``tau``-quantile of the standardised gamma shock."""
    g = float(stats.gamma.ppf(_quantile(quantile), GAMMA_SHAPE))
    return (g - GAMMA_SHAPE) / sqrt(GAMMA_SHAPE)


def population_coefficients(quantile: float | None = None) -> tuple[float, float, float]:
    """Intercept, distance and load slopes of the conditional quantile, or of the mean."""
    z = 0.0 if quantile is None else shock_quantile(quantile)
    return (BASE_MINUTES + BASE_SPREAD * z, MINUTES_PER_KM, SPREAD_PER_LOAD * z)


def conditional_quantile(quantile: float, distance: float, load: float) -> float:
    """The ``tau``-quantile of the delivery time at a given distance and load."""
    km = real(distance, name="distance")
    level = real(load, name="load")
    intercept, per_km, per_load = population_coefficients(quantile)
    return intercept + per_km * km + per_load * level


def late_share_at_mean(shape: float = GAMMA_SHAPE) -> float:
    """Chance a delivery exceeds its conditional mean: ``P(G > k)``, ``3 exp(-2)`` for ``k = 2``."""
    k = positive(shape, name="shape")
    return float(special.gammaincc(k, k))


def promise_quantile(late_cost: float, early_cost: float) -> float:
    """The newsvendor quantile ``c_L / (c_L + c_E)`` that minimises expected cost."""
    late = positive(late_cost, name="late_cost")
    early = positive(early_cost, name="early_cost")
    return late / (late + early)


def least_squares(predictors: ArrayLike, outcome: ArrayLike) -> tuple[tuple[float, ...], float]:
    """Least-squares coefficients with an intercept first, and the residuals' standard deviation.

    The standard deviation divides by the number of observations, as ``resid.std()``.
    """
    y = _vector(outcome, "outcome")
    design = _design(predictors, y.size)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual_sd = float(np.std(y - design @ coefficients))
    return tuple(float(c) for c in coefficients), residual_sd


def quantile_regression(
    predictors: ArrayLike, outcome: ArrayLike, quantile: float
) -> tuple[float, ...]:
    """Coefficients, intercept first, that minimise the pinball loss at ``quantile``.

    The dual linear program is solved with HiGHS; its equality multipliers are
    the primal coefficients.
    """
    y = _vector(outcome, "outcome")
    design = _design(predictors, y.size)
    tau = _quantile(quantile)
    if y.size < design.shape[1]:
        raise ValueError("need at least as many outcomes as coefficients")
    result = optimize.linprog(
        -y,
        A_eq=design.T,
        b_eq=(1 - tau) * design.sum(axis=0),
        bounds=(0, 1),
        method="highs",
    )
    if result.status != 0:
        raise RuntimeError(f"the quantile regression program failed: {result.message}")
    return tuple(float(-m) for m in result.eqlin.marginals)


def pinball_loss(quantile: float, outcome: ArrayLike, prediction: ArrayLike) -> float:
    """Mean pinball loss: ``tau`` per unit under-predicted, ``1 - tau`` per unit over."""
    tau = _quantile(quantile)
    y = _vector(outcome, "outcome")
    p = _vector(prediction, "prediction")
    if p.shape != y.shape:
        raise ValueError("outcome and prediction must have the same length")
    d = y - p
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def coverage(outcome: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Share of outcomes inside their interval, both edges included."""
    y = _vector(outcome, "outcome")
    lo = _vector(lower, "lower")
    hi = _vector(upper, "upper")
    if lo.shape != y.shape or hi.shape != y.shape:
        raise ValueError("outcome, lower and upper must have the same length")
    return float(np.mean((y >= lo) & (y <= hi)))


def band_coverage(
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    *,
    load_range: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Chance a new delivery with load in ``load_range`` lands between two linear bands.

    Each band is an intercept and slopes on distance and load. Given both, the
    delivery time is inside when the gamma draw lies between the edges mapped
    to its scale, and that probability is averaged over uniform distance and load.
    """
    low_load, high_load = (real(v, name="load_range") for v in load_range)
    if not 0.0 <= low_load < high_load <= 1.0:
        raise ValueError("load_range must be an increasing pair within [0, 1]")
    lo = tuple(real(v, name="lower") for v in lower)
    hi = tuple(real(v, name="upper") for v in upper)
    if len(lo) != 3 or len(hi) != 3:
        raise ValueError("each band needs an intercept, a distance slope and a load slope")
    root_k = sqrt(GAMMA_SHAPE)

    def gamma_cdf(shock: float) -> float:
        return float(special.gammainc(GAMMA_SHAPE, max(GAMMA_SHAPE + root_k * shock, 0.0)))

    def inside(distance: float, load: float) -> float:
        centre = BASE_MINUTES + MINUTES_PER_KM * distance
        scale = BASE_SPREAD + SPREAD_PER_LOAD * load
        bottom = lo[0] + lo[1] * distance + lo[2] * load
        top = hi[0] + hi[1] * distance + hi[2] * load
        if top <= bottom:
            return 0.0
        return gamma_cdf((top - centre) / scale) - gamma_cdf((bottom - centre) / scale)

    first, last = DISTANCE_RANGE
    total, _ = integrate.dblquad(inside, low_load, high_load, first, last, epsabs=1e-11)
    return float(total) / ((last - first) * (high_load - low_load))


def _as_band(values: tuple[float, ...], shift: float = 0.0) -> tuple[float, float, float]:
    """An intercept and two slopes, with the intercept moved by ``shift``."""
    return (values[0] + shift, values[1], values[2])


def example_payload() -> QuantileRegressionSummary:
    """Fit both models on the article's deliveries, score them and build the figure's bands."""
    rng = np.random.default_rng(SEED)
    data = simulate_deliveries(rng)
    n = data.minutes.size
    train = np.arange(n) < TRAINING
    test = ~train
    predictors = np.column_stack([data.distance, data.load])
    design = np.column_stack([np.ones(n), predictors])
    y, y_test = data.minutes[train], data.minutes[test]

    ols, residual_sd = least_squares(predictors[train], y)
    fitted = {q: quantile_regression(predictors[train], y, q) for q in QUANTILES}
    lower_q, median_q, upper_q = QUANTILES

    prediction = design[test] @ np.asarray(ols)
    ols_lo = prediction - NORMAL_MARGIN * residual_sd
    ols_hi = prediction + NORMAL_MARGIN * residual_sd
    qr = {q: design[test] @ np.asarray(fitted[q]) for q in QUANTILES}
    ols_lower_band = _as_band(ols, -NORMAL_MARGIN * residual_sd)
    ols_upper_band = _as_band(ols, NORMAL_MARGIN * residual_sd)

    load_test = data.load[test]
    segments = []
    for low, high in LOAD_SEGMENTS:
        m = (load_test >= low) & (load_test < high)
        segments.append(
            SegmentRow(
                lower=low,
                upper=high,
                deliveries=int(m.sum()),
                ols_coverage=coverage(y_test[m], ols_lo[m], ols_hi[m]),
                quantile_coverage=coverage(y_test[m], qr[lower_q][m], qr[upper_q][m]),
                ols_width=float(np.mean(ols_hi[m] - ols_lo[m])),
                quantile_width=float(np.mean(qr[upper_q][m] - qr[lower_q][m])),
                expected_ols_coverage=band_coverage(
                    ols_lower_band, ols_upper_band, load_range=(low, high)
                ),
                expected_quantile_coverage=band_coverage(
                    _as_band(fitted[lower_q]), _as_band(fitted[upper_q]), load_range=(low, high)
                ),
            )
        )

    fits = [FitRow("least squares", None, *_as_band(ols), population_coefficients())]
    fits.extend(
        FitRow(f"quantile {q}", q, *_as_band(fitted[q]), population_coefficients(q))
        for q in QUANTILES
    )

    fresh = simulate_fixed_distance(rng)
    grid = np.linspace(0, 1, GRID_POINTS)
    distance = np.full(GRID_POINTS, FIGURE_DISTANCE)
    grid_design = np.column_stack([np.ones(GRID_POINTS), distance, grid])
    grid_ols = grid_design @ np.asarray(ols)

    def on_grid(values: NDArray[np.float64]) -> tuple[float, ...]:
        return tuple(float(v) for v in values)

    return QuantileRegressionSummary(
        residual_sd=residual_sd,
        fits=tuple(fits),
        segments=tuple(segments),
        misses=Misses(
            ols_coverage=coverage(y_test, ols_lo, ols_hi),
            quantile_coverage=coverage(y_test, qr[lower_q], qr[upper_q]),
            ols_above=float(np.mean(y_test > ols_hi)),
            ols_below=float(np.mean(y_test < ols_lo)),
            quantile_above=float(np.mean(y_test > qr[upper_q])),
            quantile_below=float(np.mean(y_test < qr[lower_q])),
        ),
        pinball=tuple(
            PinballRow(
                quantile=q,
                ols=pinball_loss(q, y_test, prediction + residual_sd * float(stats.norm.ppf(q))),
                quantile_regression=pinball_loss(q, y_test, qr[q]),
            )
            for q in QUANTILES
        ),
        mean_minutes=float(np.mean(data.minutes)),
        median_minutes=float(np.median(data.minutes)),
        population_mean_minutes=BASE_MINUTES + MINUTES_PER_KM * sum(DISTANCE_RANGE) / 2,
        late_share=float(np.mean(y_test > prediction)),
        population_late_share=late_share_at_mean(),
        promises=tuple(
            PromiseRow(late, early, promise_quantile(late, early)) for late, early in COST_RATIOS
        ),
        figure=BandFigure(
            loads=on_grid(fresh.load),
            minutes=on_grid(fresh.minutes),
            grid=on_grid(grid),
            ols_lower=on_grid(grid_ols - NORMAL_MARGIN * residual_sd),
            ols_upper=on_grid(grid_ols + NORMAL_MARGIN * residual_sd),
            lower=on_grid(grid_design @ np.asarray(fitted[lower_q])),
            median=on_grid(grid_design @ np.asarray(fitted[median_q])),
            upper=on_grid(grid_design @ np.asarray(fitted[upper_q])),
        ),
    )
