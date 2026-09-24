"""Regression dilution and simulation-extrapolation, for the article on noisy predictors.

If ``y = beta x* + e`` but the predictor is recorded as ``x = x* + u``, with
``u`` independent noise, least squares on ``x`` converges to ``beta lambda``,
where ``lambda = Var(x*) / (Var(x*) + Var(u))`` is the reliability of the
measurement. With ``Var(x*) = 1`` a reliability ``lambda`` takes noise of standard
deviation ``sqrt((1 - lambda) / lambda)``. Correlations shrink the same way, by
the square root of the product of the two reliabilities (Spearman, 1904).

With a second predictor the bias leaks. Let ``x1 = z + e1`` and ``x2 = z + e2``
share a factor ``z``, with noise of standard deviation ``p`` in each, and let
``y = beta x1 + noise``. Adding noise of standard deviation ``s`` to ``x1`` and
regressing on both gives population coefficients

    b1 = beta ((1 + p^2)^2 - 1) / D,    b2 = beta s^2 / D,
    D = (1 + p^2 + s^2)(1 + p^2) - 1,

so the clean ``x2``, which does nothing, takes weight from the noisy ``x1``.

Simulation-extrapolation adds further noise with variance ``k Var(u)`` for
``k`` in 0, 0.5, 1, 1.5 and 2, refits, and extrapolates the slope to ``k = -1``.
For a linear model the slope at ``k`` is exactly ``beta / (1 + (1 + k) Var(u))``,
the rational form ``a / (b + k)`` with ``a = beta lambda / (1 - lambda)`` and
``b = 1 / (1 - lambda)``: at reliability 0.6, ``a = 1.5`` and ``b = 2.5``. A
quadratic in ``k`` only approximates that curve, and under-corrects a full unit
beyond the data even when fitted to the exact slopes.

The figure and the article's first three code blocks are one computation from a
generator seeded at 0, and are reproduced draw for draw: 20,000 true values and
outcomes, the five reliabilities, the two-predictor draws (which the figure
consumes without plotting), and the SIMEX refits, 20 at each multiple. The
fitted slopes, the leakage table, the extrapolations and the fitted denominator
are the article's, and are pinned at its printed precision. Its field deployment
section trains random forests on draws from a generator seeded at 1, which the
figure does not use, and is not reproduced.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from blog_reproducibility.common.validation import count, non_negative, positive, probability, real

__all__ = [
    "ADDED_NOISE_SDS",
    "EXTRAPOLATION_POINT",
    "OBSERVATIONS",
    "PROXY_NOISE_SD",
    "RATIONAL_START",
    "RELIABILITIES",
    "SEED",
    "SIMEX_MULTIPLES",
    "SIMEX_REFITS",
    "SIMEX_RELIABILITY",
    "TRUE_SLOPE",
    "AttenuationRow",
    "LeakageRow",
    "MeasurementErrorSummary",
    "SimexResult",
    "attenuated_slope",
    "example_payload",
    "fit_quadratic",
    "fit_rational",
    "leakage_coefficients",
    "noise_sd_for_reliability",
    "observed_correlation",
    "ols_coefficients",
    "predictor_correlation",
    "rational_extrapolant",
    "rational_parameters",
    "simex_slope",
    "simulate",
]

SEED: Final[int] = 0
OBSERVATIONS: Final[int] = 20_000
TRUE_SLOPE: Final[float] = 1.0
RELIABILITIES: Final[tuple[float, ...]] = (1.0, 0.8, 0.6, 0.4, 0.2)
# Two predictors sharing a factor, each with noise of this standard deviation.
PROXY_NOISE_SD: Final[float] = 0.7
ADDED_NOISE_SDS: Final[tuple[float, ...]] = (0.0, 0.5, 1.0, 1.5)
# Simulation-extrapolation: added noise variance as multiples of the existing noise.
SIMEX_RELIABILITY: Final[float] = 0.6
SIMEX_MULTIPLES: Final[tuple[float, ...]] = (0.0, 0.5, 1.0, 1.5, 2.0)
SIMEX_REFITS: Final[int] = 20
RATIONAL_START: Final[tuple[float, float]] = (1.0, 2.0)
EXTRAPOLATION_POINT: Final[float] = -1.0


@dataclass(frozen=True, slots=True)
class AttenuationRow:
    """The fitted slope at one reliability, against ``beta lambda``."""

    reliability: float
    noise_sd: float
    fitted_slope: float
    predicted_slope: float


@dataclass(frozen=True, slots=True)
class LeakageRow:
    """Both coefficients after adding noise to the first predictor, fitted and exact."""

    added_noise_sd: float
    coefficient_x1: float
    coefficient_x2: float
    exact_x1: float
    exact_x2: float


@dataclass(frozen=True, slots=True)
class SimexResult:
    """The refitted slopes, both extrapolants and regression calibration."""

    reliability: float
    multiples: tuple[float, ...]
    slopes: tuple[float, ...]
    exact_slopes: tuple[float, ...]
    quadratic: tuple[float, ...]
    quadratic_estimate: float
    rational_numerator: float
    rational_denominator: float
    rational_estimate: float
    exact_numerator: float
    exact_denominator: float
    exact_quadratic_estimate: float
    regression_calibration: float


@dataclass(frozen=True, slots=True)
class MeasurementErrorSummary:
    """The figure's two panels and the article's leakage table."""

    observations: int
    true_slope: float
    attenuation: tuple[AttenuationRow, ...]
    predictor_correlation: float
    exact_predictor_correlation: float
    leakage: tuple[LeakageRow, ...]
    simex: SimexResult


def _reliability(value: float) -> float:
    reliability = probability(value, name="reliability")
    if reliability == 0.0:
        raise ValueError("reliability must be positive")
    return reliability


def noise_sd_for_reliability(reliability: float) -> float:
    """Noise standard deviation giving ``reliability`` when the true predictor has unit variance."""
    lam = _reliability(reliability)
    return sqrt((1 - lam) / lam)


def attenuated_slope(reliability: float, true_slope: float = TRUE_SLOPE) -> float:
    """Where least squares on the noisy predictor converges: ``beta lambda``."""
    return real(true_slope, name="true_slope") * _reliability(reliability)


def observed_correlation(
    true_correlation: float, reliability_x: float, reliability_y: float
) -> float:
    """Spearman's attenuation: the true correlation times ``sqrt(lambda_x lambda_y)``."""
    rho = real(true_correlation, name="true_correlation")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("true_correlation must lie in [-1, 1]")
    return rho * sqrt(_reliability(reliability_x) * _reliability(reliability_y))


def predictor_correlation(proxy_noise_sd: float = PROXY_NOISE_SD) -> float:
    """Correlation of two predictors that share a unit-variance factor: ``1 / (1 + p^2)``."""
    return 1 / (1 + non_negative(proxy_noise_sd, name="proxy_noise_sd") ** 2)


def leakage_coefficients(
    added_noise_sd: float,
    *,
    proxy_noise_sd: float = PROXY_NOISE_SD,
    true_slope: float = TRUE_SLOPE,
) -> tuple[float, float]:
    """Population coefficients of the noisy ``x1`` and the clean ``x2``."""
    s2 = non_negative(added_noise_sd, name="added_noise_sd") ** 2
    v = 1 + positive(proxy_noise_sd, name="proxy_noise_sd") ** 2
    beta = real(true_slope, name="true_slope")
    determinant = (v + s2) * v - 1
    return beta * (v**2 - 1) / determinant, beta * s2 / determinant


def simex_slope(multiple: float, reliability: float, true_slope: float = TRUE_SLOPE) -> float:
    """Slope after adding noise with ``multiple`` times the existing noise variance."""
    k = real(multiple, name="multiple")
    noise_variance = noise_sd_for_reliability(reliability) ** 2
    if 1 + (1 + k) * noise_variance <= 0:
        raise ValueError("multiple removes more than all of the variance")
    return real(true_slope, name="true_slope") / (1 + (1 + k) * noise_variance)


def rational_parameters(reliability: float, true_slope: float = TRUE_SLOPE) -> tuple[float, float]:
    """Exact ``a = beta lambda / (1 - lambda)`` and ``b = 1 / (1 - lambda)`` of ``a / (b + k)``."""
    lam = probability(reliability, name="reliability", inclusive=False)
    return real(true_slope, name="true_slope") * lam / (1 - lam), 1 / (1 - lam)


def rational_extrapolant(
    multiple: ArrayLike, numerator: float, denominator: float
) -> NDArray[np.float64]:
    """The rational form ``a / (b + k)``, as the article fits it."""
    values: NDArray[np.float64] = numerator / (denominator + np.asarray(multiple, dtype=np.float64))
    return values


def fit_quadratic(multiples: ArrayLike, slopes: ArrayLike) -> tuple[float, ...]:
    """Least-squares quadratic in the multiple, highest power first."""
    x = np.asarray(multiples, dtype=np.float64)
    y = np.asarray(slopes, dtype=np.float64)
    return tuple(float(c) for c in np.polyfit(x, y, 2))


def fit_rational(
    multiples: ArrayLike, slopes: ArrayLike, start: tuple[float, float] = RATIONAL_START
) -> tuple[float, float]:
    """Least-squares ``a`` and ``b`` of ``a / (b + k)``, from the article's starting point."""
    (a, b), _ = optimize.curve_fit(rational_extrapolant, multiples, slopes, p0=list(start))
    return float(a), float(b)


def ols_coefficients(features: ArrayLike, outcome: ArrayLike) -> tuple[float, ...]:
    """Least-squares slopes, fitted with an intercept that is then dropped, as the article's."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("features must be (n, p) and outcome (n,)")
    design = np.column_stack([np.ones(len(x)), x])
    return tuple(float(c) for c in np.linalg.lstsq(design, y, rcond=None)[0][1:])


def simulate(
    seed: int = SEED, *, observations: int = OBSERVATIONS, refits: int = SIMEX_REFITS
) -> MeasurementErrorSummary:
    """Run the article's simulations from one generator, in the article's order."""
    rng = np.random.default_rng(count(seed, name="seed"))
    n = count(observations, name="observations", minimum=3)
    repeats = count(refits, name="refits", minimum=1)

    x_true = rng.normal(size=n)
    y = TRUE_SLOPE * x_true + rng.normal(scale=1.0, size=n)
    attenuation = []
    for reliability in RELIABILITIES:
        su = np.sqrt((1 - reliability) / reliability)
        slope = np.polyfit(x_true + rng.normal(scale=su, size=n), y, 1)[0]
        attenuation.append(
            AttenuationRow(
                reliability=reliability,
                noise_sd=float(su),
                fitted_slope=float(slope),
                predicted_slope=attenuated_slope(reliability),
            )
        )

    # Two predictors sharing a factor; only the first has an effect.
    z = rng.normal(size=n)
    x1 = z + rng.normal(scale=PROXY_NOISE_SD, size=n)
    x2 = z + rng.normal(scale=PROXY_NOISE_SD, size=n)
    y2 = TRUE_SLOPE * x1 + rng.normal(size=n)
    leakage = []
    for added in ADDED_NOISE_SDS:
        b1, b2 = ols_coefficients(np.column_stack([x1 + rng.normal(scale=added, size=n), x2]), y2)
        exact_1, exact_2 = leakage_coefficients(added)
        leakage.append(LeakageRow(added, b1, b2, exact_1, exact_2))

    su = np.sqrt((1 - SIMEX_RELIABILITY) / SIMEX_RELIABILITY)
    x_obs = x_true + rng.normal(scale=su, size=n)
    multiples = np.array(SIMEX_MULTIPLES)
    slopes = np.array(
        [
            np.mean(
                [
                    np.polyfit(x_obs + rng.normal(scale=np.sqrt(k) * su, size=n), y, 1)[0]
                    for _ in range(repeats)
                ]
            )
            for k in multiples
        ]
    )
    quadratic = fit_quadratic(multiples, slopes)
    a, b = fit_rational(multiples, slopes)
    exact = [simex_slope(float(k), SIMEX_RELIABILITY) for k in multiples]
    exact_a, exact_b = rational_parameters(SIMEX_RELIABILITY)
    point = EXTRAPOLATION_POINT
    simex = SimexResult(
        reliability=SIMEX_RELIABILITY,
        multiples=SIMEX_MULTIPLES,
        slopes=tuple(float(value) for value in slopes),
        exact_slopes=tuple(exact),
        quadratic=quadratic,
        quadratic_estimate=float(np.polyval(quadratic, point)),
        rational_numerator=a,
        rational_denominator=b,
        rational_estimate=float(rational_extrapolant(point, a, b)),
        exact_numerator=exact_a,
        exact_denominator=exact_b,
        exact_quadratic_estimate=float(np.polyval(fit_quadratic(multiples, exact), point)),
        regression_calibration=float(slopes[0] / SIMEX_RELIABILITY),
    )
    return MeasurementErrorSummary(
        observations=n,
        true_slope=TRUE_SLOPE,
        attenuation=tuple(attenuation),
        predictor_correlation=float(np.corrcoef(x1, x2)[0, 1]),
        exact_predictor_correlation=predictor_correlation(),
        leakage=tuple(leakage),
        simex=simex,
    )


def example_payload() -> MeasurementErrorSummary:
    """Return the figure's slopes and extrapolations and the article's leakage table."""
    return simulate()
