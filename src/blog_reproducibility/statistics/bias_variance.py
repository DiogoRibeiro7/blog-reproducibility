"""Model complexity and the lasso, for the article on statistics and machine learning.

The article argues that the useful boundary between the two fields is the
question asked, the loss and the validation, not the algorithm. Its two figures
make the points about validation and regularisation concrete.

**Training against test error.** Forty design points evenly spaced on
[-1, 1], a true curve ``cos(2.2 x)`` and noise with standard deviation 0.32.
At each polynomial degree from 1 to 15, 160 replications draw a training
response and an independent test response at the same points, fit the
polynomial to the training response by least squares, and record the mean
squared error on both. The design is fixed, so both averages have closed
forms. With ``H`` the hat matrix of the degree-``d`` fit, ``p = d + 1``
coefficients and ``b = (I - H) f`` the part of the truth the fit cannot
represent,

    E[training MSE] = (||b||^2 + sigma^2 (n - p)) / n,
    E[test MSE]     = (||b||^2 + sigma^2 (n + p)) / n,

so the gap between them is exactly the optimism ``2 sigma^2 p / n``, the
overfitting penalty of Mallows' Cp. A single replication's error is a quadratic
form in Gaussian noise, with variance ``(2 sigma^4 (n - p) + 4 sigma^2 ||b||^2)
/ n^2`` on the training response and ``(2 sigma^4 (n + 3 p) + 4 sigma^2 ||b||^2)
/ n^2`` on the test response.

**Lasso paths.** 120 observations of eight independent standard normal
features, coefficients (3, -2, 1.4, 0, 0, 0.8, 0, -0.5) and unit noise. The
features are standardised and the lasso is solved by coordinate descent on 120
penalties spaced geometrically from ``max |X^T y| / n``, the smallest penalty
at which every coefficient is zero, down to a thousandth of it. That is the
grid scikit-learn builds for ``n_alphas=120``; it is passed explicitly because
that argument is deprecated. Features whose coefficient at the smallest penalty
exceeds 0.25 in size are drawn in full and labelled.

Both published images drew from the site's shared global generator, whose
state depended on the figures generated before them. Each simulation here has
its own generator seeded with ``SEED``, so the bias-variance draws differ from
the published image. The published lasso image shows the same paths as these
draws, so it appears to have been drawn from a fresh generator; its penalty axis
grew to the right although its label says "decreasing", and the figure here runs
it from the largest penalty on the left to the smallest on the right. The
article prints no numbers from either simulation.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

from blog_reproducibility.common.validation import count, non_negative, positive

__all__ = [
    "COEFFICIENTS",
    "DEGREES",
    "DESIGN_POINTS",
    "FREQUENCY",
    "LABEL_THRESHOLD",
    "LASSO_NOISE_SD",
    "LASSO_SAMPLES",
    "NOISE_SD",
    "PENALTIES",
    "PENALTY_RATIO",
    "REPLICATIONS",
    "SEED",
    "BiasVarianceSummary",
    "ComplexityCurve",
    "ComplexitySummary",
    "ExpectedErrors",
    "LassoData",
    "LassoPath",
    "LassoSummary",
    "best_degree",
    "complexity_summary",
    "design_points",
    "example_payload",
    "expected_errors",
    "lasso_coefficient_path",
    "lasso_summary",
    "penalty_grid",
    "simulate_complexity_curve",
    "simulate_lasso_data",
    "true_curve",
]

SEED: Final[int] = 20260816

DESIGN_POINTS: Final[int] = 40
FREQUENCY: Final[float] = 2.2
NOISE_SD: Final[float] = 0.32
DEGREES: Final[tuple[int, ...]] = tuple(range(1, 16))
REPLICATIONS: Final[int] = 160

LASSO_SAMPLES: Final[int] = 120
COEFFICIENTS: Final[tuple[float, ...]] = (3.0, -2.0, 1.4, 0.0, 0.0, 0.8, 0.0, -0.5)
LASSO_NOISE_SD: Final[float] = 1.0
PENALTIES: Final[int] = 120
PENALTY_RATIO: Final[float] = 1e-3
LABEL_THRESHOLD: Final[float] = 0.25


@dataclass(frozen=True, slots=True)
class ComplexityCurve:
    """Mean training and test error over the replications, by polynomial degree."""

    degrees: tuple[int, ...]
    training_error: tuple[float, ...]
    test_error: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ExpectedErrors:
    """Closed-form expected errors of one polynomial degree, and one replication's spread."""

    degree: int
    squared_bias: float
    training: float
    test: float
    training_sd: float
    test_sd: float
    optimism: float


@dataclass(frozen=True, slots=True)
class ComplexitySummary:
    """The simulated curve, its expectation, and where each puts the best degree."""

    curve: ComplexityCurve
    expected: tuple[ExpectedErrors, ...]
    best_degree: int
    expected_best_degree: int


@dataclass(frozen=True, slots=True)
class LassoData:
    """Standardised features and the response for the lasso paths."""

    features: NDArray[np.float64]
    response: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LassoPath:
    """Penalties in decreasing order, and one row of coefficients per feature."""

    penalties: NDArray[np.float64]
    coefficients: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LassoSummary:
    """Where the path starts and ends, and the order in which features enter it."""

    largest_penalty: float
    smallest_penalty: float
    entry_order: tuple[int, ...]
    entry_penalties: tuple[float, ...]
    final_coefficients: tuple[float, ...]
    least_squares: tuple[float, ...]
    labelled: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BiasVarianceSummary:
    """The numbers behind both figures."""

    complexity: ComplexitySummary
    lasso: LassoSummary


def design_points(points: int = DESIGN_POINTS) -> NDArray[np.float64]:
    """Evenly spaced design points on [-1, 1]."""
    return np.linspace(-1.0, 1.0, count(points, name="points", minimum=2))


def true_curve(x: ArrayLike) -> NDArray[np.float64]:
    """The regression function ``cos(2.2 x)``."""
    return np.asarray(np.cos(FREQUENCY * np.asarray(x, dtype=np.float64)), dtype=np.float64)


def _degrees(degrees: tuple[int, ...], points: int) -> tuple[int, ...]:
    checked = tuple(count(d, name="degree", minimum=0) for d in degrees)
    if not checked:
        raise ValueError("degrees must not be empty")
    if max(checked) >= points:
        raise ValueError("every degree must leave fewer coefficients than design points")
    return checked


def simulate_complexity_curve(
    *,
    seed: int = SEED,
    replications: int = REPLICATIONS,
    noise_sd: float = NOISE_SD,
    degrees: tuple[int, ...] = DEGREES,
    points: int = DESIGN_POINTS,
) -> ComplexityCurve:
    """Mean training and test error by degree; each replication draws training, then test noise."""
    rng = np.random.default_rng(count(seed, name="seed"))
    reps = count(replications, name="replications", minimum=1)
    sd = non_negative(noise_sd, name="noise_sd")
    x = design_points(points)
    truth = true_curve(x)
    checked = _degrees(degrees, x.size)

    training: list[float] = []
    test: list[float] = []
    for degree in checked:
        fitted_errors, fresh_errors = [], []
        for _ in range(reps):
            y_train = truth + rng.normal(0, sd, x.size)
            y_test = truth + rng.normal(0, sd, x.size)
            fitted = np.polyval(np.polyfit(x, y_train, degree), x)
            fitted_errors.append(float(np.mean((fitted - y_train) ** 2)))
            fresh_errors.append(float(np.mean((fitted - y_test) ** 2)))
        training.append(float(np.mean(fitted_errors)))
        test.append(float(np.mean(fresh_errors)))
    return ComplexityCurve(degrees=checked, training_error=tuple(training), test_error=tuple(test))


def expected_errors(
    degree: int, *, noise_sd: float = NOISE_SD, points: int = DESIGN_POINTS
) -> ExpectedErrors:
    """Expected training and test error of a degree-``d`` least-squares fit on the fixed design."""
    x = design_points(points)
    (d,) = _degrees((degree,), x.size)
    variance = non_negative(noise_sd, name="noise_sd") ** 2
    n, p = x.size, d + 1
    basis, _ = np.linalg.qr(np.vander(x, p))
    truth = true_curve(x)
    residual = truth - basis @ (basis.T @ truth)
    bias = float(residual @ residual)
    return ExpectedErrors(
        degree=d,
        squared_bias=bias / n,
        training=(bias + variance * (n - p)) / n,
        test=(bias + variance * (n + p)) / n,
        training_sd=float(np.sqrt(2 * variance**2 * (n - p) + 4 * variance * bias)) / n,
        test_sd=float(np.sqrt(2 * variance**2 * (n + 3 * p) + 4 * variance * bias)) / n,
        optimism=2 * variance * p / n,
    )


def best_degree(curve: ComplexityCurve) -> int:
    """Degree with the lowest mean test error."""
    return curve.degrees[int(np.argmin(curve.test_error))]


def complexity_summary(curve: ComplexityCurve | None = None) -> ComplexitySummary:
    """The simulated curve with its closed-form expectation at every degree."""
    simulated = curve if curve is not None else simulate_complexity_curve()
    expected = tuple(expected_errors(degree) for degree in simulated.degrees)
    tests = [row.test for row in expected]
    return ComplexitySummary(
        curve=simulated,
        expected=expected,
        best_degree=best_degree(simulated),
        expected_best_degree=simulated.degrees[int(np.argmin(tests))],
    )


def simulate_lasso_data(*, seed: int = SEED, samples: int = LASSO_SAMPLES) -> LassoData:
    """Draw the features, then the noise, from one generator; standardise the features."""
    rng = np.random.default_rng(count(seed, name="seed"))
    n = count(samples, name="samples", minimum=2)
    beta = np.array(COEFFICIENTS)
    raw = rng.normal(size=(n, beta.size))
    response = raw @ beta + rng.normal(0, LASSO_NOISE_SD, n)
    features = np.asarray(StandardScaler().fit_transform(raw), dtype=np.float64)
    return LassoData(features=features, response=np.asarray(response, dtype=np.float64))


def penalty_grid(
    data: LassoData, *, points: int = PENALTIES, ratio: float = PENALTY_RATIO
) -> NDArray[np.float64]:
    """Penalties from ``max |X^T y| / n`` down to ``ratio`` times it, geometrically spaced."""
    largest = float(np.max(np.abs(data.features.T @ data.response))) / data.response.size
    shrink = positive(ratio, name="ratio")
    if shrink >= 1.0:
        raise ValueError("ratio must be below 1")
    return np.geomspace(largest, largest * shrink, num=count(points, name="points", minimum=2))


def lasso_coefficient_path(data: LassoData | None = None) -> LassoPath:
    """Lasso coefficients along the penalty grid, by coordinate descent."""
    sample = data if data is not None else simulate_lasso_data()
    penalties, coefficients, _ = lasso_path(
        sample.features, sample.response, alphas=penalty_grid(sample)
    )
    return LassoPath(
        penalties=np.asarray(penalties, dtype=np.float64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
    )


def lasso_summary(path: LassoPath | None = None, data: LassoData | None = None) -> LassoSummary:
    """Entry order and penalties, end-point coefficients and the least-squares fit they approach."""
    sample = data if data is not None else simulate_lasso_data()
    result = path if path is not None else lasso_coefficient_path(sample)
    active = result.coefficients != 0
    entered = [j for j in range(active.shape[0]) if active[j].any()]
    first = {j: int(np.argmax(active[j])) for j in entered}
    order = sorted(entered, key=lambda j: first[j])
    final = result.coefficients[:, -1]
    least_squares = np.linalg.lstsq(sample.features, sample.response, rcond=None)[0]
    return LassoSummary(
        largest_penalty=float(result.penalties[0]),
        smallest_penalty=float(result.penalties[-1]),
        entry_order=tuple(j + 1 for j in order),
        entry_penalties=tuple(float(result.penalties[first[j]]) for j in order),
        final_coefficients=tuple(float(c) for c in final),
        least_squares=tuple(float(c) for c in least_squares),
        labelled=tuple(int(j) + 1 for j in np.flatnonzero(np.abs(final) > LABEL_THRESHOLD)),
    )


def example_payload() -> BiasVarianceSummary:
    """The numbers behind both figures."""
    return BiasVarianceSummary(complexity=complexity_summary(), lasso=lasso_summary())
