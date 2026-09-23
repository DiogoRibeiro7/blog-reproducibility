"""Learning curves and power-law extrapolation, for the article on whether more data helps.

A learning curve is test error against the number of training examples. For a
wide range of models it follows a power law,

    err(n) = a + b n^(-c),

with ``a`` the floor the model class reaches with unlimited data, ``c`` the rate
of approach and ``b`` a scale. The article fits that law to the five smallest
training sizes and checks its predictions at the three largest.

The task has twelve standard normal features, five of which enter the true
log-odds, including an interaction ``x0 x1`` and a quadratic ``x3^2 - 1`` that a
linear model cannot represent. Because the true probabilities are known, the
Bayes error rate, the mean of ``min(p, 1 - p)``, is known too: it is the floor
no classifier can beat. A logistic regression and a gradient boosting
classifier are trained on three random draws from a pool at each of eight sizes
from 250 to 32,000, and scored on one fixed 50,000-example test set.

The simulation follows the article's code draw for draw. One generator seeded
at 0 draws the test set and then a pool of 96,000 examples. Each training set is
chosen without replacement from the pool by a generator seeded at
``100 * size + draw``, and the boosting model is seeded at 0. The label-noise
run flips ten percent of the pool's labels with a generator seeded at 7 and
reuses the same training indices. Symmetric noise at rate ``e`` leaves the Bayes
decision unchanged; only on noisy *test* labels does the best measurable error
rise, to ``e + (1 - 2 e)`` times the Bayes rate.
"""

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "DRAWS",
    "FEATURES",
    "FITTED_SIZES",
    "FLIP_RATE",
    "MODELS",
    "NOISE_SEED",
    "POOL_SIZE",
    "SEED",
    "TEST_SIZE",
    "TRAINING_SIZES",
    "CurvePoint",
    "LearningCurve",
    "LearningCurveSummary",
    "PowerLawFit",
    "SimulatedTask",
    "bayes_error",
    "example_payload",
    "fit_power_law",
    "flip_labels",
    "learning_curve",
    "make_model",
    "noisy_test_floor",
    "power_law",
    "simulate_examples",
    "simulate_task",
    "training_indices",
    "true_probability",
]

SEED: Final[int] = 0
NOISE_SEED: Final[int] = 7
FEATURES: Final[int] = 12
TEST_SIZE: Final[int] = 50_000
TRAINING_SIZES: Final[tuple[int, ...]] = (250, 500, 1000, 2000, 4000, 8000, 16000, 32000)
POOL_SIZE: Final[int] = 3 * TRAINING_SIZES[-1]
DRAWS: Final[int] = 3
FITTED_SIZES: Final[int] = 5
FLIP_RATE: Final[float] = 0.10
MODELS: Final[tuple[str, ...]] = ("logistic", "boosting")
# Starting point for the noisy curve's fit, as in the article's code.
NOISY_INITIAL_FLOOR: Final[float] = 0.2
# Bounds on (floor, scale, exponent) for the power-law fit.
FIT_BOUNDS: Final[tuple[tuple[float, float, float], tuple[float, float, float]]] = (
    (0.0, 0.0, 0.05),
    (1.0, 100.0, 2.0),
)


@dataclass(frozen=True, slots=True, eq=False)
class SimulatedTask:
    """The fixed test set with its true probabilities, and the pool to train from."""

    test_features: NDArray[np.float64]
    test_labels: NDArray[np.int64]
    test_probabilities: NDArray[np.float64]
    pool_features: NDArray[np.float64]
    pool_labels: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """Errors at one training size, averaged over the random training draws."""

    size: int
    test_error: float
    test_error_sd: float
    training_error: float


@dataclass(frozen=True, slots=True)
class LearningCurve:
    """One model's learning curve."""

    model: str
    points: tuple[CurvePoint, ...]

    @property
    def sizes(self) -> tuple[int, ...]:
        """Training sizes, smallest first."""
        return tuple(point.size for point in self.points)

    @property
    def test_errors(self) -> tuple[float, ...]:
        """Mean test error at each size."""
        return tuple(point.test_error for point in self.points)


@dataclass(frozen=True, slots=True)
class PowerLawFit:
    """Fitted parameters of ``floor + scale * n ** -exponent``."""

    floor: float
    scale: float
    exponent: float

    def predict(self, size: float) -> float:
        """Predicted error at ``size`` training examples."""
        return power_law(size, self.floor, self.scale, self.exponent)


@dataclass(frozen=True, slots=True)
class LearningCurveSummary:
    """The figure's curves and fits, and the article's label-noise run."""

    bayes_error: float
    positive_rate: float
    fitted_sizes: int
    curves: tuple[LearningCurve, ...]
    fits: tuple[PowerLawFit, ...]
    noisy_curve: LearningCurve
    noisy_fit: PowerLawFit
    noisy_test_floor: float

    def curve(self, model: str) -> LearningCurve:
        """The learning curve of one model."""
        return self.curves[MODELS.index(model)]

    def fit(self, model: str) -> PowerLawFit:
        """The power-law fit to one model's smallest sizes."""
        return self.fits[MODELS.index(model)]


def true_probability(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Probability of the positive class under the article's true log-odds."""
    if features.ndim != 2 or features.shape[1] < 5:
        raise ValueError("features must be a matrix with at least five columns")
    x = features
    logit = (
        1.2 * x[:, 0]
        - 1.0 * x[:, 1]
        + 0.8 * x[:, 2]
        + 1.5 * x[:, 0] * x[:, 1]
        + 1.0 * (x[:, 3] ** 2 - 1)
        - 0.6 * x[:, 4]
    )
    return np.asarray(1 / (1 + np.exp(-logit)), dtype=np.float64)


def simulate_examples(
    size: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Draw features, then labels from the true probabilities, in the article's order."""
    features = rng.normal(size=(count(size, name="size", minimum=1), FEATURES))
    probabilities = true_probability(features)
    labels = (rng.uniform(size=features.shape[0]) < probabilities).astype(np.int64)
    return features, labels, probabilities


def simulate_task(seed: int = SEED) -> SimulatedTask:
    """Draw the test set and then the training pool from one seeded generator."""
    rng = np.random.default_rng(count(seed, name="seed"))
    test_features, test_labels, test_probabilities = simulate_examples(TEST_SIZE, rng)
    pool_features, pool_labels, _ = simulate_examples(POOL_SIZE, rng)
    return SimulatedTask(test_features, test_labels, test_probabilities, pool_features, pool_labels)


def bayes_error(probabilities: NDArray[np.float64]) -> float:
    """Error of the rule that predicts the more probable class: mean of min(p, 1 - p)."""
    if probabilities.size == 0 or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be a non-empty array of values in [0, 1]")
    return float(np.mean(np.minimum(probabilities, 1 - probabilities)))


def noisy_test_floor(bayes: float, flip_rate: float = FLIP_RATE) -> float:
    """Best measurable error when a share ``flip_rate`` of test labels is flipped."""
    rate = probability(flip_rate, name="flip_rate")
    return rate + (1 - 2 * rate) * probability(bayes, name="bayes")


def training_indices(size: int, draw: int, pool_size: int = POOL_SIZE) -> NDArray[np.int64]:
    """Pool rows for one training set, chosen by a generator seeded at 100 * size + draw."""
    n = count(size, name="size", minimum=1)
    pool = count(pool_size, name="pool_size", minimum=n)
    rng = np.random.default_rng(100 * n + count(draw, name="draw"))
    return np.asarray(rng.choice(pool, n, replace=False), dtype=np.int64)


def make_model(name: str) -> Any:
    """A fresh, unfitted classifier of the article's two kinds."""
    if name == "logistic":
        return LogisticRegression(max_iter=2000)
    if name == "boosting":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0
        )
    raise ValueError(f"model must be one of {MODELS}")


def flip_labels(
    labels: NDArray[np.int64], flip_rate: float = FLIP_RATE, *, seed: int = NOISE_SEED
) -> NDArray[np.int64]:
    """Flip each binary label independently with probability ``flip_rate``."""
    rate = probability(flip_rate, name="flip_rate")
    flip = np.random.default_rng(count(seed, name="seed")).uniform(size=labels.size) < rate
    return np.where(flip, 1 - labels, labels).astype(np.int64)


def learning_curve(
    model: str,
    task: SimulatedTask,
    *,
    sizes: tuple[int, ...] = TRAINING_SIZES,
    draws: int = DRAWS,
    pool_labels: NDArray[np.int64] | None = None,
) -> LearningCurve:
    """Test and training error at each size, averaged over seeded draws from the pool.

    ``pool_labels`` replaces the pool's labels for training, as in the label-noise
    run; test error is always measured against the clean test labels.
    """
    make_model(model)
    labels = task.pool_labels if pool_labels is None else pool_labels
    if labels.shape != task.pool_labels.shape:
        raise ValueError("pool_labels must have one label per pool row")
    repeats = count(draws, name="draws", minimum=1)
    points = []
    for size in sizes:
        test_errors, training_errors = [], []
        for draw in range(repeats):
            rows = training_indices(size, draw, task.pool_labels.size)
            fitted = make_model(model).fit(task.pool_features[rows], labels[rows])
            test_errors.append(
                1 - float(np.mean(fitted.predict(task.test_features) == task.test_labels))
            )
            training_errors.append(
                1 - float(np.mean(fitted.predict(task.pool_features[rows]) == labels[rows]))
            )
        points.append(
            CurvePoint(
                size=size,
                test_error=float(np.mean(test_errors)),
                test_error_sd=float(np.std(test_errors)),
                training_error=float(np.mean(training_errors)),
            )
        )
    return LearningCurve(model, tuple(points))


def power_law(size: float, floor: float, scale: float, exponent: float) -> float:
    """``floor + scale * size ** -exponent``."""
    return float(floor + scale * positive(size, name="size") ** -exponent)


def fit_power_law(
    sizes: tuple[int, ...],
    errors: tuple[float, ...],
    *,
    initial_floor: float | None = None,
) -> PowerLawFit:
    """Least-squares fit of the power law, bounded as in the article.

    The search starts at a floor of ``initial_floor``, or 90 percent of the error
    at the largest fitted size, with scale 1 and exponent 0.5.
    """
    if len(sizes) != len(errors) or len(sizes) < 4:
        raise ValueError("the fit needs at least four sizes, each with an error")
    x = np.array([positive(size, name="size") for size in sizes])
    y = np.array([probability(error, name="error") for error in errors])
    start = 0.9 * y[-1] if initial_floor is None else probability(initial_floor, name="floor")
    parameters, _ = curve_fit(
        lambda n, a, b, c: a + b * n ** (-c),
        x,
        y,
        p0=[start, 1.0, 0.5],
        bounds=FIT_BOUNDS,
        maxfev=20000,
    )
    floor, scale, exponent = (float(value) for value in parameters)
    return PowerLawFit(floor, scale, exponent)


def _fit_curve(curve: LearningCurve, *, initial_floor: float | None = None) -> PowerLawFit:
    return fit_power_law(
        curve.sizes[:FITTED_SIZES],
        curve.test_errors[:FITTED_SIZES],
        initial_floor=initial_floor,
    )


def example_payload(*, label_noise_sizes: tuple[int, ...] = TRAINING_SIZES) -> LearningCurveSummary:
    """Return the figure's curves and fits, and the boosting curve on noisy labels.

    ``label_noise_sizes`` limits the label-noise run, the slowest part, to fewer
    sizes; the fit to it needs the five smallest.
    """
    task = simulate_task()
    curves = tuple(learning_curve(model, task) for model in MODELS)
    noisy = learning_curve(
        "boosting", task, sizes=label_noise_sizes, pool_labels=flip_labels(task.pool_labels)
    )
    bayes = bayes_error(task.test_probabilities)
    return LearningCurveSummary(
        bayes_error=bayes,
        positive_rate=float(np.mean(task.test_labels)),
        fitted_sizes=FITTED_SIZES,
        curves=curves,
        fits=tuple(_fit_curve(curve) for curve in curves),
        noisy_curve=noisy,
        noisy_fit=_fit_curve(noisy, initial_floor=NOISY_INITIAL_FLOOR),
        noisy_test_floor=noisy_test_floor(bayes),
    )
