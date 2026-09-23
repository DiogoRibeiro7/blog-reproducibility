"""Optimism of the best validation score, for the article on the winner's curse.

Each of ``k`` candidate configurations has a true accuracy, and a validation set
of ``n`` items scores it as a binomial proportion. Any one score is unbiased,
but the largest of ``k`` scores is not: the winner tends to be a candidate whose
validation error happened to be positive. The optimism is the winner's
validation score minus its true accuracy.

When all candidates are equally good the optimism has a closed form: the
standard error of one score, ``sqrt(p (1 - p) / n)``, times the expected maximum
of ``k`` standard normal variables,

    E[max] = integral of x k phi(x) Phi(x)^(k - 1) dx.

Candidates that genuinely differ reduce the optimism, because a better
candidate wins on merit more often than on luck.

The figure draws true accuracies from a normal distribution centred on 80
percent with a one-point spread, clipped to [0.5, 0.99], and averages the
optimism over 3,000 replications for each number of candidates and each of
three validation sizes, all from one generator seeded at 0. That simulation is
reproduced draw for draw. The article's tables use 4,000 replications and a
different order of draws, so they are not reproduced; the closed-form numbers
the article quotes are.
"""

from dataclasses import dataclass
from math import inf, sqrt
from typing import Final

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from blog_reproducibility.common.validation import count, non_negative, probability

__all__ = [
    "BASE_ACCURACY",
    "CANDIDATES",
    "REPLICATIONS",
    "SEED",
    "SPREAD",
    "VALIDATION_SIZES",
    "OptimismCurve",
    "WinnersCurseSummary",
    "equal_candidates_optimism",
    "example_payload",
    "expected_maximum_of_normals",
    "optimism_curves",
    "selection_optimism",
    "validation_standard_error",
]

SEED: Final[int] = 0
CANDIDATES: Final[tuple[int, ...]] = (1, 2, 5, 10, 20, 50, 100, 200, 500)
VALIDATION_SIZES: Final[tuple[int, ...]] = (200, 1000, 5000)
REPLICATIONS: Final[int] = 3000
BASE_ACCURACY: Final[float] = 0.80
SPREAD: Final[float] = 0.01
ACCURACY_LIMITS: Final[tuple[float, float]] = (0.5, 0.99)


@dataclass(frozen=True, slots=True)
class OptimismCurve:
    """Simulated optimism, in accuracy points, for one validation size."""

    validation_size: int
    candidates: tuple[int, ...]
    optimism: tuple[float, ...]
    equal_candidates_bound: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class WinnersCurseSummary:
    """The figure's curves and the closed-form numbers the article quotes."""

    curves: tuple[OptimismCurve, ...]
    expected_maxima: tuple[tuple[int, float], ...]
    standard_error_points: float
    equal_candidates_optimism_points: float


def expected_maximum_of_normals(candidates: int) -> float:
    """Expected maximum of ``candidates`` independent standard normal variables."""
    k = count(candidates, name="candidates", minimum=1)
    if k == 1:
        return 0.0

    def integrand(x: float) -> float:
        return float(x * k * norm.pdf(x) * norm.cdf(x) ** (k - 1))

    value, _ = quad(integrand, -inf, inf)
    return float(value)


def validation_standard_error(validation_size: int, accuracy: float = BASE_ACCURACY) -> float:
    """Standard error of one validation accuracy, as a proportion."""
    size = count(validation_size, name="validation_size", minimum=1)
    p = probability(accuracy, name="accuracy")
    return sqrt(p * (1 - p) / size)


def equal_candidates_optimism(
    candidates: int, validation_size: int, accuracy: float = BASE_ACCURACY
) -> float:
    """Closed-form optimism in accuracy points when every candidate is equally good."""
    return (
        100
        * validation_standard_error(validation_size, accuracy)
        * expected_maximum_of_normals(candidates)
    )


def selection_optimism(
    candidates: int,
    validation_size: int,
    rng: np.random.Generator,
    *,
    replications: int = REPLICATIONS,
    base_accuracy: float = BASE_ACCURACY,
    spread: float = SPREAD,
) -> float:
    """Mean of the winner's validation score minus its true accuracy, in points."""
    k = count(candidates, name="candidates", minimum=1)
    n = count(validation_size, name="validation_size", minimum=1)
    reps = count(replications, name="replications", minimum=1)
    centre = probability(base_accuracy, name="base_accuracy")
    scale = non_negative(spread, name="spread")
    low, high = ACCURACY_LIMITS

    total = 0.0
    for _ in range(reps):
        true = np.clip(rng.normal(centre, scale, k), low, high)
        validation = rng.binomial(n, true) / n
        winner = validation.argmax()
        total += float(validation[winner] - true[winner])
    return 100 * total / reps


def optimism_curves(
    seed: int = SEED,
    *,
    candidates: tuple[int, ...] = CANDIDATES,
    validation_sizes: tuple[int, ...] = VALIDATION_SIZES,
    replications: int = REPLICATIONS,
) -> tuple[OptimismCurve, ...]:
    """Simulate the optimism for every validation size and number of candidates."""
    rng = np.random.default_rng(count(seed, name="seed"))
    return tuple(
        OptimismCurve(
            validation_size=n,
            candidates=candidates,
            optimism=tuple(
                selection_optimism(k, n, rng, replications=replications) for k in candidates
            ),
            equal_candidates_bound=tuple(equal_candidates_optimism(k, n) for k in candidates),
        )
        for n in validation_sizes
    )


def example_payload() -> WinnersCurseSummary:
    """Return the figure's curves and the article's closed-form numbers."""
    return WinnersCurseSummary(
        curves=optimism_curves(),
        expected_maxima=tuple((k, expected_maximum_of_normals(k)) for k in (5, 20, 100)),
        standard_error_points=100 * validation_standard_error(1000),
        equal_candidates_optimism_points=equal_candidates_optimism(100, 1000),
    )
