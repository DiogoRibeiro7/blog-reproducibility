"""Lorenz curve and Gini coefficient for the article on measuring inequality.

The article's figure draws the Lorenz curve of a simulated lognormal income
distribution and reports its Gini coefficient. The Gini here is the exact
trapezoidal area of the empirical Lorenz curve, which equals the familiar
rank-weighted sample formula, so no numerical integration error enters.

A lognormal distribution with log-scale spread ``sigma`` has the closed-form
Gini ``erf(sigma / 2)``, which gives an independent check on the simulation.
"""

from dataclasses import dataclass
from math import erf
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count, positive, real

__all__ = [
    "LOG_MEAN",
    "LOG_SIGMA",
    "SAMPLE_SIZE",
    "SEED",
    "InequalitySummary",
    "example_payload",
    "gini_coefficient",
    "lognormal_gini",
    "lorenz_curve",
    "simulate_incomes",
]

SEED: Final[int] = 20260816
SAMPLE_SIZE: Final[int] = 20_000
LOG_MEAN: Final[float] = 10.2
LOG_SIGMA: Final[float] = 0.85


@dataclass(frozen=True, slots=True)
class InequalitySummary:
    """The simulated distribution and the Gini coefficient the figure reports."""

    sample_size: int
    log_mean: float
    log_sigma: float
    seed: int
    sample_gini: float
    lognormal_gini: float


def simulate_incomes(
    *,
    size: int = SAMPLE_SIZE,
    log_mean: float = LOG_MEAN,
    log_sigma: float = LOG_SIGMA,
    seed: int = SEED,
) -> NDArray[np.float64]:
    """Draw lognormal incomes from a generator seeded here."""
    rng = np.random.default_rng(count(seed, name="seed"))
    return rng.lognormal(
        mean=real(log_mean, name="log_mean"),
        sigma=positive(log_sigma, name="log_sigma"),
        size=count(size, name="size", minimum=1),
    )


def _sorted_incomes(incomes: ArrayLike) -> NDArray[np.float64]:
    """Validate incomes and return them in ascending order."""
    values = np.asarray(incomes, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("incomes must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("incomes must be finite")
    if np.any(values < 0):
        raise ValueError("incomes must be non-negative")
    if values.sum() <= 0:
        raise ValueError("incomes must have a positive total")
    return np.sort(values)


def lorenz_curve(incomes: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return cumulative population and income shares, both starting at zero."""
    values = _sorted_incomes(incomes)
    income_share = np.concatenate(([0.0], np.cumsum(values) / values.sum()))
    population_share = np.linspace(0.0, 1.0, income_share.size)
    return population_share, income_share


def gini_coefficient(incomes: ArrayLike) -> float:
    """Twice the area between the line of equality and the empirical Lorenz curve.

    For ascending incomes ``x_1..x_n`` the trapezoidal area reduces to
    ``sum((2i - n - 1) x_i) / (n * sum(x))``, evaluated directly here.
    """
    values = _sorted_incomes(incomes)
    size = values.size
    ranks = np.arange(1, size + 1, dtype=np.float64)
    return float(np.sum((2 * ranks - size - 1) * values) / (size * values.sum()))


def lognormal_gini(log_sigma: float) -> float:
    """Population Gini coefficient of a lognormal distribution."""
    return erf(positive(log_sigma, name="log_sigma") / 2)


def example_payload() -> InequalitySummary:
    """Return the numbers behind the article's figure."""
    incomes = simulate_incomes()
    return InequalitySummary(
        sample_size=SAMPLE_SIZE,
        log_mean=LOG_MEAN,
        log_sigma=LOG_SIGMA,
        seed=SEED,
        sample_gini=gini_coefficient(incomes),
        lognormal_gini=lognormal_gini(LOG_SIGMA),
    )
