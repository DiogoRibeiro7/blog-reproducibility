"""Subjects against measurements, for the longitudinal design article.

A study with a fixed budget of observations can spend it on more people or on
longer trajectories, and the two buy different things. In a random-intercept
model with between-subject SD 1 and within-subject SD 3, ``n`` people measured
``T`` times each give:

* a population mean with standard error ``sqrt((1 + 9/T) / n)`` — more people
  help most;
* an individual estimate with RMSE ``3 / sqrt(T)`` — only more measurements
  help;
* a heterogeneity estimate with SD ``sqrt(2/(n-1)) (1 + 9/T)``.

All three are closed forms. The article also reports a simulation that checks
them, and quotes its values to three places with NumPy named, so the simulation
here uses NumPy's PCG64 generator with the same seed and reproduces them.

The simulation is an exact reduction rather than an approximation: the mean of
``T`` independent Gaussian errors is Gaussian with variance ``sigma^2 / T``, so
each person's mean can be drawn directly instead of drawing every visit.

One generator runs the whole table, and each design continues the stream where
the last one left it. That is what the article's code does, and it is what its
quoted values came from, so ``simulate_designs`` takes the generator once and
threads it through rather than reseeding per design. Reseeding would give
different numbers in the third decimal and would no longer match the article.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np

from blog_reproducibility.common.validation import count

__all__ = [
    "BETWEEN_SD",
    "DESIGNS",
    "REPLICATIONS",
    "SEED",
    "WITHIN_SD",
    "DesignRow",
    "example_payload",
    "individual_rmse",
    "population_standard_error",
    "simulate_design",
    "simulate_designs",
    "variance_estimate_sd",
]

BETWEEN_SD: Final[float] = 1.0
WITHIN_SD: Final[float] = 3.0
REPLICATIONS: Final[int] = 10_000
SEED: Final[int] = 20260921
# Each design spends the same budget of 1,000 observations.
DESIGNS: Final[tuple[tuple[int, int], ...]] = ((20, 50), (50, 20), (100, 10), (200, 5))


@dataclass(frozen=True, slots=True)
class DesignRow:
    """One way of spending the budget, analytically and by simulation."""

    subjects: int
    measurements: int
    analytic_population_se: float
    simulated_population_se: float
    analytic_individual_rmse: float
    simulated_individual_rmse: float
    heterogeneity_sd: float

    @property
    def observations(self) -> int:
        """The budget this design spends."""
        return self.subjects * self.measurements


def population_standard_error(subjects: int, measurements: int) -> float:
    """Standard error of the population mean under the random-intercept model."""
    people = count(subjects, name="subjects", minimum=1)
    visits = count(measurements, name="measurements", minimum=1)
    return sqrt((BETWEEN_SD**2 + WITHIN_SD**2 / visits) / people)


def individual_rmse(measurements: int) -> float:
    """Root mean squared error of one person's own mean, without pooling.

    Only the number of measurements enters. Adding people does nothing for it,
    which is the article's point about individual-level questions.
    """
    visits = count(measurements, name="measurements", minimum=1)
    return WITHIN_SD / sqrt(visits)


def variance_estimate_sd(subjects: int, measurements: int) -> float:
    """Standard deviation of the estimated between-subject variance."""
    people = count(subjects, name="subjects", minimum=2)
    visits = count(measurements, name="measurements", minimum=1)
    return sqrt(2 / (people - 1)) * (BETWEEN_SD**2 + WITHIN_SD**2 / visits)


def simulate_design(
    subjects: int,
    measurements: int,
    *,
    replications: int = REPLICATIONS,
    generator: "np.random.Generator | None" = None,
    seed: int = SEED,
) -> tuple[float, float]:
    """Return the simulated population SE and individual RMSE for one design.

    Pass ``generator`` to continue an existing stream; otherwise one is built
    from ``seed`` here, so a single design still repeats exactly and never
    depends on global random state.
    """
    people = count(subjects, name="subjects", minimum=2)
    visits = count(measurements, name="measurements", minimum=1)
    draws = count(replications, name="replications", minimum=2)

    rng = generator if generator is not None else np.random.default_rng(count(seed, name="seed"))
    noise = rng.normal(0.0, WITHIN_SD / sqrt(visits), size=(draws, people))
    theta = rng.normal(0.0, BETWEEN_SD, size=(draws, people))
    means = theta + noise

    return (
        float(means.mean(axis=1).std(ddof=1)),
        float(sqrt(float(np.mean(noise**2)))),
    )


def simulate_designs(
    designs: tuple[tuple[int, int], ...] = DESIGNS,
    *,
    replications: int = REPLICATIONS,
    seed: int = SEED,
) -> tuple[tuple[float, float], ...]:
    """Simulate every design from one generator, in order.

    The stream carries across designs. That is deliberate and is what the
    article's numbers came from; reseeding each design would change them.
    """
    rng = np.random.default_rng(count(seed, name="seed"))
    return tuple(
        simulate_design(
            subjects,
            measurements,
            replications=replications,
            generator=rng,
        )
        for subjects, measurements in designs
    )


def example_payload() -> tuple[DesignRow, ...]:
    """Return the four designs the article compares."""
    simulated = simulate_designs()
    rows: list[DesignRow] = []
    for (subjects, measurements), (simulated_se, simulated_rmse) in zip(
        DESIGNS, simulated, strict=True
    ):
        rows.append(
            DesignRow(
                subjects=subjects,
                measurements=measurements,
                analytic_population_se=population_standard_error(subjects, measurements),
                simulated_population_se=simulated_se,
                analytic_individual_rmse=individual_rmse(measurements),
                simulated_individual_rmse=simulated_rmse,
                heterogeneity_sd=variance_estimate_sd(subjects, measurements),
            )
        )
    return tuple(rows)


def budget_curve(budget: int = 1000) -> tuple[tuple[int, float, float], ...]:
    """Return measurements, population SE, and individual RMSE along one budget."""
    total = count(budget, name="budget", minimum=1)
    return tuple(
        (
            visits,
            sqrt((BETWEEN_SD**2 + WITHIN_SD**2 / visits) / (total / visits)),
            individual_rmse(visits),
        )
        for visits in (2, 5, 10, 20, 25, 50, 100, 200)
    )
