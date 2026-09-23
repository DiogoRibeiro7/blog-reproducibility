"""Queue waits against utilisation, for the article on why 90 percent means waiting.

A single first-come-first-served server obeys the Lindley recursion

    W_i = max(0, W_(i-1) + S_(i-1) - A_i),

where ``S`` are service times and ``A`` the gaps between arrivals. It holds for
any arrival and service distributions. For Poisson arrivals the
Pollaczek-Khinchine formula gives the mean wait in multiples of the mean service
time as ``rho / (1 - rho) * (1 + c_s^2) / 2``, where ``c_s^2`` is the squared
coefficient of variation of service; exponential service has ``c_s^2 = 1``.

The article simulates 200,000 exponential jobs at each utilisation from one
generator seeded at 0, and a day with one overloaded hour from a generator
seeded at 3. Both are reproduced draw for draw here.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count, non_negative, probability

__all__ = [
    "HOURS",
    "MINUTES_PER_HOUR",
    "OVERLOAD_HOUR",
    "REPORTED_HOURS",
    "SIMULATED_UTILISATIONS",
    "HourlyWait",
    "QueueSummary",
    "UtilisationRow",
    "example_payload",
    "hourly_means",
    "lindley_waits",
    "mean_wait",
    "overloaded_day",
    "simulate_utilisations",
]

JOBS: Final[int] = 200_000
SIMULATED_UTILISATIONS: Final[tuple[float, ...]] = (0.5, 0.7, 0.8, 0.9, 0.95)
HOURS: Final[int] = 24
MINUTES_PER_HOUR: Final[int] = 60
OVERLOAD_HOUR: Final[int] = 8
OVERLOAD_LOAD: Final[float] = 1.3
BASE_LOAD: Final[float] = 0.8
REPORTED_HOURS: Final[tuple[int, ...]] = (7, 8, 9, 10, 11, 12, 14)


@dataclass(frozen=True, slots=True)
class UtilisationRow:
    """Simulated and theoretical waits at one utilisation, in service times."""

    utilisation: float
    simulated_mean: float
    formula_mean: float
    percentile_95: float


@dataclass(frozen=True, slots=True)
class HourlyWait:
    """Mean wait, in minutes, of jobs arriving in one hour of the day."""

    hour: int
    load: float
    mean_wait: float


@dataclass(frozen=True, slots=True)
class QueueSummary:
    """The simulated steady-state waits and the overloaded day."""

    utilisation: tuple[UtilisationRow, ...]
    overloaded_day: tuple[HourlyWait, ...]


def lindley_waits(interarrivals: ArrayLike, services: ArrayLike) -> NDArray[np.float64]:
    """Waiting time in queue of each job at a single first-come-first-served server."""
    gaps = np.asarray(interarrivals, dtype=np.float64)
    work = np.asarray(services, dtype=np.float64)
    if gaps.ndim != 1 or gaps.shape != work.shape or gaps.size == 0:
        raise ValueError("interarrivals and services must be equal-length, non-empty sequences")
    if np.any(gaps < 0) or np.any(work < 0) or not np.all(np.isfinite(gaps + work)):
        raise ValueError("interarrivals and services must be finite and non-negative")

    # Plain floats: the recursion is sequential, and scalar NumPy indexing is slow.
    waits = [0.0]
    for gap, previous in zip(gaps[1:].tolist(), work[:-1].tolist(), strict=True):
        waits.append(max(0.0, waits[-1] + previous - gap))
    return np.asarray(waits)


def mean_wait(utilisation: float, service_scv: float = 1.0) -> float:
    """Pollaczek-Khinchine mean wait for Poisson arrivals, in mean service times."""
    rho = probability(utilisation, name="utilisation", inclusive=False)
    variability = non_negative(service_scv, name="service_scv")
    return rho / (1 - rho) * (1 + variability) / 2


def simulate_utilisations(
    utilisations: tuple[float, ...] = SIMULATED_UTILISATIONS,
    *,
    jobs: int = JOBS,
    seed: int = 0,
) -> tuple[UtilisationRow, ...]:
    """Simulate exponential arrivals and service, dropping the first tenth as warm-up."""
    size = count(jobs, name="jobs", minimum=10)
    rng = np.random.default_rng(count(seed, name="seed"))
    rows = []
    for utilisation in utilisations:
        formula = mean_wait(utilisation)
        gaps = rng.exponential(1 / utilisation, size)
        services = rng.exponential(1.0, size)
        waits = lindley_waits(gaps, services)[size // 10 :]
        rows.append(
            UtilisationRow(
                utilisation=utilisation,
                simulated_mean=float(waits.mean()),
                formula_mean=formula,
                percentile_95=float(np.percentile(waits, 95)),
            )
        )
    return tuple(rows)


def overloaded_day(
    seed: int = 3,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Arrival minutes, waits, and hourly loads for a day with one overloaded hour.

    Service takes one minute on average; arrivals in each hour are Poisson at
    the hour's load, uniformly placed within it.
    """
    rng = np.random.default_rng(count(seed, name="seed"))
    loads = np.where(np.arange(HOURS) == OVERLOAD_HOUR, OVERLOAD_LOAD, BASE_LOAD)
    arrivals, services = [], []
    for hour in range(HOURS):
        jobs = rng.poisson(loads[hour] * MINUTES_PER_HOUR)
        start = hour * MINUTES_PER_HOUR
        arrivals.append(np.sort(rng.uniform(start, start + MINUTES_PER_HOUR, jobs)))
        services.append(rng.exponential(1.0, jobs))
    times = np.concatenate(arrivals)
    work = np.concatenate(services)
    waits = lindley_waits(np.diff(np.concatenate(([0.0], times))), work)
    return times, waits, loads


def hourly_means(times: NDArray[np.float64], waits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean wait of the jobs arriving in each hour of the day."""
    hours = (times // MINUTES_PER_HOUR).astype(np.intp)
    return np.array([waits[hours == hour].mean() for hour in range(HOURS)])


def example_payload() -> QueueSummary:
    """Return the article's utilisation table and overloaded-day table."""
    times, waits, loads = overloaded_day()
    means = hourly_means(times, waits)
    return QueueSummary(
        utilisation=simulate_utilisations(),
        overloaded_day=tuple(
            HourlyWait(hour, float(loads[hour]), float(means[hour])) for hour in REPORTED_HOURS
        ),
    )
