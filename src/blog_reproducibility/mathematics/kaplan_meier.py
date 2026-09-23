"""Kaplan-Meier estimation, for the article on survival analysis.

At each distinct event time ``t`` the estimator multiplies the running survival
by ``1 - d / r``, where ``d`` is the number of events at ``t`` and ``r`` the number
still at risk, meaning observed at or after ``t``. Censored observations leave
the risk set without contributing an event, which is how they carry partial
information instead of being dropped.

The article's figure simulates two arms of 160 subjects with exponential event
times, mean 9 for control and 15 for treatment, censored by independent
exponential times with mean 20.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count, positive

__all__ = [
    "ARMS",
    "SEED",
    "ArmSummary",
    "SimulatedArm",
    "SurvivalCurve",
    "example_payload",
    "kaplan_meier",
    "median_survival",
    "simulate_arms",
    "survival_at",
]

SEED: Final[int] = 20260816
ARM_SIZE: Final[int] = 160
CENSORING_MEAN: Final[float] = 20.0
# Arm name and mean event time.
ARMS: Final[tuple[tuple[str, float], ...]] = (("Control", 9.0), ("Treatment", 15.0))


@dataclass(frozen=True, slots=True)
class SurvivalCurve:
    """A right-continuous step function starting at survival one at time zero."""

    times: NDArray[np.float64]
    survival: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """What one simulated arm contributes to the figure."""

    name: str
    events: int
    censored: int
    median_survival: float | None
    exact_median: float


@dataclass(frozen=True, slots=True)
class SimulatedArm:
    """Observed times, event indicators, and their Kaplan-Meier curve."""

    name: str
    observed: NDArray[np.float64]
    events: NDArray[np.bool_]
    curve: SurvivalCurve


def kaplan_meier(times: ArrayLike, events: ArrayLike) -> SurvivalCurve:
    """Kaplan-Meier survival estimate from observed times and event indicators."""
    observed = np.asarray(times, dtype=np.float64)
    happened = np.asarray(events, dtype=bool)
    if observed.ndim != 1 or observed.size == 0 or observed.shape != happened.shape:
        raise ValueError("times and events must be equal-length, non-empty sequences")
    if not np.all(np.isfinite(observed)) or np.any(observed < 0):
        raise ValueError("times must be finite and non-negative")

    steps, survival, running = [0.0], [1.0], 1.0
    for time in np.unique(observed[happened]):
        at_risk = int(np.sum(observed >= time))
        deaths = int(np.sum((observed == time) & happened))
        running *= 1 - deaths / at_risk
        steps.append(float(time))
        survival.append(running)
    return SurvivalCurve(np.asarray(steps), np.asarray(survival))


def survival_at(curve: SurvivalCurve, times: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the step function, taking the value just after any step at ``t``."""
    index = np.searchsorted(curve.times, np.asarray(times, dtype=np.float64), side="right") - 1
    return curve.survival[np.maximum(index, 0)]


def median_survival(curve: SurvivalCurve) -> float | None:
    """First time the curve reaches one half or below, if it does."""
    below = np.nonzero(curve.survival <= 0.5)[0]
    return float(curve.times[below[0]]) if below.size else None


def simulate_arms(*, size: int = ARM_SIZE, seed: int = SEED) -> tuple[SimulatedArm, ...]:
    """Simulate both arms with independent exponential censoring."""
    rng = np.random.default_rng(count(seed, name="seed"))
    subjects = count(size, name="size", minimum=1)
    arms = []
    for name, mean in ARMS:
        event_times = rng.exponential(positive(mean, name="mean"), subjects)
        censoring = rng.exponential(CENSORING_MEAN, subjects)
        observed = np.minimum(event_times, censoring)
        events = event_times <= censoring
        arms.append(SimulatedArm(name, observed, events, kaplan_meier(observed, events)))
    return tuple(arms)


def example_payload() -> tuple[ArmSummary, ...]:
    """Return event counts and median survival for each arm."""
    return tuple(
        ArmSummary(
            name=arm.name,
            events=int(arm.events.sum()),
            censored=int((~arm.events).sum()),
            median_survival=median_survival(arm.curve),
            exact_median=mean * float(np.log(2)),
        )
        for arm, (_, mean) in zip(simulate_arms(), ARMS, strict=True)
    )
