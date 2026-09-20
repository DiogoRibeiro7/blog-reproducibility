"""Sequential upper-CUSUM calculations used by the blog's change-point article."""

from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Final, Iterable

DEFAULT_SEED: Final[int] = 42
DEFAULT_BASELINE: Final[float] = 0.0
DEFAULT_CHANGED_MEAN: Final[float] = 2.0
DEFAULT_SIGMA: Final[float] = 1.0
DEFAULT_FIRST_CHANGED_INDEX: Final[int] = 60
DEFAULT_SAMPLE_SIZE: Final[int] = 100
DEFAULT_THRESHOLD: Final[float] = 5.0


@dataclass(frozen=True, slots=True)
class CusumResult:
    """Result of an upper one-sided CUSUM calculation."""

    scores: tuple[float, ...]
    alarm_index: int | None


@dataclass(frozen=True, slots=True)
class SequentialCusumExample:
    """Deterministic Gaussian mean-shift example used in the published article."""

    data: tuple[float, ...]
    scores: tuple[float, ...]
    alarm_index: int | None
    first_changed_index: int
    baseline: float
    changed_mean: float
    sigma: float
    target_shift: float
    threshold: float


def _finite_float(value: object, *, name: str) -> float:
    """Return a finite numeric value as float and reject booleans/non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")

    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def upper_cusum(
    data: Iterable[float],
    *,
    baseline: float,
    target_shift: float,
    threshold: float,
) -> CusumResult:
    """Compute a one-sided upper CUSUM for a specified positive mean shift.

    The observation-scale recurrence is

        C_n = max(0, C_(n-1) + X_n - baseline - target_shift / 2).

    The first zero-based index with C_n > threshold is returned as the alarm
    index. Scores continue to be calculated after the first alarm because the
    published example displays the complete trajectory.

    Args:
        data: Finite observations in monitoring order.
        baseline: Pre-change mean.
        target_shift: Positive target increase in the mean.
        threshold: Positive alarm threshold on the observation-scale CUSUM.

    Returns:
        The complete CUSUM score trajectory and first alarm index, if any.

    Raises:
        TypeError: If a supplied parameter or observation is not numeric.
        ValueError: If values are non-finite or positive parameters are invalid.
    """
    mu_0 = _finite_float(baseline, name="baseline")
    delta = _finite_float(target_shift, name="target_shift")
    h = _finite_float(threshold, name="threshold")

    if delta <= 0.0:
        raise ValueError("target_shift must be positive")
    if h <= 0.0:
        raise ValueError("threshold must be positive")

    score = 0.0
    scores: list[float] = []
    alarm_index: int | None = None

    for index, value in enumerate(data):
        observation = _finite_float(value, name=f"data[{index}]")
        score = max(0.0, score + observation - mu_0 - delta / 2.0)
        scores.append(score)

        if alarm_index is None and score > h:
            alarm_index = index

    return CusumResult(scores=tuple(scores), alarm_index=alarm_index)


def seeded_mean_shift_example(*, seed: int = DEFAULT_SEED) -> SequentialCusumExample:
    """Return the deterministic Gaussian example reported in the article."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    rng = Random(seed)
    baseline = DEFAULT_BASELINE
    changed_mean = DEFAULT_CHANGED_MEAN
    sigma = DEFAULT_SIGMA
    first_changed_index = DEFAULT_FIRST_CHANGED_INDEX

    data = tuple(
        [rng.gauss(baseline, sigma) for _ in range(first_changed_index)]
        + [
            rng.gauss(changed_mean, sigma)
            for _ in range(DEFAULT_SAMPLE_SIZE - first_changed_index)
        ]
    )

    target_shift = changed_mean - baseline
    result = upper_cusum(
        data,
        baseline=baseline,
        target_shift=target_shift,
        threshold=DEFAULD_THRESHOLD,
    )

    return SequentialCusumExample(
        data=data,
        scores=result.scores,
        alarm_index=result.alarm_index,
        first_changed_index=first_changed_index,
        baseline=baseline,
        changed_mean=changed_mean,
        sigma=sigma,
        target_shift=target_shift,
        threshold=DEFAULT_THRESHOLD,
    )
