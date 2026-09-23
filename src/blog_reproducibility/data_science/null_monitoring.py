"""Null-rate against metric monitoring, for the article on silent data-quality failures.

Each day 20,000 rows arrive from three customer segments with different means.
Some rows arrive null, and the heavier segments lose theirs more often, so
dropping nulls biases the metric. Two monitors watch the same feed, each firing
when its statistic moves more than three standard deviations from its own quiet
history: one on the metric, one on the share of rows that are null.

The null rate moves as soon as rows start going missing; the metric moves only
once enough of the heavy segment is lost to shift the average. The figure shows
how often each monitor fires as the null rate rises from its quiet 2 percent.
The generator is the figure's own, seeded at 79 and shared by the quiet history
and every day that follows, reproduced draw for draw.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from blog_reproducibility.common.validation import count, probability

__all__ = [
    "NULL_RATES",
    "QUIET_NULL_RATE",
    "DetectionRow",
    "MonitorBaseline",
    "MonitoringSummary",
    "detection_rates",
    "example_payload",
    "quiet_baseline",
    "simulate_day",
]

ROWS: Final[int] = 20_000
SEGMENT_MEANS: Final[tuple[float, ...]] = (12.0, 18.0, 30.0)
SEGMENT_SHARES: Final[tuple[float, ...]] = (0.55, 0.30, 0.15)
# Each segment's null probability relative to the overall rate.
SEGMENT_NULL_WEIGHTS: Final[tuple[float, ...]] = (0.6, 1.0, 2.2)
VALUE_SD: Final[float] = 6.0
QUIET_NULL_RATE: Final[float] = 0.02
QUIET_DAYS: Final[int] = 60
DAYS_PER_RATE: Final[int] = 300
NULL_RATES: Final[tuple[float, ...]] = tuple(float(rate) for rate in np.linspace(0.02, 0.16, 15))
THRESHOLD_SDS: Final[float] = 3.0
SEED: Final[int] = 79


@dataclass(frozen=True, slots=True)
class MonitorBaseline:
    """Means and day-to-day spreads of both statistics over the quiet history."""

    metric_mean: float
    metric_sd: float
    null_rate_mean: float
    null_rate_sd: float


@dataclass(frozen=True, slots=True)
class DetectionRow:
    """Share of days each monitor fires at one null rate."""

    null_rate: float
    metric_monitor: float
    null_rate_monitor: float


@dataclass(frozen=True, slots=True)
class MonitoringSummary:
    """The quiet baseline and the detection curves the figure plots."""

    baseline: MonitorBaseline
    detection: tuple[DetectionRow, ...]


def simulate_day(
    null_rate: float, rng: np.random.Generator, *, rows: int = ROWS
) -> tuple[float, float]:
    """One day's metric, dropping nulls, and its share of null rows."""
    rate = probability(null_rate, name="null_rate")
    size = count(rows, name="rows", minimum=1)
    segment = rng.choice(len(SEGMENT_MEANS), size, p=SEGMENT_SHARES)
    values = np.asarray(SEGMENT_MEANS)[segment] + rng.normal(0, VALUE_SD, size)
    chance = np.clip(rate * np.asarray(SEGMENT_NULL_WEIGHTS)[segment], 0, 1)
    missing = rng.random(size) < chance
    if missing.all():
        raise ValueError("every row is null, so the metric is undefined")
    return float(values[~missing].mean()), float(missing.mean())


def quiet_baseline(rng: np.random.Generator, *, days: int = QUIET_DAYS) -> MonitorBaseline:
    """Both statistics over quiet days at the normal null rate."""
    history = np.array(
        [simulate_day(QUIET_NULL_RATE, rng) for _ in range(count(days, name="days", minimum=2))]
    )
    return MonitorBaseline(
        metric_mean=float(history[:, 0].mean()),
        metric_sd=float(history[:, 0].std()),
        null_rate_mean=float(history[:, 1].mean()),
        null_rate_sd=float(history[:, 1].std()),
    )


def detection_rates(
    baseline: MonitorBaseline,
    null_rates: tuple[float, ...],
    rng: np.random.Generator,
    *,
    days: int = DAYS_PER_RATE,
) -> tuple[DetectionRow, ...]:
    """Share of simulated days on which each monitor crosses three standard deviations."""
    rows = []
    for rate in null_rates:
        runs: NDArray[np.float64] = np.array(
            [simulate_day(rate, rng) for _ in range(count(days, name="days", minimum=1))]
        )
        metric = np.abs(runs[:, 0] - baseline.metric_mean) > THRESHOLD_SDS * baseline.metric_sd
        nulls = np.abs(runs[:, 1] - baseline.null_rate_mean) > THRESHOLD_SDS * baseline.null_rate_sd
        rows.append(DetectionRow(rate, float(metric.mean()), float(nulls.mean())))
    return tuple(rows)


def example_payload(*, days: int = DAYS_PER_RATE) -> MonitoringSummary:
    """Return the baseline and detection curves behind the figure."""
    rng = np.random.default_rng(SEED)
    baseline = quiet_baseline(rng)
    return MonitoringSummary(baseline, detection_rates(baseline, NULL_RATES, rng, days=days))
