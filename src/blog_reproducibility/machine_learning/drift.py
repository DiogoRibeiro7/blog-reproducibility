"""Four kinds of drift, for the article on drift in production machine learning.

The article separates what changed from whether it matters. Its four figures
each isolate one piece of that argument, and this module holds the data each
one draws.

**Monitoring architecture.** A diagram, not a simulation: production inputs
pass data-quality checks, then feature monitoring, then prediction monitoring;
mature labels feed outcome monitoring and decision monitoring; both branches
end in a response. The stages and the arrows between them are the model, and
``monitoring_order`` checks that they form a pipeline with one source and one
sink.

**Feature drift**, :math:`P_{train}(X) \\neq P_{prod}(X)`. A sensor scaling
change: 6000 training values from N(0, 1) and 6000 production values from
N(0.85, 1.15^2). For two normals the 1-Wasserstein distance is the mean of a
folded normal, E|0.85 + 0.15 Z|, and the two-sample Kolmogorov-Smirnov
statistic estimates the largest gap between the two CDFs; both are reported
next to the sample means.

**Concept drift**, :math:`P_{train}(Y \\mid X) \\neq P_{prod}(Y \\mid X)`.
Two logistic risk curves with no noise: slope 1.6 and midpoint 0.1 during
training, slope 1.15 and midpoint 1.0 in production. A threshold at the
training midpoint still splits scores at the same place but no longer marks
50% risk.

**Prediction drift.** Ninety days of alert counts from a fixed score threshold:
a seasonal baseline 95 + 8 sin(day / 5.5) plus N(0, 5) noise, and from day 55
a ramp from 25 to 95 extra alerts. Review capacity is 140 a day. The threshold
never changes, so every extra alert comes from the moved score distribution.

The published feature- and prediction-drift images drew from a shared global
generator, so they cannot be reproduced exactly; each simulation here has its
own generator seeded with ``SEED``. The article prints no numbers from these
simulations.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import ks_2samp, norm, wasserstein_distance

from blog_reproducibility.common.validation import count, non_negative, positive, real

__all__ = [
    "ALERT_DAYS",
    "CAPACITY",
    "EDGES",
    "FEATURE_SAMPLE_SIZE",
    "PRODUCTION_CURVE",
    "PRODUCTION_FEATURE",
    "RAMP",
    "SEED",
    "SHIFT_DAY",
    "STAGES",
    "TRAINING_CURVE",
    "TRAINING_FEATURE",
    "AlertSeries",
    "ConceptDriftSummary",
    "DriftSummary",
    "FeatureDriftSample",
    "FeatureDriftSummary",
    "LogisticCurve",
    "NormalFeature",
    "PredictionDriftSummary",
    "concept_drift_summary",
    "event_probability",
    "example_payload",
    "feature_drift_summary",
    "monitoring_order",
    "normal_wasserstein",
    "prediction_drift_summary",
    "simulate_alerts",
    "simulate_feature_drift",
]

SEED: Final[int] = 20260816

# Monitoring architecture: stage names and the arrows between them.
STAGES: Final[tuple[str, ...]] = (
    "Production inputs",
    "Data quality checks",
    "Feature drift monitoring",
    "Prediction monitoring",
    "Labels and outcomes",
    "Decision monitoring",
    "Response",
)
EDGES: Final[tuple[tuple[str, str], ...]] = (
    ("Production inputs", "Data quality checks"),
    ("Data quality checks", "Feature drift monitoring"),
    ("Feature drift monitoring", "Prediction monitoring"),
    ("Prediction monitoring", "Response"),
    ("Feature drift monitoring", "Labels and outcomes"),
    ("Labels and outcomes", "Decision monitoring"),
    ("Decision monitoring", "Response"),
)


@dataclass(frozen=True, slots=True)
class NormalFeature:
    """A normally distributed feature in one period."""

    mean: float
    sd: float


@dataclass(frozen=True, slots=True)
class LogisticCurve:
    """Event probability 1 / (1 + exp(-slope (x - midpoint)))."""

    slope: float
    midpoint: float


FEATURE_SAMPLE_SIZE: Final[int] = 6000
TRAINING_FEATURE: Final[NormalFeature] = NormalFeature(mean=0.0, sd=1.0)
PRODUCTION_FEATURE: Final[NormalFeature] = NormalFeature(mean=0.85, sd=1.15)

TRAINING_CURVE: Final[LogisticCurve] = LogisticCurve(slope=1.6, midpoint=0.1)
PRODUCTION_CURVE: Final[LogisticCurve] = LogisticCurve(slope=1.15, midpoint=1.0)

ALERT_DAYS: Final[int] = 90
BASELINE_LEVEL: Final[float] = 95.0
SEASONAL_AMPLITUDE: Final[float] = 8.0
SEASONAL_PERIOD: Final[float] = 5.5
ALERT_NOISE_SD: Final[float] = 5.0
# The shift starts on day 55 (index 54) and ramps the extra alerts from 25 to 95.
SHIFT_DAY: Final[int] = 55
RAMP: Final[tuple[float, float]] = (25.0, 95.0)
CAPACITY: Final[float] = 140.0


@dataclass(frozen=True, slots=True)
class FeatureDriftSample:
    """Training and production draws of the drifted feature."""

    training: NDArray[np.float64]
    production: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FeatureDriftSummary:
    """Sample moments and distances between the two periods."""

    training_mean: float
    production_mean: float
    training_sd: float
    production_sd: float
    ks_statistic: float
    ks_p_value: float
    wasserstein: float
    population_wasserstein: float


@dataclass(frozen=True, slots=True)
class ConceptDriftSummary:
    """Where the two risk curves sit relative to a threshold fixed at training."""

    threshold: float
    training_risk_at_threshold: float
    production_risk_at_threshold: float
    production_midpoint: float
    crossing: float
    risk_at_crossing: float


@dataclass(frozen=True, slots=True)
class AlertSeries:
    """Daily alerts with and without the distribution shift."""

    days: NDArray[np.int64]
    baseline: NDArray[np.float64]
    shifted: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PredictionDriftSummary:
    """Alert workload before and after the shift, against review capacity."""

    mean_before: float
    mean_after: float
    max_before: float
    max_after: float
    days_over_capacity_before: int
    days_over_capacity_after: int
    first_day_over_capacity: int | None


@dataclass(frozen=True, slots=True)
class DriftSummary:
    """Every number behind the four figures."""

    monitoring_order: tuple[str, ...]
    feature: FeatureDriftSummary
    concept: ConceptDriftSummary
    prediction: PredictionDriftSummary


def monitoring_order(
    stages: tuple[str, ...] = STAGES, edges: tuple[tuple[str, str], ...] = EDGES
) -> tuple[str, ...]:
    """Topological order of the stages; reject cycles, unknown stages, extra sources or sinks."""
    known = set(stages)
    if len(known) != len(stages):
        raise ValueError("stage names must be unique")
    for source, target in edges:
        if source not in known or target not in known:
            raise ValueError(f"edge {source!r} -> {target!r} names an unknown stage")

    incoming = {stage: 0 for stage in stages}
    outgoing: dict[str, list[str]] = {stage: [] for stage in stages}
    for source, target in edges:
        incoming[target] += 1
        outgoing[source].append(target)

    sources = [stage for stage in stages if incoming[stage] == 0]
    sinks = [stage for stage in stages if not outgoing[stage]]
    if len(sources) != 1 or len(sinks) != 1:
        raise ValueError("the pipeline must have exactly one source and one sink")

    order: list[str] = []
    ready = list(sources)
    while ready:
        stage = ready.pop(0)
        order.append(stage)
        for target in outgoing[stage]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if len(order) != len(stages):
        raise ValueError("the pipeline must not contain a cycle")
    return tuple(order)


def simulate_feature_drift(
    *, seed: int = SEED, size: int = FEATURE_SAMPLE_SIZE
) -> FeatureDriftSample:
    """Draw the training period, then the production period, from one generator."""
    rng = np.random.default_rng(count(seed, name="seed"))
    points = count(size, name="size", minimum=1)
    training = rng.normal(TRAINING_FEATURE.mean, TRAINING_FEATURE.sd, points)
    production = rng.normal(PRODUCTION_FEATURE.mean, PRODUCTION_FEATURE.sd, points)
    return FeatureDriftSample(training=training, production=production)


def normal_wasserstein(first: NormalFeature, second: NormalFeature) -> float:
    """1-Wasserstein distance between two normals: the mean of |dmu + dsigma Z|."""
    shift = real(second.mean, name="mean") - real(first.mean, name="mean")
    spread = abs(positive(second.sd, name="sd") - positive(first.sd, name="sd"))
    if spread == 0.0:
        return abs(shift)
    ratio = shift / spread
    folded = spread * np.sqrt(2.0 / np.pi) * np.exp(-(ratio**2) / 2.0) + shift * (
        1.0 - 2.0 * norm.cdf(-ratio)
    )
    return float(folded)


def feature_drift_summary(sample: FeatureDriftSample | None = None) -> FeatureDriftSummary:
    """Moments, KS test, and Wasserstein distance between the two periods."""
    data = sample if sample is not None else simulate_feature_drift()
    test = ks_2samp(data.training, data.production)
    return FeatureDriftSummary(
        training_mean=float(np.mean(data.training)),
        production_mean=float(np.mean(data.production)),
        training_sd=float(np.std(data.training, ddof=1)),
        production_sd=float(np.std(data.production, ddof=1)),
        ks_statistic=float(test.statistic),
        ks_p_value=float(test.pvalue),
        wasserstein=float(wasserstein_distance(data.training, data.production)),
        population_wasserstein=normal_wasserstein(TRAINING_FEATURE, PRODUCTION_FEATURE),
    )


def event_probability(x: ArrayLike, curve: LogisticCurve) -> NDArray[np.float64]:
    """Logistic event probability along ``x``."""
    slope = positive(curve.slope, name="slope")
    midpoint = real(curve.midpoint, name="midpoint")
    points = np.asarray(x, dtype=np.float64)
    return np.asarray(1.0 / (1.0 + np.exp(-slope * (points - midpoint))), dtype=np.float64)


def concept_drift_summary(
    training: LogisticCurve = TRAINING_CURVE, production: LogisticCurve = PRODUCTION_CURVE
) -> ConceptDriftSummary:
    """Risk at the training threshold under each curve, and where the curves cross."""
    threshold = real(training.midpoint, name="midpoint")
    slope_gap = positive(training.slope, name="slope") - positive(production.slope, name="slope")
    if slope_gap == 0.0:
        raise ValueError("curves with equal slopes never cross")
    # Equal logits: slope_t (x - m_t) = slope_p (x - m_p), solved for x.
    crossing = (training.slope * training.midpoint - production.slope * production.midpoint) / (
        slope_gap
    )
    return ConceptDriftSummary(
        threshold=threshold,
        training_risk_at_threshold=float(event_probability(threshold, training)),
        production_risk_at_threshold=float(event_probability(threshold, production)),
        production_midpoint=float(production.midpoint),
        crossing=float(crossing),
        risk_at_crossing=float(event_probability(crossing, training)),
    )


def simulate_alerts(*, seed: int = SEED, noise_sd: float = ALERT_NOISE_SD) -> AlertSeries:
    """Seasonal alert counts, then the same series with the post-shift ramp added."""
    rng = np.random.default_rng(count(seed, name="seed"))
    days = np.arange(1, ALERT_DAYS + 1)
    noise = rng.normal(0, non_negative(noise_sd, name="noise_sd"), days.size)
    baseline = BASELINE_LEVEL + SEASONAL_AMPLITUDE * np.sin(days / SEASONAL_PERIOD) + noise
    shifted = baseline.copy()
    start = SHIFT_DAY - 1
    shifted[start:] += np.linspace(RAMP[0], RAMP[1], days.size - start)
    return AlertSeries(days=days, baseline=baseline, shifted=np.maximum(shifted, 0.0))


def prediction_drift_summary(series: AlertSeries | None = None) -> PredictionDriftSummary:
    """Workload before and after the shift, counted against review capacity."""
    data = series if series is not None else simulate_alerts()
    start = SHIFT_DAY - 1
    before, after = data.shifted[:start], data.shifted[start:]
    over = np.flatnonzero(data.shifted > CAPACITY)
    return PredictionDriftSummary(
        mean_before=float(np.mean(before)),
        mean_after=float(np.mean(after)),
        max_before=float(np.max(before)),
        max_after=float(np.max(after)),
        days_over_capacity_before=int(np.sum(before > CAPACITY)),
        days_over_capacity_after=int(np.sum(after > CAPACITY)),
        first_day_over_capacity=int(data.days[over[0]]) if over.size else None,
    )


def example_payload() -> DriftSummary:
    """The numbers behind all four figures."""
    return DriftSummary(
        monitoring_order=monitoring_order(),
        feature=feature_drift_summary(),
        concept=concept_drift_summary(),
        prediction=prediction_drift_summary(),
    )
