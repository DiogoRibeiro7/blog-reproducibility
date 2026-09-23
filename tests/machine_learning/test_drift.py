"""Check the drift models against closed forms, then each figure's claim.

The article prints no numbers from these simulations, so nothing is pinned to
it. The monitoring diagram is checked as a graph: one source, one sink, no
cycle, and the path its alt text describes. Feature drift is checked against
the population distances between the two normals. Concept drift has no noise,
so its risks are hand-computed. Prediction drift is checked against the
deterministic ramp it adds.

One detail the concept-drift figure does not show: the two logistic curves
cross at x = -2.2, so for very low feature values the production risk is the
higher one. The shift is to the right everywhere the threshold matters.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from blog_reproducibility.machine_learning.drift import (
    CAPACITY,
    EDGES,
    FEATURE_SAMPLE_SIZE,
    PRODUCTION_CURVE,
    PRODUCTION_FEATURE,
    RAMP,
    SEED,
    SHIFT_DAY,
    STAGES,
    TRAINING_CURVE,
    TRAINING_FEATURE,
    LogisticCurve,
    NormalFeature,
    concept_drift_summary,
    event_probability,
    example_payload,
    monitoring_order,
    normal_wasserstein,
    simulate_alerts,
    simulate_feature_drift,
)

SUMMARY = example_payload()


def test_monitoring_graph_is_a_pipeline_from_inputs_to_response() -> None:
    """Every stage is ordered, inputs come first, and the response comes last."""
    order = SUMMARY.monitoring_order

    assert sorted(order) == sorted(STAGES)
    assert order[0] == "Production inputs"
    assert order[-1] == "Response"
    position = {stage: index for index, stage in enumerate(order)}
    assert all(position[source] < position[target] for source, target in EDGES)


def test_monitoring_graph_has_the_path_the_alt_text_describes() -> None:
    """Validation, feature, and prediction monitoring in sequence; labels feed decisions."""
    successors = set(EDGES)

    for chain in (
        ("Production inputs", "Data quality checks", "Feature drift monitoring"),
        ("Feature drift monitoring", "Prediction monitoring", "Response"),
        ("Feature drift monitoring", "Labels and outcomes", "Decision monitoring", "Response"),
    ):
        assert all(pair in successors for pair in zip(chain, chain[1:], strict=False))


def test_monitoring_order_rejects_malformed_graphs() -> None:
    """Cycles, unknown stages, and a second sink are refused."""
    with pytest.raises(ValueError, match="cycle"):
        monitoring_order(("a", "b", "c", "d"), (("a", "b"), ("b", "c"), ("c", "b"), ("c", "d")))
    with pytest.raises(ValueError, match="unknown"):
        monitoring_order(("a", "b"), (("a", "z"),))
    with pytest.raises(ValueError, match="one source and one sink"):
        monitoring_order(("a", "b", "c"), (("a", "b"), ("a", "c")))
    with pytest.raises(ValueError, match="unique"):
        monitoring_order(("a", "a"), ())


def test_normal_wasserstein_matches_hand_worked_cases() -> None:
    """Equal spreads give |shift|; equal means give |dsigma| sqrt(2 / pi)."""
    assert normal_wasserstein(NormalFeature(0, 1), NormalFeature(2, 1)) == pytest.approx(2.0)
    assert normal_wasserstein(NormalFeature(0, 1), NormalFeature(0, 3)) == pytest.approx(
        2 * math.sqrt(2 / math.pi)
    )


def test_normal_wasserstein_matches_the_quantile_integral() -> None:
    """W1 is the integral of the gap between quantile functions."""
    first, second = NormalFeature(0.0, 1.0), NormalFeature(0.2, 1.8)
    u = (np.arange(200_000) + 0.5) / 200_000
    gap = np.abs(norm.ppf(u, second.mean, second.sd) - norm.ppf(u, first.mean, first.sd))

    assert normal_wasserstein(first, second) == pytest.approx(float(gap.mean()), rel=1e-3)


def test_feature_sample_matches_its_design() -> None:
    """The sample moments sit close to the design means and spreads."""
    feature = SUMMARY.feature
    standard_error = 1.15 / math.sqrt(FEATURE_SAMPLE_SIZE)

    assert feature.training_mean == pytest.approx(TRAINING_FEATURE.mean, abs=4 * standard_error)
    assert feature.production_mean == pytest.approx(PRODUCTION_FEATURE.mean, abs=4 * standard_error)
    assert feature.training_sd == pytest.approx(TRAINING_FEATURE.sd, rel=0.05)
    assert feature.production_sd == pytest.approx(PRODUCTION_FEATURE.sd, rel=0.05)


def test_feature_distances_match_the_population() -> None:
    """Sample KS and Wasserstein estimate the gaps between the two normal distributions."""
    grid = np.linspace(-8, 8, 200_001)
    population_ks = float(
        np.max(
            np.abs(
                norm.cdf(grid, TRAINING_FEATURE.mean, TRAINING_FEATURE.sd)
                - norm.cdf(grid, PRODUCTION_FEATURE.mean, PRODUCTION_FEATURE.sd)
            )
        )
    )
    feature = SUMMARY.feature

    assert feature.ks_statistic == pytest.approx(population_ks, abs=0.03)
    assert feature.population_wasserstein == pytest.approx(0.85, abs=1e-6)
    assert feature.wasserstein == pytest.approx(feature.population_wasserstein, abs=0.05)


def test_feature_drift_is_detected() -> None:
    """The figure's claim: the input distribution moved, and a KS test sees it."""
    feature = SUMMARY.feature

    assert feature.production_mean - feature.training_mean > 0.75
    assert feature.ks_p_value < 1e-100


def test_feature_simulation_is_deterministic_and_draws_training_first() -> None:
    """Same seed, same draws; the training block comes before the production block."""
    first, second = simulate_feature_drift(), simulate_feature_drift()
    rng = np.random.default_rng(SEED)

    np.testing.assert_array_equal(first.training, second.training)
    np.testing.assert_array_equal(first.training, rng.normal(0.0, 1.0, FEATURE_SAMPLE_SIZE))
    np.testing.assert_array_equal(first.production, rng.normal(0.85, 1.15, FEATURE_SAMPLE_SIZE))


def test_concept_drift_risks_are_hand_computed() -> None:
    """At the training threshold 0.1: 50% then, 1 / (1 + e^1.035) now."""
    concept = SUMMARY.concept

    assert concept.threshold == 0.1
    assert concept.training_risk_at_threshold == pytest.approx(0.5)
    assert concept.production_risk_at_threshold == pytest.approx(1 / (1 + math.exp(1.035)))
    assert round(concept.production_risk_at_threshold, 3) == 0.262


def test_concept_curves_cross_where_the_logits_agree() -> None:
    """1.6 (x - 0.1) = 1.15 (x - 1.0) at x = -2.2, where both give the same risk."""
    concept = SUMMARY.concept

    assert concept.crossing == pytest.approx(-2.2)
    at_crossing = [
        float(event_probability(concept.crossing, curve))
        for curve in (TRAINING_CURVE, PRODUCTION_CURVE)
    ]
    assert at_crossing == pytest.approx([concept.risk_at_crossing] * 2)


def test_same_threshold_means_different_risk() -> None:
    """The figure's claim: right of the crossing, production risk is below training risk."""
    x = np.linspace(-2.1, 4, 500)

    assert np.all(event_probability(x, PRODUCTION_CURVE) < event_probability(x, TRAINING_CURVE))
    assert SUMMARY.concept.production_midpoint > SUMMARY.concept.threshold


def test_concept_validation() -> None:
    """Equal slopes never cross; a non-positive slope is refused."""
    with pytest.raises(ValueError):
        concept_drift_summary(LogisticCurve(1.0, 0.0), LogisticCurve(1.0, 1.0))
    with pytest.raises(ValueError):
        event_probability(0.0, LogisticCurve(0.0, 0.0))


def test_alert_shift_adds_exactly_the_ramp() -> None:
    """Before day 55 the series is untouched; after it the extra alerts average (25 + 95) / 2."""
    series = simulate_alerts()
    start = SHIFT_DAY - 1
    extra = series.shifted - series.baseline

    assert series.days[start] == SHIFT_DAY
    np.testing.assert_array_equal(extra[:start], 0.0)
    assert extra[start] == pytest.approx(RAMP[0])
    assert extra[-1] == pytest.approx(RAMP[1])
    assert float(np.mean(extra[start:])) == pytest.approx(sum(RAMP) / 2)


def test_alert_baseline_without_noise_is_the_seasonal_curve() -> None:
    """Zero noise leaves 95 + 8 sin(day / 5.5)."""
    series = simulate_alerts(noise_sd=0.0)

    np.testing.assert_allclose(series.baseline, 95 + 8 * np.sin(series.days / 5.5))
    with pytest.raises(ValueError):
        simulate_alerts(noise_sd=-1.0)


def test_workload_crosses_capacity_only_after_the_shift() -> None:
    """The figure's claim: alert volume rises past review capacity after the shift."""
    prediction = SUMMARY.prediction

    assert prediction.days_over_capacity_before == 0
    assert prediction.max_before < CAPACITY
    assert prediction.days_over_capacity_after > 0
    assert prediction.first_day_over_capacity is not None
    assert prediction.first_day_over_capacity > SHIFT_DAY
    assert prediction.mean_after - prediction.mean_before > 50
