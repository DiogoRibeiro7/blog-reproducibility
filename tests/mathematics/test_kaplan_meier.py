"""Check the Kaplan-Meier estimator against hand-computed products.

Small examples with ties and censoring are worked by hand. Without censoring
the estimator must equal the empirical survival function, which is computed
independently here.
"""

import numpy as np
import pytest

from blog_reproducibility.mathematics.kaplan_meier import (
    example_payload,
    kaplan_meier,
    median_survival,
    simulate_arms,
    survival_at,
)

SUMMARY = example_payload()


def test_hand_computed_example_with_censoring_and_ties() -> None:
    """Times 1, 2+, 3, 3, 4+, 5: survival 5/6, then 5/6 * 2/4, then 0."""
    curve = kaplan_meier([1, 2, 3, 3, 4, 5], [1, 0, 1, 1, 0, 1])

    assert curve.times.tolist() == [0.0, 1.0, 3.0, 5.0]
    assert curve.survival.tolist() == pytest.approx([1.0, 5 / 6, 5 / 6 * 2 / 4, 0.0])


def test_without_censoring_equals_empirical_survival() -> None:
    """With every event observed, S(t) is the share of times beyond t."""
    times = np.random.default_rng(2).exponential(5.0, 50)
    curve = kaplan_meier(times, np.ones(50, dtype=bool))
    grid = np.linspace(0, times.max() + 1, 500)

    empirical = [(times > point).mean() for point in grid]
    np.testing.assert_allclose(survival_at(curve, grid), empirical, atol=1e-12)


def test_censoring_only_leaves_the_curve_at_one() -> None:
    """No events means no steps."""
    curve = kaplan_meier([1.0, 2.0], [0, 0])

    assert curve.survival.tolist() == [1.0]
    assert median_survival(curve) is None


def test_step_function_is_right_continuous() -> None:
    """At an event time the curve already takes its post-event value."""
    curve = kaplan_meier([1.0, 2.0], [1, 1])

    assert survival_at(curve, [0.5, 1.0, 1.5, 2.0]).tolist() == [1.0, 0.5, 0.5, 0.0]


def test_treatment_stays_above_control() -> None:
    """The figure's claim, apart from a sliver before the first control event.

    With this seed the treatment arm's first event precedes control's, so for
    a moment near zero it sits below by less than 0.001, which the plot cannot show.
    """
    control, treatment = simulate_arms()
    grid = np.linspace(0, 40, 40_001)
    gap = survival_at(treatment.curve, grid) - survival_at(control.curve, grid)

    after_start = grid >= 0.2
    control_at_risk = grid < control.observed.max()
    assert np.all(gap[after_start] >= 0)
    assert np.all(gap[after_start & control_at_risk] > 0)
    assert gap.min() > -0.001


def test_arms_follow_their_design() -> None:
    """The longer-lived arm has fewer events and a later median."""
    control, treatment = SUMMARY

    assert control.events + control.censored == treatment.events + treatment.censored == 160
    assert control.events > treatment.events
    assert control.median_survival is not None and treatment.median_survival is not None
    assert control.median_survival < treatment.median_survival
    # Medians land near the exponential median, mean * log 2.
    assert control.median_survival == pytest.approx(control.exact_median, rel=0.25)
    assert treatment.median_survival == pytest.approx(treatment.exact_median, rel=0.25)


def test_invalid_inputs_are_rejected() -> None:
    """Empty, mismatched, negative, and non-finite inputs are refused."""
    with pytest.raises(ValueError):
        kaplan_meier([], [])
    with pytest.raises(ValueError):
        kaplan_meier([1.0, 2.0], [1])
    with pytest.raises(ValueError):
        kaplan_meier([-1.0], [1])
    with pytest.raises(ValueError):
        kaplan_meier([float("inf")], [1])
