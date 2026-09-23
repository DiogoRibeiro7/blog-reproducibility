"""Check synthetic control against its constraints and reproduce the article's numbers.

The weights are checked against the conditions that define them. The seeded
panel then reproduces the article's estimator table and placebo checks. SLSQP
results can move in the last digits between SciPy releases, so values are
compared at the precision the article prints.
"""

import numpy as np
import pytest

from blog_reproducibility.data_science.synthetic_control import (
    DONORS,
    EFFECT,
    POST_MONTHS,
    PRE_MONTHS,
    example_payload,
    placebo_gaps,
    simulate_panel,
    synthetic_weights,
)

SUMMARY = example_payload()


def test_weights_are_convex() -> None:
    """Non-negative weights summing to one: an interpolation of donors, never beyond."""
    outcomes = simulate_panel()
    weights = synthetic_weights(outcomes[0, :PRE_MONTHS], outcomes[1:, :PRE_MONTHS])

    assert weights.shape == (DONORS,)
    assert np.all(weights >= -1e-9)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_weights_recover_an_exact_combination() -> None:
    """A treated path that is a convex mix of donors is matched with those weights."""
    donors = np.random.default_rng(3).normal(size=(4, 30)).cumsum(axis=1)
    target = np.array([0.5, 0.3, 0.2, 0.0])

    weights = synthetic_weights(target @ donors, donors)
    assert weights == pytest.approx(target, abs=1e-3)


def test_estimator_table_matches_the_article() -> None:
    """Before-after finds nothing, difference-in-differences two thirds, SC overshoots a little."""
    assert round(SUMMARY.before_after_effect, 2) == 0.30
    assert round(SUMMARY.difference_in_differences_effect, 2) == -6.64
    assert round(SUMMARY.synthetic_control_effect, 2) == -9.29


def test_fit_matches_the_article() -> None:
    """Four donors carry weight, the largest at 0.38, and the pre-period fit is 1.81."""
    assert SUMMARY.donors_with_weight == 4
    assert [round(weight, 2) for weight in SUMMARY.largest_weights[:2]] == [0.38, 0.30]
    assert round(SUMMARY.pre_period_rmspe, 2) == 1.81


def test_placebo_checks_match_the_article() -> None:
    """The treated ratio ranks first of 21; the placebo in time finds almost nothing."""
    placebo = SUMMARY.placebo_in_space

    assert round(placebo.treated_ratio, 1) == 5.2
    assert round(placebo.donor_median_ratio, 1) == 1.2
    assert round(placebo.donor_max_ratio, 1) == 2.0
    assert placebo.rank == 1
    assert placebo.p_value == pytest.approx(1 / 21)
    assert round(SUMMARY.placebo_in_time_gap, 2) == 0.72


def test_panel_design() -> None:
    """Twenty-one units over 48 months, with the effect applied only to the treated unit."""
    with_effect = simulate_panel()
    without = simulate_panel(effect=0.0)

    assert with_effect.shape == (DONORS + 1, PRE_MONTHS + POST_MONTHS)
    difference = with_effect - without
    assert np.all(difference[0, PRE_MONTHS:] == EFFECT)
    assert np.all(difference[0, :PRE_MONTHS] == 0)
    assert np.all(difference[1:] == 0)


def test_placebo_gaps_have_one_row_per_unit() -> None:
    """Every unit gets its own synthetic control from all the others."""
    gaps = placebo_gaps(simulate_panel(donors=4, seed=2))

    assert gaps.shape == (5, PRE_MONTHS + POST_MONTHS)


def test_invalid_inputs_are_rejected() -> None:
    """Mismatched shapes and empty designs are refused."""
    with pytest.raises(ValueError):
        synthetic_weights(np.zeros(5), np.zeros((3, 4)))
    with pytest.raises(ValueError):
        simulate_panel(donors=0)
