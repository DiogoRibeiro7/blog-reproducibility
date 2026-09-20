"""Check the CRP measurement arithmetic of the inflammation article.

The critical difference is checked against one a published paper reports from
its own components, and the log-normal interval against the multiplicative
structure it comes from.
"""

from math import exp, log, sqrt

import pytest

from blog_reproducibility.health.inflammation_markers import (
    CANTOS,
    EFLM_WITHIN,
    ESTIMATES,
    MACY,
    Z,
    example_payload,
    lognormal_change_value,
    single_result_interval,
    symmetric_change_value,
)

SUMMARY = example_payload()


def test_critical_difference_reproduces_a_published_one() -> None:
    """Macy's components should give back the critical difference the paper reports."""
    assert SUMMARY.critical_difference_from_macy_percent == pytest.approx(
        MACY["published critical difference"], abs=0.5
    )
    assert SUMMARY.critical_difference_from_macy_percent == 118


def test_symmetric_change_value_identities() -> None:
    """The formula is a root-sum-of-squares scaled by z sqrt(2)."""
    assert symmetric_change_value(0.0) == 0.0
    assert symmetric_change_value(10.0) == pytest.approx(Z * sqrt(2) * 10.0)
    assert symmetric_change_value(3.0, 4.0) == pytest.approx(symmetric_change_value(5.0))


def test_lognormal_rise_and_fall_are_the_same_multiplicative_step() -> None:
    """A 155% rise and a 61% fall are one factor applied in opposite directions."""
    rise, fall = lognormal_change_value(EFLM_WITHIN)

    assert rise > 0 > fall
    up = 1 + rise / 100
    down = 1 + fall / 100
    assert up * down == pytest.approx(1.0)

    # The asymmetry is real: the fall can never exceed 100%, the rise is unbounded.
    assert fall > -100
    assert lognormal_change_value(0.0) == pytest.approx((0.0, 0.0))


def test_lognormal_change_value_against_the_defining_sigma() -> None:
    """Recomputing the log-scale spread by hand gives the same factor."""
    within = 34.7
    sigma = sqrt(log(1 + (within / 100) ** 2))
    expected_factor = exp(Z * sqrt(2) * sigma)
    rise, _ = lognormal_change_value(within)

    assert 1 + rise / 100 == pytest.approx(expected_factor)


def test_single_result_interval_is_multiplicative_around_the_median() -> None:
    """The interval is the median times a factor and its reciprocal."""
    low, high = single_result_interval(3.0, EFLM_WITHIN)

    assert low < 3.0 < high
    assert sqrt(low * high) == pytest.approx(3.0)  # geometric mean is the median
    # Scaling the usual level scales the whole interval.
    scaled_low, scaled_high = single_result_interval(6.0, EFLM_WITHIN)
    assert scaled_low == pytest.approx(2 * low)
    assert scaled_high == pytest.approx(2 * high)
    # No variation means no interval.
    assert single_result_interval(3.0, 0.0) == pytest.approx((3.0, 3.0))


def test_published_numbers() -> None:
    """Every value the article prints should come back unchanged."""
    assert SUMMARY.change_needed_lognormal_percent == (155, -61)
    assert SUMMARY.single_results_from_3_mg_per_litre == (1.5, 5.8)
    assert SUMMARY.single_results_from_1_mg_per_litre == (0.52, 1.94)
    assert SUMMARY.cantos_events_prevented_per_1000 == 6.4
    assert SUMMARY.cantos_extra_fatal_infections_per_1000 == 1.3
    assert SUMMARY.cantos_events_prevented_per_extra_infection == 4.9

    # The band around a usual level of 3 straddles the cut-off of 3 used for risk.
    low, high = SUMMARY.single_results_from_3_mg_per_litre
    assert low < 3.0 < high


def test_cantos_trade_off_follows_from_the_published_rates() -> None:
    """The prevented events and the extra infections come from the same two rates."""
    prevented = CANTOS["placebo"] - CANTOS["150 mg"]
    infections = CANTOS["fatal infection, canakinumab"] - CANTOS["fatal infection, placebo"]

    assert prevented > 0
    assert infections > 0
    assert SUMMARY.cantos_events_prevented_per_extra_infection == pytest.approx(
        prevented / infections, abs=0.05
    )


def test_the_forest_plot_rows_are_well_formed() -> None:
    """Every interval contains its point estimate, and the genetic row spans one."""
    for estimate in ESTIMATES:
        assert estimate.low <= estimate.ratio <= estimate.high
        assert estimate.low > 0

    genetic = ESTIMATES[0]
    assert "Mendelian" in genetic.label
    assert genetic.low <= 1.0 <= genetic.high  # raising CRP itself shows no effect

    # The trials that lowered inflammation by other routes did show one.
    lodoco2 = ESTIMATES[-1]
    assert lodoco2.high < 1.0


def test_invalid_inputs() -> None:
    """Negative variation and non-positive usual levels are rejected."""
    with pytest.raises(ValueError):
        symmetric_change_value(-1.0)
    with pytest.raises(ValueError):
        lognormal_change_value(-1.0)
    with pytest.raises(ValueError):
        single_result_interval(0.0, EFLM_WITHIN)
    with pytest.raises(ValueError):
        single_result_interval(3.0, -1.0)
