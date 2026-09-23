"""Check the annotator ceiling against the binomial closed form and the article's claims.

The majority-vote error is checked by hand for small panels and by enumerating
every vote pattern. The figure's simulation (seed 0) is then checked against
the closed form within Monte Carlo error, and against the claims in its alt
text. The article's agreement, ceiling, gap and voting tables come from a
different stream of draws, so they are not pinned; its sample-size table
involves no draws and is pinned at the printed precision.
"""

from itertools import product
from math import sqrt

import pytest

from blog_reproducibility.machine_learning.annotator_ceiling import (
    ERROR_RATES,
    ITEMS,
    PANEL_SIZES,
    CeilingCurve,
    ceiling,
    example_payload,
    items_for_power,
    majority_error,
    simulate_ceilings,
)

SUMMARY = example_payload()
CURVES = {curve.annotators: curve for curve in SUMMARY.curves}


def test_majority_error_by_hand() -> None:
    """One annotator errs at e; three err when at least two do: 3e^2(1-e) + e^3."""
    assert majority_error(0.1, 1) == pytest.approx(0.1)
    assert majority_error(0.1, 3) == pytest.approx(3 * 0.01 * 0.9 + 0.001)
    assert majority_error(0.0, 7) == 0.0
    assert majority_error(1.0, 7) == pytest.approx(1.0)


@pytest.mark.parametrize("annotators", [1, 3, 5, 7])
def test_a_coin_flipping_panel_is_right_half_the_time(annotators: int) -> None:
    """At e = 1/2 the vote carries no information, whatever the panel size."""
    assert majority_error(0.5, annotators) == pytest.approx(0.5)


def test_majority_error_matches_enumeration() -> None:
    """Summing the probability of every losing vote pattern gives the same number."""
    error, annotators = 0.2, 5
    total = 0.0
    for pattern in product((0, 1), repeat=annotators):
        wrong = sum(pattern)
        if wrong > annotators / 2:
            total += error**wrong * (1 - error) ** (annotators - wrong)
    assert majority_error(error, annotators) == pytest.approx(total)


def test_simulated_ceilings_match_the_closed_form() -> None:
    """Every simulated point lies within 4.5 binomial standard errors of 1 - P(majority wrong)."""
    for curve in SUMMARY.curves:
        for simulated, exact in zip(curve.simulated, curve.closed_form, strict=True):
            standard_error = sqrt(exact * (1 - exact) / ITEMS)
            assert abs(simulated - exact) <= 4.5 * standard_error + 1e-12


def test_one_annotator_caps_the_score_at_one_minus_the_error_rate() -> None:
    """The alt text's first claim: the single-annotator ceiling is 1 - e."""
    single = CURVES[1]
    assert single.closed_form == pytest.approx([1 - error for error in ERROR_RATES])
    assert single.simulated == pytest.approx([1 - error for error in ERROR_RATES], abs=0.002)


def test_majority_voting_raises_the_ceiling_sharply() -> None:
    """At every error rate each larger panel scores higher; three voters cut 10% error below 3%."""
    for position in range(len(ERROR_RATES)):
        heights = [CURVES[panel].simulated[position] for panel in PANEL_SIZES]
        assert heights == sorted(heights)
        assert len(set(heights)) == len(heights)
    assert 1 - CURVES[3].simulated[ERROR_RATES.index(0.10)] < 0.03
    assert 1 - CURVES[7].simulated[ERROR_RATES.index(0.30)] < 0.13


def test_every_point_is_inside_the_plotted_range() -> None:
    """The figure's y-limits, 0.65 to 1.01, clip nothing."""
    for curve in SUMMARY.curves:
        assert all(0.65 < value <= 1.0 for value in curve.simulated)


def test_sample_size_table_matches_the_article() -> None:
    """785, 969, 1,226 and 2,180 items; 1.2x, 1.6x and 2.8x the clean requirement."""
    assert [round(row.items_needed) for row in SUMMARY.power] == [785, 969, 1226, 2180]
    assert [round(row.measurable_gap, 4) for row in SUMMARY.power] == [0.05, 0.045, 0.04, 0.03]
    assert [round(row.relative_to_clean, 1) for row in SUMMARY.power[1:]] == [1.2, 1.6, 2.8]


def test_required_items_grow_as_one_minus_twice_the_error_squared() -> None:
    """Noise compresses the gap by 1 - 2e, so the requirement scales by (1 - 2e)^-2."""
    for error in (0.05, 0.15, 0.3, 0.45):
        ratio = items_for_power(error) / items_for_power(0.0)
        assert ratio == pytest.approx((1 - 2 * error) ** -2)


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the curves; another seed does not."""

    def small(seed: int) -> tuple[CeilingCurve, ...]:
        return simulate_ceilings(seed, items=2000, error_rates=(0.1, 0.2), panel_sizes=(1, 3))

    assert small(5) == small(5)
    assert small(5) != small(6)


def test_invalid_inputs_are_rejected() -> None:
    """Even panels, impossible error rates, booleans and empty designs are refused."""
    with pytest.raises(ValueError):
        majority_error(0.1, 4)
    with pytest.raises(ValueError):
        ceiling(1.5)
    with pytest.raises(TypeError):
        ceiling(0.1, True)
    with pytest.raises(ValueError):
        items_for_power(0.5)
    with pytest.raises(ValueError):
        items_for_power(0.1, power=1.0)
    with pytest.raises(ValueError):
        simulate_ceilings(items=0)
    with pytest.raises(ValueError):
        simulate_ceilings(panel_sizes=(2,))
