"""Check the aether-drift prediction for a rotating interferometer.

Nothing here reproduces a measurement. The checks are on the physics: that the
exact and leading-order expressions agree where they should, that the effect is
second order in the speed ratio, that the inversion undoes the prediction, and
that the scale comes out where the standard account puts it.
"""

from math import sqrt

import pytest

from blog_reproducibility.physics.michelson_morley import (
    ARM_LENGTH_METRES,
    EARTH_ORBITAL_SPEED_M_PER_S,
    SODIUM_WAVELENGTH_METRES,
    SPEED_OF_LIGHT_M_PER_S,
    example_payload,
    expected_fringe_shift,
    leading_order_fringe_shift,
    prediction,
    speed_from_fringe_shift,
)

SUMMARY = example_payload()


def test_exact_and_leading_order_differ_by_exactly_the_next_term() -> None:
    """Expanding the exact form gives a relative correction of 5 beta^2 / 4.

    At orbital speed that is about 1.2e-8, which is why the leading-order
    expression is the one worth quoting.
    """
    for speed in (10_000.0, EARTH_ORBITAL_SPEED_M_PER_S, 100_000.0):
        beta = speed / SPEED_OF_LIGHT_M_PER_S
        exact = expected_fringe_shift(speed)
        leading = leading_order_fringe_shift(speed)

        assert exact > leading
        assert exact / leading - 1 == pytest.approx(1.25 * beta**2, rel=1e-3)

    assert expected_fringe_shift() / leading_order_fringe_shift() - 1 < 1e-7


def test_exact_expression_against_the_light_travel_times_written_out() -> None:
    """Recomputing both arms from their travel times gives the same shift.

    Written as a difference of travel times the calculation cancels badly: the
    two arms agree to eight places at orbital speed. The agreement below is
    therefore only to about one part in a million, which is the cancellation
    and not a disagreement about the physics.
    """
    for speed in (3_000.0, EARTH_ORBITAL_SPEED_M_PER_S, 3_000_000.0):
        beta = speed / SPEED_OF_LIGHT_M_PER_S
        length = ARM_LENGTH_METRES

        along = 2 * length / (1 - beta**2)
        across = 2 * length / sqrt(1 - beta**2)
        # A quarter turn swaps the arms, so the fringes move by twice the difference.
        naive = 2 * (along - across) / SODIUM_WAVELENGTH_METRES

        assert expected_fringe_shift(speed) == pytest.approx(naive, rel=1e-5)


def test_the_stable_identity_matches_the_naive_difference_where_it_can() -> None:
    """At a speed where nothing cancels, both routes agree to full precision."""
    speed = 0.5 * SPEED_OF_LIGHT_M_PER_S
    beta = 0.5
    length, wavelength = ARM_LENGTH_METRES, SODIUM_WAVELENGTH_METRES

    naive = 4 * length / wavelength * (1 / (1 - beta**2) - 1 / sqrt(1 - beta**2))
    assert expected_fringe_shift(speed) == pytest.approx(naive, rel=1e-12)


def test_the_effect_is_second_order_in_the_speed_ratio() -> None:
    """Doubling the speed quadruples the predicted shift."""
    base = expected_fringe_shift(10_000.0)
    doubled = expected_fringe_shift(20_000.0)

    assert doubled / base == pytest.approx(4.0, rel=1e-6)
    # And it is linear in the arm length and inverse in the wavelength.
    assert expected_fringe_shift(arm_length_metres=22.0) == pytest.approx(
        2 * expected_fringe_shift(arm_length_metres=11.0)
    )
    assert expected_fringe_shift(wavelength_metres=2 * SODIUM_WAVELENGTH_METRES) == pytest.approx(
        expected_fringe_shift() / 2
    )


def test_a_stationary_apparatus_predicts_nothing() -> None:
    """The shift vanishes as the speed goes to zero."""
    assert expected_fringe_shift(1e-6) == pytest.approx(0.0, abs=1e-20)
    assert leading_order_fringe_shift(1e-6) == pytest.approx(0.0, abs=1e-20)


def test_the_inversion_undoes_the_prediction() -> None:
    """Feeding a predicted shift back gives the speed that produced it."""
    for speed in (3_000.0, 10_000.0, EARTH_ORBITAL_SPEED_M_PER_S):
        shift = leading_order_fringe_shift(speed)
        assert speed_from_fringe_shift(shift) == pytest.approx(speed, rel=1e-9)

    # A null result at a hundredth of a fringe bounds the speed well below orbital.
    bound = SUMMARY.speed_implied_by_the_resolvable_shift
    assert bound < EARTH_ORBITAL_SPEED_M_PER_S / 5


def test_the_predicted_scale_matches_the_standard_account() -> None:
    """Eleven metres of arm and sodium light give about four tenths of a fringe."""
    row = SUMMARY.at_earth_orbital_speed

    assert row.beta == pytest.approx(9.93e-5, rel=0.01)
    assert row.exact_shift == pytest.approx(0.37, abs=0.01)
    assert row.leading_order_shift == pytest.approx(row.exact_shift, rel=1e-6)

    # Well above a hundredth of a fringe, which is what makes the null result mean something.
    assert row.exact_shift > 30 * SUMMARY.resolvable_shift


def test_the_prediction_rises_with_speed() -> None:
    """Every faster hypothesis predicts a larger shift."""
    shifts = [row.exact_shift for row in SUMMARY.by_speed.values()]
    assert shifts == sorted(shifts)
    assert all(shift > 0 for shift in shifts)


def test_prediction_bundles_both_forms() -> None:
    """The bundled record carries the same values the functions return."""
    row = prediction(10_000.0)

    assert row.speed_m_per_s == 10_000.0
    assert row.exact_shift == expected_fringe_shift(10_000.0)
    assert row.leading_order_shift == leading_order_fringe_shift(10_000.0)
    assert row.beta == pytest.approx(10_000.0 / SPEED_OF_LIGHT_M_PER_S)


def test_invalid_inputs() -> None:
    """Non-positive lengths and speeds at or above light speed are rejected."""
    with pytest.raises(ValueError):
        expected_fringe_shift(0.0)
    with pytest.raises(ValueError):
        expected_fringe_shift(SPEED_OF_LIGHT_M_PER_S)
    with pytest.raises(ValueError):
        expected_fringe_shift(arm_length_metres=0.0)
    with pytest.raises(ValueError):
        expected_fringe_shift(wavelength_metres=-1.0)
    with pytest.raises(ValueError):
        speed_from_fringe_shift(0.0)
    with pytest.raises(ValueError):
        example_payload(resolvable_shift=0.0)
