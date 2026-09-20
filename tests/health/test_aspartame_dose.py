"""Check the dose arithmetic and the numbers the aspartame essay quotes.

The website test compared each value against the text of the published article.
That article is in the other repository, so the values it prints are written out
here instead; a change in the model still has to be noticed and explained.
"""

import pytest

from blog_reproducibility.health.aspartame_dose import (
    ADI,
    BLOOD_THRESHOLDS,
    CAN_LITRES,
    DETECTION_LIMIT,
    DRINK,
    METHANOL_SHARE,
    STEGINK,
    cans_to_reach,
    example_payload,
    methanol_from,
    predicted_peak,
)

SUMMARY = example_payload()


def test_methanol_share_is_close_to_the_stoichiometric_value() -> None:
    """Methanol over aspartame by mass: the published 10% rounds the exact figure down."""
    stoichiometric = 32.04 / 294.30
    assert pytest.approx(METHANOL_SHARE, abs=0.01) == stoichiometric
    assert stoichiometric > METHANOL_SHARE


def test_the_kinetic_bound_matches_the_measured_peaks() -> None:
    """The one-compartment ceiling lands inside one SD of the volunteers at every dose."""
    for dose, (observed, sd) in STEGINK.items():
        predicted = predicted_peak(dose)
        assert abs(predicted - observed) < sd
        assert abs(predicted / observed - 1) < 0.10

    # The dose at which nothing was detected is predicted to sit at the detection limit.
    assert predicted_peak(34) == pytest.approx(DETECTION_LIMIT, rel=0.15)


def test_prediction_is_linear_in_the_dose() -> None:
    """The model is one division, so doubling the dose doubles the peak exactly."""
    assert predicted_peak(0) == 0.0
    assert predicted_peak(200) == pytest.approx(2 * predicted_peak(100))
    assert predicted_peak(40) == pytest.approx(40 * METHANOL_SHARE / 0.77)


def test_one_can_is_far_below_every_threshold() -> None:
    """One can at the maximum permitted level is 198 mg of aspartame."""
    aspartame = DRINK["maximum permitted"] * CAN_LITRES
    assert aspartame == pytest.approx(198)

    rise = predicted_peak(aspartame / 70)
    assert rise < 0.5  # below the body's own background
    assert BLOOD_THRESHOLDS["effects on the nervous system"] / rise > 500
    assert SUMMARY.nervous_system_threshold_over_one_can == 544


def test_cans_to_reach_the_acceptable_intake() -> None:
    """The published counts: 14.1 cans for an adult, 4.0 for a 20 kg child."""
    cans = SUMMARY.cans_to_reach_the_adi

    assert cans["70 kg, maximum permitted"] == pytest.approx(40 * 70 / 198, abs=0.05)
    assert cans["20 kg, maximum permitted"] == pytest.approx(40 * 20 / 198, abs=0.05)
    assert cans["70 kg, maximum permitted"] == 14.1
    assert cans["20 kg, maximum permitted"] == 4.0

    # At measured concentrations the article quotes 52 to 95 cans, and 15 to 27 for a child.
    assert 52 <= cans["70 kg, measured mean, 2020"] < 53
    assert 95 <= cans["70 kg, measured mean, 2008"] < 96
    assert 15 <= cans["20 kg, measured mean, 2020"] < 16
    assert 27 <= cans["20 kg, measured mean, 2008"] < 28


def test_published_numbers() -> None:
    """Every value the article prints should come back unchanged."""
    assert SUMMARY.methanol_from_one_can_mg["maximum permitted"] == 19.8
    assert SUMMARY.methanol_from_a_glass_of_juice_mg == {"mean": 35.0, "low": 3.0, "high": 160.0}
    assert SUMMARY.methanol_at_the_adi_70kg_mg == 280
    assert SUMMARY.lethal_dose_70kg_mg == (21_000, 70_000)
    assert SUMMARY.predicted_peak_at_34_mg_per_kg == 4.4
    assert SUMMARY.predicted_peak_at_the_adi == 5.2
    assert SUMMARY.predicted_rise_from_one_can_70kg == 0.37
    assert SUMMARY.phenylalanine_from_one_can_mg == 111
    assert SUMMARY.nutrinet_share_of_adi_percent == 1.7
    assert SUMMARY.high_exposure_share_of_adi_percent == {
        "toddlers": 90,
        "children": 81,
        "adults": 69,
    }

    predicted = [peak.predicted_mg_per_litre for peak in SUMMARY.measured_peaks]
    assert predicted == [13.0, 19.5, 26.0]


def test_the_daily_limit_is_far_below_the_lethal_dose() -> None:
    """A whole day at the limit yields less methanol than the body makes, by a wide margin."""
    at_the_limit = SUMMARY.methanol_at_the_adi_70kg_mg
    lethal_low, _ = SUMMARY.lethal_dose_70kg_mg

    assert lethal_low / at_the_limit == pytest.approx(75, abs=1)
    # And one kilogram of apples is in the same range as a day at the limit.
    assert 400 <= at_the_limit * 5 <= 1400 * 2


def test_invalid_doses_and_concentrations_are_rejected() -> None:
    """Negative masses and zero concentrations have no meaning here."""
    with pytest.raises(ValueError):
        methanol_from(-1.0)
    with pytest.raises(ValueError):
        predicted_peak(-1.0)
    with pytest.raises(ValueError):
        cans_to_reach(ADI["EFSA and JECFA"], 70, 0.0)
    with pytest.raises(ValueError):
        cans_to_reach(ADI["EFSA and JECFA"], 0.0, 600)
    with pytest.raises(TypeError):
        methanol_from(True)
