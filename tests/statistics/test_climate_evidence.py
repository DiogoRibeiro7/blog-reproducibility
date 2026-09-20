"""Check the climate distribution shift and the evidence it accumulates.

The closed-form KL divergence is checked against direct numerical integration of
the density ratio, and the information cost of thresholding against the data
processing inequality it has to obey.
"""

from math import log
from statistics import NormalDist

import pytest

from blog_reproducibility.statistics.climate_evidence import (
    COOLER_MEAN,
    SPREAD,
    WARMER_MEAN,
    binary_kl,
    climate_log_ratio,
    climate_tails,
    evidence_after,
    example_payload,
    normal_kl,
)

SUMMARY = example_payload()


def test_normal_kl_matches_density_ratio_integration() -> None:
    """Integrating p(x) log(p(x)/q(x)) reproduces the closed form."""
    samples = 40_000
    for mean_p, sd_p, mean_q, sd_q in (
        (7.0, 5.0, 5.0, 5.0),
        (7.0, 7.0, 5.0, 5.0),
        (5.0, 5.0, 7.0, 7.0),
        (5.0, 5.0, 5.0, 5.0),
    ):
        first, second = NormalDist(mean_p, sd_p), NormalDist(mean_q, sd_q)
        step = 18 * sd_p / samples
        total = 0.0
        for index in range(samples):
            x = mean_p - 9 * sd_p + (index + 0.5) * step
            total += first.pdf(x) * log(first.pdf(x) / second.pdf(x)) * step

        assert total == pytest.approx(normal_kl(mean_p, sd_p, mean_q, sd_q), abs=1e-10)


def test_log_ratio_matches_the_densities_it_comes_from() -> None:
    """The shortcut is the log of the two densities, at every reading."""
    warmer, cooler = NormalDist(WARMER_MEAN, SPREAD), NormalDist(COOLER_MEAN, SPREAD)

    for temperature in (-10.0, 0.0, 6.0, 15.0, 25.0):
        assert climate_log_ratio(temperature) == pytest.approx(
            log(warmer.pdf(temperature) / cooler.pdf(temperature)), abs=1e-12
        )

    # It crosses zero exactly midway between the two means.
    assert climate_log_ratio((COOLER_MEAN + WARMER_MEAN) / 2) == pytest.approx(0.0)
    assert climate_log_ratio(0.0) < 0 < climate_log_ratio(20.0)


def test_processing_loses_information_but_changing_units_does_not() -> None:
    """Thresholding can only lose evidence; rescaling the axis cannot."""
    warmer, cooler = NormalDist(WARMER_MEAN, SPREAD), NormalDist(COOLER_MEAN, SPREAD)
    full = normal_kl(WARMER_MEAN, SPREAD, COOLER_MEAN, SPREAD)

    for threshold in (-10.0, 0.0, 6.0, 15.0, 25.0):
        reduced = binary_kl(warmer.cdf(threshold), cooler.cdf(threshold))
        assert 0 <= reduced <= full

    fahrenheit = normal_kl(
        WARMER_MEAN * 1.8 + 32, SPREAD * 1.8, COOLER_MEAN * 1.8 + 32, SPREAD * 1.8
    )
    assert full == pytest.approx(fahrenheit, abs=1e-12)


def test_kl_identities() -> None:
    """A divergence from a distribution to itself is zero, and none is negative."""
    assert normal_kl(5.0, 5.0, 5.0, 5.0) == pytest.approx(0.0)
    assert binary_kl(0.3, 0.3) == pytest.approx(0.0)
    assert normal_kl(7.0, 5.0, 5.0, 5.0) > 0
    assert binary_kl(0.3, 0.5) > 0

    # For equal spreads the divergence is half the squared standardised gap.
    gap = (WARMER_MEAN - COOLER_MEAN) / SPREAD
    assert normal_kl(WARMER_MEAN, SPREAD, COOLER_MEAN, SPREAD) == pytest.approx(gap**2 / 2)


def test_the_warmer_climate_still_freezes() -> None:
    """The chance of a sub-zero day halves; it does not disappear."""
    cooler, warmer = SUMMARY.tails

    assert cooler.below_freezing == pytest.approx(0.158655, abs=5e-7)
    assert warmer.below_freezing == pytest.approx(0.080757, abs=5e-7)
    assert warmer.below_freezing > 0.05
    assert cooler.below_freezing / warmer.below_freezing == pytest.approx(1.96, abs=0.01)

    # And hot days more than double.
    assert cooler.above_hot_day == pytest.approx(0.022750, abs=5e-7)
    assert warmer.above_hot_day == pytest.approx(0.054799, abs=5e-7)
    assert warmer.above_hot_day > 2 * cooler.above_hot_day


def test_evidence_grows_faster_than_its_own_spread() -> None:
    """The total grows like n while its standard deviation grows like sqrt(n)."""
    rows = [evidence_after(number) for number in (1, 10, 25, 100)]

    assert rows[0].mean == pytest.approx(0.08)
    assert rows[0].standard_deviation == pytest.approx(0.4)
    assert rows[-1].mean == pytest.approx(100 * rows[0].mean)
    assert rows[-1].standard_deviation == pytest.approx(10 * rows[0].standard_deviation)

    wrong_way = [row.probability_of_pointing_the_wrong_way for row in rows]
    assert wrong_way == sorted(wrong_way, reverse=True)
    assert wrong_way[0] > 0.4  # one reading is nearly uninformative
    assert wrong_way[-1] < 0.03  # a season is not


def test_the_per_observation_mean_is_the_divergence() -> None:
    """Each reading contributes exactly the KL divergence, in expectation."""
    assert evidence_after(1).mean == pytest.approx(SUMMARY.full_reading_kl)
    assert evidence_after(37).mean == pytest.approx(37 * SUMMARY.full_reading_kl)


def test_published_numbers() -> None:
    """Every value the article prints should come back unchanged."""
    assert SUMMARY.full_reading_kl == pytest.approx(0.08)
    assert SUMMARY.freezing_indicator_kl == pytest.approx(0.026864, abs=5e-7)
    # Reducing the reading to "did it freeze" throws away two thirds of it.
    assert SUMMARY.freezing_indicator_kl / SUMMARY.full_reading_kl == pytest.approx(0.336, abs=5e-4)

    rows = SUMMARY.evidence
    assert [row.observations for row in rows] == [1, 10, 25, 100]
    assert rows[1].standard_deviation == pytest.approx(1.264911, abs=5e-7)
    assert rows[2].probability_of_pointing_the_wrong_way == pytest.approx(0.158655, abs=5e-7)
    assert rows[3].probability_of_pointing_the_wrong_way == pytest.approx(0.022750, abs=5e-7)


def test_invalid_inputs() -> None:
    """Non-positive spreads and boundary probabilities are rejected."""
    with pytest.raises(ValueError):
        normal_kl(5.0, 0.0, 5.0, 5.0)
    with pytest.raises(ValueError):
        normal_kl(5.0, 5.0, 5.0, -1.0)
    for value in (0.0, 1.0):
        with pytest.raises(ValueError):
            binary_kl(value, 0.5)
        with pytest.raises(ValueError):
            binary_kl(0.5, value)
    with pytest.raises(ValueError):
        climate_tails(5.0, sd=0.0)
    with pytest.raises(ValueError):
        evidence_after(0)
