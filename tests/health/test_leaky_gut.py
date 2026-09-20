"""Check the surrogate-test and absolute-risk arithmetic of the leaky-gut article.

The tail-agreement integral is checked against a seeded simulation of correlated
pairs, and the absolute risks against the weighted average they must reproduce.
"""

from math import atanh, sqrt, tanh
from random import Random

import pytest

from blog_reproducibility.common.bivariate import STANDARD
from blog_reproducibility.health.leaky_gut import (
    POWER,
    TOP,
    TURPIN,
    ZHOU,
    correlation_interval,
    example_payload,
    risks_by_test,
    share_truly_high,
)

SUMMARY = example_payload()


def test_share_truly_high_against_a_seeded_simulation() -> None:
    """Simulating correlated pairs reproduces the integral."""
    rng = Random(20260603)
    correlation = 0.5
    cut = STANDARD.inv_cdf(1 - TOP)
    positive = both = 0

    for _ in range(300_000):
        test = rng.gauss(0.0, 1.0)
        if test > cut:
            positive += 1
            truth = correlation * test + sqrt(1 - correlation**2) * rng.gauss(0.0, 1.0)
            both += truth > cut

    assert share_truly_high(correlation) == pytest.approx(both / positive, abs=0.012)


def test_share_truly_high_at_the_ends() -> None:
    """With no correlation the test is worth nothing; with full correlation it is exact."""
    assert share_truly_high(0.0) == pytest.approx(TOP, abs=1e-6)
    assert share_truly_high(1.0) == 1.0

    values = [share_truly_high(value) for value in (0.0, 0.2, 0.5, 0.9)]
    assert values == sorted(values)


def test_correlation_interval_against_fishers_transformation() -> None:
    """Recomputing the transformation by hand gives the same interval."""
    estimate = correlation_interval(0.004, 39)
    correlation = sqrt(0.004)
    half = 1.959964 / sqrt(39 - 3)

    assert estimate.correlation == pytest.approx(correlation)
    assert estimate.low == pytest.approx(tanh(atanh(correlation) - half))
    assert estimate.high == pytest.approx(tanh(atanh(correlation) + half))

    # The interval reaches below zero: the kit measuring nothing is not ruled out.
    assert estimate.low < 0 < estimate.high
    # More participants narrow it.
    wider = correlation_interval(0.004, 39)
    narrower = correlation_interval(0.004, 400)
    assert narrower.high - narrower.low < wider.high - wider.low


def test_risks_by_test_reproduce_the_overall_risk() -> None:
    """The two risks must average back to the overall risk at the stated share."""
    overall = TURPIN["developed Crohn's disease"] / TURPIN["relatives"]
    ratio = TURPIN["hazard ratio"]

    for share in (0.1, 0.2, 0.3):
        normal, abnormal = risks_by_test(share, overall, ratio)
        assert (1 - share) * normal + share * abnormal == pytest.approx(overall)
        assert abnormal / normal == pytest.approx(ratio)
        assert normal < overall < abnormal

    # A ratio of one makes the test uninformative and both risks the base risk.
    assert risks_by_test(0.2, overall, 1.0) == pytest.approx((overall, overall))


def test_a_threefold_risk_is_still_a_small_risk() -> None:
    """Even after an abnormal test most relatives do not develop the disease."""
    for low, high in SUMMARY.risk_by_test_percent.values():
        assert low < high < 10.0


def test_published_numbers() -> None:
    """Every value the article prints should come back unchanged."""
    assert SUMMARY.correlation.correlation == 0.063
    assert SUMMARY.correlation.low == -0.26
    assert SUMMARY.correlation.high == 0.37
    assert SUMMARY.variance_explained_percent == 0.4

    assert SUMMARY.truly_high_among_positives_percent == {
        "at the observed correlation": 22.5,
        "at the top of its interval": 36.6,
        "at 0.5": 43.6,
        "at 0.9": 75.0,
        "by chance": 20.0,
    }
    assert SUMMARY.relatives_who_developed_crohns_percent == 3.5
    assert SUMMARY.risk_by_test_percent == {
        "0.1": (2.9, 8.9),
        "0.2": (2.5, 7.6),
        "0.3": (2.2, 6.6),
    }
    assert SUMMARY.glutamine_trial_responders_percent == (79.6, 5.8)

    # The kit explains less than half a percent of the variance in permeability.
    assert SUMMARY.variance_explained_percent == pytest.approx(100 * POWER["r squared"])
    # The published response rates, recomputed from the counts.
    assert SUMMARY.glutamine_trial_responders_percent[0] == pytest.approx(
        100 * ZHOU["glutamine"][0] / ZHOU["glutamine"][1], abs=0.05
    )


def test_invalid_inputs() -> None:
    """Impossible correlations, shares, and ratios are rejected."""
    with pytest.raises(ValueError):
        correlation_interval(1.5, 39)
    with pytest.raises(ValueError):
        correlation_interval(0.004, 3)
    with pytest.raises(ValueError):
        share_truly_high(1.5)
    with pytest.raises(ValueError):
        risks_by_test(1.5, 0.035, 3.0)
    with pytest.raises(ValueError):
        risks_by_test(0.2, 0.035, 0.0)
