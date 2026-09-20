"""Check the panel and reference-change arithmetic of the hormone article.

The flag probability is checked against the binomial distribution written out
term by term, and the correlated version against the independent formula it must
reduce to. The reference change value is checked against one a published paper
reports from its own components.
"""

from itertools import pairwise
from math import comb, sqrt

import pytest

from blog_reproducibility.health.hormone_testing import (
    CASALS,
    NAUGLER,
    WITHIN_SUBJECT_CV,
    Z,
    any_flag,
    any_flag_correlated,
    example_payload,
    flag_curve,
    reference_change_value,
    sorted_reference_changes,
)

SUMMARY = example_payload()


def test_chance_of_a_flag_against_the_binomial_distribution() -> None:
    """One minus the no-flag term should equal the sum of every other term."""
    for analytes in (1, 5, 12, 20):
        at_least_one = sum(
            comb(analytes, k) * 0.05**k * 0.95 ** (analytes - k) for k in range(1, analytes + 1)
        )
        assert any_flag(analytes) == pytest.approx(at_least_one)

    assert any_flag(0) == 0.0
    assert any_flag(1) == pytest.approx(0.05)


def test_simulation_agrees_with_the_formula_when_analytes_are_independent() -> None:
    """At zero correlation the simulation must reproduce the closed form."""
    assert any_flag_correlated(12, 0.0, repeats=30_000) == pytest.approx(any_flag(12), abs=0.012)


def test_correlation_lowers_the_chance_but_not_to_one_in_twenty() -> None:
    """Shared variation reduces the flag rate without collapsing it to a single test."""
    correlated = any_flag_correlated(12, 0.5, repeats=30_000)

    assert 0.05 < correlated < any_flag(12)
    assert correlated > 0.25
    # Perfect correlation makes the panel behave like one analyte.
    assert any_flag_correlated(12, 1.0, repeats=20_000) == pytest.approx(0.05, abs=0.01)


def test_simulation_is_seeded_and_independent_of_global_state() -> None:
    """The same seed repeats; a different seed gives a different estimate."""
    import random

    random.seed(5)
    first = any_flag_correlated(6, 0.4, repeats=2_000)
    random.seed(6)

    assert any_flag_correlated(6, 0.4, repeats=2_000) == first
    assert any_flag_correlated(6, 0.4, repeats=2_000, seed=99) != first


def test_reference_change_value_reproduces_a_published_one() -> None:
    """Casals's components should give back the value the paper reports."""
    assert SUMMARY.salivary_cortisol_rcv_percent == pytest.approx(CASALS["published RCV"], abs=0.5)
    assert SUMMARY.salivary_cortisol_rcv_percent == 103.7


def test_reference_change_value_identities() -> None:
    """The formula is a root-sum-of-squares scaled by z sqrt(2)."""
    assert reference_change_value(0.0) == 0.0
    assert reference_change_value(10.0) == pytest.approx(Z * sqrt(2) * 10.0)
    # Adding analytical variation adds in quadrature, never linearly.
    combined = reference_change_value(3.0, 4.0)
    assert combined == pytest.approx(reference_change_value(5.0))
    assert combined < reference_change_value(3.0) + reference_change_value(4.0)
    # Doubling both CVs doubles the threshold.
    assert reference_change_value(20.0, 10.0) == pytest.approx(
        2 * reference_change_value(10.0, 5.0)
    )


def test_false_positive_share_follows_from_the_two_rates() -> None:
    """Five in every 8.6 abnormal results are expected in healthy people."""
    assert SUMMARY.false_positive_share_percent == round(
        100 * NAUGLER["expected in the healthy"] / NAUGLER["observed abnormal"]
    )
    assert SUMMARY.false_positive_share_percent == 58


def test_published_numbers() -> None:
    """Every value the article prints should come back unchanged."""
    flags = SUMMARY.chance_of_a_flag_percent
    assert flags[1] == 5.0
    assert flags[5] == 22.6
    assert flags[12] == 46.0
    assert flags[20] == 64.2
    assert flags[40] == 87.1
    assert SUMMARY.chance_of_a_flag_correlated_percent == 31.7

    change = SUMMARY.reference_change_values_percent
    assert change == {
        "free T4": 13,
        "SHBG": 23,
        "FSH": 26,
        "testosterone": 40,
        "estradiol": 42,
        "cortisol": 45,
        "TSH": 49,
        "progesterone": 51,
        "DHEAS": 55,
        "LH": 70,
        "insulin": 70,
        "prolactin": 125,
        "free testosterone": 152,
    }

    assert SUMMARY.estradiol_capsule_against_label_percent == (-27, 10)
    assert SUMMARY.progesterone_capsule_against_label_percent == (-9, 35)
    assert SUMMARY.progesterone_luteal_over_follicular == 136


def test_ordering_follows_the_within_subject_variation() -> None:
    """The reference change value is monotone in the within-subject CV."""
    rows = sorted_reference_changes()

    assert [name for name, _ in rows] == sorted(WITHIN_SUBJECT_CV, key=WITHIN_SUBJECT_CV.get)  # type: ignore[arg-type]
    assert all(left <= right for (_, left), (_, right) in pairwise(rows))


def test_flag_curve_is_increasing_and_starts_at_one_test() -> None:
    """The curve the figure draws runs from one analyte upwards."""
    curve = flag_curve()

    assert curve[0][0] == 1
    assert curve[0][1] == pytest.approx(5.0)
    assert len(curve) == 40
    assert all(left < right for (_, left), (_, right) in pairwise(curve))


def test_invalid_inputs() -> None:
    """Negative variation, impossible correlations, and bad counts are rejected."""
    with pytest.raises(ValueError):
        reference_change_value(-1.0)
    with pytest.raises(ValueError):
        any_flag(-1)
    with pytest.raises(ValueError):
        any_flag(5, coverage=1.5)
    with pytest.raises(ValueError):
        any_flag_correlated(5, -0.1)
    with pytest.raises(ValueError):
        any_flag_correlated(5, 0.5, repeats=0)
    with pytest.raises(TypeError):
        any_flag(True)
