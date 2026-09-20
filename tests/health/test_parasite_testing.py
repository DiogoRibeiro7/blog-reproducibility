"""Check the diagnostic arithmetic of the parasite article.

Bayes' rule is checked against direct counting of a hypothetical cohort, the
compounding of negatives against the odds form, and the exact specificity bound
against the rule of three it approximates.
"""

from itertools import pairwise

import pytest

from blog_reproducibility.health.parasite_testing import (
    BRANDA_PUBLISHED_NPV,
    BRANDA_SENSITIVITY,
    PRIORS,
    TAPE,
    after_positive,
    detected_by,
    example_payload,
    left_after_negatives,
    likelihood_ratio,
    specificity_lower_bound,
)

SUMMARY = example_payload()


def test_bayes_rule_reproduces_the_published_negative_predictive_values() -> None:
    """The model should return the values Branda's paper reports, to its own rounding."""
    computed = SUMMARY.negative_predictive_value_percent

    for prevalence, published in BRANDA_PUBLISHED_NPV.items():
        assert abs(computed[str(prevalence)] - published) < 0.6, prevalence

    assert computed == {"0.05": 98.5, "0.1": 97.0, "0.15": 95.3, "0.2": 93.5}


def test_repeat_sampling_is_close_to_independent() -> None:
    """The observed two-specimen yield is near, and below, the independent bound."""
    observed = SUMMARY.two_specimens_observed_percent
    independent = SUMMARY.two_specimens_if_independent_percent

    assert abs(independent - observed) < 3
    assert observed < independent  # correlated shedding costs a little
    tape_gap = SUMMARY.tape_three_mornings_if_independent_percent - 100 * TAPE["three mornings"]
    assert abs(tape_gap) < 3


def test_detected_by_identities() -> None:
    """No samples detect nothing; a perfect test detects everything at once."""
    assert detected_by(0, 0.72) == 0.0
    assert detected_by(1, 0.72) == pytest.approx(0.72)
    assert detected_by(3, 1.0) == 1.0
    assert detected_by(3, 0.0) == 0.0

    yields = [detected_by(number, 0.5) for number in range(1, 6)]
    assert all(left < right for left, right in pairwise(yields))


def test_negative_results_compound() -> None:
    """In odds, every negative result divides by 1 / (1 - sensitivity)."""
    for prior in PRIORS:
        one, two, three = (
            left_after_negatives(prior, number, BRANDA_SENSITIVITY) for number in (1, 2, 3)
        )
        assert prior > one > two > three

        def odds(probability: float) -> float:
            return probability / (1 - probability)

        assert odds(one) / odds(two) == pytest.approx(1 / (1 - BRANDA_SENSITIVITY))
        assert odds(two) / odds(three) == pytest.approx(1 / (1 - BRANDA_SENSITIVITY))

    assert pytest.approx(3.6, abs=0.05) == 1 / (1 - BRANDA_SENSITIVITY)


def test_left_after_negatives_against_direct_counting() -> None:
    """Counting a cohort of 100,000 people gives the same posterior."""
    infected, healthy = 5_000, 95_000
    still_negative = infected * (1 - 0.72) ** 3

    assert left_after_negatives(0.05, 3, 0.72) == pytest.approx(
        still_negative / (still_negative + healthy)
    )
    # No test result leaves the prior alone, and a certain prior stays certain.
    assert left_after_negatives(0.05, 0, 0.72) == pytest.approx(0.05)
    assert left_after_negatives(1.0, 3, 0.72) == 1.0
    assert left_after_negatives(0.0, 3, 0.72) == 0.0


def test_imperfect_specificity_raises_what_a_negative_leaves() -> None:
    """A test that also flags healthy people makes its negatives less informative."""
    perfect = left_after_negatives(0.05, 1, 0.72, 1.0)
    imperfect = left_after_negatives(0.05, 1, 0.72, 0.90)

    assert imperfect > perfect


def test_a_common_checklist_barely_moves_a_prior_and_a_laboratory_test_does() -> None:
    """A checklist half the healthy population would pass is worth almost nothing."""
    assert after_positive(0.01, likelihood_ratio(0.95, 0.5)) < 0.02
    assert SUMMARY.antigen_likelihood_ratio > 8 * SUMMARY.checklist_likelihood_ratio["0.5"]

    # Zero false positives in fifty: the exact one-sided limit, with the rule of
    # three as an independent check on it.
    assert specificity_lower_bound(50) == pytest.approx(0.05 ** (1 / 50))
    assert specificity_lower_bound(50) == pytest.approx(1 - 3 / 50, abs=0.003)
    # More negative specimens tighten the bound.
    assert specificity_lower_bound(200) > specificity_lower_bound(50)


def test_likelihood_ratio_and_update_identities() -> None:
    """A ratio of one leaves the prior alone; the update is monotone in the ratio."""
    assert after_positive(0.01, 1.0) == pytest.approx(0.01)
    assert after_positive(0.5, 3.0) == pytest.approx(0.75)
    assert likelihood_ratio(0.95, 0.95) == pytest.approx(1.0)

    posteriors = [after_positive(0.01, ratio) for ratio in (1.0, 2.0, 5.0, 20.0)]
    assert all(left < right for left, right in pairwise(posteriors))


def test_published_numbers() -> None:
    """Every value the article prints should come back unchanged."""
    assert SUMMARY.first_specimen_percent == 75.9
    assert SUMMARY.two_specimens_observed_percent == 92.0
    assert SUMMARY.two_specimens_if_independent_percent == 94.2
    assert SUMMARY.tape_three_mornings_if_independent_percent == 87.5

    assert SUMMARY.left_after_negatives_percent["0.05"] == (1.452, 0.411, 0.115)
    assert SUMMARY.left_after_negatives_percent["0.01"] == (0.282, 0.079, 0.022)
    assert SUMMARY.left_after_three_at_half_sensitivity_percent["0.05"] == 0.65
    assert SUMMARY.left_after_three_at_half_sensitivity_percent["0.01"] == 0.13

    assert SUMMARY.after_a_positive_checklist_percent == {"0.8": 1.2, "0.5": 1.9, "0.2": 4.6}
    assert SUMMARY.antigen_specificity_lower_bound_percent == 94.2
    assert SUMMARY.antigen_likelihood_ratio == 16.2
    assert SUMMARY.after_a_positive_antigen_test_percent == 14.0


def test_invalid_inputs() -> None:
    """Impossible priors, sensitivities, and sample counts are rejected."""
    with pytest.raises(ValueError):
        detected_by(-1, 0.5)
    with pytest.raises(ValueError):
        detected_by(2, 1.5)
    with pytest.raises(ValueError):
        left_after_negatives(1.5, 1, 0.72)
    with pytest.raises(ValueError):
        likelihood_ratio(0.95, 0.0)
    with pytest.raises(ValueError):
        after_positive(0.0, 2.0)
    with pytest.raises(ValueError):
        after_positive(0.01, 0.0)
    with pytest.raises(ValueError):
        specificity_lower_bound(0)
    with pytest.raises(TypeError):
        detected_by(True, 0.5)
