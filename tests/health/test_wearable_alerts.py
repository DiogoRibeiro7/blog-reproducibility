"""Check the alert arithmetic against direct counting of the cohort."""

import pytest

from blog_reproducibility.health.wearable_alerts import (
    COHORT,
    SENSITIVITY,
    SPECIFICITY,
    alert_counts,
    example_payload,
)

LOW, MEDIUM, HIGH = example_payload()


def test_the_table_accounts_for_everybody() -> None:
    """The four cells add back to the cohort at every prevalence."""
    for row in example_payload():
        total = row.true_positive + row.false_positive + row.false_negative + row.true_negative
        assert total == pytest.approx(COHORT)
        assert row.true_positive + row.false_negative == pytest.approx(COHORT * row.prevalence)


def test_predictive_value_follows_from_the_counts() -> None:
    """The share of alerts that are real is the true ones over all of them."""
    for row in example_payload():
        assert row.positive_predictive_value == pytest.approx(row.true_positive / row.alerts)
        assert 0.0 < row.positive_predictive_value < 1.0
        assert row.negative_predictive_value > row.positive_predictive_value


def test_the_same_test_means_three_different_things() -> None:
    """Predictive value rises with prevalence while the test never changes."""
    assert LOW.positive_predictive_value < MEDIUM.positive_predictive_value
    assert MEDIUM.positive_predictive_value < HIGH.positive_predictive_value

    assert LOW.positive_predictive_value == pytest.approx(0.153846, abs=5e-7)
    assert MEDIUM.positive_predictive_value == pytest.approx(0.486486, abs=5e-7)
    assert HIGH.positive_predictive_value == pytest.approx(0.818182, abs=5e-7)

    # At one percent, more than five alerts in six are false.
    assert LOW.false_positive > 5 * LOW.true_positive


def test_published_counts() -> None:
    """The table the article prints, per ten thousand people."""
    assert (LOW.true_positive, LOW.false_positive) == pytest.approx((90, 495))
    assert (MEDIUM.true_positive, MEDIUM.false_positive) == pytest.approx((450, 475))
    assert (HIGH.true_positive, HIGH.false_positive) == pytest.approx((1800, 400))
    assert LOW.true_negative == pytest.approx(9405)
    assert HIGH.true_negative == pytest.approx(7600)


def test_a_perfect_test_makes_prevalence_irrelevant() -> None:
    """With no false positives every alert is real, whatever the prevalence."""
    for prevalence in (0.01, 0.2):
        row = alert_counts(prevalence, sensitivity=1.0, specificity=1.0)
        assert row.positive_predictive_value == pytest.approx(1.0)
        assert row.false_positive == 0.0
        assert row.false_negative == 0.0


def test_a_useless_test_returns_the_prevalence() -> None:
    """A coin flip that ignores the person tells you only what you knew."""
    row = alert_counts(0.2, sensitivity=0.5, specificity=0.5)
    assert row.positive_predictive_value == pytest.approx(0.2)


def test_invalid_inputs() -> None:
    """Boundary prevalences and impossible accuracies are rejected."""
    for prevalence in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            alert_counts(prevalence)
    with pytest.raises(ValueError):
        alert_counts(0.1, sensitivity=1.5)
    with pytest.raises(ValueError):
        alert_counts(0.1, specificity=-0.1)
    with pytest.raises(ValueError):
        alert_counts(0.1, cohort=0.0)
    assert SENSITIVITY == 0.90
    assert SPECIFICITY == 0.95
