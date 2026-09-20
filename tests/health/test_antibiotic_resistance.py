"""Check that selection changes the composition without anything growing."""

from itertools import pairwise

import pytest

from blog_reproducibility.health.antibiotic_resistance import (
    RESISTANT_SURVIVAL,
    SENSITIVE_SURVIVAL,
    example_payload,
    selection_rows,
)

SUMMARY = example_payload()


def test_selection_changes_composition_without_population_growth() -> None:
    """Both counts fall at every step and the resistant share rises anyway."""
    rows = SUMMARY.rows

    for before, after in pairwise(rows):
        assert after.total < before.total
        assert after.resistant < before.resistant
        assert after.sensitive < before.sensitive
        assert after.resistant_share > before.resistant_share


def test_the_odds_of_resistance_multiply_by_exactly_eighty() -> None:
    """The odds ratio is the ratio of the two survival rates, every time."""
    rows = SUMMARY.rows

    for before, after in pairwise(rows):
        before_odds = before.resistant_share / (1 - before.resistant_share)
        after_odds = after.resistant_share / (1 - after.resistant_share)
        assert after_odds / before_odds == pytest.approx(80.0, rel=1e-10)

    assert SUMMARY.odds_multiplier == pytest.approx(RESISTANT_SURVIVAL / SENSITIVE_SURVIVAL)
    assert SUMMARY.odds_multiplier == 80.0


def test_counts_follow_the_survival_rates_exactly() -> None:
    """Each row is the previous one scaled by the two survival rates."""
    rows = SUMMARY.rows

    for before, after in pairwise(rows):
        assert after.sensitive == pytest.approx(before.sensitive * SENSITIVE_SURVIVAL)
        assert after.resistant == pytest.approx(before.resistant * RESISTANT_SURVIVAL)


def test_equal_survival_rates_leave_the_composition_alone() -> None:
    """Without a difference in survival there is nothing for selection to do."""
    rows = selection_rows(3, sensitive_survival=0.5, resistant_survival=0.5)

    shares = [row.resistant_share for row in rows]
    assert all(share == pytest.approx(shares[0]) for share in shares)
    assert rows[-1].total < rows[0].total  # the population still shrank


def test_a_single_bottleneck_and_none_at_all() -> None:
    """Zero bottlenecks returns the starting population alone."""
    assert len(selection_rows(0)) == 1
    assert len(selection_rows(5)) == 6
    assert selection_rows(0)[0].resistant_share == pytest.approx(0.001)


def test_published_numbers() -> None:
    """Every value the article prints should come back unchanged."""
    rows = SUMMARY.rows

    assert [row.sensitive for row in rows] == [
        pytest.approx(99_900.0),
        pytest.approx(999.0),
        pytest.approx(9.99),
        pytest.approx(0.0999),
    ]
    assert [row.resistant for row in rows] == [
        pytest.approx(100.0),
        pytest.approx(80.0),
        pytest.approx(64.0),
        pytest.approx(51.2),
    ]
    assert [round(row.resistant_share, 6) for row in rows] == [
        0.001,
        0.074143,
        0.864982,
        0.998053,
    ]

    # One in a thousand becomes almost all of what is left, from one in a thousand.
    assert rows[0].resistant_share < 0.002
    assert rows[-1].resistant_share > 0.99
    assert rows[-1].total < rows[0].total / 1_000


def test_invalid_inputs() -> None:
    """Negative rounds and survival rates outside [0, 1] are rejected."""
    with pytest.raises(ValueError):
        selection_rows(-1)
    with pytest.raises(ValueError):
        selection_rows(3, sensitive_survival=1.5)
    with pytest.raises(ValueError):
        selection_rows(3, resistant_survival=-0.1)
    with pytest.raises(TypeError):
        selection_rows(True)
