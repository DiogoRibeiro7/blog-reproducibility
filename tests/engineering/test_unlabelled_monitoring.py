"""Check that the three monitoring worlds are what the article claims.

The point of the example is an equality and an inequality: two worlds with
identical observable signals and very different accuracy, and a third with a
different signal and identical accuracy. Both are asserted directly.
"""

import pytest

from blog_reproducibility.engineering.unlabelled_monitoring import (
    WORLDS,
    example_payload,
    summarise_world,
)

REFERENCE, REVERSAL, SHIFT = example_payload()


def test_every_world_is_a_probability_distribution() -> None:
    """The four cells sum to one in each of them."""
    for mass in WORLDS.values():
        assert sum(mass) == pytest.approx(1.0)
        assert all(value >= 0 for value in mass)


def test_identical_signals_can_hide_opposite_accuracy() -> None:
    """The reference and the reversal look the same before labels arrive."""
    assert REFERENCE.positive_share == pytest.approx(REVERSAL.positive_share)
    assert REFERENCE.mean_confidence == pytest.approx(REVERSAL.mean_confidence)

    # And they are as different as they could be once the labels do arrive.
    assert REFERENCE.accuracy == pytest.approx(0.9)
    assert REVERSAL.accuracy == pytest.approx(0.1)
    assert REFERENCE.accuracy + REVERSAL.accuracy == pytest.approx(1.0)


def test_a_moved_signal_can_mean_nothing_at_all() -> None:
    """The input shift moves the observable share and leaves accuracy alone."""
    assert SHIFT.positive_share == pytest.approx(0.9)
    assert SHIFT.positive_share != pytest.approx(REFERENCE.positive_share)
    assert SHIFT.accuracy == pytest.approx(REFERENCE.accuracy)
    assert SHIFT.brier_score == pytest.approx(REFERENCE.brier_score)


def test_mean_confidence_cannot_move_at_all_in_this_model() -> None:
    """The model scores 0.9 or 0.1, so its confidence is 0.9 whatever happens."""
    assert all(row.mean_confidence == pytest.approx(0.9) for row in example_payload())


def test_the_brier_score_needs_outcomes_and_ranks_the_worlds() -> None:
    """Scoring against the labels separates what the signals could not."""
    assert REFERENCE.brier_score == pytest.approx(0.09)
    assert REVERSAL.brier_score == pytest.approx(0.73)
    assert REVERSAL.brier_score > REFERENCE.brier_score
    assert all(0.0 <= row.brier_score <= 1.0 for row in example_payload())


def test_published_values() -> None:
    """The table the article prints, to three places."""
    expected = {
        "Reference": (0.500, 0.900, 0.900, 0.090),
        "Hidden reversal": (0.500, 0.900, 0.100, 0.730),
        "Input shift only": (0.900, 0.900, 0.900, 0.090),
    }
    for row in example_payload():
        share, confidence, accuracy, brier = expected[row.name]
        assert row.positive_share == pytest.approx(share, abs=5e-4)
        assert row.mean_confidence == pytest.approx(confidence, abs=5e-4)
        assert row.accuracy == pytest.approx(accuracy, abs=5e-4)
        assert row.brier_score == pytest.approx(brier, abs=5e-4)


def test_invalid_worlds() -> None:
    """A world needs four cells that sum to one."""
    with pytest.raises(ValueError):
        summarise_world("short", (0.5, 0.5))
    with pytest.raises(ValueError):
        summarise_world("unnormalised", (0.5, 0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        summarise_world("negative", (-0.1, 0.4, 0.4, 0.3))
