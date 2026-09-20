"""Check the adaptive baseline against its closed form and its missing-data rule.

The recursion is geometric, so the exact solution is available and the test
compares the two rather than comparing the recursion with itself.
"""

import pytest

from blog_reproducibility.health.adaptive_baseline import (
    adaptive_baseline,
    exact_baseline,
    example_payload,
)

SUMMARY = example_payload()


def test_the_recursion_matches_the_exact_geometric_solution() -> None:
    """Every step of every learning rate agrees with the closed form."""
    for alpha in (0.0, 0.02, 0.1, 0.25, 1.0):
        rows = adaptive_baseline((4.0,) * 30, alpha)
        for step, row in enumerate(rows, start=1):
            assert row.after == pytest.approx(exact_baseline(step, alpha, 4.0))
            assert row.before == pytest.approx(exact_baseline(step - 1, alpha, 4.0))


def test_scoring_happens_before_the_update() -> None:
    """The first observation after a step is compared against the old baseline."""
    rows = adaptive_baseline((4.0,) * 3, 0.5)

    assert rows[0].before == 0.0
    assert rows[0].score == 4.0
    assert rows[0].after == 2.0
    assert rows[1].before == 2.0
    assert rows[1].score == 2.0


def test_a_missing_observation_neither_scores_nor_updates() -> None:
    """A gap in the record is not evidence, so the baseline does not move."""
    rows = adaptive_baseline((4.0, None, 4.0), 0.5)

    assert rows[1].observation is None
    assert rows[1].score is None
    assert rows[1].before == rows[1].after == rows[0].after
    # And the next real observation is scored against that unchanged baseline.
    assert rows[2].before == rows[0].after


def test_the_learning_rate_decides_how_long_a_change_is_flagged() -> None:
    """A frozen baseline keeps scoring the change; a fast one stops almost at once."""
    frozen = SUMMARY["0.0"]
    slow = SUMMARY["0.02"]
    fast = SUMMARY["0.25"]

    assert all(row.score == 4.0 for row in frozen)
    assert all(row.after == 0.0 for row in frozen)

    assert slow[-1].score is not None and slow[-1].score > 2.0
    assert fast[-1].score is not None and fast[-1].score < 0.01

    # Larger rates always leave a smaller score at the same step.
    scores = [SUMMARY[alpha][10].score for alpha in ("0.0", "0.02", "0.1", "0.25")]
    assert all(score is not None for score in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


def test_a_rate_of_one_jumps_straight_to_the_observation() -> None:
    """With alpha of one the baseline is the last observation and nothing is flagged twice."""
    rows = adaptive_baseline((4.0, 4.0, 4.0), 1.0)

    assert rows[0].score == 4.0
    assert rows[0].after == 4.0
    assert rows[1].score == 0.0
    assert rows[2].score == 0.0


def test_the_baseline_approaches_but_never_reaches_the_level() -> None:
    """A geometric approach never arrives, which is why a small rate keeps flagging."""
    for alpha in (0.02, 0.1, 0.25):
        rows = adaptive_baseline((4.0,) * 30, alpha)
        assert all(row.after < 4.0 for row in rows)
        afters = [row.after for row in rows]
        assert afters == sorted(afters)


def test_published_values() -> None:
    """The draft's four rates over thirty days."""
    assert set(SUMMARY) == {"0.0", "0.02", "0.1", "0.25"}
    assert all(len(rows) == 30 for rows in SUMMARY.values())

    assert SUMMARY["0.1"][-1].after == pytest.approx(3.83044, abs=5e-6)
    assert SUMMARY["0.25"][-1].after == pytest.approx(3.99929, abs=5e-6)
    assert SUMMARY["0.02"][-1].after == pytest.approx(1.81806, abs=5e-6)


def test_invalid_inputs() -> None:
    """Rates outside [0, 1], non-finite starts, and infinite observations are rejected."""
    with pytest.raises(ValueError):
        adaptive_baseline((1.0,), 1.5)
    with pytest.raises(ValueError):
        adaptive_baseline((1.0,), -0.1)
    with pytest.raises(ValueError):
        adaptive_baseline((1.0,), 0.5, initial=float("nan"))
    with pytest.raises(ValueError):
        adaptive_baseline((float("inf"),), 0.5)
    with pytest.raises(ValueError):
        exact_baseline(-1, 0.5, 4.0)
