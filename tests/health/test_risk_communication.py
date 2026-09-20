"""Check the concentration and starting-risk examples.

Both are small enough to verify by hand, so the checks are on the identities the
articles rest on: that dose is concentration times volume, and that a relative
reduction fixes neither of the absolute numbers.
"""

import pytest

from blog_reproducibility.health.risk_communication import (
    example_payload,
    exposure_rows,
    risk_rows,
)

SUMMARY = example_payload()


def test_dose_is_concentration_times_volume() -> None:
    """Each row's amount follows from its own two inputs."""
    for row in SUMMARY.exposures:
        assert row.amount_mg == pytest.approx(row.concentration_mg_per_ml * row.volume_ml)


def test_a_lower_concentration_can_deliver_a_larger_amount() -> None:
    """Sample B is a tenth as concentrated as A and delivers half as much again."""
    by_name = {row.name: row for row in SUMMARY.exposures}
    a, b, c = by_name["A"], by_name["B"], by_name["C"]

    assert b.concentration_mg_per_ml < a.concentration_mg_per_ml
    assert b.amount_mg > a.amount_mg
    assert b.amount_mg / a.amount_mg == pytest.approx(1.5)

    # And equal amounts can come from different concentrations entirely.
    assert c.amount_mg == pytest.approx(a.amount_mg)
    assert c.concentration_mg_per_ml != a.concentration_mg_per_ml


def test_the_same_relative_reduction_prevents_very_different_numbers() -> None:
    """Both populations halve their risk; one prevents ten times as many events."""
    high, low = SUMMARY.risks

    assert high.relative_reduction == pytest.approx(low.relative_reduction)
    assert high.relative_reduction == pytest.approx(0.5)
    assert high.difference_per_1000 == pytest.approx(10 * low.difference_per_1000)


def test_absolute_and_relative_views_are_consistent() -> None:
    """The difference and the ratio are two descriptions of the same two numbers."""
    for row in SUMMARY.risks:
        assert row.difference_per_1000 == pytest.approx(row.before_per_1000 - row.after_per_1000)
        assert row.relative_reduction == pytest.approx(
            row.difference_per_1000 / row.before_per_1000
        )
        assert 0 <= row.relative_reduction <= 1


def test_published_numbers() -> None:
    """Every value the articles print should come back unchanged."""
    assert [(row.name, row.amount_mg) for row in exposure_rows()] == [
        ("A", 20.0),
        ("B", 30.0),
        ("C", 20.0),
    ]
    assert [
        (row.name, row.before_per_1000, row.after_per_1000, row.difference_per_1000)
        for row in risk_rows()
    ] == [
        ("Population A", 20.0, 10.0, 10.0),
        ("Population B", 2.0, 1.0, 1.0),
    ]
    assert all(row.relative_reduction == pytest.approx(0.5) for row in risk_rows())
