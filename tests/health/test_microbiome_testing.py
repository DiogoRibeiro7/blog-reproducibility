"""Check the closure, flag, and repeatability arithmetic of the microbiome article.

Closure is checked against a community counted directly rather than against the
formula. Repeatability is checked against a seeded simulation of two samples per
person, which does not use the integral the model evaluates.
"""

from math import sqrt
from random import Random

import pytest

from blog_reproducibility.health.microbiome_testing import (
    BERRY,
    STANDARD,
    any_flag,
    apparent_change,
    closure_curve,
    example_payload,
    flagged_again,
)

SUMMARY = example_payload()


def test_closure_against_a_community_counted_directly() -> None:
    """Recomputing shares from counts must give the same apparent change."""
    counts = [300.0, 250.0, 200.0, 150.0, 100.0]  # the first taxon holds 30%
    before = [value / sum(counts) for value in counts]
    counts[0] *= 10
    after = [value / sum(counts) for value in counts]

    # Nothing happened to these four taxa.
    for old, new in zip(before[1:], after[1:], strict=True):
        assert new / old - 1 == pytest.approx(apparent_change(0.30, 10))

    assert apparent_change(0.30, 1) == 0.0


def test_closure_identities() -> None:
    """Shrinking a taxon raises every other share, and a whole community cannot move."""
    assert apparent_change(0.30, 0.5) > 0
    assert apparent_change(0.0, 10) == 0.0  # a taxon holding nothing changes nothing
    assert apparent_change(1.0, 10) == pytest.approx(-0.9)  # one taxon is the whole community

    # A larger incumbent or a larger fold change both push the others further down.
    assert apparent_change(0.30, 10) < apparent_change(0.05, 10) < 0
    assert apparent_change(0.30, 10) < apparent_change(0.30, 2) < 0


def test_flag_counts() -> None:
    """A hundred taxa at a 95% range flag five by construction."""
    assert SUMMARY.expected_flags_among_100_taxa == 5
    assert SUMMARY.chance_of_a_flag_among_100_taxa_percent == pytest.approx(
        100 * (1 - 0.95**100), abs=0.05
    )
    assert SUMMARY.expected_flags_with_interquartile_ranges == 50
    assert any_flag(0) == 0.0


def test_flagged_again_at_the_ends_and_against_simulation() -> None:
    """With no repeatability a flag is chance; with perfect repeatability it is certain."""
    assert flagged_again(0.0) == pytest.approx(0.05, abs=1e-4)
    assert flagged_again(1.0) == 1.0

    rng = Random(20260919)
    correlation = 0.5
    cut = STANDARD.inv_cdf(0.95)
    flagged = again = 0
    for _ in range(400_000):
        first = rng.gauss(0.0, 1.0)
        if first > cut:
            flagged += 1
            second = correlation * first + sqrt(1 - correlation**2) * rng.gauss(0.0, 1.0)
            again += second > cut

    assert flagged_again(correlation) == pytest.approx(again / flagged, abs=0.012)


def test_flagged_again_is_increasing_in_repeatability() -> None:
    """More repeatable taxa keep their flags more often."""
    values = [flagged_again(icc) for icc in (0.0, 0.3, 0.5, 0.7, 0.9)]
    assert values == sorted(values)
    # The floor is the top share itself, reached to within the grid's accuracy.
    assert all(0.05 <= value <= 1.0 for value in values[1:])
    assert values[0] == pytest.approx(0.05, abs=1e-6)


def test_published_numbers() -> None:
    """Every value the article prints should come back unchanged."""
    assert SUMMARY.other_shares_when_a_30_percent_taxon_doubles == -23
    assert SUMMARY.other_shares_when_a_30_percent_taxon_rises_tenfold == -73
    assert SUMMARY.other_shares_when_a_5_percent_taxon_rises_tenfold == -31
    assert SUMMARY.chance_of_a_flag_among_100_taxa_percent == 99.4
    assert SUMMARY.flagged_again_percent == {
        "0.3": 14.3,
        "0.5": 24.4,
        "0.7": 39.2,
        "0.9": 63.7,
    }
    assert SUMMARY.microbiome_over_macronutrients == 0.39

    # The article's point about retesting: more than a third of flags disappear
    # even for the most repeatable taxa.
    assert 100 - SUMMARY.flagged_again_percent["0.9"] > 33
    assert BERRY["gut microbiome"] < BERRY["meal macronutrients"]


def test_closure_curve_spans_one_to_tenfold() -> None:
    """The curve the figure draws starts at no change and ends at tenfold."""
    curve = closure_curve(0.30)

    assert curve[0][0] == pytest.approx(1.0)
    assert curve[-1][0] == pytest.approx(10.0)
    assert curve[0][1] == pytest.approx(0.0)
    assert curve[-1][1] == pytest.approx(100 * apparent_change(0.30, 10))


def test_invalid_inputs() -> None:
    """Impossible shares, folds, and grids are rejected."""
    with pytest.raises(ValueError):
        apparent_change(1.5, 2)
    with pytest.raises(ValueError):
        apparent_change(0.3, 0.0)
    with pytest.raises(ValueError):
        flagged_again(1.5)
    with pytest.raises(ValueError):
        flagged_again(0.5, top=0.0)
    with pytest.raises(ValueError):
        flagged_again(0.5, step=0.0)
    with pytest.raises(TypeError):
        any_flag(True)
