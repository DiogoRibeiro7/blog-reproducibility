"""Check point-in-time selection against the release dates it depends on."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from blog_reproducibility.time_series.release_vintages import (
    DECISION_THRESHOLD,
    annualised_to_quarterly,
    as_of,
    example_payload,
    load_releases,
)

RELEASES = load_releases()
DECISIONS = example_payload()


def test_the_releases_are_loaded_in_order_with_their_sources() -> None:
    """Three dated publications of one reference period, oldest first."""
    assert len(RELEASES) == 3
    assert [release.release_name for release in RELEASES] == ["Advance", "Second", "Third"]
    assert [release.value for release in RELEASES] == [-0.3, -0.2, -0.5]

    dates = [release.released_at for release in RELEASES]
    assert dates == sorted(dates)
    for release in RELEASES:
        assert release.reference_period == "2025Q1"
        assert release.source_url.startswith("https://www.bea.gov/")


def test_nothing_is_available_before_the_first_release() -> None:
    """A cutoff before the advance estimate returns None, not a zero."""
    before = RELEASES[0].released_at - timedelta(seconds=1)
    assert as_of(RELEASES, before) is None
    assert DECISIONS[0].value is None
    assert DECISIONS[0].below_threshold is None


def test_each_release_becomes_available_at_its_own_timestamp() -> None:
    """The value changes exactly when the release lands, not before."""
    for release in RELEASES:
        just_before = release.released_at - timedelta(seconds=1)
        assert as_of(RELEASES, release.released_at) == release.value
        assert as_of(RELEASES, just_before) != release.value or release is RELEASES[0]


def test_the_decision_flips_only_once_the_third_estimate_lands() -> None:
    """The rule says no twice and yes once, and hindsight would say yes throughout."""
    assert [decision.below_threshold for decision in DECISIONS] == [None, False, False, True]
    assert [decision.value for decision in DECISIONS] == [None, -0.3, -0.2, -0.5]

    # Reading the series by reference period alone returns the final figure always.
    final = RELEASES[-1].value
    assert final < DECISION_THRESHOLD
    assert all(final < DECISION_THRESHOLD for _ in DECISIONS)


def test_the_revision_moved_in_both_directions() -> None:
    """The second estimate was better than the first and the third was worse than both."""
    advance, second, third = (release.value for release in RELEASES)
    assert second > advance > third


def test_annualised_and_quarterly_rates_are_consistent() -> None:
    """Compounding the quarterly rate four times returns the annualised one."""
    for annualised in (-0.3, -0.5, 2.0):
        quarterly = annualised_to_quarterly(annualised)
        assert (1 + quarterly / 100) ** 4 == pytest.approx(1 + annualised / 100)

    assert annualised_to_quarterly(-0.3) == pytest.approx(-0.0750845, abs=5e-8)
    assert annualised_to_quarterly(0.0) == pytest.approx(0.0)


def test_invalid_inputs() -> None:
    """A non-finite rate and a missing file are rejected."""
    with pytest.raises(ValueError):
        annualised_to_quarterly(float("nan"))
    with pytest.raises(FileNotFoundError):
        load_releases(Path("no-such-file.csv"))
    assert as_of((), datetime.fromisoformat("2025-07-01T12:00:00+00:00")) is None
