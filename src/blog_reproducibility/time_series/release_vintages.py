"""Point-in-time data, for the article on economic data having two dates.

An economic series has a reference period and a release date, and a backtest
that ignores the second one is asking what a decision *would* have been with
information nobody had. The article's example is US real GDP growth for 2025Q1,
which was published three times with three different numbers.

The model answers one question: given a cutoff, what was the best estimate
actually available then? Reading the same series by reference period alone
returns the final figure at every cutoff, and the decision rule flips.

The three releases are transcribed from BEA news releases, with their source
URLs, into ``data/economics/gdp_q1_2025_release_vintages.csv``. Later revisions
exist and are outside this example. Availability is treated as immediate at the
release timestamp, which is an assumption, not a fact about how fast anyone
reads a press release.
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from blog_reproducibility.common.validation import real

__all__ = [
    "DATA_PATH",
    "DECISION_THRESHOLD",
    "Release",
    "VintageDecision",
    "annualised_to_quarterly",
    "as_of",
    "example_payload",
    "load_releases",
]

DATA_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data" / "economics" / "gdp_q1_2025_release_vintages.csv"
)
# The rule the article puts to the data: act when growth is below -0.4%.
DECISION_THRESHOLD: Final[float] = -0.4


@dataclass(frozen=True, slots=True)
class Release:
    """One dated publication of one reference period."""

    series: str
    reference_period: str
    release_name: str
    released_at: datetime
    value: float
    units: str
    source_url: str


@dataclass(frozen=True, slots=True)
class VintageDecision:
    """What was known at one cutoff, and what the rule said then."""

    cutoff: datetime
    value: float | None
    below_threshold: bool | None


def load_releases(path: Path = DATA_PATH) -> tuple[Release, ...]:
    """Load the transcribed releases, oldest first."""
    if not path.is_file():
        raise FileNotFoundError(f"No release vintages at {path}")

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no releases")

    releases = [
        Release(
            series=row["series"],
            reference_period=row["reference_period"],
            release_name=row["release_name"],
            released_at=datetime.fromisoformat(row["released_at"]),
            value=float(row["value"]),
            units=row["units"],
            source_url=row["source_url"],
        )
        for row in rows
    ]
    return tuple(sorted(releases, key=lambda release: release.released_at))


def as_of(releases: tuple[Release, ...], cutoff: datetime) -> float | None:
    """Return the most recent value published on or before the cutoff.

    ``None`` means nothing had been published yet, which is a real state and not
    a missing value to be filled in.
    """
    eligible = [release for release in releases if release.released_at <= cutoff]
    if not eligible:
        return None
    return max(eligible, key=lambda release: release.released_at).value


def annualised_to_quarterly(annualised_percent: float) -> float:
    """Convert an annualised percentage rate to the quarterly rate behind it."""
    rate = real(annualised_percent, name="annualised_percent")
    # float ** float is Any to mypy, because a negative base can go complex.
    # The base here is positive for any rate above -100%.
    return float(100 * ((1 + rate / 100) ** 0.25 - 1))


def example_payload(
    cutoffs: tuple[str, ...] = ("2025-04-29", "2025-05-01", "2025-06-01", "2025-07-01"),
) -> tuple[VintageDecision, ...]:
    """Return what the rule decided at each cutoff the article uses."""
    releases = load_releases()
    decisions: list[VintageDecision] = []

    for date in cutoffs:
        cutoff = datetime.fromisoformat(date + "T12:00:00+00:00")
        value = as_of(releases, cutoff)
        decisions.append(
            VintageDecision(
                cutoff=cutoff,
                value=value,
                below_threshold=None if value is None else value < DECISION_THRESHOLD,
            )
        )
    return tuple(decisions)
