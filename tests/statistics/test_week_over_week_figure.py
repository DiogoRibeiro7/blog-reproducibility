"""Tests for the week-over-week noise distribution figure rendering."""

from math import exp, pi, sqrt
from pathlib import Path

from blog_reproducibility.statistics.week_over_week import (
    BIN_EDGES,
    ComparisonRow,
    ComparisonStats,
    WeekOverWeekSummary,
)
from blog_reproducibility.statistics.week_over_week_figure import render_week_over_week_figure

# Rendering needs only the densities and spreads; normal curves stand in for the histograms.
CENTRES = [(a + b) / 2 for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:], strict=True)]


def _normal(sd: float) -> tuple[float, ...]:
    return tuple(exp(-0.5 * (x / sd) ** 2) / (sd * sqrt(2 * pi)) for x in CENTRES)


def _row(name: str, sd: float) -> ComparisonRow:
    stats = ComparisonStats(0.0, sd, 0.3, 0.04, 0.1, 0.13)
    return ComparisonRow(name, stats, stats)


SUMMARY = WeekOverWeekSummary(
    comparisons=85_000,
    same_weekday=_row("same weekday last week", 0.049),
    seven_day=_row("seven-day averages", 0.029),
    same_weekday_last_year=ComparisonStats(0.0, 0.05, 0.31, 0.044, 0.097, 0.129),
    bin_edges=BIN_EDGES,
    same_weekday_density=_normal(0.049),
    seven_day_density=_normal(0.029),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_week_over_week_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "wow_noise_distribution.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
