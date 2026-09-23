"""Tests for the novelty tenure-versus-calendar figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.novelty_effects import (
    DAYS,
    NoveltyRow,
    NoveltySummary,
    calendar_expectation,
    novelty_effect,
)
from blog_reproducibility.statistics.novelty_effects_figure import render_novelty_figure

# Rendering needs only the curves; the closed forms stand in for the simulation.
_TRUTH = novelty_effect(range(DAYS))
_MIXTURE = calendar_expectation()
SUMMARY = NoveltySummary(
    runs=1,
    rows=tuple(
        NoveltyRow(
            day=k + 1,
            true_effect=float(_TRUTH[k]),
            by_tenure=float(_TRUTH[k]),
            by_calendar=float(_MIXTURE[k]),
            calendar_expectation=float(_MIXTURE[k]),
        )
        for k in range(DAYS)
    ),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_novelty_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "novelty_tenure_vs_calendar.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
