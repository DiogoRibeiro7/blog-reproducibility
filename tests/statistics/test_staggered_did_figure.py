"""Tests for the staggered difference-in-differences figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.staggered_did import (
    EXPOSURES,
    EventStudyRow,
    SinglePanel,
    StaggeredSummary,
)
from blog_reproducibility.statistics.staggered_did_figure import render_staggered_did_figure

# Rendering needs only the event study and the two lines; this stands in for 200 panels.
SUMMARY = StaggeredSummary(
    replications=200,
    rows=tuple(
        EventStudyRow(k, 0.2 * (k + 1), 0.2 * (k + 1) + 0.01 * (-1) ** k, 0.02) for k in EXPOSURES
    ),
    true_att=19 / 15,
    twfe=0.58,
    expected_twfe=0.60,
    twfe_standard_error=0.007,
    scenarios=(),
    single_panel=SinglePanel(1, 19 / 15, 0.39, 1.14, 1.13, 1.0, 0.0),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_staggered_did_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "staggered_did_event_study.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
