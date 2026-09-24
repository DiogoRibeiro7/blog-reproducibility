"""Tests for the recurrent events mean cumulative function figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.recurrent_events import example_payload
from blog_reproducibility.statistics.recurrent_events_figure import (
    render_recurrent_events_figure,
)

# The fleet is small and simulates in milliseconds, so the figure renders the real curves.
SUMMARY = example_payload()


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_recurrent_events_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "recurrent_events_mcf.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)


def test_renderer_computes_the_payload_when_none_is_given(tmp_path: Path) -> None:
    """Without a summary the renderer simulates the fleet itself."""
    artifact = render_recurrent_events_figure(output_dir=tmp_path)

    assert artifact.path.exists()
