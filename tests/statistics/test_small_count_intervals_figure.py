"""Tests for the small-count interval coverage figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.small_count_intervals_figure import render_small_count_figure


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_small_count_figure(output_dir=tmp_path)

    assert artifact.path == tmp_path / "small_count_interval_coverage.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
