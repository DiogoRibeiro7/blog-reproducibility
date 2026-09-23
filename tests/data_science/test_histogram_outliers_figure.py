"""Tests for the 2D histogram outlier figure rendering."""

from pathlib import Path

from blog_reproducibility.data_science.histogram_outliers_figure import render_histogram_figure


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_histogram_figure(output_dir=tmp_path)

    assert artifact.path == tmp_path / "hist2d_outliers.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1536, 624)
