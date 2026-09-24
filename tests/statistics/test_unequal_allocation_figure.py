"""Tests for the unequal-allocation figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.unequal_allocation_figure import render_allocation_figure


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_allocation_figure(output_dir=tmp_path)

    assert artifact.path == tmp_path / "allocation_variance_cost.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
