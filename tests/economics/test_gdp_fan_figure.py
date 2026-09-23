"""Tests for the GDP fan chart figure rendering."""

from pathlib import Path

from blog_reproducibility.economics.gdp_fan_figure import render_fan_figure


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_fan_figure(output_dir=tmp_path)

    assert artifact.path == tmp_path / "monte_carlo_fan.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
