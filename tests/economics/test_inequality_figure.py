"""Tests for the Lorenz curve figure rendering."""

from pathlib import Path

from blog_reproducibility.economics.inequality_figure import render_lorenz_gini_figure


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_lorenz_gini_figure(output_dir=tmp_path)

    assert artifact.path == tmp_path / "lorenz_gini.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (896, 800)
