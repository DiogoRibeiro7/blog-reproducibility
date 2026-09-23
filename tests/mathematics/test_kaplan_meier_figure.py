"""Tests for the Kaplan-Meier figure rendering."""

from pathlib import Path

from blog_reproducibility.mathematics.kaplan_meier_figure import render_kaplan_meier_figure


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_kaplan_meier_figure(output_dir=tmp_path)

    assert artifact.path == tmp_path / "kaplan_meier.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
