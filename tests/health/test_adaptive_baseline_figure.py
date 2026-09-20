"""Tests for adaptive baseline draft figure rendering."""

from pathlib import Path

from blog_reproducibility.health.adaptive_baseline_figure import render_baseline_figure


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The draft figure should render as a non-empty PNG file."""
    artifact = render_baseline_figure(output_dir=tmp_path)

    assert artifact.path == tmp_path / "healthcare_adaptive_baseline.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1760, 800)
