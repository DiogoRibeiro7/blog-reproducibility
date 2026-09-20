"""Tests for the sequential CUSUM figure renderer."""

from pathlib import Path

from blog_reproducibility.time_series.sequential_cusum_figure import (
    render_sequential_cusum_figure,
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The renderer should produce a non-empty PNG with stable nominal dimensions."""
    artifact = render_sequential_cusum_figure(output_dir=tmp_path)

    assert artifact.path == tmp_path / "sequential_cusum_worked_example.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert artifact.width == 1600
    assert artifact.height == 960
