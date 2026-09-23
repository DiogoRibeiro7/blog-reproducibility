"""Tests for the distance-concentration figure rendering."""

from pathlib import Path

from blog_reproducibility.mathematics.distance_concentration import (
    DIMENSIONS,
    NOISE_DIMENSIONS,
    AccuracyRow,
    ContrastRow,
    DistanceSummary,
)
from blog_reproducibility.mathematics.distance_concentration_figure import (
    render_concentration_figure,
)

# Rendering needs only the tables; recomputing them would retrain every classifier.
SUMMARY = DistanceSummary(
    contrast=tuple(ContrastRow(d, 10 / d, 11 / d) for d in DIMENSIONS),
    accuracy=tuple(AccuracyRow(n, 0.8 - n / 2000, 0.82, 0.85) for n in NOISE_DIMENSIONS),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_concentration_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "distance_concentration.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1664, 640)
