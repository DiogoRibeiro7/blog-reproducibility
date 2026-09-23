"""Tests for the percentile-precision figure rendering."""

from math import sqrt
from pathlib import Path

from blog_reproducibility.statistics.percentile_metrics import SAMPLE_SIZES, PrecisionRow
from blog_reproducibility.statistics.percentile_metrics_figure import render_percentile_figure

# Rendering needs only the relative errors; simulating them takes 33 million draws.
ROWS = tuple(
    PrecisionRow(n, 2.2 / sqrt(n), 0.78 / sqrt(n), 2.6 / sqrt(n), 6.4 / sqrt(n))
    for n in SAMPLE_SIZES
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_percentile_figure(output_dir=tmp_path, rows=ROWS)

    assert artifact.path == tmp_path / "percentile_precision.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
