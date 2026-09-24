"""Tests for the CUPED variance-reduction figure rendering."""

from math import sqrt
from pathlib import Path

from blog_reproducibility.statistics.cuped import CORRELATIONS, StandardErrorRow
from blog_reproducibility.statistics.cuped_figure import render_cuped_figure

# Rendering needs only the standard errors; simulating them takes 12,000 experiments.
ROWS = tuple(
    StandardErrorRow(
        correlation=rho,
        difference_in_means=0.316,
        cuped=0.316 * sqrt(1 - rho**2),
        stratified=0.316 * sqrt(1 - 0.8 * rho**2),
        theory=0.316 * sqrt(1 - rho**2),
    )
    for rho in CORRELATIONS
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_cuped_figure(output_dir=tmp_path, rows=ROWS)

    assert artifact.path == tmp_path / "cuped_variance_reduction.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
