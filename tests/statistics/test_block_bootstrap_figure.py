"""Tests for the block bootstrap coverage figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.block_bootstrap import (
    AUTOCORRELATIONS,
    BLOCK_LENGTHS,
    BlockBootstrapSummary,
    CoverageRow,
    dependence_row,
)
from blog_reproducibility.statistics.block_bootstrap_figure import render_block_bootstrap_figure

# Rendering needs only the coverage; recomputing it would resample 1,600 series.
SUMMARY = BlockBootstrapSummary(
    length=200,
    resamples=400,
    rows=tuple(
        CoverageRow(
            autocorrelation=phi,
            block_lengths=BLOCK_LENGTHS,
            replications=400,
            covered=tuple(round(400 * (0.94 - phi / (2 + block))) for block in BLOCK_LENGTHS),
            coverage=tuple(0.94 - phi / (2 + block) for block in BLOCK_LENGTHS),
        )
        for phi in AUTOCORRELATIONS
    ),
    dependence=tuple(dependence_row(phi) for phi in AUTOCORRELATIONS),
    suggested_block_length=200 ** (1 / 3),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_block_bootstrap_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "block_bootstrap_coverage.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
