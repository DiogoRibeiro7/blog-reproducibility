"""Tests for the Benford spread figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.benford import (
    ARTICLE_SIGMAS,
    BenfordSummary,
    SpreadCurve,
    percentile_span,
    spread_row,
)
from blog_reproducibility.statistics.benford_figure import render_benford_figure

# Rendering needs only the curve; a smooth stand-in replaces 22 columns of 50,000 values.
SIGMAS = tuple(0.15 + 0.1 * i for i in range(25))
MADS = tuple(0.15 * 2.718 ** (-3 * sigma**2) + 0.001 for sigma in SIGMAS)
SUMMARY = BenfordSummary(
    benford=(0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046),
    curve=SpreadCurve(
        sigmas=SIGMAS,
        spans=tuple(percentile_span(sigma) for sigma in SIGMAS),
        mads=MADS,
        population_mads=MADS,
        expected_mads=MADS,
    ),
    close_conformity_span=37.0,
    acceptable_conformity_span=24.0,
    article_rows=tuple(spread_row(sigma) for sigma in ARTICLE_SIGMAS[:1]),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_benford_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "benford_spread.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
