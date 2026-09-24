"""Tests for the cluster-randomisation false positive figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.cluster_randomisation import (
    INTRACLASS_CORRELATIONS,
    ArticleNumbers,
    ClusterSummary,
    FalsePositiveRow,
)
from blog_reproducibility.statistics.cluster_randomisation_figure import render_cluster_figure

# Rendering needs only the rates; recomputing them would rerun 5,600 A/A experiments.
SUMMARY = ClusterSummary(
    rows=tuple(
        FalsePositiveRow(
            intraclass_correlation=icc,
            design_effect=1 + 199 * icc,
            predicted_customer_level=0.05 + 3 * icc,
            customer_level=min(1.0, 0.05 + 3 * icc),
            store_level=0.05,
        )
        for icc in INTRACLASS_CORRELATIONS
    ),
    article=ArticleNumbers((), 365.3, 1.05, 0.32, 0.30, ()),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_cluster_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "cluster_randomisation_false_positives.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
