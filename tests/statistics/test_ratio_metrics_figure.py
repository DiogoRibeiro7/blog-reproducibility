"""Tests for the ratio-metric false positive figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.ratio_metrics import (
    CONCENTRATIONS,
    MEAN_SESSIONS,
    ArticleNumbers,
    FalsePositiveRow,
    RatioMetricSummary,
)
from blog_reproducibility.statistics.ratio_metrics_figure import render_ratio_metric_figure

# Rendering needs only the rates; recomputing them would rerun 12,000 A/A tests.
SUMMARY = RatioMetricSummary(
    rows=tuple(
        FalsePositiveRow(
            concentration=kappa,
            mean_sessions=m,
            variance_ratio=1 + m / (kappa + 1),
            predicted_session_level=0.05 + m / (kappa + 1) / 10,
            session_level=0.05 + m / (kappa + 1) / 10,
            delta_method=0.05,
        )
        for kappa in CONCENTRATIONS
        for m in MEAN_SESSIONS
    ),
    article=ArticleNumbers(0.2, 0.13, 1.4, 1.53, 0.19),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_ratio_metric_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "ratio_metric_false_positives.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
