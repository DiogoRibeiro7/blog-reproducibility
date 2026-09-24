"""Tests for the group aggregation figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.group_aggregation import (
    INTRACLASS_CORRELATIONS,
    AggregationCurve,
    GroupAggregationSummary,
    group_sizes,
    levels_row,
    predicted_group_mean_correlation,
)
from blog_reproducibility.statistics.group_aggregation_figure import (
    render_group_aggregation_figure,
)

# Rendering needs only the curves; the closed form with a small offset stands in for the draws.
SIZES = group_sizes()
SUMMARY = GroupAggregationSummary(
    groups=600,
    between_correlation=0.9,
    within_correlation=0.0,
    curves=tuple(
        AggregationCurve(
            intraclass_correlation=share,
            group_sizes=SIZES,
            measured=tuple(
                predicted_group_mean_correlation(share, share, 0.9, 0.0, size) + 0.01
                for size in SIZES
            ),
            predicted=tuple(
                predicted_group_mean_correlation(share, share, 0.9, 0.0, size) for size in SIZES
            ),
            individual=0.9 * share,
        )
        for share in INTRACLASS_CORRELATIONS
    ),
    size_rows=(levels_row(0.9, 0.0, 0.1, 5),),
    sign_rows=(levels_row(0.8, -0.3, 0.15, 200),),
    slope_rows=(levels_row(0.8, -0.2, 0.05, 200),),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_group_aggregation_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "aggregation_group_size.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
