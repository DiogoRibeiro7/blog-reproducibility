"""Tests for the survey weighting figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.post_stratification import (
    ARTICLE_SLOPES,
    AttitudeRow,
    PostStratificationSummary,
    WeightingCurve,
    expected_effective_share,
    expected_unweighted_bias,
    imbalance_row,
)
from blog_reproducibility.statistics.post_stratification_figure import (
    render_post_stratification_figure,
)

# Rendering needs only the curve; the closed forms stand in for 120,000 people.
SLOPES = tuple(0.05 + 0.1 * i for i in range(16))
EFFECTIVE = tuple(expected_effective_share(slope) for slope in SLOPES)
RATIO = tuple(imbalance_row(slope).weight_ratio for slope in SLOPES)
BIAS = tuple(expected_unweighted_bias(slope) for slope in SLOPES)
SUMMARY = PostStratificationSummary(
    curve=WeightingCurve(
        slopes=SLOPES,
        population_mean=7.1,
        respondents=(30_000,) * 16,
        effective_share=EFFECTIVE,
        surviving_bias=tuple(0.02 * (-1) ** i for i in range(16)),
        weight_ratio=RATIO,
        unweighted_bias=BIAS,
        weighted_bias=(0.0,) * 16,
        expected_effective_share=EFFECTIVE,
        expected_weight_ratio=RATIO,
        expected_unweighted_bias=BIAS,
    ),
    population_mean=7.102,
    survey=imbalance_row(0.55),
    imbalance=tuple(imbalance_row(slope) for slope in ARTICLE_SLOPES),
    attitude=(AttitudeRow(0.0, 0.26, 0.0),),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_post_stratification_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "weighting_effective_sample.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
