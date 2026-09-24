"""Tests for the regression discontinuity bandwidth figure rendering."""

from math import sqrt
from pathlib import Path

from blog_reproducibility.statistics.regression_discontinuity import (
    BANDWIDTHS,
    TRUE_EFFECT,
    UNITS,
    BandwidthRow,
    RegressionDiscontinuitySummary,
    local_linear_bias,
    local_linear_sd,
)
from blog_reproducibility.statistics.regression_discontinuity_figure import (
    render_regression_discontinuity_figure,
)


def _row(h: float) -> BandwidthRow:
    bias, sd = local_linear_bias(h), local_linear_sd(h)
    rmse = sqrt(bias**2 + sd**2)
    return BandwidthRow(h, 600, TRUE_EFFECT + bias, -bias, sd, rmse, bias, sd, rmse, UNITS * h / 50)


# Rendering needs only the three curves; the closed forms stand in for 6,000 samples.
SUMMARY = RegressionDiscontinuitySummary(
    effect=TRUE_EFFECT,
    units=UNITS,
    rows=tuple(_row(h) for h in BANDWIDTHS),
    naive_difference=8.5,
    optimal_bandwidth=19.1,
    polynomials=(),
    placebos=(),
    manipulation=(),
    sample_sizes=(),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_regression_discontinuity_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "rdd_bandwidth_tradeoff.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
