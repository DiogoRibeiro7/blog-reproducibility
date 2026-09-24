"""Tests for the quantile regression bands figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.quantile_regression import (
    BandFigure,
    FitRow,
    Misses,
    PinballRow,
    PromiseRow,
    QuantileRegressionSummary,
    SegmentRow,
)
from blog_reproducibility.statistics.quantile_regression_figure import (
    render_quantile_regression_figure,
)

# Rendering needs only the figure's points and bands; population lines stand in for the fits.
GRID = tuple(i / 99 for i in range(100))
LOADS = tuple(i / 199 for i in range(200))
SUMMARY = QuantileRegressionSummary(
    residual_sd=8.8,
    fits=(FitRow("least squares", None, 20.0, 3.0, 0.0, (20.0, 3.0, 0.0)),),
    segments=(SegmentRow(0.0, 0.25, 500, 0.99, 0.87, 29.0, 10.1, 0.99, 0.89),),
    misses=Misses(0.93, 0.89, 0.05, 0.02, 0.05, 0.06),
    pinball=(PinballRow(0.05, 0.74, 0.50),),
    mean_minutes=36.6,
    median_minutes=35.9,
    population_mean_minutes=36.5,
    late_share=0.40,
    population_late_share=0.41,
    promises=(PromiseRow(4.0, 1.0, 0.8),),
    figure=BandFigure(
        loads=LOADS,
        minutes=tuple(35.0 + (2 + 12 * load) * ((i % 7) - 2) / 2 for i, load in enumerate(LOADS)),
        grid=GRID,
        ols_lower=tuple(20.5 + 0.9 * g for g in GRID),
        ols_upper=tuple(49.5 + 0.9 * g for g in GRID),
        lower=tuple(32.7 - 14.0 * g for g in GRID),
        median=tuple(34.5 - 2.7 * g for g in GRID),
        upper=tuple(38.9 + 23.3 * g for g in GRID),
    ),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_quantile_regression_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "quantile_regression_bands.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 736)
