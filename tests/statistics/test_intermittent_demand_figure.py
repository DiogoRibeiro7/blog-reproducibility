"""Tests for the intermittent demand forecast bias figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.intermittent_demand import (
    METHODS,
    IntermittentSummary,
    MethodRow,
    closed_forms,
)
from blog_reproducibility.statistics.intermittent_demand_figure import (
    render_intermittent_demand_figure,
)

# Rendering needs only the averages and their errors; the series are not simulated.
AVERAGES = (0.0, 0.796, 0.800, 0.828, 0.787)
SUMMARY = IntermittentSummary(
    series=40,
    forecasts_per_series=234,
    rows=tuple(
        MethodRow(name, average, 0.01 if average else 0.0, (average,), average - 0.8, 1.27, 1.8)
        for name, average in zip(METHODS, AVERAGES, strict=True)
    ),
    closed_forms=closed_forms(),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_intermittent_demand_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "intermittent_forecast_bias.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
