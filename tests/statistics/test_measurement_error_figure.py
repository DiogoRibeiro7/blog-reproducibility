"""Tests for the measurement error figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.measurement_error import (
    RELIABILITIES,
    SIMEX_MULTIPLES,
    AttenuationRow,
    LeakageRow,
    MeasurementErrorSummary,
    SimexResult,
    leakage_coefficients,
    noise_sd_for_reliability,
    simex_slope,
)
from blog_reproducibility.statistics.measurement_error_figure import (
    render_measurement_error_figure,
)

# Rendering needs only the slopes and fits; the closed forms stand in for the simulation.
EXACT = tuple(simex_slope(k, 0.6) for k in SIMEX_MULTIPLES)
SUMMARY = MeasurementErrorSummary(
    observations=20_000,
    true_slope=1.0,
    attenuation=tuple(
        AttenuationRow(rel, noise_sd_for_reliability(rel), rel + 0.003, rel)
        for rel in RELIABILITIES
    ),
    predictor_correlation=0.67,
    exact_predictor_correlation=1 / 1.49,
    leakage=(LeakageRow(0.5, 0.76, 0.16, *leakage_coefficients(0.5)),),
    simex=SimexResult(
        reliability=0.6,
        multiples=SIMEX_MULTIPLES,
        slopes=EXACT,
        exact_slopes=EXACT,
        quadratic=(0.037, -0.204, 0.595),
        quadratic_estimate=0.836,
        rational_numerator=1.5,
        rational_denominator=2.5,
        rational_estimate=1.0,
        exact_numerator=1.5,
        exact_denominator=2.5,
        exact_quadratic_estimate=0.845,
        regression_calibration=1.0,
    ),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_measurement_error_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "measurement_error_attenuation_simex.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1664, 640)
