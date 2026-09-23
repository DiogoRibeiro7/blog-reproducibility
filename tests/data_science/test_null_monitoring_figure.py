"""Tests for the null-rate monitoring figure rendering."""

from pathlib import Path

from blog_reproducibility.data_science.null_monitoring import (
    NULL_RATES,
    DetectionRow,
    MonitorBaseline,
    MonitoringSummary,
)
from blog_reproducibility.data_science.null_monitoring_figure import render_detection_figure

# Rendering needs only the curves; recomputing them would simulate 4,500 days.
SUMMARY = MonitoringSummary(
    baseline=MonitorBaseline(16.4, 0.06, 0.019, 0.001),
    detection=tuple(DetectionRow(rate, min(1.0, rate * 8), 1.0) for rate in NULL_RATES),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_detection_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "null_monitor_detection.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
