"""Tests for the monitor detection delay figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.run_length import (
    CHARTS,
    SHIFTS,
    DelayQuantiles,
    DetectionCurve,
    FalseAlarms,
    RunLengthSummary,
    ShewhartRow,
    shewhart_average_run_length,
)
from blog_reproducibility.statistics.run_length_figure import render_run_length_figure

# Rendering needs only the mean delays; the payload would simulate 31,500 runs.
EXACT = tuple(shewhart_average_run_length(s) for s in SHIFTS)
SUMMARY = RunLengthSummary(
    replications=1500,
    max_days=2000,
    curves=tuple(
        DetectionCurve(
            chart=chart.name,
            shifts=SHIFTS,
            mean_days=tuple(1 + (days - 1) / speed for days in EXACT),
            median_days=EXACT,
            sd_days=EXACT,
        )
        for chart, speed in zip(CHARTS, (1.0, 4.5, 4.0), strict=True)
    ),
    shewhart_exact=EXACT,
    false_alarms=FalseAlarms(0.0027, 370.4, 256.4, (0.1, 0.25, 0.5), (39.0, 106.4, 256.4)),
    shewhart_table=(ShewhartRow(0.0, 370.4, 257),),
    one_sigma_delay=DelayQuantiles(1.0, 43.9, (0.1, 0.9), (5, 100)),
    tighter_in_control=22.0,
    tighter_one_sigma=6.2,
    correlations=(),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_run_length_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "monitor_detection_delay.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
