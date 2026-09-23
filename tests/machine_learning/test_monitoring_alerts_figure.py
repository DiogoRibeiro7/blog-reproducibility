"""Tests for the drift-monitoring figure rendering."""

from pathlib import Path

from blog_reproducibility.machine_learning.monitoring_alerts import (
    SHIFTS,
    AlertCounts,
    BurstComparison,
    PowerRow,
)
from blog_reproducibility.machine_learning.monitoring_alerts_figure import (
    render_alert_bursts_figure,
    render_power_curves_figure,
)

# Rendering needs only the summaries; recomputing them runs 60,000 KS tests.
ROWS = tuple(
    PowerRow(
        s, min(1.0, 4 * s), min(1.0, 2 * s), min(1.0, 2.5 * s), min(1.0, 1.5 * s), 5.0, 9.0, 0.6
    )
    for s in SHIFTS
)
BURSTS = BurstComparison(
    independent=AlertCounts((8, 10, 12, 9, 11), 10.0, 1.4, 12, 0, 0),
    correlated=AlertCounts((1, 25, 4, 16, 4), 10.0, 9.0, 25, 1, 1),
)


def test_power_curves_render_the_expected_png(tmp_path: Path) -> None:
    """The power-curve figure should render as a non-empty PNG file."""
    artifact = render_power_curves_figure(output_dir=tmp_path, rows=ROWS)

    assert artifact.path == tmp_path / "drift_alert_power_curves.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)


def test_alert_bursts_render_the_expected_png(tmp_path: Path) -> None:
    """The burst histogram should render as a non-empty PNG file."""
    artifact = render_alert_bursts_figure(output_dir=tmp_path, bursts=BURSTS)

    assert artifact.path == tmp_path / "drift_alert_bursts.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
