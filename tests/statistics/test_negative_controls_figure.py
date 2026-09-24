"""Tests for the negative control tracking figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.negative_controls import STRENGTHS, TrackingCurve
from blog_reproducibility.statistics.negative_controls_figure import (
    render_negative_control_figure,
)

# Rendering needs only the curve; recomputing it would draw 204 populations of 30,000.
CURVE = TrackingCurve(
    users=30_000,
    runs=12,
    control_exposure=0.6,
    strengths=STRENGTHS,
    biases=tuple(0.83 * s for s in STRENGTHS),
    signals=tuple(0.5 + 0.004 * (-1) ** i for i in range(len(STRENGTHS))),
    expected_biases=tuple(0.83 * s for s in STRENGTHS),
    expected_signal=0.498,
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_negative_control_figure(output_dir=tmp_path, curve=CURVE)

    assert artifact.path == tmp_path / "negative_control_tracking.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
