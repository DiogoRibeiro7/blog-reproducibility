"""Tests for the digit heaping threshold figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.digit_heaping import (
    HeapingSummary,
    ThresholdCurve,
    recorded_share_above,
    true_share_above,
)
from blog_reproducibility.statistics.digit_heaping_figure import render_digit_heaping_figure

# Rendering needs only the curves; the closed forms stand in for 400,000 records.
THRESHOLDS = tuple(15 + 0.5 * i for i in range(62))
TRUE = tuple(true_share_above(t) for t in THRESHOLDS)
RECORDED = tuple(recorded_share_above(t) for t in THRESHOLDS)
SUMMARY = HeapingSummary(
    curve=ThresholdCurve(0.5, THRESHOLDS, TRUE, RECORDED, TRUE, RECORDED),
    on_round=(),
    off_round=(),
    whipple=(),
    true_mean=20.94,
    true_quantiles=(18.0, 36.4, 64.7),
    quantiles=(),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_digit_heaping_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "heaping_threshold_error.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
