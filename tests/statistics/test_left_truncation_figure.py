"""Tests for the survivorship and left-truncation figure rendering."""

from math import exp
from pathlib import Path

from blog_reproducibility.statistics.left_truncation import (
    ClosedForms,
    LeftTruncationSummary,
    MedianEstimates,
    SiteRow,
    SnapshotNumbers,
    SurvivalCurves,
)
from blog_reproducibility.statistics.left_truncation_figure import (
    render_left_truncation_figure,
)

# Rendering needs only the curves; smooth stand-ins replace the Kaplan-Meier steps.
AGES = tuple(i / 20 for i in range(201))
SUMMARY = LeftTruncationSummary(
    curves=SurvivalCurves(
        ages=AGES,
        reference=tuple(exp(-((age / 5.0) ** 1.5)) for age in AGES),
        naive=tuple(exp(-((age / 10.0) ** 1.5)) for age in AGES),
        truncated=tuple(exp(-((age / 4.9) ** 1.5)) for age in AGES),
    ),
    snapshot=SnapshotNumbers(20_000, 7644, 0.38, 0.40, 0.32, 3.4, 6.9, 4.7, 2940, 0.34, 0.38, 0.22),
    medians=MedianEstimates(None, 7.7, 3.8, 4.0),
    sites=(SiteRow("normal", 4.7, 8.8, 4.6, 0.20, 0.10, 0.09),),
    closed_forms=ClosedForms(0.38, 0.31, 3.3, 6.9, 4.7, 4.0, 0.33, 0.38),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_left_truncation_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "survivorship_left_truncation.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
