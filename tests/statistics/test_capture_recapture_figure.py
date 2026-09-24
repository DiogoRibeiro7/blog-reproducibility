"""Tests for the capture-recapture heterogeneity figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.capture_recapture import (
    CaptureRecaptureSummary,
    HeterogeneityCurve,
    expected_capture,
)
from blog_reproducibility.statistics.capture_recapture_figure import (
    render_capture_recapture_figure,
)

# Rendering needs only the curves; the closed forms stand in for 3,000 replications.
SPREADS = (0.0, 0.35, 0.7, 1.05, 1.4)
EXPECTED = tuple(expected_capture(spread) for spread in SPREADS)
SUMMARY = CaptureRecaptureSummary(
    curve=HeterogeneityCurve(
        spreads=SPREADS,
        two_pass=tuple(e.two_pass_limit for e in EXPECTED),
        chao=tuple(e.chao_limit or 0.0 for e in EXPECTED),
        seen=tuple(e.seen for e in EXPECTED),
        two_pass_sd=(20.0,) * 5,
        chao_sd=(25.0,) * 5,
        smallest_twice=30,
    ),
    expected=EXPECTED,
    homogeneous=(),
    two_pass=(),
    three_pass=(),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_capture_recapture_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "capture_recapture_heterogeneity.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
