"""Tests for the sample ratio mismatch figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.sample_ratio_mismatch import DROP_SHARES, DropRow
from blog_reproducibility.statistics.sample_ratio_mismatch_figure import render_mismatch_figure

# Rendering needs only the rows; recomputing them would simulate 360 million users.
ROWS = tuple(
    DropRow(
        drop=drop,
        relative_bias=50 * drop,
        alarm_rate=min(1.0, 100 * drop),
        mean_lift=0.01 * (1 + 50 * drop),
        mean_treated_share=(1 - drop) / (2 - drop),
    )
    for drop in DROP_SHARES
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_mismatch_figure(output_dir=tmp_path, rows=ROWS)

    assert artifact.path == tmp_path / "srm_detection_and_bias.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
