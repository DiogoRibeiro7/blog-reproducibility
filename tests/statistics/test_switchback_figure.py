"""Tests for the switchback period-length figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.switchback import PERIODS, PeriodRow
from blog_reproducibility.statistics.switchback_figure import render_switchback_figure

# Rendering needs only the rows; recomputing them would rerun 720 simulated fortnights.
ROWS = tuple(
    PeriodRow(
        period=period,
        bias=-0.07 / period,
        bias_monte_carlo_error=0.0001,
        spread=0.0035 * period**0.5,
        spread_with_burn_in=0.0036 * period**0.5,
    )
    for period in PERIODS
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_switchback_figure(output_dir=tmp_path, rows=ROWS)

    assert artifact.path == tmp_path / "switchback_period_length.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
