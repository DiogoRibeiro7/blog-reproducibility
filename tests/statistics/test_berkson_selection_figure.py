"""Tests for the Berkson selection figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.berkson_selection import (
    SELECTION_SHARES,
    SelectionCurve,
    selected_correlation,
)
from blog_reproducibility.statistics.berkson_selection_figure import (
    render_berkson_selection_figure,
)

# Rendering needs only the curve; the closed form stands in for 200,000 tickets.
CURVE = SelectionCurve(
    tickets=200_000,
    population_correlation=0.0,
    shares=SELECTION_SHARES,
    selected=tuple(round(200_000 * share) for share in SELECTION_SHARES),
    correlations=tuple(selected_correlation(share) for share in SELECTION_SHARES),
    expected_correlations=tuple(selected_correlation(share) for share in SELECTION_SHARES),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_berkson_selection_figure(output_dir=tmp_path, curve=CURVE)

    assert artifact.path == tmp_path / "berkson_selection_correlation.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
