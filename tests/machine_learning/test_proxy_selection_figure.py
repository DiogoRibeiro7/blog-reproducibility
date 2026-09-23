"""Tests for the proxy-selection figure rendering."""

from pathlib import Path

from blog_reproducibility.machine_learning.proxy_selection import (
    COSTS,
    SHARES,
    ProxySelectionSummary,
    expected_goal_gain,
    expected_proxy_gain,
)
from blog_reproducibility.machine_learning.proxy_selection_figure import (
    render_proxy_selection_figure,
)

# Rendering needs only the gains; simulating them draws 3,750 pools of 20,000.
SUMMARY = ProxySelectionSummary(
    shares=SHARES,
    costs=COSTS,
    proxy_gain=tuple(expected_proxy_gain(share) for share in SHARES),
    goal_gain=tuple(tuple(expected_goal_gain(share, cost) for share in SHARES) for cost in COSTS),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_proxy_selection_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "proxy_goodhart_selection.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
