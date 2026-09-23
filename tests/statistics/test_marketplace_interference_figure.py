"""Tests for the marketplace interference figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.marketplace_interference import INVENTORIES, InventoryRow
from blog_reproducibility.statistics.marketplace_interference_figure import (
    render_marketplace_figure,
)

# Rendering needs only the rows; recomputing them would rerun 960 simulated experiments.
ROWS = tuple(
    InventoryRow(
        inventory=inventory,
        buyer_level=0.2,
        buyer_level_spread=0.035,
        market_level=min(0.2, max(0.0, (inventory - 190) / 250)),
        market_level_spread=0.005,
        true_lift=min(0.2, max(0.0, (inventory - 190) / 250)),
        true_lift_spread=0.02,
        rollout_lift=min(0.2, max(0.0, (inventory - 190) / 250)),
    )
    for inventory in INVENTORIES
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_marketplace_figure(output_dir=tmp_path, rows=ROWS)

    assert artifact.path == tmp_path / "marketplace_interference_lift.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
