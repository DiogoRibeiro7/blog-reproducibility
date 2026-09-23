"""Figure renderer for the article on interference in marketplace experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.marketplace_interference import (
    BUYERS,
    CONTROL_INTENT,
    MARKETS,
    TREATED_INTENT,
    InventoryRow,
    inventory_rows,
)


def render_marketplace_figure(
    *, output_dir: Path, rows: tuple[InventoryRow, ...] | None = None
) -> FigureArtifact:
    """Plot the buyer-level and city-level estimates and the true lift against inventory."""
    use_house_style()
    result = rows if rows is not None else inventory_rows()
    inventories = [row.inventory for row in result]

    figure, axis = plt.subplots()
    axis.plot(
        inventories,
        [row.buyer_level for row in result],
        marker="o",
        color=PALETTE[1],
        lw=2,
        label="Buyer-level split inside one shared market",
    )
    axis.plot(
        inventories,
        [row.market_level for row in result],
        marker="o",
        color=PALETTE[2],
        lw=2,
        label=f"Separate cities randomised ({MARKETS} cities, {MARKETS // 2} treated)",
    )
    axis.plot(
        inventories,
        [row.true_lift for row in result],
        color=PALETTE[0],
        lw=2.4,
        label="True lift from rolling out to everyone",
    )
    axis.invert_xaxis()
    axis.set_xlabel(
        f"daily inventory (units); control demand is {BUYERS * CONTROL_INTENT:.0f}, "
        f"treated demand {BUYERS * TREATED_INTENT:.0f}"
    )
    axis.set_ylabel("estimated lift in sales")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_title("Shared inventory makes the buyer-level test overstate the lift")
    axis.legend(loc="center left")

    return save_figure(figure, slug="marketplace_interference_lift", output_dir=output_dir)
