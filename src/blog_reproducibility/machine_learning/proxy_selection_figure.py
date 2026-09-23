"""Figure renderer for the article on proxy metrics under optimisation."""

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.proxy_selection import (
    ProxySelectionSummary,
    example_payload,
)

# The published figure draws its zero line in this grey rather than a house ink.
ZERO_LINE: Final[str] = "#8a8f98"
GOAL_COLOURS: Final[tuple[str, ...]] = (PALETTE[0], PALETTE[2], PALETTE[5], PALETTE[1])


def render_proxy_selection_figure(
    *, output_dir: Path, summary: ProxySelectionSummary | None = None
) -> FigureArtifact:
    """Plot goal gain at each cost, and the proxy's gain, against the share kept."""
    use_house_style()
    result = summary if summary is not None else example_payload()

    figure, axis = plt.subplots()
    axis.plot(
        result.shares,
        result.proxy_gain,
        color=PALETTE[3],
        lw=1.6,
        ls=":",
        label="Proxy gain (what the dashboard shows)",
    )
    for cost, gains, colour in zip(result.costs, result.goal_gain, GOAL_COLOURS, strict=False):
        axis.plot(
            result.shares,
            gains,
            marker="o",
            color=colour,
            lw=2,
            label=f"Goal gain, manipulation cost {cost:.1f}",
        )
    axis.axhline(0, color=ZERO_LINE, lw=1)
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_xticks(result.shares)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("share of candidates kept (harder optimisation to the right)")
    axis.set_ylabel("change in the metric, standard deviations")
    axis.set_title("Harder optimisation of a proxy need not move the goal")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="proxy_goodhart_selection", output_dir=output_dir)
