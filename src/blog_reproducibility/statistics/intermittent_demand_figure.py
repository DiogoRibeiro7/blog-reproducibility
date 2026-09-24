"""Figure renderer for the article on forecasting intermittent demand."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.intermittent_demand import (
    IntermittentSummary,
    example_payload,
)

# Zero apart; the rolling mean and smoothing together; the two Croston forms together.
COLOURS = (PALETTE[1], PALETTE[0], PALETTE[0], PALETTE[2], PALETTE[2])
CONFIDENCE_Z = 1.96


def render_intermittent_demand_figure(
    *, output_dir: Path, summary: IntermittentSummary | None = None
) -> FigureArtifact:
    """Plot each method's average weekly forecast against the true demand rate."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    rows = result.rows
    truth = result.closed_forms.true_rate

    figure, axis = plt.subplots()
    positions = np.arange(len(rows))
    axis.barh(
        positions,
        [row.average_forecast for row in rows],
        xerr=np.array([row.standard_error for row in rows]) * CONFIDENCE_Z,
        color=COLOURS[: len(rows)],
        height=0.6,
        error_kw={"ecolor": INK_SECONDARY, "lw": 1.2},
    )
    axis.axvline(truth, color=INK_PRIMARY, lw=1.5, ls="--")
    axis.text(
        truth,
        -0.75,
        f"true rate, {truth:.2f} a week",
        color=INK_SECONDARY,
        fontsize=9,
        ha="center",
        va="bottom",
    )
    axis.set_ylim(len(rows) - 0.4, -0.95)
    axis.set_yticks(positions)
    axis.set_yticklabels([row.method for row in rows])
    axis.set_xlabel("average weekly forecast, units")
    # Undebiased Croston is 3.5 percent above the rate too, so "only one method is
    # wrong" overstated it; the forecast of zero is the one that is far off. The
    # title is kept short enough to fit the axes in DejaVu Sans, the fallback font
    # wherever the house style's Segoe UI is not installed.
    axis.set_title("Zero misses the rate but has the least absolute error")

    return save_figure(figure, slug="intermittent_forecast_bias", output_dir=output_dir)
