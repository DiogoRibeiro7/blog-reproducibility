"""Figure renderer for the article on regression to the mean in operational analytics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    BASELINE,
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.regression_to_the_mean import ScatterData, scatter_data


def render_regression_to_the_mean_figure(
    *, output_dir: Path, data: ScatterData | None = None
) -> FigureArtifact:
    """Scatter month two against month one, the worst decile highlighted, with both lines."""
    use_house_style()
    result = data if data is not None else scatter_data()

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    axis.scatter(
        result.rest_month_one,
        result.rest_month_two,
        s=14,
        color=INK_MUTED,
        alpha=0.5,
        label="Other machines",
    )
    axis.scatter(
        result.worst_month_one,
        result.worst_month_two,
        s=20,
        color=PALETTE[1],
        alpha=0.9,
        label="Worst 10% in month 1",
    )
    lim = np.array(result.limits)
    axis.plot(lim, lim, color=BASELINE, lw=1.4, label="No change (y = x)")
    axis.plot(
        lim,
        result.mean + result.correlation * (lim - result.mean),
        color=PALETTE[0],
        lw=2.2,
        label=f"Expected month 2 (rho = {result.correlation:.2f})",
    )
    mx, my = result.worst_mean_month_one, result.worst_mean_month_two
    axis.plot(
        [mx], [my], marker="D", markersize=9, color=PALETTE[1], markeredgecolor=SURFACE, zorder=6
    )
    axis.annotate(
        f"worst-decile mean\n{mx:.1f} in month 1, {my:.1f} in month 2",
        xy=(mx, my),
        xytext=(-150, 60),
        textcoords="offset points",
        fontsize=9.5,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 1},
    )
    axis.set_xlabel("failures in month 1")
    axis.set_ylabel("failures in month 2")
    axis.set_title("Selected on month 1, the worst decile lands on the regression line")
    axis.set_xlim(result.limits)
    axis.set_ylim(result.limits)
    axis.grid(axis="both")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="regression_to_the_mean_scatter", output_dir=output_dir)
