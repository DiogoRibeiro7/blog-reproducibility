"""Figure renderer for the article on week-over-week comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.week_over_week import (
    SMALL_MOVE,
    WeekOverWeekSummary,
    example_payload,
)


def render_week_over_week_figure(
    *, output_dir: Path, summary: WeekOverWeekSummary | None = None
) -> FigureArtifact:
    """Plot the histograms of quiet single-day and seven-day comparisons."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    edges = np.array(result.bin_edges)
    centres = (edges[:-1] + edges[1:]) / 2

    figure, axis = plt.subplots()
    # Each bin's centre weighted by its density draws the website's density histogram.
    axis.hist(
        centres,
        bins=result.bin_edges,
        weights=result.same_weekday_density,
        alpha=0.55,
        color=PALETTE[1],
        label=f"Same weekday last week, spread {result.same_weekday.simulated.spread:.1%}",
    )
    axis.hist(
        centres,
        bins=result.bin_edges,
        weights=result.seven_day_density,
        alpha=0.55,
        color=PALETTE[0],
        label=f"Seven-day averages, spread {result.seven_day.simulated.spread:.1%}",
    )
    for x in (-SMALL_MOVE, SMALL_MOVE):
        axis.axvline(x, color=INK_MUTED, lw=1.4, ls="--")
    axis.annotate(
        "the five percent that starts an investigation",
        (SMALL_MOVE, axis.get_ylim()[1] * 0.92),
        color=INK_SECONDARY,
        fontsize=9,
        ha="left",
        va="top",
        xytext=(6, 0),
        textcoords="offset points",
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("measured change against the comparison period")
    axis.set_ylabel("density")
    axis.set_title("What the comparison does when nothing has happened")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="wow_noise_distribution", output_dir=output_dir)
