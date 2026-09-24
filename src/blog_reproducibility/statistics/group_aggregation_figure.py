"""Figure renderer for the article on group averages and the ecological fallacy."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.group_aggregation import (
    GroupAggregationSummary,
    example_payload,
)

COLOURS = (PALETTE[3], PALETTE[0], PALETTE[2])
TICKS = (2, 5, 10, 25, 100, 500, 2000)


def render_group_aggregation_figure(
    *, output_dir: Path, summary: GroupAggregationSummary | None = None
) -> FigureArtifact:
    """Plot the correlation of group means against group size, measured and predicted."""
    use_house_style()
    result = summary if summary is not None else example_payload()

    figure, axis = plt.subplots()
    for curve, colour in zip(result.curves, COLOURS, strict=True):
        axis.plot(
            curve.group_sizes,
            curve.measured,
            marker="o",
            markersize=4,
            color=colour,
            lw=2,
            label=f"{curve.intraclass_correlation:.0%} of each variable is group",
        )
        axis.plot(curve.group_sizes, curve.predicted, color=colour, lw=1, ls=":")
    axis.axhline(result.between_correlation, color=INK_MUTED, lw=1.5, ls="--")
    axis.annotate(
        f"correlation between the group parts, {result.between_correlation:.2f}",
        (result.curves[0].group_sizes[-1], result.between_correlation),
        color=INK_SECONDARY,
        fontsize=9,
        ha="right",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points",
    )
    axis.set_xscale("log")
    axis.set_xticks(TICKS)
    axis.set_xticklabels([f"{tick:,}" for tick in TICKS])
    axis.set_ylim(0, 1)
    axis.set_xlabel("people averaged into each group")
    axis.set_ylabel("correlation between group averages")
    axis.set_title("Bigger buckets, stronger correlation, same people")
    axis.legend(loc="lower right")

    return save_figure(figure, slug="aggregation_group_size", output_dir=output_dir)
