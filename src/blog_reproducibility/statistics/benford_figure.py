"""Figure renderer for the article on Benford's law as a screen."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.benford import (
    ACCEPTABLE_CONFORMITY,
    CLOSE_CONFORMITY,
    VALUES,
    BenfordSummary,
    example_payload,
)

TICKS = (2, 10, 100, 1000, 10000, 100000)
TICK_LABELS = ("2x", "10x", "100x", "1,000x", "10,000x", "100,000x")


def render_benford_figure(
    *, output_dir: Path, summary: BenfordSummary | None = None
) -> FigureArtifact:
    """Plot the mean absolute deviation from Benford against the range of the column."""
    use_house_style()
    curve = (summary if summary is not None else example_payload()).curve

    figure, axis = plt.subplots()
    axis.plot(
        curve.spans,
        curve.mads,
        marker="o",
        color=PALETTE[0],
        lw=2,
        label=f"Lognormal column, {VALUES:,} values",
    )
    axis.axhline(
        CLOSE_CONFORMITY,
        color=PALETTE[1],
        lw=1.5,
        ls="--",
        label=f"Close conformity, {CLOSE_CONFORMITY}",
    )
    axis.axhline(
        ACCEPTABLE_CONFORMITY,
        color=PALETTE[3],
        lw=1.5,
        ls=":",
        label=f"Acceptable conformity, {ACCEPTABLE_CONFORMITY}",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(TICKS)
    axis.set_xticklabels(TICK_LABELS)
    axis.set_xlabel("range the data covers, 2.5th to 97.5th percentile")
    axis.set_ylabel("mean absolute deviation from Benford")
    axis.set_title("Whether a column follows the law is decided by its range")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="benford_spread", output_dir=output_dir)
