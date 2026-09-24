"""Figure renderer for the article on capture-recapture estimation."""

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
from blog_reproducibility.statistics.capture_recapture import (
    TRUE_POPULATION,
    CaptureRecaptureSummary,
    example_payload,
)


def render_capture_recapture_figure(
    *, output_dir: Path, summary: CaptureRecaptureSummary | None = None
) -> FigureArtifact:
    """Plot the two- and three-pass estimates and the number seen against the spread."""
    use_house_style()
    curve = (summary if summary is not None else example_payload()).curve

    figure, axis = plt.subplots()
    axis.axhline(TRUE_POPULATION, color=INK_MUTED, lw=1.5, ls=":")
    axis.annotate(
        f"true population, {TRUE_POPULATION}",
        (curve.spreads[-1], TRUE_POPULATION),
        color=INK_SECONDARY,
        fontsize=9,
        ha="right",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points",
    )
    axis.plot(
        curve.spreads,
        curve.two_pass,
        marker="o",
        color=PALETTE[1],
        lw=2,
        label="Two passes, Chapman",
    )
    axis.plot(
        curve.spreads,
        curve.chao,
        marker="s",
        color=PALETTE[0],
        lw=2,
        label="Three passes, Chao lower bound",
    )
    axis.plot(
        curve.spreads,
        curve.seen,
        marker="^",
        color=PALETTE[2],
        lw=2,
        ls="--",
        label="Found by at least one pass",
    )
    axis.set_xlabel("spread in how easy items are to find")
    axis.set_ylabel("estimated number of items")
    axis.set_ylim(200, 600)
    axis.set_title("Uneven difficulty pulls the estimate below the truth")
    axis.legend(loc="lower left")

    return save_figure(figure, slug="capture_recapture_heterogeneity", output_dir=output_dir)
