"""Figure renderer for the article on empirical Bayes shrinkage across experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.empirical_bayes import ShrunkProgramme, figure_programme

AXIS_LOW = -0.005
X_HIGH = 0.045
Y_HIGH = 0.05


def render_empirical_bayes_figure(
    *, output_dir: Path, programme: ShrunkProgramme | None = None
) -> FigureArtifact:
    """Plot the raw and shrunk estimates of the winners against their true effects."""
    use_house_style()
    result = programme if programme is not None else figure_programme()
    win = result.winners
    truth = result.programme.effects[win]

    figure, axis = plt.subplots()
    axis.plot(
        [AXIS_LOW, Y_HIGH],
        [AXIS_LOW, Y_HIGH],
        color=INK_MUTED,
        lw=1.4,
        ls="--",
        label="Perfect agreement",
    )
    axis.scatter(
        truth,
        result.programme.estimates[win],
        s=14,
        alpha=0.45,
        color=PALETTE[1],
        label="Raw estimate",
    )
    axis.scatter(
        truth, result.shrunk[win], s=14, alpha=0.45, color=PALETTE[0], label="Shrunk estimate"
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    axis.set_xlim(AXIS_LOW, X_HIGH)
    axis.set_ylim(AXIS_LOW, Y_HIGH)
    axis.set_xlabel("true effect")
    axis.set_ylabel("reported effect")
    axis.set_title("Winners reported raw, and the same winners shrunk")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="empirical_bayes_shrinkage", output_dir=output_dir)
