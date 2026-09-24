"""Figure renderer for the article on Berkson's paradox in operational data."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.berkson_selection import SelectionCurve, selection_curve


def render_berkson_selection_figure(
    *, output_dir: Path, curve: SelectionCurve | None = None
) -> FigureArtifact:
    """Plot the correlation among escalated tickets against the share escalated."""
    use_house_style()
    result = curve if curve is not None else selection_curve()
    shares = list(result.shares)

    figure, axis = plt.subplots()
    axis.plot(
        shares,
        result.correlations,
        marker="o",
        color=PALETTE[1],
        label="Correlation among escalated tickets",
    )
    axis.axhline(
        0,
        color=PALETTE[0],
        lw=1.5,
        ls="--",
        label="Correlation in all tickets (independent attributes)",
    )
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_xticks(shares)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("share of tickets escalated (more selective to the right)")
    axis.set_ylabel("correlation of severity and customer value")
    axis.set_ylim(-0.8, 0.15)
    for share, value in zip(shares, result.correlations, strict=True):
        axis.annotate(
            f"{value:+.2f}",
            (share, value),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=9,
            color=PALETTE[1],
        )
    axis.set_title("Selecting on a sum makes independent things look opposed")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="berkson_selection_correlation", output_dir=output_dir)
