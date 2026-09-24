"""Figure renderer for the article on trigger dilution in feature experiments."""

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.trigger_dilution import annotated_gaps, sample_size_curve


def render_trigger_dilution_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot users per arm for each analysis against the trigger rate, on log scales."""
    use_house_style()
    rates, needed_all, needed_triggered = sample_size_curve()

    figure, axis = plt.subplots()
    axis.plot(rates, needed_all, color=PALETTE[1], lw=2, label="All users, diluted effect")
    axis.plot(rates, needed_triggered, color=PALETTE[0], lw=2, label="Triggered users only")
    for gap in annotated_gaps():
        axis.plot(
            [gap.trigger_rate, gap.trigger_rate],
            [gap.users_triggered, gap.users_all],
            color=INK_MUTED,
            lw=1,
            ls=":",
        )
        axis.annotate(
            f"{gap.ratio:.0f}x",
            (gap.trigger_rate, sqrt(gap.users_all * gap.users_triggered)),
            color=INK_SECONDARY,
            fontsize=9,
            ha="left",
            xytext=(4, 0),
            textcoords="offset points",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks([0.01, 0.02, 0.05, 0.10, 0.20, 0.50])
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("share of users who reach the feature")
    axis.set_ylabel("users per arm for 80% power")
    axis.set_title("What it costs to measure a feature on everybody")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="trigger_dilution_cost", output_dir=output_dir)
