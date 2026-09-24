"""Figure renderer for the article on recurrent failures and the mean cumulative function."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.recurrent_events import (
    RecurrentEventsSummary,
    example_payload,
)


def render_recurrent_events_figure(
    *, output_dir: Path, summary: RecurrentEventsSummary | None = None
) -> FigureArtifact:
    """Plot the mean cumulative function by site and overall against the naive average count."""
    use_house_style()
    curves = (summary if summary is not None else example_payload()).curves

    figure, axis = plt.subplots()
    axis.plot(
        curves.ages,
        curves.harsh,
        color=PALETTE[1],
        lw=2,
        label="Harsh sites, mean cumulative function",
    )
    axis.plot(
        curves.ages,
        curves.all_machines,
        color=PALETTE[0],
        lw=2.2,
        label="All machines, mean cumulative function",
    )
    axis.plot(
        curves.ages,
        curves.normal,
        color=PALETTE[2],
        lw=2,
        label="Normal sites, mean cumulative function",
    )
    axis.plot(
        curves.ages,
        curves.naive,
        color=PALETTE[3],
        lw=2,
        ls="--",
        label="All machines, naive average count",
    )
    axis.set_xlabel("age (years)")
    axis.set_ylabel("cumulative failures per machine")
    axis.set_title("Counting failures per machine needs a risk set")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="recurrent_events_mcf", output_dir=output_dir)
