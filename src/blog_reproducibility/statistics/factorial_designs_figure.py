"""Figure renderer for the article on one factor at a time against factorial designs."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.factorial_designs import FactorialSummary, example_payload


def render_factorial_designs_figure(
    *, output_dir: Path, summary: FactorialSummary | None = None
) -> FigureArtifact:
    """Plot the share of experiments ending at the best setting against the runs spent."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    factorial_runs = [row.runs for row in result.factorial]
    ofat_runs = [row.runs for row in result.ofat]

    figure, axis = plt.subplots()
    axis.plot(
        factorial_runs,
        [row.share for row in result.factorial],
        marker="o",
        color=PALETTE[0],
        label="Factorial design (8-run half fraction, then full 2⁴ replicated)",
    )
    axis.plot(
        ofat_runs,
        [row.share for row in result.ofat],
        marker="o",
        color=PALETTE[1],
        label="One factor at a time from the baseline",
    )
    axis.set_xscale("log")
    ticks = [*factorial_runs, ofat_runs[-1]]
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(runs) for runs in ticks])
    axis.set_xlabel("experimental runs")
    axis.set_ylabel("share of experiments ending at the best setting")
    axis.set_ylim(0, 1.04)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title("An interaction traps one-factor-at-a-time experiments")
    axis.legend(loc="center right")

    return save_figure(figure, slug="factorial_vs_ofat_optimum", output_dir=output_dir)
