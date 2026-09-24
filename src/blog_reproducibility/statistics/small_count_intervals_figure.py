"""Figure renderer for the article on zero failures, small counts and the rule of three."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.small_count_intervals import (
    SmallCountSummary,
    example_payload,
)


def render_small_count_figure(
    *, output_dir: Path, summary: SmallCountSummary | None = None
) -> FigureArtifact:
    """Plot the simulated coverage of three 95 percent intervals against expected events."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    rows = result.simulated
    expected = [row.expected_events for row in rows]

    figure, axis = plt.subplots()
    axis.plot(
        expected,
        [row.coverage.wald for row in rows],
        marker="o",
        color=PALETTE[1],
        lw=2,
        label="Wald (normal approximation)",
    )
    axis.plot(
        expected,
        [row.coverage.wilson for row in rows],
        marker="o",
        color=PALETTE[0],
        lw=2,
        label="Wilson score",
    )
    axis.plot(
        expected,
        [row.coverage.clopper_pearson for row in rows],
        marker="o",
        color=PALETTE[2],
        lw=2,
        label="Exact (Clopper-Pearson)",
    )
    axis.axhline(0.95, color=PALETTE[3], lw=1, ls="--", label="Nominal 95 percent")
    axis.set_xscale("log")
    axis.set_xticks(expected)
    axis.set_xticklabels([f"{events:g}" for events in expected])
    axis.set_xlabel("expected number of events in the sample (true rate 0.5 percent)")
    axis.set_ylabel("share of intervals containing the true rate")
    axis.set_ylim(0, 1.02)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_title("Below a handful of events, the textbook interval fails")
    axis.legend(loc="lower right")

    return save_figure(figure, slug="small_count_interval_coverage", output_dir=output_dir)
