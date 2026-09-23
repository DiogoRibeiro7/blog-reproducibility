"""Figure renderer for the article on novelty effects and experiment duration."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.novelty_effects import (
    LONG_RUN_EFFECT,
    NoveltySummary,
    example_payload,
)


def render_novelty_figure(
    *, output_dir: Path, summary: NoveltySummary | None = None
) -> FigureArtifact:
    """Plot the true tenure effect against the tenure and calendar-day estimates."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    days = [row.day for row in result.rows]

    figure, axis = plt.subplots()
    axis.plot(
        days,
        [row.true_effect for row in result.rows],
        color=PALETTE[0],
        lw=2.4,
        label="True effect at that tenure",
    )
    axis.plot(
        days,
        [row.by_tenure for row in result.rows],
        marker="o",
        ms=4,
        ls="--",
        color=PALETTE[2],
        label="Estimated, grouped by user tenure",
    )
    axis.plot(
        days,
        [row.by_calendar for row in result.rows],
        marker="o",
        ms=4,
        color=PALETTE[1],
        label="Estimated, grouped by calendar day",
    )
    axis.axhline(
        LONG_RUN_EFFECT,
        color=PALETTE[3],
        lw=1,
        ls=":",
        label=f"Long-run effect ({100 * LONG_RUN_EFFECT:.0f} percent)",
    )
    axis.set_xlabel("day")
    axis.set_ylabel("estimated lift")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_title("Calendar days mix tenures and hide the decay")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="novelty_tenure_vs_calendar", output_dir=output_dir)
