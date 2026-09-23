"""Figure renderer for the article on silent data-quality failures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.data_science.null_monitoring import MonitoringSummary, example_payload


def render_detection_figure(
    *, output_dir: Path, summary: MonitoringSummary | None = None
) -> FigureArtifact:
    """Plot how often each monitor fires against the null rate."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    rates = [row.null_rate for row in result.detection]

    figure, axis = plt.subplots()
    axis.plot(
        rates,
        [row.null_rate_monitor for row in result.detection],
        marker="o",
        color=PALETTE[0],
        lw=2,
        label="Monitor on the null rate",
    )
    axis.plot(
        rates,
        [row.metric_monitor for row in result.detection],
        marker="s",
        color=PALETTE[1],
        lw=2,
        label="Monitor on the metric",
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set(
        xlabel="share of rows arriving with a null value",
        ylabel="share of days the monitor fires",
        title="The input monitor knows before the output monitor",
    )
    axis.legend(loc="lower right")

    return save_figure(figure, slug="null_monitor_detection", output_dir=output_dir)
