"""Figure renderer for the article on regression discontinuity designs."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.regression_discontinuity import (
    RegressionDiscontinuitySummary,
    example_payload,
)


def render_regression_discontinuity_figure(
    *, output_dir: Path, summary: RegressionDiscontinuitySummary | None = None
) -> FigureArtifact:
    """Plot the error, noise and bias of the local linear estimate against the bandwidth."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    bandwidths = [row.bandwidth for row in result.rows]

    figure, axis = plt.subplots()
    axis.plot(
        bandwidths,
        [row.root_mean_squared_error for row in result.rows],
        marker="o",
        color=PALETTE[0],
        lw=2.2,
        label="Root mean squared error",
    )
    axis.plot(
        bandwidths,
        [row.standard_deviation for row in result.rows],
        marker="o",
        color=PALETTE[2],
        label="Standard deviation (noise)",
    )
    axis.plot(
        bandwidths,
        [row.absolute_bias for row in result.rows],
        marker="o",
        color=PALETTE[1],
        label="Absolute bias (curvature)",
    )
    axis.set_xscale("log")
    axis.set_xticks(bandwidths)
    axis.set_xticklabels([f"{h:g}" for h in bandwidths])
    # Only the chosen ticks are labelled, as on the website; log axes add minor labels otherwise.
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("bandwidth on each side of the cutoff (units of the running variable)")
    axis.set_ylabel("error of the estimated effect")
    axis.set_title("The bandwidth trades curvature bias against noise")
    axis.legend(loc="upper center")

    return save_figure(figure, slug="rdd_bandwidth_tradeoff", output_dir=output_dir)
