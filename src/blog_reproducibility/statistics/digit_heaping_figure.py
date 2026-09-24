"""Figure renderer for the article on digit heaping and thresholds."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    GRID,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.digit_heaping import (
    ROUND_THRESHOLDS,
    HeapingSummary,
    example_payload,
)


def render_digit_heaping_figure(
    *, output_dir: Path, summary: HeapingSummary | None = None
) -> FigureArtifact:
    """Plot the true and recorded shares above each threshold, round numbers marked."""
    use_house_style()
    curve = (summary if summary is not None else example_payload()).curve

    figure, axis = plt.subplots()
    axis.plot(
        curve.thresholds,
        curve.true_shares,
        color=PALETTE[0],
        lw=2,
        label="True share above the threshold",
    )
    axis.plot(
        curve.thresholds,
        curve.recorded_shares,
        color=PALETTE[1],
        lw=2,
        label="Recorded share, half the entries rounded",
    )
    for threshold in ROUND_THRESHOLDS:
        axis.axvline(threshold, color=GRID, lw=1, zorder=0)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("threshold, minutes")
    axis.set_ylabel("share of records above it")
    axis.set_title("The gap is widest exactly where thresholds are written")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="heaping_threshold_error", output_dir=output_dir)
