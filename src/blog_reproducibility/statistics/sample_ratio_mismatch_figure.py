"""Figure renderer for the article on sample ratio mismatch."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.sample_ratio_mismatch import DropRow, simulate_drops


def render_mismatch_figure(
    *, output_dir: Path, rows: tuple[DropRow, ...] | None = None
) -> FigureArtifact:
    """Plot the bias in the measured lift and the alarm rate against the drop share."""
    use_house_style()
    result = rows if rows is not None else simulate_drops()
    drops = [row.drop for row in result]

    figure, axis = plt.subplots()
    axis.plot(
        drops,
        [row.relative_bias for row in result],
        marker="o",
        color=PALETTE[1],
        lw=2,
        label="Relative bias in the measured lift",
    )
    axis.plot(
        drops,
        [row.alarm_rate for row in result],
        marker="o",
        color=PALETTE[0],
        lw=2,
        label="Experiments whose sample ratio check fires",
    )
    axis.set_xlabel("share of treated users lost before logging")
    axis.set_ylabel("bias as a share of the true lift; alarm rate")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_title("The alarm fires where the damage becomes serious")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="srm_detection_and_bias", output_dir=output_dir)
