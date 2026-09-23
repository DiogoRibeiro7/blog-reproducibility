"""Figure renderer for the article on percentile metrics in latency experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.percentile_metrics import PrecisionRow, precision_rows

_SERIES = (
    ("mean", "Mean", PALETTE[0]),
    ("median", "Median", PALETTE[2]),
    ("p95", "95th percentile", PALETTE[3]),
    ("p99", "99th percentile", PALETTE[1]),
)


def render_percentile_figure(
    *, output_dir: Path, rows: tuple[PrecisionRow, ...] | None = None
) -> FigureArtifact:
    """Plot the relative standard error of each statistic against the requests measured."""
    use_house_style()
    result = rows if rows is not None else precision_rows()
    sizes = [row.requests for row in result]

    figure, axis = plt.subplots()
    for field, label, colour in _SERIES:
        axis.plot(
            sizes,
            [float(getattr(row, field)) for row in result],
            marker="o",
            color=colour,
            lw=2,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([f"{size:,}" for size in sizes])
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    axis.set_xlabel("requests measured")
    axis.set_ylabel("relative standard error of the statistic")
    axis.set_title("The tail is the hardest part of the distribution to measure")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="percentile_precision", output_dir=output_dir)
