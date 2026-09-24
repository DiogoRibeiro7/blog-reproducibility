"""Figure renderer for the article on how long a monitor takes to notice."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.run_length import RunLengthSummary, example_payload

COLOURS = (PALETTE[3], PALETTE[0], PALETTE[2])
TICKS = (1, 2, 5, 10, 25, 50, 100, 250)


def render_run_length_figure(
    *, output_dir: Path, summary: RunLengthSummary | None = None
) -> FigureArtifact:
    """Plot mean days to detection against the size of the shift, one line per chart."""
    use_house_style()
    curves = (summary if summary is not None else example_payload()).curves

    figure, axis = plt.subplots()
    for curve, colour in zip(curves, COLOURS, strict=True):
        axis.plot(curve.shifts, curve.mean_days, marker="o", color=colour, lw=2, label=curve.chart)
    axis.set_yscale("log")
    axis.set_yticks(TICKS)
    axis.set_yticklabels([str(tick) for tick in TICKS])
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("size of the shift, in standard deviations")
    axis.set_ylabel("mean days until the monitor signals")
    axis.set_title("Memory buys speed on small shifts, not on large ones")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="monitor_detection_delay", output_dir=output_dir)
