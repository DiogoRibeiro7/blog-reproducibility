"""Figure renderer for the article on Type S and Type M errors in small studies."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, PercentFormatter

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.design_analysis import (
    SIGNIFICANCE,
    DesignAnalysisSummary,
    example_payload,
)

_COLOURS = {0.10: PALETTE[2], 0.05: PALETTE[1], 0.01: PALETTE[0]}


def render_design_analysis_figure(
    *, output_dir: Path, summary: DesignAnalysisSummary | None = None
) -> FigureArtifact:
    """Plot the exaggeration of a significant estimate against power at three thresholds."""
    use_house_style()
    result = summary if summary is not None else example_payload()

    figure, axis = plt.subplots()
    for curve in result.curves:
        colour = _COLOURS[curve.significance]
        axis.plot(
            curve.powers,
            curve.exaggerations,
            color=colour,
            lw=2,
            label=f"Significance at {curve.significance:.2f}",
        )
        if curve.significance == SIGNIFICANCE:
            for point in result.annotated:
                axis.plot([point.power], [point.exaggeration], marker="o", color=colour, ms=6)
                axis.annotate(
                    f"{point.exaggeration:.2f}x",
                    (point.power, point.exaggeration),
                    color=INK_SECONDARY,
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    xytext=(6, 4),
                    textcoords="offset points",
                )
    axis.axhline(1, color=INK_MUTED, lw=1.2, ls=":")
    axis.set_yscale("log")
    axis.set_yticks([1, 1.5, 2, 3, 5, 8])
    axis.set_yticklabels(["1x", "1.5x", "2x", "3x", "5x", "8x"])
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_ylim(0.95, 9)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("power of the study")
    axis.set_ylabel("size of a significant estimate, relative to the truth")
    axis.set_title("A significant result from a small study is mostly filter")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="design_analysis_type_sm", output_dir=output_dir)
