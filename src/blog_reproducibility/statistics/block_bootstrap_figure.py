"""Figure renderer for the article on when the bootstrap fails."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    BASELINE,
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.block_bootstrap import (
    NOMINAL_COVERAGE,
    BlockBootstrapSummary,
    example_payload,
)

COLOURS = (INK_MUTED, PALETTE[2], PALETTE[0], PALETTE[1])


def render_block_bootstrap_figure(
    *, output_dir: Path, summary: BlockBootstrapSummary | None = None
) -> FigureArtifact:
    """Plot interval coverage against block length, one line per autocorrelation."""
    use_house_style()
    result = summary if summary is not None else example_payload()

    figure, axis = plt.subplots()
    for row, colour in zip(result.rows, COLOURS, strict=True):
        axis.plot(
            row.block_lengths,
            row.coverage,
            marker="o",
            color=colour,
            label=f"autocorrelation {row.autocorrelation:.1f}",
        )
    blocks = result.rows[0].block_lengths
    axis.axhline(NOMINAL_COVERAGE, color=BASELINE, lw=1.2)
    axis.annotate(
        "nominal 95%",
        xy=(1, NOMINAL_COVERAGE),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    axis.set_xscale("log")
    axis.set_xticks(blocks)
    axis.set_xticklabels([str(block) for block in blocks])
    axis.set_xlabel("block length (1 = ordinary bootstrap)")
    axis.set_ylabel("coverage of the 95% interval")
    axis.set_ylim(0.25, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title("The ordinary bootstrap breaks as soon as observations are dependent")
    axis.legend(loc="lower right")

    return save_figure(figure, slug="block_bootstrap_coverage", output_dir=output_dir)
