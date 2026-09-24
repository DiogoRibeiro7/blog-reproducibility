"""Figure renderer for the article on what a clean sample of fifty proves."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.acceptance_sampling import (
    ACCEPTABLE_QUALITY,
    LOT_TOLERANCE,
    AcceptanceSamplingSummary,
    example_payload,
)

COLOURS = (PALETTE[3], PALETTE[1], PALETTE[2], PALETTE[0])


def render_acceptance_sampling_figure(
    *, output_dir: Path, summary: AcceptanceSamplingSummary | None = None
) -> FigureArtifact:
    """Plot the operating characteristic of four plans against the lot's true defect rate."""
    use_house_style()
    curves = (summary if summary is not None else example_payload()).curves

    figure, axis = plt.subplots()
    for curve, colour in zip(curves, COLOURS, strict=True):
        axis.plot(
            curve.rates,
            curve.acceptance,
            color=colour,
            lw=2,
            label=f"Inspect {curve.sample_size}, allow {curve.accept_number}",
        )
    for rate, text in (
        (ACCEPTABLE_QUALITY, "agreed quality, 1%"),
        (LOT_TOLERANCE, "must be caught, 5%"),
    ):
        axis.axvline(rate, color=INK_MUTED, lw=1.2, ls=":")
        axis.annotate(
            text,
            (rate, 0.03),
            color=INK_SECONDARY,
            fontsize=9,
            ha="left",
            va="bottom",
            xytext=(4, 0),
            textcoords="offset points",
        )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("true defect rate of the lot")
    axis.set_ylabel("chance the lot is accepted")
    axis.set_title("Bigger samples separate good lots from bad, not just reject more")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="acceptance_sampling_oc", output_dir=output_dir)
