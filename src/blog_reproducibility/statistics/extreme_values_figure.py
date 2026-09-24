"""Figure renderer for the article on estimating the tail you have not seen."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    BASELINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.extreme_values import (
    HOURS_PER_YEAR,
    RETURN_PERIODS,
    ExtremeValueSummary,
    example_payload,
)


def render_extreme_values_figure(
    *, output_dir: Path, summary: ExtremeValueSummary | None = None
) -> FigureArtifact:
    """Plot hourly exceedance probability against load: observed, true, Pareto and normal."""
    use_house_style()
    curves = (summary if summary is not None else example_payload()).curves

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.scatter(
        curves.observed_levels,
        curves.observed_probabilities,
        s=9,
        color=INK_MUTED,
        alpha=0.6,
        label="Observed (3 years)",
    )
    axis.plot(curves.levels, curves.true, color=INK_PRIMARY, lw=1.6, label="True tail")
    axis.plot(curves.levels, curves.pareto, color=PALETTE[0], label="Generalised Pareto fit")
    axis.plot(curves.levels, curves.normal, color=PALETTE[1], label="Normal fit")
    for years in RETURN_PERIODS:
        p = 1 / (years * HOURS_PER_YEAR)
        axis.axhline(p, color=BASELINE, lw=1.0)
        axis.annotate(
            f"{years}-year level",
            xy=(132, p),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=9,
            color=INK_SECONDARY,
        )
    axis.set_yscale("log")
    axis.set_ylim(3e-7, 3e-2)
    axis.set_xlim(125, 620)
    axis.set_xlabel("hourly peak load")
    axis.set_ylabel("probability an hour exceeds the level")
    axis.set_title("The data end at the sample maximum; the tail does not")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="extreme_value_tail_plot", output_dir=output_dir)
