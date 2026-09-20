"""Figure renderers for the microbiome-testing article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SEQUENTIAL,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.health.microbiome_testing import closure_curve, flagged_again


def render_closure_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot how every other share falls when one taxon's absolute count grows."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(7.2, 4.4))

    shades = (SEQUENTIAL[2], SEQUENTIAL[4], SEQUENTIAL[7])
    for share, colour in zip((0.05, 0.15, 0.30), shades, strict=True):
        curve = closure_curve(share)
        axis.plot(
            [fold for fold, _ in curve],
            [change for _, change in curve],
            color=colour,
            label=f"the taxon that grew held {100 * share:.0f}% of the community",
        )

    axis.set_xscale("log")
    axis.set_xticks([1, 2, 5, 10])
    axis.set_xticklabels(["no change", "doubles", "fivefold", "tenfold"])
    axis.set_ylim(-80, 5)
    axis.set_xlabel("what happened to one taxon's absolute count")
    axis.set_ylabel("apparent change in every other taxon's share, %")
    axis.set_title("A percentage falls when its neighbour grows")
    axis.legend(loc="lower left")

    return save_figure(figure, slug="microbiome_compositional_closure", output_dir=output_dir)


def render_repeatability_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot how often a flagged taxon is flagged again in a second sample."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(7.2, 4.4))

    axis.axhline(
        5,
        color=INK_MUTED,
        linewidth=1.2,
        linestyle=(0, (3, 3)),
        label="no relation between the two samples",
    )
    grid = [index / 100 for index in range(100)]
    axis.plot(
        grid,
        [100 * flagged_again(value) for value in grid],
        color=PALETTE[0],
        label="a taxon with this repeatability",
    )

    half = 100 * flagged_again(0.5)
    axis.plot(
        [0.5],
        [half],
        "o",
        color=PALETTE[1],
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    axis.annotate(
        f"intraclass correlation 0.5: {half:.0f}%",
        xy=(0.5, half),
        xytext=(-12, 26),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 100)
    axis.set_xlabel("repeatability of the taxon within one person (intraclass correlation)")
    axis.set_ylabel("flagged results that are flagged again, %")
    axis.set_title("A flag that would not survive a second sample")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="microbiome_flag_repeatability", output_dir=output_dir)


def render_microbiome_testing_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the microbiome-testing article."""
    return (
        render_closure_figure(output_dir=output_dir),
        render_repeatability_figure(output_dir=output_dir),
    )
