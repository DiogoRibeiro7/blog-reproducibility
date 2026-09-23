"""Figure renderer for the article on outlier algorithms."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SURFACE,
    FigureArtifact,
    save_figure,
    sequential_cmap,
    use_house_style,
)
from blog_reproducibility.data_science.histogram_outliers import BINS, EXTENT, flagged_sample


def render_histogram_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the 2D histogram and the points flagged for sitting in near-empty cells."""
    use_house_style()
    sample = flagged_sample()
    points, flagged = sample.points, sample.flagged

    figure, (density, flags) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    histogram = density.hist2d(
        points[:, 0],
        points[:, 1],
        bins=BINS,
        range=[list(EXTENT), list(EXTENT)],
        cmap=sequential_cmap(),
        cmin=1,
    )
    density.set_title("2D histogram: density per cell", fontsize=11)
    colourbar = figure.colorbar(histogram[3], ax=density)
    colourbar.outline.set_visible(False)
    colourbar.set_label("count", color=INK_SECONDARY)

    flags.plot(
        points[~flagged, 0],
        points[~flagged, 1],
        "o",
        markersize=3,
        color=INK_MUTED,
        alpha=0.35,
        markeredgecolor="none",
        label="Dense cells",
    )
    flags.plot(
        points[flagged, 0],
        points[flagged, 1],
        "o",
        markersize=5,
        color=PALETTE[1],
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label="Sparse cells (flagged)",
    )
    flags.set_title("Points falling in near-empty cells", fontsize=11)
    flags.set(xlim=EXTENT, ylim=EXTENT)
    flags.legend()
    for axis in (density, flags):
        axis.set_xlabel("feature 1")
    density.set_ylabel("feature 2")

    return save_figure(figure, slug="hist2d_outliers", output_dir=output_dir)
