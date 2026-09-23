"""Figure renderer for the article on kernel density estimation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.data_science.kernel_density import GRID, gaussian_kde, mixture_sample

LABELS = {0.12: "undersmoothed", 0.45: "about right", 1.40: "oversmoothed"}


def render_bandwidth_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the same sample's density estimate at three bandwidths, with a rug."""
    use_house_style()
    sample = mixture_sample()

    figure, axis = plt.subplots()
    axis.plot(
        sample,
        np.full(sample.size, -0.006),
        "|",
        color=INK_MUTED,
        markeredgecolor=INK_MUTED,
        markeredgewidth=0.8,
        markersize=6,
    )
    for index, (bandwidth, verdict) in enumerate(LABELS.items()):
        axis.plot(
            GRID,
            gaussian_kde(sample, bandwidth),
            color=PALETTE[index],
            label=f"h = {bandwidth:.2f} ({verdict})",
        )
    axis.set(
        title="Bandwidth decides whether you see one mode or two",
        xlabel="value",
        ylabel="density",
    )
    axis.legend()

    return save_figure(figure, slug="kde_bandwidth", output_dir=output_dir)
