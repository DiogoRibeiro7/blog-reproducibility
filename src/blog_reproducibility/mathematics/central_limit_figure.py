"""Figure renderer for the article on the central limit theorem."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.mathematics.central_limit import (
    SAMPLE_SIZES,
    normal_reference,
    sample_means,
)


def render_convergence_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the sampling distribution of the mean at three sample sizes."""
    use_house_style()
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    grid = np.linspace(0, 3, 400)

    # One hue across the panels: the same quantity at three sample sizes.
    for axis, size, means in zip(axes, SAMPLE_SIZES, sample_means(), strict=True):
        axis.hist(means, bins=60, range=(0, 3), density=True, color=PALETTE[0], alpha=0.85)
        axis.plot(grid, normal_reference(grid, size), color=INK_PRIMARY, lw=1.6)
        axis.set_title(f"n = {size}", fontsize=11)
        axis.set_xlabel("sample mean")
    axes[0].set_ylabel("density")
    figure.suptitle(
        "The mean becomes normal long before the data does",
        x=0.012,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )
    axes[2].annotate(
        "normal reference",
        xy=(1.55, 1.2),
        xytext=(1.8, 1.7),
        fontsize=9,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "lw": 1, "color": INK_MUTED},
    )

    return save_figure(figure, slug="clt_convergence", output_dir=output_dir)
