"""Figure renderer for the article on optional stopping and sequential tests."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.sequential_testing import (
    FigureBoundaries,
    figure_boundaries,
)


def render_sequential_figure(
    *, output_dir: Path, boundaries: FigureBoundaries | None = None
) -> FigureArtifact:
    """Plot the critical z each stopping rule demands against the fraction of the sample."""
    use_house_style()
    result = boundaries if boundaries is not None else figure_boundaries()
    pocock = result.pocock.critical_values
    looks = len(pocock)
    fraction = np.arange(1, looks + 1) / looks
    days = len(result.mixture)

    figure, axis = plt.subplots()
    axis.plot(
        fraction,
        np.full(looks, result.naive),
        marker="o",
        color=PALETTE[3],
        lw=2,
        label="Naive 0.05 at every look",
    )
    axis.plot(
        fraction,
        pocock,
        marker="s",
        color=PALETTE[2],
        lw=2,
        label=f"Pocock, flat {pocock[0]:.2f}",
    )
    axis.plot(
        fraction,
        result.obrien_fleming.critical_values,
        marker="D",
        color=PALETTE[0],
        lw=2,
        label="O'Brien-Fleming",
    )
    axis.plot(
        np.arange(1, days + 1) / days,
        result.mixture,
        color=PALETTE[1],
        lw=2,
        ls="--",
        label="Mixture rule, every day",
    )
    axis.set_xlabel("fraction of the planned sample collected")
    axis.set_ylabel("critical z value to stop")
    axis.set_ylim(1.5, 7.0)
    axis.set_title("What each rule demands before it lets you stop")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="sequential_boundaries", output_dir=output_dir)
