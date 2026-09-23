"""Figure renderer for the article on splines."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.data_science.splines import POLYNOMIAL_DEGREE, fit_curves


def render_splines_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the observations with the line, polynomial, and spline fits."""
    use_house_style()
    fits = fit_curves()

    figure, axis = plt.subplots()
    axis.plot(
        fits.x,
        fits.y,
        "o",
        color=INK_MUTED,
        markersize=4,
        alpha=0.6,
        markeredgecolor="none",
        label="Observations",
    )
    axis.plot(fits.x, fits.line, color=PALETTE[3], label="Linear fit")
    axis.plot(
        fits.x, fits.polynomial, color=PALETTE[1], label=f"Degree-{POLYNOMIAL_DEGREE} polynomial"
    )
    axis.plot(fits.x, fits.spline, color=PALETTE[0], label="Cubic spline")
    axis.set(
        title="A spline bends locally; a high-degree polynomial wobbles globally",
        xlabel="x",
        ylabel="y",
    )
    axis.legend(ncol=2)

    return save_figure(figure, slug="splines_fit", output_dir=output_dir)
