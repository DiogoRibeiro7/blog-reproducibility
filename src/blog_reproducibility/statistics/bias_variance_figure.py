"""Figure renderers for the article on statistics and machine learning."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    BASELINE,
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.bias_variance import (
    LABEL_THRESHOLD,
    best_degree,
    lasso_coefficient_path,
    simulate_complexity_curve,
)

__all__ = ["render_bias_variance_figure", "render_regularization_paths_figure"]


def render_bias_variance_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot mean training and test error against polynomial degree."""
    use_house_style()
    curve = simulate_complexity_curve()
    degrees = curve.degrees
    best = best_degree(curve)
    lowest = min(curve.test_error)

    figure, axis = plt.subplots()
    axis.plot(
        degrees,
        curve.training_error,
        color=PALETTE[0],
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label="Training error",
    )
    axis.plot(
        degrees,
        curve.test_error,
        color=PALETTE[1],
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label="Test error",
    )
    axis.axvline(best, color=INK_MUTED, lw=1.0)
    axis.annotate(
        f"test error minimised\nat degree {best}",
        xy=(best, lowest),
        xytext=(best + 1.2, lowest + 0.10),
        fontsize=9,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "lw": 1, "color": INK_MUTED},
    )
    axis.set_title("Training error keeps falling; test error does not")
    axis.set_xlabel("polynomial degree")
    axis.set_ylabel("mean squared error")
    axis.legend()
    return save_figure(figure, slug="bias_variance", output_dir=output_dir)


def render_regularization_paths_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot every lasso coefficient against the penalty, labelling those that survive."""
    use_house_style()
    path = lasso_coefficient_path()
    penalties, coefficients = path.penalties, path.coefficients

    # All eight paths are the same kind of thing, so they share one hue;
    # identity comes from direct labels on the survivors, not from eight
    # colours (four of which would sit indistinguishably on top of zero).
    figure, axis = plt.subplots()
    axis.set_xscale("log")
    # Largest penalty on the left, as the axis label says, so the paths read
    # from everything-zero to near least squares; the extra room past the
    # smallest penalty on the right holds the labels.
    axis.set_xlim(float(penalties[0]), float(penalties[-1]) * 0.45)
    for j, row in enumerate(coefficients):
        survives = bool(np.abs(row[-1]) > LABEL_THRESHOLD)
        axis.plot(
            penalties,
            row,
            color=PALETTE[0],
            alpha=1.0 if survives else 0.30,
            lw=2.0 if survives else 1.2,
        )
        if survives:
            # The final value sits at the right end of the path: label it in the margin.
            axis.annotate(
                f"$x_{{{j + 1}}}$",
                xy=(float(penalties[-1]), float(row[-1])),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=9.5,
                color=INK_SECONDARY,
                va="center",
                ha="left",
            )
    axis.axhline(0, color=BASELINE, lw=0.8)
    axis.set_title("Lasso drives coefficients to exactly zero, one by one")
    axis.set_xlabel("regularisation strength $\\alpha$ (log scale, decreasing)")
    axis.set_ylabel("coefficient")
    return save_figure(figure, slug="regularization_paths", output_dir=output_dir)
