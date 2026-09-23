"""Figure renderer for the article on learning curves and whether more data helps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.learning_curves import (
    LearningCurveSummary,
    example_payload,
)

_SERIES = (
    ("logistic", PALETTE[1], "Logistic regression"),
    ("boosting", PALETTE[0], "Gradient boosting"),
)


def _label_end(axis: Axes, x: float, y: float, text: str, colour: str) -> None:
    """Direct-label a series at its final point, as the website's house style does."""
    axis.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=INK_SECONDARY,
        fontsize=9.5,
    )
    axis.plot(
        [x],
        [y],
        marker="o",
        markersize=5,
        color=colour,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=5,
    )


def render_learning_curves_figure(
    *, output_dir: Path, summary: LearningCurveSummary | None = None
) -> FigureArtifact:
    """Plot both learning curves with power laws fitted to the smallest sizes."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    fitted = result.fitted_sizes

    figure, axis = plt.subplots()
    for model, colour, label in _SERIES:
        curve = result.curve(model)
        fit = result.fit(model)
        sizes = curve.sizes
        errors = curve.test_errors
        grid = np.logspace(np.log10(sizes[0]), np.log10(sizes[-1]), 200)
        axis.plot(grid, [fit.predict(n) for n in grid], color=colour, lw=1.2, alpha=0.7)
        axis.plot(
            sizes[:fitted],
            errors[:fitted],
            "o",
            color=colour,
            markersize=6,
            label=f"{label}, used for fit",
        )
        axis.plot(
            sizes[fitted:],
            errors[fitted:],
            "o",
            markerfacecolor=SURFACE,
            markeredgecolor=colour,
            markeredgewidth=1.8,
            markersize=6,
            label=f"{label}, held out",
        )
        end = fit.predict(sizes[-1])
        _label_end(axis, sizes[-1], end, f"fit predicts {end:.3f}", colour)

    sizes = result.curves[0].sizes
    axis.axhline(result.bayes_error, color=INK_MUTED, lw=1.2)
    axis.annotate(
        f"Bayes error {result.bayes_error:.3f}",
        xy=(260, result.bayes_error),
        xytext=(0, -12),
        textcoords="offset points",
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    axis.set_xscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([f"{size:,}" for size in sizes])
    axis.set_xlabel("training examples")
    axis.set_ylabel("test error rate")
    axis.set_ylim(0.19, 0.34)
    axis.set_title("A power law fitted to small sizes predicts the large ones")
    axis.legend(loc="upper right", ncol=2)

    return save_figure(figure, slug="learning_curves_extrapolation", output_dir=output_dir)
