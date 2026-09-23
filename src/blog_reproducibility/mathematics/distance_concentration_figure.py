"""Figure renderer for the article on distance concentration in high dimensions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    BASELINE,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.mathematics.distance_concentration import (
    NOISE_DIMENSIONS,
    DistanceSummary,
    example_payload,
)


def render_concentration_figure(
    *, output_dir: Path, summary: DistanceSummary | None = None
) -> FigureArtifact:
    """Plot relative contrast against dimension and classifier accuracy against noise."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    figure, (contrast, accuracy) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    dimensions = [row.dimension for row in result.contrast]
    contrast.plot(
        dimensions,
        [row.uniform for row in result.contrast],
        marker="o",
        color=PALETTE[0],
        label="Uniform",
    )
    contrast.plot(
        dimensions,
        [row.gaussian for row in result.contrast],
        marker="o",
        color=PALETTE[1],
        label="Gaussian",
    )
    contrast.set(
        xscale="log",
        yscale="log",
        xlabel="dimension",
        ylabel="relative contrast (farthest - nearest) / nearest",
    )
    contrast.set_title("Distances concentrate", fontsize=11.5)
    contrast.legend(loc="upper right")

    positions = np.arange(len(NOISE_DIMENSIONS))
    for values, colour, label in (
        ([row.knn for row in result.accuracy], PALETTE[1], "k-NN on all dimensions"),
        ([row.knn_after_pca for row in result.accuracy], PALETTE[0], "k-NN after PCA to 2"),
        ([row.logistic for row in result.accuracy], PALETTE[2], "Logistic regression"),
    ):
        accuracy.plot(positions, values, marker="o", color=colour, label=label)
    accuracy.axhline(0.5, color=BASELINE, lw=1.0)
    accuracy.set_xticks(positions)
    accuracy.set_xticklabels([str(noise) for noise in NOISE_DIMENSIONS])
    accuracy.set(
        xlabel="noise dimensions added to two informative ones",
        ylabel="test accuracy",
        ylim=(0.45, 1.0),
    )
    accuracy.set_title("Nearest neighbours drown in irrelevant dimensions", fontsize=11.5)
    accuracy.legend(loc="lower left")
    figure.suptitle(
        "High dimensions make every neighbour equally far",
        x=0.012,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )

    return save_figure(figure, slug="distance_concentration", output_dir=output_dir)
