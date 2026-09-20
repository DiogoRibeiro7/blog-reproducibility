"""Figure renderer for the longitudinal design article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.longitudinal_design import budget_curve


def render_budget_figure(*, output_dir: Path) -> FigureArtifact:
    """Show the same budget buying population precision or individual precision."""
    use_house_style()
    curve = budget_curve()
    measurements = [visits for visits, _, _ in curve]

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(measurements, [se for _, se, _ in curve], marker="o", color=PALETTE[0])
    axes[1].plot(measurements, [rmse for _, _, rmse in curve], marker="o", color=PALETTE[1])

    axes[0].set(title="Population mean", ylabel="Standard error")
    axes[1].set(title="Individual level, without pooling", ylabel="RMSE")
    for axis in axes:
        axis.set(xscale="log", xlabel="Measurements per person")

    figure.suptitle("One budget of 1,000 observations, two different questions")

    return save_figure(figure, slug="subjects_vs_measurements_2026", output_dir=output_dir)
