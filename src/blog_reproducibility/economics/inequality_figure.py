"""Figure renderer for the article on Lorenz curves and Gini coefficients."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_PRIMARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.economics.inequality import (
    gini_coefficient,
    lorenz_curve,
    simulate_incomes,
)


def render_lorenz_gini_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the Lorenz curve against equality and label the Gini coefficient."""
    use_house_style()
    incomes = simulate_incomes()
    population, income = lorenz_curve(incomes)

    figure, axis = plt.subplots(figsize=(5.6, 5.0))
    axis.plot([0, 1], [0, 1], color=INK_MUTED, lw=1.6, label="Perfect equality")
    axis.plot(population, income, color=PALETTE[0], label="Observed distribution")
    axis.fill_between(population, income, population, color=PALETTE[0], alpha=0.10)
    axis.annotate(
        f"Gini = {gini_coefficient(incomes):.2f}",
        xy=(0.36, 0.56),
        fontsize=11,
        color=INK_PRIMARY,
        fontweight="semibold",
    )
    axis.set(
        xlabel="cumulative share of population",
        ylabel="cumulative share of income",
        title="The Gini coefficient is twice the shaded area",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.set_aspect("equal")
    axis.grid(axis="both")
    axis.legend()

    return save_figure(figure, slug="lorenz_gini", output_dir=output_dir)
