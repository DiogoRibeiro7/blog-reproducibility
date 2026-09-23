"""Figure renderer for the article on survival analysis."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.mathematics.kaplan_meier import simulate_arms, survival_at


def render_kaplan_meier_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot both arms' survival curves with censored observations ticked."""
    use_house_style()
    figure, axis = plt.subplots()

    for index, arm in enumerate(simulate_arms()):
        colour = PALETTE[index]
        axis.step(arm.curve.times, arm.curve.survival, where="post", color=colour, label=arm.name)
        censored = arm.observed[~arm.events]
        axis.plot(
            censored,
            survival_at(arm.curve, censored),
            linestyle="none",
            marker="|",
            markersize=7,
            color=colour,
            markeredgecolor=colour,
            markeredgewidth=1.4,
        )
    axis.set(
        title="Kaplan-Meier estimate, censored observations ticked",
        xlabel="time",
        ylabel="survival probability",
        ylim=(0, 1.02),
        xlim=(0, 40),
    )
    axis.legend()

    return save_figure(figure, slug="kaplan_meier", output_dir=output_dir)
