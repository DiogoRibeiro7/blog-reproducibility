"""Figure renderers for the concentration and the starting-risk articles."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.health.risk_communication import exposure_rows, risk_rows


def render_concentration_figure(*, output_dir: Path) -> FigureArtifact:
    """Show a lower concentration delivering a larger amount."""
    use_house_style()
    rows = exposure_rows()
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.1))

    axes[0].bar(
        [row.name for row in rows],
        [row.concentration_mg_per_ml for row in rows],
        color=PALETTE[0],
        width=0.6,
    )
    axes[1].bar(
        [row.name for row in rows],
        [row.amount_mg for row in rows],
        color=PALETTE[1],
        width=0.6,
    )

    axes[0].set(title="Concentration", ylabel="mg per mL", ylim=(0, 12))
    axes[1].set(title="Amount in the chosen volume", ylabel="mg", ylim=(0, 36))
    for axis in axes:
        axis.set(xlabel="Hypothetical sample")
    for position, row in enumerate(rows):
        axes[1].text(
            position,
            row.amount_mg + 1,
            f"{row.volume_ml:g} mL: {row.amount_mg:g} mg",
            ha="center",
            fontsize=9,
        )

    figure.suptitle("A lower concentration can deliver a larger amount")

    return save_figure(figure, slug="science_concentration_and_amount", output_dir=output_dir)


def render_relative_risk_figure(*, output_dir: Path) -> FigureArtifact:
    """Show the same relative reduction meaning very different absolute ones."""
    use_house_style()
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)

    for axis, row in zip(axes, risk_rows(), strict=True):
        axis.bar(
            [0, 1],
            [row.before_per_1000, row.after_per_1000],
            color=list(PALETTE[:2]),
            width=0.55,
        )
        axis.set_xticks([0, 1], ["Comparison", "Intervention"])
        axis.set(
            title=f"{row.name}: {row.relative_reduction:.0%} lower risk",
            ylim=(0, 24),
            xlabel=f"{row.difference_per_1000:g} fewer events per 1,000 people",
        )
        for position, value in enumerate((row.before_per_1000, row.after_per_1000)):
            axis.text(position, value + 0.5, f"{value:g}", ha="center")

    axes[0].set_ylabel("Events per 1,000 people over five years")

    return save_figure(figure, slug="science_relative_absolute_risk", output_dir=output_dir)
