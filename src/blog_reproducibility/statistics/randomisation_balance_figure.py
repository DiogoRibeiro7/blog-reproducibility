"""Figure renderer for the randomisation balance draft."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.randomisation_balance import (
    adjustment_curve,
    balance_probabilities,
    example_payload,
)


def render_balance_figure(*, output_dir: Path) -> FigureArtifact:
    """Put the exact allocation distribution next to what adjustment buys."""
    use_house_style()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    allocations = balance_probabilities()
    axes[0].bar(
        [row.high_treated for row in allocations],
        [100 * row.probability for row in allocations],
        color=PALETTE[0],
    )
    perfect = 100 * example_payload().perfect_balance_probability
    axes[0].set(
        xlabel="High-covariate units in treatment (of 10)",
        ylabel="Probability (%)",
        title=f"Perfect balance occurs in {perfect:.1f}% of allocations",
        xticks=range(0, 11, 2),
    )

    curve = adjustment_curve()
    axes[1].plot([slope for slope, _ in curve], [value for _, value in curve])
    unadjusted = next(value for slope, value in curve if slope == 0.0)
    axes[1].axhline(unadjusted, ls="--", color=PALETTE[1], label="Unadjusted")
    axes[1].set(
        xlabel="Fixed adjustment coefficient",
        ylabel="SD of estimated effect",
        title="Adjustment helps when the coefficient is useful",
    )
    axes[1].legend()

    return save_figure(figure, slug="research_randomisation_balance", output_dir=output_dir)
