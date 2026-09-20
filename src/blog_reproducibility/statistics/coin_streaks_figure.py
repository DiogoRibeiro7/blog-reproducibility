"""Figure renderer for the article on randomness owing no reversal."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.coin_streaks import next_head_probabilities


def render_mechanisms_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot what each mechanism predicts after the same observed streak."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(8.2, 4.4))

    rows = [next_head_probabilities(heads) for heads in range(5)]
    series = (
        ("Known fair coin", [row.fair_coin for row in rows]),
        ("Bag without replacement", [row.bag_without_replacement for row in rows]),
        ("Unknown 25% or 75% coin", [row.unknown_coin for row in rows]),
    )
    for index, (label, values) in enumerate(series):
        axis.plot(range(5), values, marker="o", color=PALETTE[index], label=label)

    axis.set(
        xlabel="Initial consecutive heads observed",
        ylabel="Probability of heads next",
        ylim=(0, 1),
        title="The same streak means different things in different mechanisms",
    )
    axis.set_xticks(range(5))
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.19), fontsize=9)

    return save_figure(figure, slug="science_streaks_three_mechanisms", output_dir=output_dir)
