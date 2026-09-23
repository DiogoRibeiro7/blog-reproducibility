"""Figure renderer for the article on bandits against A/B tests."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.bandit_regret import (
    BanditRegretSummary,
    example_payload,
)


def render_bandit_regret_figure(
    *, output_dir: Path, summary: BanditRegretSummary | None = None
) -> FigureArtifact:
    """Plot mean cumulative regret against users for the three policies."""
    use_house_style()
    result = summary if summary is not None else example_payload()

    figure, axis = plt.subplots()
    for curve, colour, label in (
        (result.thompson, PALETTE[0], "Thompson sampling"),
        (result.quarter_then_exploit, PALETTE[2], "Even split for a quarter, then exploit"),
        (result.even_split, PALETTE[1], "Even split for the whole test"),
    ):
        axis.plot(result.users, curve, color=colour, lw=2, label=label)
    axis.set_xlabel("users")
    axis.set_ylabel("expected conversions lost to the worse variant")
    axis.set_title("What exploration costs, and when it stops costing")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="bandit_cumulative_regret", output_dir=output_dir)
