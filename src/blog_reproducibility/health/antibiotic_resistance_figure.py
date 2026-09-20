"""Figure renderer for the antibiotic resistance article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.health.antibiotic_resistance import selection_rows


def render_selection_figure(*, output_dir: Path) -> FigureArtifact:
    """Put the falling counts next to the rising share."""
    use_house_style()
    rows = selection_rows()
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    series = (
        ("Sensitive", [row.sensitive for row in rows], PALETTE[0]),
        ("Resistant", [row.resistant for row in rows], PALETTE[1]),
    )
    for name, values, colour in series:
        axes[0].semilogy(
            [row.bottleneck for row in rows], values, marker="o", label=name, color=colour
        )
    axes[0].set(ylabel="Expected number (logarithmic scale)", title="Both numbers decrease")
    axes[0].legend()

    axes[1].plot(
        [row.bottleneck for row in rows],
        [100 * row.resistant_share for row in rows],
        marker="o",
        color=PALETTE[1],
    )
    axes[1].set(ylabel="Resistant share (%)", ylim=(0, 105), title="The resistant share increases")

    for axis in axes:
        axis.set(xlabel="Hypothetical selective bottlenecks")
        axis.set_xticks(range(len(rows)))

    return save_figure(figure, slug="science_antibiotic_selection", output_dir=output_dir)
