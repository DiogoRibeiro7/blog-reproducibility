"""Figure renderer for the unlabelled-monitoring article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.engineering.unlabelled_monitoring import example_payload

SHORT_LABELS = ("Reference", "Hidden\nreversal", "Input shift\nonly")


def render_worlds_figure(*, output_dir: Path) -> FigureArtifact:
    """Put what monitoring can see beside what it cannot."""
    use_house_style()
    rows = example_payload()
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    positions = list(range(len(rows)))

    axes[0].bar(
        [position - 0.17 for position in positions],
        [row.positive_share for row in rows],
        width=0.34,
        color=PALETTE[0],
        label="Positive prediction share",
    )
    axes[0].bar(
        [position + 0.17 for position in positions],
        [row.mean_confidence for row in rows],
        width=0.34,
        color=PALETTE[1],
        label="Mean confidence",
    )
    axes[1].bar(positions, [row.accuracy for row in rows], color=PALETTE[0], width=0.5)

    axes[0].set_title("Observable before labels arrive")
    axes[1].set_title("Accuracy requires outcomes")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.19), fontsize=8)

    for axis in axes:
        axis.set_xticks(positions, SHORT_LABELS)
        axis.set(ylim=(0, 1.05), ylabel="Proportion")
    for position, row in zip(positions, rows, strict=True):
        axes[1].text(position, row.accuracy + 0.025, f"{row.accuracy:.0%}", ha="center")

    figure.suptitle("Identical monitoring signals can conceal a different accuracy")

    return save_figure(figure, slug="unlabelled_monitoring_worlds_2026", output_dir=output_dir)
