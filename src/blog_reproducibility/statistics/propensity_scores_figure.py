"""Figure renderer for the article on propensity scores, matching and weighting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.propensity_scores import (
    ESTIMATORS,
    PropensitySummary,
    example_payload,
)

LABELS = (
    "Both models\ncorrect",
    "Outcome model\nmisspecified",
    "Propensity model\nmisspecified",
    "Strong confounding,\npoor overlap",
)
COLOURS = (PALETTE[3], PALETTE[4], PALETTE[2], PALETTE[0])
# The website draws the zero line in this grey rather than a house ink.
ZERO_LINE = "#8a8f98"


def render_propensity_scores_figure(
    *, output_dir: Path, summary: PropensitySummary | None = None
) -> FigureArtifact:
    """Plot the absolute bias of the four adjusted estimators in each condition, as lollipops."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    biases = (
        [row.absolute_bias.regression for row in result.rows],
        [row.absolute_bias.matching for row in result.rows],
        [row.absolute_bias.weighting for row in result.rows],
        [row.absolute_bias.doubly_robust for row in result.rows],
    )
    positions = np.arange(len(LABELS))

    figure, axis = plt.subplots()
    # The unadjusted difference is off this scale; the comparison that matters is among
    # the four adjusted estimators, whose biases are an order of magnitude smaller.
    for i, (name, values, colour) in enumerate(zip(ESTIMATORS[1:], biases, COLOURS, strict=True)):
        offset = (i - 1.5) * 0.13
        axis.plot(positions + offset, values, marker="o", ms=9, ls="none", color=colour, label=name)
        for position, value in zip(positions + offset, values, strict=True):
            axis.plot([position, position], [0, value], color=colour, lw=2, alpha=0.45)
    axis.axhline(0, color=ZERO_LINE, lw=1)
    axis.set_xticks(positions)
    axis.set_xticklabels(LABELS)
    axis.set_xlim(-0.5, len(LABELS) - 0.5)
    axis.set_ylabel(f"absolute bias (true effect is {result.effect:.2f})")
    axis.set_ylim(-0.015, 0.36)
    axis.set_title("Each method fails on its own assumption")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="propensity_estimator_bias", output_dir=output_dir)
