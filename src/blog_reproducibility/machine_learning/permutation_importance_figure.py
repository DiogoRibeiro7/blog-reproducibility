"""Figure renderer for the article on permutation importance with correlated features."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    BASELINE,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.permutation_importance import (
    ImportanceSummary,
    example_payload,
)


def render_permutation_importance_figure(
    *, output_dir: Path, summary: ImportanceSummary | None = None
) -> FigureArtifact:
    """Plot the four importance methods side by side as bar charts."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    names = [name.replace(" ", "\n") for name in result.features]
    panels = (
        ("Permutation", names, list(result.permutation)),
        (
            "Group permutation",
            ["x1 + x2", "x3", "x4"],
            [result.group_pair, result.permutation[2], result.permutation[3]],
        ),
        ("Conditional permutation", names, list(result.conditional)),
        ("Drop-column refit", names, list(result.drop_column)),
    )

    figure, axes = plt.subplots(1, 4, figsize=(10.4, 3.4))
    for axis, (title, labels, values) in zip(axes, panels, strict=True):
        axis.bar(labels, values, color=PALETTE[0], width=0.62)
        axis.set_title(title, fontsize=11)
        axis.axhline(0, color=BASELINE, lw=0.8)
        axis.tick_params(axis="x", labelsize=8.5)
    axes[0].set_ylabel("increase in test MSE")
    figure.suptitle(
        "Same model, same data, four different answers",
        x=0.012,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )

    return save_figure(figure, slug="permutation_importance_methods", output_dir=output_dir)
