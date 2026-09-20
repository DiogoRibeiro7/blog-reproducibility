"""Figure renderer for the numerical verification article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.engineering.numerical_verification import STEP_COUNTS, example_payload


def render_refinement_figure(*, output_dir: Path) -> FigureArtifact:
    """Show two methods converging and a third converging to the wrong answer."""
    use_house_style()
    rows = example_payload().rows
    labels: list[str] = []
    for row in rows:
        if row.method not in labels:
            labels.append(row.method)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    widths = [1 / steps for steps in STEP_COUNTS]

    for index, label in enumerate(labels):
        errors = [row.endpoint_error for row in rows if row.method == label]
        axis.loglog(widths, errors, marker="o", color=PALETTE[index], label=label)

    axis.set(
        xlabel="Step size h (smaller to the right)",
        ylabel="Maximum component error at t = 1",
        title="Refinement exposes an implementation converging to the wrong answer",
    )
    axis.invert_xaxis()
    axis.set_xticks(widths, labels=[f"{width:g}" for width in widths])
    axis.legend()

    return save_figure(figure, slug="numerical_verification_2026", output_dir=output_dir)
