"""Figure renderer for the article on survivorship bias and left truncation."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.left_truncation import (
    LeftTruncationSummary,
    example_payload,
)


def render_left_truncation_figure(
    *, output_dir: Path, summary: LeftTruncationSummary | None = None
) -> FigureArtifact:
    """Plot the reference, naive and left-truncated survival curves against age."""
    use_house_style()
    curves = (summary if summary is not None else example_payload()).curves

    figure, axis = plt.subplots()
    axis.plot(
        curves.ages,
        curves.reference,
        color=PALETTE[0],
        lw=2.2,
        label="Reference: every unit ever installed",
    )
    axis.plot(
        curves.ages,
        curves.naive,
        color=PALETTE[1],
        lw=2,
        label="Survivors at the snapshot, counted from age zero",
    )
    axis.plot(
        curves.ages,
        curves.truncated,
        color=PALETTE[2],
        lw=2,
        ls="--",
        label="Survivors with left truncation (risk set by age)",
    )
    axis.axhline(0.5, color=PALETTE[3], lw=1, ls=":")
    axis.set_xlabel("age (years)")
    axis.set_ylabel("share still in service")
    axis.set_ylim(0, 1.02)
    axis.set_title("Counting survivors from age zero doubles the apparent lifetime")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="survivorship_left_truncation", output_dir=output_dir)
