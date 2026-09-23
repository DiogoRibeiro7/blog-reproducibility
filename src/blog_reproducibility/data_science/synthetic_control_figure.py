"""Figure renderer for the article on synthetic control."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    BASELINE,
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.data_science.synthetic_control import (
    PRE_MONTHS,
    placebo_gaps,
    simulate_panel,
)


def render_synthetic_control_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the treated unit against its synthetic control, and the placebo gaps."""
    use_house_style()
    outcomes = simulate_panel()
    gaps = placebo_gaps(outcomes)
    months = np.arange(1, outcomes.shape[1] + 1)
    boundary = PRE_MONTHS + 0.5

    figure, (paths, placebo) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    paths.plot(months, outcomes[0], color=PALETTE[0], label="Treated unit")
    paths.plot(months, outcomes[0] - gaps[0], color=PALETTE[1], label="Synthetic control")
    paths.axvline(boundary, color=INK_MUTED, lw=1.2)
    paths.annotate(
        "intervention",
        xy=(boundary, paths.get_ylim()[1]),
        xytext=(-6, -14),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    paths.set(xlabel="month", ylabel="outcome")
    paths.set_title("Treated unit and its synthetic control", fontsize=11.5)
    paths.legend(loc="upper left")

    for gap in gaps[1:]:
        placebo.plot(months, gap, color=INK_MUTED, lw=1.0, alpha=0.45)
    placebo.plot(months, gaps[0], color=PALETTE[0], lw=2.2, label="Treated unit")
    placebo.plot([], [], color=INK_MUTED, lw=1.0, label="Each donor treated as a placebo")
    placebo.axvline(boundary, color=INK_MUTED, lw=1.2)
    placebo.axhline(0, color=BASELINE, lw=1.0)
    placebo.set(xlabel="month", ylabel="gap: unit minus its synthetic control")
    placebo.set_title("Placebo gaps", fontsize=11.5)
    placebo.legend(loc="lower left")
    figure.suptitle(
        "The effect is the gap after the intervention, judged against placebo gaps",
        x=0.012,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )

    return save_figure(figure, slug="synthetic_control_paths", output_dir=output_dir)
