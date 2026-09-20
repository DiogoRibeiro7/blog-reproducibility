"""Figure renderers for the inflammation-marker article."""

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    BASELINE,
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SEQUENTIAL,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.health.inflammation_markers import (
    EFLM_WITHIN,
    ESTIMATES,
    single_result_interval,
)


def render_scale_figure(*, output_dir: Path) -> FigureArtifact:
    """Put reference points and one person's noise band on the same CRP scale."""
    use_house_style()
    low, high = single_result_interval(3.0, EFLM_WITHIN)

    rows = (
        ("acute infection or injury", [(500.0, "above 500")], PALETTE[7]),
        (
            "current smokers and never-smokers",
            [(1.35, "1.35 never"), (2.53, "2.53 smokers")],
            PALETTE[1],
        ),
        (
            "healthy blood donors",
            [(0.8, "median 0.8"), (3.0, "90th centile 3.0"), (10.0, "99th centile 10")],
            PALETTE[0],
        ),
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.axvspan(low, high, color=SEQUENTIAL[0], alpha=0.6, linewidth=0)
    axis.text(
        sqrt(low * high),
        3.62,
        f"single results of one person whose\nusual level is 3 mg/L: {low:.1f} to {high:.1f}",
        ha="center",
        va="top",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    # Stop the cut-off lines below the band's own label.
    axis.vlines([3, 10], -0.7, 2.62, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))

    for position, (_, points, colour) in enumerate(rows):
        axis.plot(
            [value for value, _ in points],
            [position] * len(points),
            "o",
            color=colour,
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        for index, (value, text) in enumerate(points):
            axis.annotate(
                text,
                xy=(value, position),
                xytext=(0, 11 if index % 2 == 0 else -17),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=INK_SECONDARY,
            )

    axis.set_xscale("log")
    axis.set_xlim(0.05, 1500)
    axis.set_ylim(-0.7, 3.8)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([row[0] for row in rows], fontsize=9)
    axis.set_xticks([0.1, 1, 3, 10, 100, 1000])
    axis.set_xticklabels(["0.1", "1", "3", "10", "100", "1,000"])
    axis.minorticks_off()
    axis.set_xlabel("C-reactive protein, mg/L (logarithmic scale)")
    axis.set_title("One marker, four orders of magnitude, and a wide band of noise")
    axis.grid(axis="y", visible=False)

    return save_figure(figure, slug="inflammation_crp_scale", output_dir=output_dir)


def render_trials_figure(*, output_dir: Path) -> FigureArtifact:
    """Draw the forest plot of anti-inflammatory strategies."""
    use_house_style()
    rows = tuple(reversed(ESTIMATES))
    figure, axis = plt.subplots(figsize=(7.2, 4.4))

    for position, estimate in enumerate(rows):
        colour = INK_MUTED if "Mendelian" in estimate.label else PALETTE[0]
        axis.plot([estimate.low, estimate.high], [position, position], color=colour, linewidth=2)
        axis.plot(
            [estimate.ratio],
            [position],
            "o",
            color=colour,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        axis.text(
            1.27,
            position,
            f"{estimate.ratio:.2f} ({estimate.low:.2f} to {estimate.high:.2f})",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
        axis.text(0.52, position - 0.34, estimate.note, va="center", fontsize=8, color=INK_MUTED)

    axis.axvline(1.0, color=BASELINE, linewidth=1.2)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([estimate.label for estimate in rows], fontsize=9)
    axis.set_xscale("log")
    axis.set_xlim(0.5, 1.75)
    axis.set_ylim(-0.75, len(rows) - 0.4)
    axis.set_xticks([0.5, 0.7, 1.0, 1.4])
    axis.set_xticklabels(["0.5", "0.7", "1.0", "1.4"])
    axis.minorticks_off()
    axis.set_xlabel("ratio of cardiovascular events (below 1 favours the intervention)")
    axis.set_title("Which anti-inflammatory strategies prevented heart attacks")
    axis.grid(axis="y", visible=False)

    return save_figure(figure, slug="inflammation_trials_forest", output_dir=output_dir)


def render_inflammation_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the inflammation article."""
    return (
        render_scale_figure(output_dir=output_dir),
        render_trials_figure(output_dir=output_dir),
    )
