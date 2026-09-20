"""Figure renderers for the before-and-after testimonial article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.testimonial_selection import (
    selected_moments,
    simulate_pairs,
)

CUTOFF = 65.0
SLOPE = 0.64


def render_selection_figure(*, output_dir: Path) -> FigureArtifact:
    """Scatter simulated baseline and follow-up scores, marking the selected cohort."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(9, 5.8))
    pairs = simulate_pairs()

    for selected, colour, label in (
        (False, INK_MUTED, "Not selected"),
        (True, PALETTE[1], "Selected: baseline at least 65"),
    ):
        points = [(x, y) for x, y in pairs if (x >= CUTOFF) == selected]
        axis.scatter(
            [x for x, _ in points],
            [y for _, y in points],
            s=13,
            color=colour,
            alpha=0.55,
            edgecolors="none",
            label=label,
        )

    axis.plot(
        [10, 90],
        [10, 90],
        color=INK_MUTED,
        linestyle=":",
        label="No change in observed score",
    )
    axis.plot(
        [10, 90],
        [50 + SLOPE * (x - 50) for x in (10, 90)],
        color=PALETTE[0],
        label="Expected follow-up at each baseline",
    )
    axis.axvline(CUTOFF, color=PALETTE[1], linestyle="--", linewidth=1)
    axis.set(
        xlim=(10, 90),
        ylim=(10, 90),
        xlabel="Baseline score",
        ylabel="Follow-up score",
        title="Selection creates apparent improvement without an intervention",
    )
    axis.legend(loc="upper left", fontsize=8.5)

    return save_figure(figure, slug="science_testimonial_selection", output_dir=output_dir)


def render_counterfactual_figure(*, output_dir: Path) -> FigureArtifact:
    """Show all three intervention scenarios improving from the selected baseline."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(9, 5.3))

    scenarios = (
        (3.0, PALETTE[1], "Harmful: +3 relative to no intervention"),
        (0.0, PALETTE[0], "No intervention"),
        (-3.0, PALETTE[2], "Beneficial: -3 relative to no intervention"),
    )
    for effect, colour, label in scenarios:
        row = selected_moments(effect=effect)
        axis.plot([0, 1], [row.baseline_mean, row.followup_mean], marker="o", color=colour)
        axis.annotate(
            f"{row.followup_mean:.2f}  {label}",
            (1, row.followup_mean),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
        )

    control = selected_moments()
    axis.annotate(
        f"{control.baseline_mean:.2f}",
        (0, control.baseline_mean),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
    )
    axis.set(
        xlim=(-0.1, 2.5),
        ylim=(57, 72),
        xticks=[0, 1],
        xticklabels=["Selected baseline", "Follow-up"],
        ylabel="Expected group mean score",
        title="All three trajectories improve from the selected baseline",
    )
    axis.text(
        0.01,
        0.02,
        "Exact model expectations; lower scores are preferable. "
        "These are not observed trial results.",
        transform=axis.transAxes,
        fontsize=8,
        color=INK_MUTED,
    )

    return save_figure(figure, slug="science_testimonial_counterfactual", output_dir=output_dir)


def render_testimonial_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the testimonial article."""
    return (
        render_selection_figure(output_dir=output_dir),
        render_counterfactual_figure(output_dir=output_dir),
    )
