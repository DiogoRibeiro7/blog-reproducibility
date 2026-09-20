"""Figure renderers for the article on results as rhetoric."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.health.results_rhetoric import (
    FINLEY_COHORT_LOSS,
    FINLEY_RETAINED,
    clients_needed,
    wall_mean,
)


def render_retention_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot retention alongside the weight loss reported at each point."""
    use_house_style()
    weeks = list(FINLEY_RETAINED)
    figure, axis = plt.subplots(figsize=(7.2, 4.4))

    axis.bar(
        range(len(weeks)),
        [FINLEY_RETAINED[week] for week in weeks],
        color=PALETTE[0],
        width=0.62,
    )
    for position, week in enumerate(weeks):
        label = f"{FINLEY_RETAINED[week]:g}%"
        if week in FINLEY_COHORT_LOSS:
            label += f"\nlost {FINLEY_COHORT_LOSS[week][0]:g}%"
        axis.text(
            position,
            FINLEY_RETAINED[week] + 2,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_SECONDARY,
        )

    axis.set_xticks(range(len(weeks)))
    axis.set_xticklabels(["enrolled"] + [f"week {week}" for week in weeks[1:]])
    axis.set_ylim(0, 116)
    axis.set_xlabel("weight lost: mean among clients still attending, % of initial body weight")
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_ylabel("clients still attending, % of those enrolled")
    axis.set_title("The better the result, the fewer clients it describes")
    axis.grid(axis="x", visible=False)

    return save_figure(figure, slug="results_rhetoric_retention", output_dir=output_dir)


def render_wall_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the expected wall against the size of the business."""
    use_house_style()
    # One hundred to a million clients, six points to a decade.
    sizes = [round(100 * 10 ** (index / 6)) for index in range(25)]
    base = [wall_mean(size) for size in sizes]

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    series = (
        (0.0, "a programme with no effect", PALETTE[0]),
        (0.5, "a moderate effect, 0.5 SD", PALETTE[1]),
        (1.0, "a large effect, 1 SD", PALETTE[2]),
    )
    for effect, label, colour in series:
        axis.plot(sizes, [effect + value for value in base], color=colour, label=label)

    small = wall_mean(1_000, effect=1.0)
    match = clients_needed(round(small, 2))
    axis.plot([1_000, match], [small, small], color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
    for position, colour in ((1_000, PALETTE[2]), (match, PALETTE[0])):
        axis.plot(
            [position],
            [small],
            "o",
            color=colour,
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
    axis.annotate(
        f"same wall: a large effect with 1,000 clients,\nno effect with {match:,}",
        xy=(match, small),
        xytext=(0, -78),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8, "shrinkB": 6},
    )

    axis.set_xscale("log")
    axis.set_xlabel("number of clients the best twenty are chosen from")
    axis.set_ylabel("mean result on the wall, standard deviations")
    axis.set_ylim(0, 5.6)
    axis.set_title("A wall of the best twenty measures the size of the business")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="results_rhetoric_testimonial_wall", output_dir=output_dir)


def render_results_rhetoric_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the article."""
    return (
        render_retention_figure(output_dir=output_dir),
        render_wall_figure(output_dir=output_dir),
    )
