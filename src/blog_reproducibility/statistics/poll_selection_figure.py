"""Figure renderers for the poll-selection article."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.layout_engine import ConstrainedLayoutEngine

from blog_reproducibility.common.plotting import (
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.poll_selection import (
    sample_size_examples,
    weighting_example,
)


def render_precision_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the recorded share and its naive interval against the number of responses."""
    rows = sample_size_examples()
    use_house_style()

    figure, axis = plt.subplots(figsize=(9, 5.5))
    # The house style uses constrained layout; the footnote needs room below.
    layout = figure.get_layout_engine()
    if isinstance(layout, ConstrainedLayoutEngine):
        layout.set(rect=(0, 0.11, 1, 0.84))

    axis.errorbar(
        [row.summary.sample for row in rows],
        [100.0 * row.summary.respondent_share for row in rows],
        yerr=[100.0 * row.naive_interval.half_width for row in rows],
        fmt="o-",
        color=PALETTE[0],
        capsize=5,
        label="Respondent share with naive binomial intervals",
    )
    axis.axhline(60, color=PALETTE[1], linestyle="--", label="Population share: 60%")
    axis.annotate(
        "85.714% among respondents",
        xy=(7_000, 85.714),
        xytext=(7_000, 91),
        ha="center",
        fontsize=10,
    )
    axis.set(
        xscale="log",
        ylim=(53, 99),
        xlabel="Number of recorded responses (log scale)",
        ylabel="Support (%)",
        title="More responses narrow an interval around a selected population",
    )
    axis.legend(loc="lower left", fontsize=9)
    figure.text(
        0.02,
        0.008,
        "Synthetic population: 20 million people. Supporters are recorded at four times the rate "
        "of other people.\nBars show q ± 1.96 sqrt[q(1−q)/n]; these are not valid uncertainty "
        "bounds for population support.",
        fontsize=8.5,
        color=INK_SECONDARY,
    )

    return save_figure(figure, slug="science_poll_selection_precision", output_dir=output_dir)


def render_weighting_figure(*, output_dir: Path) -> FigureArtifact:
    """Contrast weighting when selection differs by group and when it differs by answer."""
    rows = (weighting_example(), weighting_example(outcome_dependent=True))
    use_house_style()

    figure, axis = plt.subplots(figsize=(9, 5.5))
    layout = figure.get_layout_engine()
    if isinstance(layout, ConstrainedLayoutEngine):
        layout.set(rect=(0, 0.12, 1, 0.83))

    series = (
        (-0.19, [100.0 * row.unweighted_share for row in rows], PALETTE[0], "Unweighted"),
        (
            0.19,
            [100.0 * row.weighted_share for row in rows],
            PALETTE[1],
            "Weighted to group totals",
        ),
    )
    for offset, values, colour, label in series:
        positions = [index + offset for index in range(len(rows))]
        axis.bar(positions, values, width=0.34, color=colour, label=label)
        for position, value in zip(positions, values, strict=True):
            axis.text(position, value + 1.1, f"{value:.2f}%", ha="center", fontsize=10)

    axis.axhline(
        60,
        color=INK_SECONDARY,
        linestyle="--",
        label="Population share: 60%",
        linewidth=1.3,
    )
    axis.set(
        ylim=(0, 116),
        xticks=[0, 1],
        yticks=[0, 20, 40, 60, 80, 100],
        xticklabels=[
            "Selection differs by group only",
            "Selection also differs by answer\nwithin each group",
        ],
        ylabel="Estimated support (%)",
        title="Matching group totals does not guarantee matching opinions",
    )
    axis.legend(loc="upper left", ncol=3, fontsize=8.5)
    figure.text(
        0.02,
        0.008,
        "Synthetic population: group A is 40% of people with 90% support; group B is 60% with 40% "
        "support.\nBoth weighted samples reproduce the population's group proportions exactly.",
        fontsize=8.5,
        color=INK_SECONDARY,
    )

    return save_figure(figure, slug="science_poll_selection_weighting", output_dir=output_dir)


def render_poll_selection_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the poll-selection article."""
    return (
        render_precision_figure(output_dir=output_dir),
        render_weighting_figure(output_dir=output_dir),
    )
