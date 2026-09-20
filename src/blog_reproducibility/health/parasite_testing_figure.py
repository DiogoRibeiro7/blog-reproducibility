"""Figure renderers for the parasite-testing article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SEQUENTIAL,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.health.parasite_testing import (
    BRANDA_SENSITIVITY,
    CARTWRIGHT,
    TAPE,
    detected_by,
    left_after_negatives,
)


def render_repeat_sampling_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot what a second and a third specimen add, against what was observed."""
    use_house_style()
    first = CARTWRIGHT["found by the first specimen"] / CARTWRIGHT["cases"]
    two = CARTWRIGHT["found by the first two"] / CARTWRIGHT["cases"]
    samples = [1, 2, 3]

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    series = (
        (
            "stool examination for ova and parasites",
            first,
            PALETTE[0],
            {1: first, 2: two},
        ),
        (
            "adhesive tape test for pinworm",
            TAPE["one morning"],
            PALETTE[1],
            {1: TAPE["one morning"], 3: TAPE["three mornings"]},
        ),
    )

    for label, sensitivity, colour, observed in series:
        axis.plot(
            samples,
            [100 * detected_by(number, sensitivity) for number in samples],
            color=colour,
            label=f"{label}: if samples were independent",
        )
        axis.plot(
            list(observed),
            [100 * value for value in observed.values()],
            "o",
            color=colour,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=f"{label}: reported",
        )
        for number, value in observed.items():
            # Keep the label clear of the line it sits next to.
            above = value > detected_by(number, sensitivity) + 1e-9
            axis.annotate(
                f"{100 * value:.0f}%",
                xy=(number, 100 * value),
                xytext=(0, 9 if above else -17),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=INK_SECONDARY,
            )

    axis.set_xlim(0.8, 3.25)
    axis.set_xticks(samples)
    axis.set_xticklabels(["one sample", "two, on separate days", "three, on separate days"])
    axis.set_ylim(40, 104)
    axis.set_ylabel("infections detected, %")
    axis.set_title("A missed infection is intermittent shedding, and sampling again finds it")
    axis.legend(loc="lower right", fontsize=8.5)
    axis.grid(axis="x", visible=False)

    return save_figure(figure, slug="parasite_repeat_sampling", output_dir=output_dir)


def render_negative_results_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot what a run of negative results leaves, against the prior."""
    use_house_style()
    priors = [10 ** (-3 + 2.5 * index / 60) for index in range(61)]  # 0.1% to about 32%
    figure, axis = plt.subplots(figsize=(7.2, 4.6))

    axis.plot(
        [100 * prior for prior in priors],
        [100 * prior for prior in priors],
        color=INK_MUTED,
        linewidth=1.2,
        linestyle=(0, (3, 3)),
        label="before any test",
    )
    shades = (SEQUENTIAL[2], SEQUENTIAL[4], SEQUENTIAL[7])
    for negatives, colour in zip((1, 2, 3), shades, strict=True):
        plural = "" if negatives == 1 else "s"
        axis.plot(
            [100 * prior for prior in priors],
            [100 * left_after_negatives(prior, negatives, BRANDA_SENSITIVITY) for prior in priors],
            color=colour,
            label=f"after {negatives} negative stool examination{plural}",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks([0.1, 1, 10])
    axis.set_xticklabels(["0.1%", "1%", "10%"])
    axis.set_yticks([0.001, 0.01, 0.1, 1, 10])
    axis.set_yticklabels(["0.001%", "0.01%", "0.1%", "1%", "10%"])
    axis.set_xlabel("probability of infection before testing")
    axis.set_ylabel("probability of infection that remains")
    axis.set_title("Each negative result divides what is left by about 3.6")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="parasite_negative_results", output_dir=output_dir)


def render_parasite_testing_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the parasite-testing article."""
    return (
        render_repeat_sampling_figure(output_dir=output_dir),
        render_negative_results_figure(output_dir=output_dir),
    )
