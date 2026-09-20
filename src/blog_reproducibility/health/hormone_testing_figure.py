"""Figure renderers for the hormone-testing article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_SECONDARY,
    PALETTE,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.health.hormone_testing import (
    DANESE_RCV,
    any_flag,
    any_flag_correlated,
    flag_curve,
    sorted_reference_changes,
)

CORRELATED_PANEL_SIZES = (1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40)


def render_panel_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the chance of at least one flag against the size of the panel."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(7.2, 4.4))

    curve = flag_curve()
    axis.plot(
        [number for number, _ in curve],
        [value for _, value in curve],
        color=PALETTE[0],
        label="independent analytes",
    )
    axis.plot(
        CORRELATED_PANEL_SIZES,
        [
            100 * any_flag_correlated(number, 0.5, repeats=20_000)
            for number in CORRELATED_PANEL_SIZES
        ],
        color=PALETTE[1],
        label="every pair correlated at 0.5",
    )

    twelve = 100 * any_flag(12)
    axis.plot(
        [12],
        [twelve],
        "o",
        color=PALETTE[0],
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    axis.annotate(
        f"a dozen hormones: {twelve:.0f}%",
        xy=(12, twelve),
        xytext=(-10, 10),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK_SECONDARY,
    )

    axis.set_xlim(0, 41)
    axis.set_ylim(0, 100)
    axis.set_xlabel("analytes measured in one healthy person")
    axis.set_ylabel("chance of at least one result outside its range, %")
    axis.set_title("The larger the panel, the more certain it is to find something")
    axis.legend(loc="lower right")

    return save_figure(figure, slug="hormone_panel_false_flags", output_dir=output_dir)


def render_change_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the reference change value for each analyte."""
    use_house_style()
    rows = sorted_reference_changes()
    figure, axis = plt.subplots(figsize=(7.2, 5.0))

    positions = range(len(rows))
    axis.barh(
        list(positions),
        [value for _, value in rows],
        color=PALETTE[0],
        height=0.62,
        label="serum, biological variation alone",
    )
    for position, (_, value) in zip(positions, rows, strict=True):
        axis.text(
            value + 3, position, f"{value:.0f}%", va="center", fontsize=9, color=INK_SECONDARY
        )

    top = len(rows)
    axis.barh(
        [top],
        [DANESE_RCV[1] - DANESE_RCV[0]],
        left=[DANESE_RCV[0]],
        color=PALETTE[1],
        height=0.62,
        label="salivary cortisol, measured, by time of day",
    )
    axis.text(
        DANESE_RCV[1] + 3,
        top,
        f"{DANESE_RCV[0]}% to {DANESE_RCV[1]}%",
        va="center",
        fontsize=9,
        color=INK_SECONDARY,
    )

    axis.set_yticks([*positions, top])
    axis.set_yticklabels([name for name, _ in rows] + ["salivary cortisol"])
    axis.set_xlim(0, 300)
    axis.set_xlabel("difference between two results needed before it means a change, %")
    axis.set_title("Two hormone results have to differ by a lot to differ at all")
    axis.legend(loc="lower right", fontsize=8.5)
    axis.grid(axis="y", visible=False)

    return save_figure(figure, slug="hormone_reference_change_values", output_dir=output_dir)


def render_hormone_testing_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the hormone-testing article."""
    return (
        render_panel_figure(output_dir=output_dir),
        render_change_figure(output_dir=output_dir),
    )
