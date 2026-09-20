"""Figure renderers for the quantum measurement article."""

from math import pi
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
from blog_reproducibility.physics.quantum_observer import joint_probabilities

PHASES = tuple(2 * pi * index / 400 for index in range(401))


def render_marker_visibility_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot interference against phase for three marker overlaps."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(9, 5.4))
    layout = figure.get_layout_engine()
    if isinstance(layout, ConstrainedLayoutEngine):
        layout.set(rect=(0, 0.12, 1, 0.83))

    for overlap, colour, style in (
        (1.0, PALETTE[0], "-"),
        (0.6, PALETTE[1], "--"),
        (0.0, PALETTE[2], ":"),
    ):
        values = [sum(joint_probabilities(phase, overlap)[0]) for phase in PHASES]
        axis.plot(
            [phase / pi for phase in PHASES],
            values,
            color=colour,
            ls=style,
            label=f"Marker overlap {overlap:g}; visibility {overlap:g}",
        )

    axis.set(
        xlim=(0, 2),
        ylim=(-0.02, 1.2),
        yticks=[0, 0.25, 0.5, 0.75, 1],
        xticks=[0, 0.5, 1, 1.5, 2],
        xlabel="Controlled phase / π",
        ylabel="Probability of the + output",
        title="Interference follows the overlap of physical marker states",
    )
    axis.legend(loc="upper center", ncol=1, fontsize=9)
    figure.text(
        0.02,
        0.008,
        "Ideal equal-amplitude two-path model; all marker outcomes are included.\n"
        "The calculation changes physical correlations. It contains no parameter for awareness "
        "or intention.",
        fontsize=8.5,
        color=INK_SECONDARY,
    )

    return save_figure(figure, slug="science_quantum_marker_visibility", output_dir=output_dir)


def render_eraser_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot conditional fringes that appear only once the marker record is used."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(9, 5.4))
    layout = figure.get_layout_engine()
    if isinstance(layout, ConstrainedLayoutEngine):
        layout.set(rect=(0, 0.12, 1, 0.83))

    tables = [joint_probabilities(phase, 0.0, beta=pi / 4) for phase in PHASES]
    for marker, colour, style, label in (
        (0, PALETTE[0], "-", "Given marker +"),
        (1, PALETTE[1], "--", "Given marker −"),
    ):
        values = [
            table[0][marker] / sum(table[port][marker] for port in range(2)) for table in tables
        ]
        axis.plot([phase / pi for phase in PHASES], values, color=colour, ls=style, label=label)

    axis.plot(
        [phase / pi for phase in PHASES],
        [sum(table[0]) for table in tables],
        color=PALETTE[2],
        ls=":",
        lw=2.5,
        label="All marker outcomes combined",
    )
    axis.set(
        xlim=(0, 2),
        ylim=(-0.02, 1.2),
        yticks=[0, 0.25, 0.5, 0.75, 1],
        xticks=[0, 0.5, 1, 1.5, 2],
        xlabel="Controlled phase / π",
        ylabel="Probability of the + path output",
        title="Quantum erasure reveals complementary conditional fringes",
    )
    axis.legend(loc="upper center", ncol=1, fontsize=9)
    figure.text(
        0.02,
        0.008,
        "Orthogonal path markers measured in a complementary basis; each marker result has "
        "probability 1/2.\nSorting by the marker record changes the subset. The unsorted path "
        "probability stays at 1/2.",
        fontsize=8.5,
        color=INK_SECONDARY,
    )

    return save_figure(figure, slug="science_quantum_eraser_conditioning", output_dir=output_dir)


def render_quantum_observer_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the quantum measurement article."""
    return (
        render_marker_visibility_figure(output_dir=output_dir),
        render_eraser_figure(output_dir=output_dir),
    )
