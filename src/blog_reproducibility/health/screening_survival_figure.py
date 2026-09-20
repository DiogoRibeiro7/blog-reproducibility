"""Figure renderers for the screening and survival article."""

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
from blog_reproducibility.health.screening_survival import example_payload, sampling

SHORT_LABELS = (
    "Clinical diagnosis",
    "Earlier diagnosis only",
    "Additional diagnoses only",
    "Both, no effect on death",
    "Both, 18 deaths postponed",
)


def render_survival_and_mortality_figure(*, output_dir: Path) -> FigureArtifact:
    """Put five-year survival and cancer deaths side by side across the scenarios."""
    use_house_style()
    rows = list(example_payload().cohort_scenarios.values())

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5.9),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    layout = figure.get_layout_engine()
    if isinstance(layout, ConstrainedLayoutEngine):
        layout.set(rect=(0, 0.09, 1, 0.80))
    figure.suptitle(
        "Diagnosis statistics and mortality answer different questions", fontsize=13, y=0.98
    )

    panels = (
        (
            axes[0],
            [(row.five_year_survival or 0.0) * 100 for row in rows],
            PALETTE[0],
            "Among diagnosed people",
            "Alive beyond five years after diagnosis (%)",
            118,
        ),
        (
            axes[1],
            [float(row.cancer_deaths) for row in rows],
            PALETTE[1],
            "Among all 1,000 people",
            "Cancer deaths by year 12 (count)",
            92,
        ),
    )
    for axis, values, colour, title, xlabel, xmax in panels:
        axis.barh(range(5), values, color=colour, height=0.55)
        for position, value in enumerate(values):
            axis.text(value + 2, position, f"{value:g}", va="center", fontsize=10)
        axis.set(xlim=(0, xmax), title=title, xlabel=xlabel, yticks=range(5))
        axis.grid(axis="y", visible=False)
        axis.grid(axis="x", visible=True)

    axes[0].set_yticklabels(SHORT_LABELS, fontsize=9.5)
    axes[0].invert_yaxis()
    figure.text(
        0.02,
        0.008,
        "Synthetic paired histories; complete follow-up. Diagnosed denominators: 120, 120, 300, "
        "300, 300.\nOnly the final scenario changes a death time. The horizon for mortality starts "
        "at common eligibility.",
        fontsize=8.5,
        color=INK_SECONDARY,
    )

    return save_figure(
        figure, slug="science_screening_survival_and_mortality", output_dir=output_dir
    )


def render_duration_selection_figure(*, output_dir: Path) -> FigureArtifact:
    """Show how the mix of fast and slow disease differs between the three views."""
    use_house_style()
    selection = sampling()

    figure, axis = plt.subplots(figsize=(9, 5.3))
    layout = figure.get_layout_engine()
    if isinstance(layout, ConstrainedLayoutEngine):
        layout.set(rect=(0, 0.13, 1, 0.82))

    positions = range(3)
    slow = [
        100 * weights[1]
        for weights in (
            selection.incident_weights,
            selection.snapshot_weights,
            selection.repeated_weights,
        )
    ]
    fast = [100 - value for value in slow]

    axis.bar(positions, fast, color=PALETTE[0], width=0.56, label="Fast: 1-year detectable window")
    axis.bar(
        positions,
        slow,
        bottom=fast,
        color=PALETTE[1],
        width=0.56,
        label="Slow: 4-year detectable window",
    )
    for position, quick, slower in zip(positions, fast, slow, strict=True):
        axis.text(
            position,
            quick / 2,
            f"Fast {quick:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
        axis.text(
            position,
            quick + slower / 2,
            f"Slow {slower:.1f}%",
            ha="center",
            va="center",
            fontweight="bold",
        )

    axis.set(
        ylim=(0, 122),
        ylabel="Composition of cases (%)",
        yticks=[0, 20, 40, 60, 80, 100],
        xticks=list(positions),
        xticklabels=[
            "New detectable cases\n40 fast + 40 slow / year",
            "First-round snapshot\n40 fast + 160 slow present",
            "Repeated 2-year screens\n20 fast + 40 slow detected / year",
        ],
        title="Detectable duration changes which disease histories are observed",
    )
    axis.legend(loc="upper center", ncol=2, fontsize=8.5)
    figure.text(
        0.02,
        0.006,
        "Synthetic stationary model; perfect sensitivity within the detectable window, uniform "
        "entry phases.\nThe first-round stock and subsequent annual detection flow have different "
        "denominators.",
        fontsize=8.5,
        color=INK_SECONDARY,
    )

    return save_figure(figure, slug="science_screening_duration_selection", output_dir=output_dir)


def render_screening_survival_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the screening article."""
    return (
        render_survival_and_mortality_figure(output_dir=output_dir),
        render_duration_selection_figure(output_dir=output_dir),
    )
