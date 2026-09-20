"""Figure renderers for the aspartame dose essay."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.health.aspartame_dose import (
    ADI,
    APPLES_1KG,
    BLOOD_THRESHOLDS,
    CAN_LITRES,
    DETECTION_LIMIT,
    DRINK,
    ENDOGENOUS,
    JUICE,
    LETHAL_PER_KG,
    STEGINK,
    dose_grid,
    methanol_from,
    predicted_peak,
)


def render_sources_figure(*, output_dir: Path) -> FigureArtifact:
    """Put every methanol source on one logarithmic scale."""
    use_house_style()
    can_low = methanol_from(DRINK["measured mean, 2008"] * CAN_LITRES)
    can_high = methanol_from(DRINK["maximum permitted"] * CAN_LITRES)
    adi = methanol_from(ADI["EFSA and JECFA"] * 70)

    rows = (
        ("one can of diet drink", can_low, can_high, PALETTE[1]),
        ("one glass of fruit juice", 0.25 * JUICE["low"], 0.25 * JUICE["high"], PALETTE[0]),
        ("aspartame at the daily limit, 70 kg", adi, adi, PALETTE[1]),
        ("made by the body in a day", float(ENDOGENOUS[0]), float(ENDOGENOUS[1]), PALETTE[0]),
        ("one kilogram of apples", float(APPLES_1KG[0]), float(APPLES_1KG[1]), PALETTE[0]),
        (
            "minimum lethal dose, 70 kg",
            float(70 * LETHAL_PER_KG[0]),
            float(70 * LETHAL_PER_KG[1]),
            PALETTE[7],
        ),
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for position, (_, low, high, colour) in enumerate(reversed(rows)):
        if low == high:
            axis.plot(
                [low],
                [position],
                "o",
                color=colour,
                markersize=9,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
            )
            text = f"{low:,.0f} mg"
        else:
            axis.plot(
                [low, high], [position, position], color=colour, linewidth=9, solid_capstyle="round"
            )
            text = f"{low:,.0f} to {high:,.0f} mg"
        axis.text(high * 1.35, position, text, va="center", fontsize=9, color=INK_SECONDARY)

    axis.set_xscale("log")
    axis.set_xlim(1, 2_000_000)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([row[0] for row in reversed(rows)], fontsize=9)
    axis.set_xticks([1, 10, 100, 1_000, 10_000, 100_000])
    axis.set_xticklabels(["1", "10", "100", "1,000", "10,000", "100,000"])
    axis.minorticks_off()
    axis.set_xlabel("methanol, mg (logarithmic scale)")
    axis.set_title("Where methanol comes from, and how much")
    axis.legend(
        handles=[
            Line2D([], [], color=PALETTE[1], linewidth=6, label="from aspartame"),
            Line2D([], [], color=PALETTE[0], linewidth=6, label="from food and the body"),
            Line2D([], [], color=PALETTE[7], linewidth=6, label="toxic dose"),
        ],
        loc="lower right",
        fontsize=8.5,
        bbox_to_anchor=(1.0, 0.12),
    )
    axis.grid(axis="y", visible=False)

    return save_figure(figure, slug="aspartame_methanol_sources", output_dir=output_dir)


def render_kinetics_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the one-compartment bound against the concentrations measured in volunteers."""
    use_house_style()
    doses = dose_grid()
    figure, axis = plt.subplots(figsize=(7.2, 4.4))

    axis.plot(
        doses,
        [predicted_peak(dose) for dose in doses],
        color=PALETTE[0],
        label="predicted: a tenth of the dose in 0.77 L per kg",
    )
    observed = list(STEGINK)
    axis.errorbar(
        observed,
        [STEGINK[dose][0] for dose in observed],
        yerr=[STEGINK[dose][1] for dose in observed],
        fmt="o",
        color=PALETTE[1],
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        capsize=3,
        label="measured in volunteers, mean and SD",
    )

    axis.axhline(DETECTION_LIMIT, color=INK_MUTED, linewidth=1.2, linestyle=(0, (3, 3)))
    axis.text(
        1.1,
        DETECTION_LIMIT * 1.12,
        "detection limit in the studies, 4 mg/L",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    nervous = BLOOD_THRESHOLDS["effects on the nervous system"]
    axis.axhline(nervous, color=PALETTE[7], linewidth=1.2, linestyle=(0, (3, 3)))
    axis.text(
        1.1,
        nervous * 1.12,
        "effects on the nervous system begin, 200 mg/L",
        fontsize=8.5,
        color=INK_SECONDARY,
    )

    markers = (
        (float(ADI["EFSA and JECFA"]), "the acceptable\ndaily intake"),
        (DRINK["maximum permitted"] * CAN_LITRES / 70.0, "one can,\n70 kg adult"),
    )
    for dose, text in markers:
        axis.plot(
            [dose],
            [predicted_peak(dose)],
            "o",
            color=PALETTE[0],
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        axis.annotate(
            text,
            xy=(dose, predicted_peak(dose)),
            xytext=(8, -24),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SECONDARY,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1, 300)
    axis.set_ylim(0.1, 600)
    axis.set_xticks([1, 3, 10, 40, 100, 200])
    axis.set_xticklabels(["1", "3", "10", "40", "100", "200"])
    axis.set_yticks([0.1, 1, 10, 100])
    axis.set_yticklabels(["0.1", "1", "10", "100"])
    axis.minorticks_off()
    axis.set_xlabel("single dose of aspartame, mg per kg of body weight")
    axis.set_ylabel("peak blood methanol, mg/L")
    axis.set_title("The dose decides: predicted and measured blood methanol")
    axis.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.86))

    return save_figure(figure, slug="aspartame_blood_methanol", output_dir=output_dir)


def render_aspartame_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the aspartame essay."""
    return (
        render_sources_figure(output_dir=output_dir),
        render_kinetics_figure(output_dir=output_dir),
    )
