"""Figure renderer for the article on measurement error in predictors."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    BASELINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.measurement_error import (
    EXTRAPOLATION_POINT,
    MeasurementErrorSummary,
    example_payload,
    rational_extrapolant,
)


def render_measurement_error_figure(
    *, output_dir: Path, summary: MeasurementErrorSummary | None = None
) -> FigureArtifact:
    """Plot attenuation against reliability, and the SIMEX refits with both extrapolants."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    simex = result.simex

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    reliability = np.linspace(0.1, 1.0, 100)
    left.plot(
        reliability,
        result.true_slope * reliability,
        color=INK_MUTED,
        lw=1.4,
        label="Theory: slope = reliability",
    )
    left.plot(
        [row.reliability for row in result.attenuation],
        [row.fitted_slope for row in result.attenuation],
        "o",
        color=PALETTE[0],
        markersize=7,
        label="Simulated fit",
    )
    left.axhline(result.true_slope, color=BASELINE, lw=1.0)
    left.set_xlabel("reliability of the predictor (signal share of its variance)")
    left.set_ylabel("fitted slope (true slope = 1)")
    left.set_xlim(0, 1.05)
    left.set_ylim(0, 1.1)
    left.set_title("Attenuation", fontsize=11.5)
    left.legend(loc="upper left")

    multiples = np.linspace(EXTRAPOLATION_POINT, 2, 200)
    right.plot(
        multiples,
        np.polyval(simex.quadratic, multiples),
        color=PALETTE[1],
        lw=1.6,
        label="Quadratic extrapolant",
    )
    right.plot(
        multiples,
        rational_extrapolant(multiples, simex.rational_numerator, simex.rational_denominator),
        color=PALETTE[0],
        lw=1.6,
        label="Rational extrapolant",
    )
    right.plot(
        simex.multiples,
        simex.slopes,
        "o",
        color=INK_PRIMARY,
        markersize=6,
        label="Refits with added noise",
    )
    right.plot(
        [EXTRAPOLATION_POINT],
        [result.true_slope],
        marker="D",
        markersize=8,
        color=PALETTE[2],
        markeredgecolor=SURFACE,
        label="True slope",
    )
    right.axvline(0, color=BASELINE, lw=1.0)
    right.annotate(
        "observed data",
        xy=(0, 0.35),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=9,
        color=INK_SECONDARY,
    )
    right.set_xlabel("added noise variance, as a multiple of the existing noise")
    right.set_ylabel("fitted slope")
    right.set_ylim(0.2, 1.1)
    right.set_title("Simulation-extrapolation", fontsize=11.5)
    right.legend(loc="upper right")
    figure.suptitle(
        "Noise in a predictor pulls its slope toward zero, predictably",
        x=0.012,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )

    return save_figure(figure, slug="measurement_error_attenuation_simex", output_dir=output_dir)
