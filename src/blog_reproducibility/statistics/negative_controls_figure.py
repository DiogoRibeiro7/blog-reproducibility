"""Figure renderer for the article on negative controls."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    BASELINE,
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.negative_controls import TrackingCurve, tracking_curve


def render_negative_control_figure(
    *, output_dir: Path, curve: TrackingCurve | None = None
) -> FigureArtifact:
    """Plot the outcome's bias and the control's signal against the outcome's exposure."""
    use_house_style()
    result = curve if curve is not None else tracking_curve()

    figure, axis = plt.subplots()
    axis.plot(
        result.strengths,
        result.biases,
        marker="o",
        color=PALETTE[1],
        lw=2,
        label="Bias in the real outcome",
    )
    axis.plot(
        result.strengths,
        result.signals,
        marker="s",
        color=PALETTE[0],
        lw=2,
        label="Signal on the negative control",
    )
    axis.axvline(result.control_exposure, color=INK_MUTED, lw=1.2, ls=":")
    axis.annotate(
        "the control's own exposure",
        (result.control_exposure, max(result.biases) * 0.92),
        color=INK_SECONDARY,
        fontsize=9,
        ha="left",
        va="top",
        xytext=(6, 0),
        textcoords="offset points",
    )
    axis.axhline(0, color=BASELINE, lw=1)
    axis.set_xlabel("how strongly the hidden variable affects the outcome")
    axis.set_ylabel("difference between adopters and others")
    axis.set_title("The control announces the bias without measuring it")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="negative_control_tracking", output_dir=output_dir)
