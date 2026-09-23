"""Figure renderer for the article on annotator agreement and the accuracy ceiling."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.annotator_ceiling import (
    CeilingCurve,
    simulate_ceilings,
)

_COLOURS = (PALETTE[1], PALETTE[3], PALETTE[2], PALETTE[0])


def render_annotator_ceiling_figure(
    *, output_dir: Path, curves: tuple[CeilingCurve, ...] | None = None
) -> FigureArtifact:
    """Plot the apparent accuracy of a perfect model against single-annotator error."""
    use_house_style()
    result = curves if curves is not None else simulate_ceilings()

    figure, axis = plt.subplots()
    for curve, colour in zip(result, _COLOURS, strict=False):
        suffix = "s, majority vote" if curve.annotators > 1 else ""
        axis.plot(
            curve.error_rates,
            curve.simulated,
            marker="o",
            color=colour,
            lw=2,
            label=f"{curve.annotators} annotator{suffix}",
        )
    axis.set_xlabel("error rate of a single annotator")
    axis.set_ylabel("accuracy a perfect model appears to reach")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_ylim(0.65, 1.01)
    axis.set_title("Label noise caps the score before the model does")
    axis.legend(loc="lower left")

    return save_figure(figure, slug="annotator_noise_ceiling", output_dir=output_dir)
