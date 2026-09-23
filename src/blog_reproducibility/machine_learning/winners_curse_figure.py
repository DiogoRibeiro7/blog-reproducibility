"""Figure renderer for the article on the winner's curse in model selection."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.winners_curse import OptimismCurve, optimism_curves

_COLOURS = (PALETTE[1], PALETTE[0], PALETTE[2])


def render_winners_curse_figure(
    *, output_dir: Path, curves: tuple[OptimismCurve, ...] | None = None
) -> FigureArtifact:
    """Plot the winner's optimism against the number of configurations compared."""
    use_house_style()
    result = curves if curves is not None else optimism_curves()

    figure, axis = plt.subplots()
    for curve, colour in zip(result, _COLOURS, strict=False):
        axis.plot(
            curve.candidates,
            curve.optimism,
            marker="o",
            color=colour,
            label=f"validation set of {curve.validation_size:,} items",
        )
    axis.axhline(0, color=PALETTE[3], lw=1, ls="--", label="honest estimate (fresh test set)")
    candidates = result[0].candidates
    axis.set_xscale("log")
    axis.set_xticks(candidates)
    axis.set_xticklabels([str(k) for k in candidates])
    axis.set_xlabel("configurations compared on the validation set")
    axis.set_ylabel("winner's validation accuracy minus its true accuracy (points)")
    axis.set_title("The best validation score is an overestimate")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="winners_curse_optimism", output_dir=output_dir)
