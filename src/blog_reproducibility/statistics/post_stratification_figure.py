"""Figure renderer for the article on weighting a survey by post-stratification."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    BASELINE,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.post_stratification import (
    PostStratificationSummary,
    example_payload,
)


def render_post_stratification_figure(
    *, output_dir: Path, summary: PostStratificationSummary | None = None
) -> FigureArtifact:
    """Plot the effective sample and the surviving bias against the spread of the weights."""
    use_house_style()
    curve = (summary if summary is not None else example_payload()).curve

    figure, axis = plt.subplots()
    axis.plot(
        curve.weight_ratio,
        curve.effective_share,
        marker="o",
        color=PALETTE[0],
        lw=2,
        label="Effective sample, share of respondents",
    )
    axis.plot(
        curve.weight_ratio,
        curve.surviving_bias,
        marker="s",
        color=PALETTE[1],
        lw=2,
        label="Bias surviving, share of the unweighted bias",
    )
    axis.axhline(0, color=BASELINE, lw=1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_ylim(-0.1, 1.05)
    axis.set_xlabel("largest weight divided by smallest")
    axis.set_ylabel("share")
    axis.set_title("Weighting removes the bias and spends precision doing it")
    axis.legend(loc="center right")

    return save_figure(figure, slug="weighting_effective_sample", output_dir=output_dir)
