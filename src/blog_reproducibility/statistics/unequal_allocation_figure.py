"""Figure renderer for the article on what an uneven traffic split costs."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.unequal_allocation import (
    ANNOTATED_SHARES,
    variance_curve,
    variance_factor,
)


def render_allocation_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the variance inflation of a split against the share sent to treatment."""
    use_house_style()
    shares, factors = variance_curve()

    figure, axis = plt.subplots()
    axis.plot(shares, factors, color=PALETTE[0], lw=2, label="Variance, relative to an even split")
    axis.axhspan(1.0, 1.1, color=PALETTE[2], alpha=0.18, label="Within 10% of the even split")
    for share in ANNOTATED_SHARES:
        factor = variance_factor(share)
        axis.plot([share], [factor], marker="o", color=PALETTE[1], ms=6)
        axis.annotate(
            f"{share:.0%}: {factor:.2f}x",
            (share, factor),
            color=INK_SECONDARY,
            fontsize=9,
            ha="left",
            va="bottom",
            xytext=(6, 3),
            textcoords="offset points",
        )
    axis.set_ylim(0.8, 7)
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("share of traffic sent to the treatment arm")
    axis.set_ylabel("variance, relative to an even split")
    axis.set_title("Anything from thirty to seventy percent is nearly free")
    axis.legend(loc="upper center")

    return save_figure(figure, slug="allocation_variance_cost", output_dir=output_dir)
