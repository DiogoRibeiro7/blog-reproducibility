"""Figure renderer for the article on cluster-randomised experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.cluster_randomisation import (
    ClusterSummary,
    example_payload,
)


def render_cluster_figure(
    *, output_dir: Path, summary: ClusterSummary | None = None
) -> FigureArtifact:
    """Plot A/A false positive rates of both tests against the intraclass correlation."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    correlations = [row.intraclass_correlation for row in result.rows]

    figure, axis = plt.subplots()
    axis.plot(
        correlations,
        [row.customer_level for row in result.rows],
        marker="o",
        color=PALETTE[1],
        lw=2,
        label="Customers treated as independent",
    )
    axis.plot(
        correlations,
        [row.store_level for row in result.rows],
        marker="o",
        color=PALETTE[0],
        lw=2,
        label="Test on store means (20 stores)",
    )
    axis.axhline(0.05, color=PALETTE[3], lw=1, ls="--", label="Nominal 5 percent")
    axis.set_xlabel("intraclass correlation (share of variance between stores)")
    axis.set_ylabel("A/A experiments declared significant")
    axis.set_ylim(0, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_title("Randomise stores, analyse customers, and the test breaks")
    axis.legend(loc="center right")

    return save_figure(figure, slug="cluster_randomisation_false_positives", output_dir=output_dir)
