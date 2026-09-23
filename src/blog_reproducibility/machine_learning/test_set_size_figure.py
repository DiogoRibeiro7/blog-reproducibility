"""Figure renderer for the article on how big a test set needs to be."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.test_set_size import RankingRow, ranking_rates

_SERIES = (
    ("same_set", "Same test set: B measures higher", PALETTE[0]),
    ("separate_sets", "Separate test sets: B measures higher", PALETTE[2]),
    ("paired_significant", "Paired (McNemar) test significant", PALETTE[1]),
    ("intervals_disjoint", "Independent 95% intervals disjoint", PALETTE[3]),
)


def render_test_set_size_figure(
    *, output_dir: Path, rows: tuple[RankingRow, ...] | None = None
) -> FigureArtifact:
    """Plot how often a test set ranks two classifiers correctly against its size."""
    use_house_style()
    result = rows if rows is not None else ranking_rates()
    sizes = [row.test_size for row in result]

    figure, axis = plt.subplots()
    for field, label, colour in _SERIES:
        axis.plot(
            sizes,
            [float(getattr(row, field)) for row in result],
            marker="o",
            color=colour,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([f"{size:,}" for size in sizes])
    axis.set_xlabel("test-set size")
    axis.set_ylabel("share of test sets")
    axis.set_ylim(0, 1.04)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title("A one-point difference needs thousands of test cases to show")
    axis.legend(loc="center right")

    return save_figure(figure, slug="test_set_size_ranking", output_dir=output_dir)
