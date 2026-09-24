"""Figure renderer for the article on equivalence and non-inferiority testing."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.equivalence_testing import (
    EquivalenceSummary,
    OutcomeRow,
    example_payload,
)


def _series(rows: tuple[OutcomeRow, ...], true_difference: float) -> tuple[OutcomeRow, ...]:
    return tuple(row for row in rows if row.true_difference == true_difference)


def render_equivalence_figure(
    *, output_dir: Path, summary: EquivalenceSummary | None = None
) -> FigureArtifact:
    """Plot the share of comparisons reaching each conclusion against the number of test cases."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    identical = _series(result.rows, 0.0)
    much_worse = _series(result.rows, -1.5)
    slightly_worse = _series(result.rows, -0.5)
    sizes = [row.test_cases for row in identical]

    figure, axis = plt.subplots()
    axis.plot(
        [row.test_cases for row in much_worse],
        [1 - row.t_test_significant for row in much_worse],
        marker="o",
        color=PALETTE[1],
        label="t-test not significant, challenger 1.5 points worse",
    )
    axis.plot(
        sizes,
        [row.equivalent for row in identical],
        marker="o",
        color=PALETTE[0],
        label="Equivalence shown, models identical",
    )
    axis.plot(
        [row.test_cases for row in slightly_worse],
        [row.equivalent for row in slightly_worse],
        marker="o",
        color=PALETTE[2],
        label="Equivalence shown, challenger 0.5 points worse",
    )
    axis.set_xscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([f"{size:,}" for size in sizes])
    axis.set_xlabel("paired test cases")
    axis.set_ylabel("share of comparisons")
    axis.set_ylim(0, 1.04)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title("A non-significant difference is not equivalence")
    axis.legend(loc="center right")

    return save_figure(figure, slug="equivalence_testing_power", output_dir=output_dir)
