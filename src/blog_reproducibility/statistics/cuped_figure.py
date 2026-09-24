"""Figure renderer for the article on CUPED and regression adjustment."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.cuped import StandardErrorRow, standard_error_rows


def render_cuped_figure(
    *, output_dir: Path, rows: tuple[StandardErrorRow, ...] | None = None
) -> FigureArtifact:
    """Plot each estimator's standard error against the covariate's correlation."""
    use_house_style()
    result = rows if rows is not None else standard_error_rows()
    correlations = [row.correlation for row in result]

    figure, axis = plt.subplots()
    axis.plot(
        correlations,
        [row.difference_in_means for row in result],
        marker="o",
        color=PALETTE[1],
        label="Difference in means",
    )
    axis.plot(
        correlations,
        [row.stratified for row in result],
        marker="o",
        color=PALETTE[3],
        label="Stratified by covariate quartile",
    )
    axis.plot(
        correlations,
        [row.cuped for row in result],
        marker="o",
        color=PALETTE[0],
        label="CUPED (regression adjustment gives the same)",
    )
    axis.plot(
        correlations,
        [row.theory for row in result],
        color=PALETTE[0],
        lw=1,
        ls="--",
        label="Theory: SE × √(1 − ρ²)",
    )
    axis.set_xlabel("correlation between the pre-experiment covariate and the outcome")
    axis.set_ylabel("standard error of the estimated effect")
    axis.set_ylim(0, 0.36)
    axis.set_title("A correlated covariate buys the precision of more users")
    axis.legend(loc="lower left")

    return save_figure(figure, slug="cuped_variance_reduction", output_dir=output_dir)
