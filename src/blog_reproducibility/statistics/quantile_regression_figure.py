"""Figure renderer for the article on quantile regression and prediction intervals."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.quantile_regression import (
    QuantileRegressionSummary,
    example_payload,
)


def render_quantile_regression_figure(
    *, output_dir: Path, summary: QuantileRegressionSummary | None = None
) -> FigureArtifact:
    """Plot deliveries at 5 km against load with the least-squares and quantile bands."""
    use_house_style()
    data = (summary if summary is not None else example_payload()).figure

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.scatter(
        data.loads, data.minutes, s=7, color=INK_MUTED, alpha=0.35, label="Deliveries at 5 km"
    )
    axis.plot(data.grid, data.ols_upper, color=PALETTE[1], label="OLS + normal, 5th to 95th")
    axis.plot(data.grid, data.ols_lower, color=PALETTE[1])
    axis.plot(data.grid, data.upper, color=PALETTE[0], label="Quantile regression, 5th to 95th")
    axis.plot(data.grid, data.lower, color=PALETTE[0])
    axis.plot(
        data.grid,
        data.median,
        color=PALETTE[0],
        lw=1.2,
        alpha=0.7,
        label="Quantile regression, median",
    )
    axis.set_xlabel("network load")
    axis.set_ylabel("delivery time (minutes)")
    axis.set_ylim(0, 110)
    axis.set_title("One interval width cannot fit a spread that changes with load")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="quantile_regression_bands", output_dir=output_dir)
