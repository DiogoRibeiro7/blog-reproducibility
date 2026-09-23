"""Figure renderer for the article on Monte Carlo macroeconomic modelling."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.economics.gdp_fan import fan_percentiles, simulate_gdp_paths


def render_fan_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the median GDP path with its 50% and 90% bands."""
    use_house_style()
    p05, p25, p50, p75, p95 = fan_percentiles(simulate_gdp_paths())
    years = np.arange(p50.size)

    figure, axis = plt.subplots()
    axis.fill_between(years, p05, p95, color=PALETTE[0], alpha=0.12, label="90% interval")
    axis.fill_between(years, p25, p75, color=PALETTE[0], alpha=0.24, label="50% interval")
    axis.plot(years, p50, color=PALETTE[0], label="Median path")
    axis.set(
        title="Persistent shocks make the fan widen faster than the horizon",
        xlabel="year",
        ylabel="GDP index (start = 100)",
    )
    axis.legend()

    return save_figure(figure, slug="monte_carlo_fan", output_dir=output_dir)
