"""Figure renderers for the cold-days-in-a-warming-climate article."""

from math import sqrt
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.climate_evidence import (
    COOLER_MEAN,
    SPREAD,
    WARMER_MEAN,
    climate_log_ratio,
)

TEMPERATURES = tuple(index / 10 for index in range(-150, 251))


def render_distribution_shift_figure(*, output_dir: Path) -> FigureArtifact:
    """Show that a warmer distribution still puts mass below freezing."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(8.2, 4.3))

    for index, mean in enumerate((COOLER_MEAN, WARMER_MEAN)):
        distribution = NormalDist(mean, SPREAD)
        density = [distribution.pdf(value) for value in TEMPERATURES]
        axis.plot(TEMPERATURES, density, color=PALETTE[index], label=f"Mean {mean:g} C")
        axis.fill_between(
            TEMPERATURES,
            density,
            where=[value <= 0 for value in TEMPERATURES],
            color=PALETTE[index],
            alpha=0.18,
        )

    axis.axvline(0, color=INK_SECONDARY, linestyle="--", linewidth=1)
    axis.set(
        xlabel="Temperature (C)",
        ylabel="Probability density",
        title="A warmer distribution still includes freezing days",
    )
    axis.legend()

    return save_figure(figure, slug="science_weather_climate_shift", output_dir=output_dir)


def render_evidence_figure(*, output_dir: Path) -> FigureArtifact:
    """Contrast one ambiguous reading with a season of them."""
    use_house_style()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].plot(
        TEMPERATURES,
        [climate_log_ratio(value) for value in TEMPERATURES],
        color=PALETTE[1],
    )
    axes[0].axhline(0, color=INK_SECONDARY, linewidth=1)
    axes[0].axvline(6, color=INK_SECONDARY, linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Observed temperature (C)",
        ylabel="Log likelihood ratio (nats)",
        title="One reading can favour either model",
        xlim=(-10, 20),
    )

    sample_sizes = list(range(1, 101))
    critical = NormalDist().inv_cdf(0.95)
    gap = WARMER_MEAN - COOLER_MEAN
    per_observation_mean = gap**2 / (2 * SPREAD**2)
    per_observation_sd = gap / SPREAD

    for sign, name, colour in (
        (1, "Warmer generates data", PALETTE[1]),
        (-1, "Cooler generates data", PALETTE[0]),
    ):
        means = [sign * per_observation_mean * number for number in sample_sizes]
        widths = [critical * per_observation_sd * sqrt(number) for number in sample_sizes]
        axes[1].plot(sample_sizes, means, color=colour, label=name)
        axes[1].fill_between(
            sample_sizes,
            [mean - width for mean, width in zip(means, widths, strict=True)],
            [mean + width for mean, width in zip(means, widths, strict=True)],
            color=colour,
            alpha=0.15,
        )

    axes[1].axhline(0, color=INK_SECONDARY, linewidth=1)
    axes[1].set(
        xlabel="Independent observations",
        ylabel="Total log likelihood ratio (nats)",
        title="Expected evidence and central 90% bands",
    )
    axes[1].legend(fontsize=8, loc="upper left")

    return save_figure(figure, slug="science_climate_kl_evidence", output_dir=output_dir)


def render_climate_evidence_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the article."""
    return (
        render_distribution_shift_figure(output_dir=output_dir),
        render_evidence_figure(output_dir=output_dir),
    )
