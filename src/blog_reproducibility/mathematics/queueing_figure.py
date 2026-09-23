"""Figure renderer for the article on queueing, utilisation, and waiting time."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.mathematics.queueing import (
    HOURS,
    MINUTES_PER_HOUR,
    OVERLOAD_HOUR,
    SIMULATED_UTILISATIONS,
    hourly_means,
    mean_wait,
    overloaded_day,
    simulate_utilisations,
)


def render_queueing_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot waiting against utilisation and the backlog left by one overloaded hour."""
    use_house_style()
    figure, (curve, day) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    utilisation = np.linspace(0.02, 0.97, 300)
    for variability, colour, label in (
        (0.0, PALETTE[2], "Deterministic service"),
        (1.0, PALETTE[0], "Exponential service"),
        (4.0, PALETTE[1], "High-variability service (c² = 4)"),
    ):
        formula = [mean_wait(float(rho), variability) for rho in utilisation]
        curve.plot(utilisation, formula, color=colour, label=label)
    simulated = [row.simulated_mean for row in simulate_utilisations()]
    curve.plot(
        SIMULATED_UTILISATIONS,
        simulated,
        "o",
        color=PALETTE[0],
        markersize=7,
        label="Simulated, exponential",
    )
    curve.set(
        xlim=(0, 1),
        ylim=(0, 25),
        xlabel="utilisation",
        ylabel="mean wait (multiples of service time)",
    )
    curve.set_title("Waiting against utilisation", fontsize=11.5)
    curve.legend(loc="upper left")

    times, waits, _ = overloaded_day()
    day.scatter(
        times / MINUTES_PER_HOUR, waits, s=5, color=INK_MUTED, alpha=0.35, label="Each arrival"
    )
    day.plot(
        np.arange(HOURS) + 0.5,
        hourly_means(times, waits),
        color=PALETTE[0],
        marker="o",
        markersize=4,
        label="Hourly mean",
    )
    day.axvspan(OVERLOAD_HOUR, OVERLOAD_HOUR + 1, color=PALETTE[1], alpha=0.18, lw=0)
    day.annotate(
        "130% load",
        xy=(OVERLOAD_HOUR + 0.5, day.get_ylim()[1] * 0.93),
        ha="center",
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    day.set(xlabel="hour of day", ylabel="wait (minutes)", xlim=(0, HOURS))
    day.set_title("One overloaded hour, four hours of backlog", fontsize=11.5)
    day.legend(loc="upper right")
    figure.suptitle(
        "Utilisation is cheap until it is not",
        x=0.012,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )

    return save_figure(figure, slug="queue_wait_vs_utilisation", output_dir=output_dir)
