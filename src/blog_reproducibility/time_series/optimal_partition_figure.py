"""Figure renderer for the article on offline change-point detection."""

from math import log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    BASELINE,
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.time_series.optimal_partition import (
    AUTOCORRELATION,
    FIGURE_MULTIPLES,
    TRUE_CHANGE_POINTS,
    make_series,
    optimal_partition,
    partition_sweep,
    robust_sigma,
)


def render_segmentation_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the default segmentation and the change-point count against the penalty."""
    use_house_style()
    series, _ = make_series(np.random.default_rng(0))
    autocorrelated, _ = make_series(np.random.default_rng(1), phi=AUTOCORRELATION)
    sigma = robust_sigma(series)
    found = optimal_partition(series, 2 * log(series.size) * sigma**2)

    figure, (segments, counts) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    segments.plot(series, color=INK_MUTED, lw=0.9, alpha=0.8, label="Series")
    bounds = (0, *found, series.size)
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        segments.plot([start, end - 1], [series[start:end].mean()] * 2, color=PALETTE[0], lw=2.4)
    segments.plot([], [], color=PALETTE[0], lw=2.4, label="Segment means")
    for point in found:
        segments.axvline(point, color=PALETTE[1], lw=1.2)
    segments.plot([], [], color=PALETTE[1], lw=1.2, label="Detected change points")
    segments.set(xlabel="time", ylabel="value")
    segments.set_title("Independent noise, penalty 2 log n", fontsize=11.5)
    segments.legend(loc="lower left")

    for values, sigma_hat, colour, label in (
        (series, sigma, PALETTE[0], "Independent noise"),
        (autocorrelated, robust_sigma(autocorrelated), PALETTE[1], "Autocorrelated noise (0.6)"),
    ):
        rows = partition_sweep(values, FIGURE_MULTIPLES, sigma_hat)
        counts.plot(
            FIGURE_MULTIPLES,
            [len(row.change_points) for row in rows],
            marker="o",
            color=colour,
            label=label,
        )
    counts.axhline(len(TRUE_CHANGE_POINTS), color=BASELINE, lw=1.2)
    counts.annotate(
        f"true count: {len(TRUE_CHANGE_POINTS)}",
        xy=(FIGURE_MULTIPLES[-1], len(TRUE_CHANGE_POINTS)),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    counts.set(xscale="log", yscale="log")
    counts.set_xticks(FIGURE_MULTIPLES)
    counts.set_xticklabels([f"{multiple:g}" for multiple in FIGURE_MULTIPLES])
    counts.set_yticks([1, 2, 4, 10, 30, 100])
    counts.set_yticklabels(["1", "2", "4", "10", "30", "100"])
    counts.set(xlabel="penalty, as a multiple of log n", ylabel="change points found")
    counts.set_title("The penalty decides how many", fontsize=11.5)
    counts.legend(loc="upper right")
    figure.suptitle(
        "Offline change-point detection is a penalised partition",
        x=0.012,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )

    return save_figure(figure, slug="changepoint_segmentation_penalty", output_dir=output_dir)
