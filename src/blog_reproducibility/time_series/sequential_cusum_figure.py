"""Figure renderer for the sequential CUSUM worked example."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.time_series.sequential_cusum import seeded_mean_shift_example


def render_sequential_cusum_figure(*, output_dir: Path) -> FigureArtifact:
    """Render the article's deterministic Gaussian mean-shift CUSUM example."""
    example = seeded_mean_shift_example()
    use_house_style()

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        constrained_layout=True,
    )
    indices = range(len(example.data))

    axes[0].plot(indices, example.data, linewidth=1.5, label="Observed value")
    axes[0].set(
        ylabel="Observation",
        title="A mean shift and a later alarm are different events",
    )

    axes[1].plot(indices, example.scores, label="Upper CUSUM")
    axes[1].axhline(
        example.threshold,
        linestyle=":",
        color=PALETTE[1],
        label=f"Threshold h = {example.threshold:g}",
    )
    axes[1].set(
        xlabel="Observation index (zero-based)",
        ylabel="CUSUM score",
    )

    for axis in axes:
        axis.axvline(
            example.first_changed_index,
            linestyle="--",
            color=PALETTE[1],
            label=f"First changed index: {example.first_changed_index}",
        )
        if example.alarm_index is not None:
            axis.axvline(
                example.alarm_index,
                linestyle="-.",
                color=PALETTE[2],
                label=f"First alarm index: {example.alarm_index}",
            )
        axis.legend(loc="upper left", fontsize=8)

    return save_figure(
        figure,
        slug="sequential_cusum_worked_example",
        output_dir=output_dir,
    )
