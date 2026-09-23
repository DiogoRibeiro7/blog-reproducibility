"""Figure renderers for the article on drift in production machine learning."""

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from blog_reproducibility.common.plotting import (
    BASELINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.drift import (
    CAPACITY,
    EDGES,
    PRODUCTION_CURVE,
    SHIFT_DAY,
    STAGES,
    TRAINING_CURVE,
    event_probability,
    simulate_alerts,
    simulate_feature_drift,
)

__all__ = [
    "ARROWS",
    "BOXES",
    "render_architecture_figure",
    "render_concept_drift_figure",
    "render_feature_drift_figure",
    "render_prediction_drift_figure",
]

# Box text, lower-left corner, width, height, and palette index for each stage.
BOXES: Final[dict[str, tuple[str, float, float, float, float, int]]] = {
    "Production inputs": ("Production\ninputs", 0.25, 3.25, 1.55, 0.75, 0),
    "Data quality checks": ("Data quality\nchecks", 2.15, 3.25, 1.55, 0.75, 2),
    "Feature drift monitoring": ("Feature drift\nmonitoring", 4.05, 3.25, 1.55, 0.75, 0),
    "Prediction monitoring": ("Prediction\nmonitoring", 5.95, 3.25, 1.55, 0.75, 3),
    "Labels and outcomes": ("Labels and\noutcomes", 4.05, 1.45, 1.55, 0.75, 1),
    "Decision monitoring": ("Decision\nmonitoring", 5.95, 1.45, 1.55, 0.75, 4),
    "Response": ("Response:\nfix, recalibrate,\nretrain, pause", 8.0, 2.35, 1.75, 1.05, 7),
}

# Hand-placed start and end points for each edge of the pipeline.
ARROWS: Final[dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]]] = {
    ("Production inputs", "Data quality checks"): ((1.8, 3.63), (2.15, 3.63)),
    ("Data quality checks", "Feature drift monitoring"): ((3.7, 3.63), (4.05, 3.63)),
    ("Feature drift monitoring", "Prediction monitoring"): ((5.6, 3.63), (5.95, 3.63)),
    ("Prediction monitoring", "Response"): ((7.5, 3.63), (8.0, 3.0)),
    ("Feature drift monitoring", "Labels and outcomes"): ((4.83, 3.25), (4.83, 2.2)),
    ("Labels and outcomes", "Decision monitoring"): ((5.6, 1.83), (5.95, 1.83)),
    ("Decision monitoring", "Response"): ((7.5, 1.83), (8.0, 2.75)),
}


def render_architecture_figure(*, output_dir: Path) -> FigureArtifact:
    """Draw the monitoring pipeline from production inputs to a response."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(10.6, 5.2))
    axis.set_axis_off()
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 5)

    for stage in STAGES:
        text, x, y, width, height, colour_index = BOXES[stage]
        colour = PALETTE[colour_index]
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.025,rounding_size=0.08",
                facecolor=colour,
                alpha=0.16,
                edgecolor=colour,
                linewidth=1.8,
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=10.2,
            color=INK_PRIMARY,
            fontweight="semibold",
        )

    for edge in EDGES:
        start, end = ARROWS[edge]
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color=INK_MUTED,
            )
        )

    axis.text(
        0.25,
        4.55,
        "Monitoring should separate signal, impact, and action",
        fontsize=13.5,
        fontweight="semibold",
        color=INK_PRIMARY,
    )
    axis.text(
        0.25,
        4.25,
        "Feature changes are early signals; mature labels and decision metrics tell "
        "whether action is needed.",
        fontsize=10.2,
        color=INK_SECONDARY,
    )
    axis.text(
        4.15,
        0.72,
        "Label delay means outcome monitoring lags production.",
        fontsize=9.6,
        color=INK_SECONDARY,
    )
    return save_figure(figure, slug="drift_monitoring_architecture", output_dir=output_dir)


def render_feature_drift_figure(*, output_dir: Path) -> FigureArtifact:
    """Overlay the training and production histograms of the drifted feature."""
    use_house_style()
    sample = simulate_feature_drift()
    bins = np.linspace(-4, 5, 80).tolist()
    training_mean = float(np.mean(sample.training))
    production_mean = float(np.mean(sample.production))

    figure, axis = plt.subplots()
    axis.hist(
        sample.training,
        bins=bins,
        density=True,
        alpha=0.38,
        color=PALETTE[0],
        label="Training period",
    )
    axis.hist(
        sample.production,
        bins=bins,
        density=True,
        alpha=0.42,
        color=PALETTE[1],
        label="Production period",
    )
    axis.axvline(training_mean, color=PALETTE[0], lw=2)
    axis.axvline(production_mean, color=PALETTE[1], lw=2)
    axis.annotate(
        "mean shifted",
        xy=(production_mean, 0.32),
        xytext=(1.95, 0.42),
        fontsize=9.5,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 1},
    )
    axis.set(
        title="Feature drift: the input distribution moved",
        xlabel="standardized sensor value",
        ylabel="density",
    )
    axis.legend()
    return save_figure(figure, slug="feature_drift_distribution", output_dir=output_dir)


def render_concept_drift_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the training and production risk curves against the feature."""
    use_house_style()
    x = np.linspace(-4, 4, 400)

    figure, axis = plt.subplots()
    axis.plot(
        x, event_probability(x, TRAINING_CURVE), color=PALETTE[0], label="Training relationship"
    )
    axis.plot(
        x,
        event_probability(x, PRODUCTION_CURVE),
        color=PALETTE[1],
        label="Production relationship",
    )
    axis.axhline(0.5, color=BASELINE, lw=1)
    axis.axvline(TRAINING_CURVE.midpoint, color=PALETTE[0], lw=1.4, alpha=0.75)
    axis.axvline(PRODUCTION_CURVE.midpoint, color=PALETTE[1], lw=1.4, alpha=0.75)
    axis.annotate(
        "same score threshold,\ndifferent real risk",
        xy=(0.55, 0.5),
        xytext=(-2.8, 0.64),
        fontsize=9.5,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 1},
    )
    axis.set(
        title="Concept drift: the feature-target relationship changed",
        xlabel="feature value",
        ylabel="probability of event",
        ylim=(0, 1),
    )
    axis.legend()
    return save_figure(figure, slug="concept_drift_boundary", output_dir=output_dir)


def render_prediction_drift_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot daily alerts against review capacity across the distribution shift."""
    use_house_style()
    series = simulate_alerts()
    days, shifted = series.days, series.shifted

    figure, axis = plt.subplots()
    axis.plot(days, shifted, color=PALETTE[0], label="Alerts above fixed threshold")
    axis.axvline(SHIFT_DAY, color=PALETTE[1], lw=2, label="Distribution shift")
    axis.axhline(CAPACITY, color=PALETTE[7], lw=1.8, label="Review capacity")
    axis.fill_between(
        days, CAPACITY, shifted, where=shifted > CAPACITY, color=PALETTE[7], alpha=0.14
    )
    axis.annotate(
        "threshold unchanged,\nworkload changed",
        xy=(75, float(shifted[74])),
        xytext=(43, 188),
        fontsize=9.5,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 1},
    )
    axis.set(
        title="Prediction drift: scores cross the action threshold more often",
        xlabel="day",
        ylabel="daily alerts",
    )
    axis.legend()
    return save_figure(figure, slug="prediction_drift_threshold", output_dir=output_dir)
