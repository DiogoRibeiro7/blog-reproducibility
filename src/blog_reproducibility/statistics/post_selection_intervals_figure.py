"""Figure renderers for the article on exact post-selection confidence intervals."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SEQUENTIAL,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.post_selection_intervals import (
    DENSITY_MEANS,
    DENSITY_UPPER,
    OBSERVED,
    ORDINARY_Z,
    THRESHOLD,
    ProcedureCurves,
    WidthCurve,
    conditional_density,
    procedure_curves,
    selective_interval,
    tail_constant,
    width_curve,
)

__all__ = [
    "render_selective_interval_runaway_figure",
    "render_selective_width_by_procedure_figure",
]


def render_selective_interval_runaway_figure(
    *, output_dir: Path, summary: WidthCurve | None = None
) -> FigureArtifact:
    """Plot conditional densities at three means, and the exact width against the distance."""
    use_house_style()
    curve = summary if summary is not None else width_curve()
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # Same quantity at three parameter values: one hue, light to dark.
    grid = np.linspace(THRESHOLD, DENSITY_UPPER, 600)
    shades = (SEQUENTIAL[2], SEQUENTIAL[4], SEQUENTIAL[6])
    for mean, shade in zip(DENSITY_MEANS, shades, strict=True):
        left.plot(
            grid, conditional_density(grid, mean), color=shade, lw=2.2, label=f"mean {mean:g}"
        )
    left.axvline(OBSERVED, color=INK_MUTED, lw=1, ls=":")
    left.annotate(
        f"observed {OBSERVED:g}",
        (OBSERVED, 19.5),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        color=INK_SECONDARY,
        va="center",
    )
    left.set_xlim(THRESHOLD, DENSITY_UPPER)
    left.set_ylim(0, 23)
    left.set_xlabel(f"observation, given that it exceeded {THRESHOLD:g}")
    left.set_ylabel("conditional density")
    left.set_title(f"A very negative mean predicts a value just above {THRESHOLD:g}")
    left.legend(loc="upper right")

    distances = np.array(curve.distances)
    widths = np.array(curve.widths)
    constant = tail_constant()
    ordinary = 2 * ORDINARY_Z
    observed_distance = OBSERVED - THRESHOLD
    observed_width = selective_interval(OBSERVED).width
    right.plot(distances, widths, color=PALETTE[0], lw=2.4, label="Exact selective interval")
    right.plot(
        distances,
        constant / distances,
        color=PALETTE[1],
        lw=2,
        ls="--",
        label=f"{constant:.2f} / distance",
    )
    right.axhline(
        ordinary, color=INK_MUTED, lw=1.4, ls=":", label=f"Ordinary interval, {ordinary:.2f}"
    )
    nearest = int(np.argmin(np.abs(distances - observed_distance)))
    right.plot(
        [observed_distance],
        [widths[nearest]],
        marker="o",
        ms=7,
        color=PALETTE[0],
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=5,
    )
    right.annotate(
        f"observed {OBSERVED:g}: width {observed_width:.0f}",
        (observed_distance, observed_width),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=9,
        color=INK_SECONDARY,
    )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("distance of the observation above the threshold")
    right.set_ylabel("width of the 95% interval")
    right.set_title("Width grows as one over the distance")
    right.legend(loc="upper right")
    # The house style turns constrained layout on; these panels were laid out tight.
    figure.set_layout_engine("tight")
    return save_figure(figure, slug="selective_interval_runaway", output_dir=output_dir)


def render_selective_width_by_procedure_figure(
    *, output_dir: Path, summary: ProcedureCurves | None = None
) -> FigureArtifact:
    """Plot the median and 90th percentile width given selection under three procedures."""
    use_house_style()
    curves = summary if summary is not None else procedure_curves()
    means = np.array(curves.means)

    figure, axis = plt.subplots()
    axis.fill_between(
        means, curves.hard_median, curves.hard_p90, color=PALETTE[0], alpha=0.16, lw=0
    )
    axis.fill_between(
        means, curves.randomised_median, curves.randomised_p90, color=PALETTE[1], alpha=0.2, lw=0
    )
    axis.plot(
        means, curves.hard_median, color=PALETTE[0], lw=2.4, label="Hard threshold, conditional"
    )
    axis.plot(
        means, curves.randomised_median, color=PALETTE[1], lw=2.2, label="Randomised selection"
    )
    axis.plot(
        means,
        np.full_like(means, curves.split),
        color=PALETTE[2],
        lw=2,
        ls="--",
        label="Data splitting",
    )
    axis.axhline(
        2 * ORDINARY_Z, color=INK_MUTED, lw=1.4, ls=":", label="Ordinary interval, not valid here"
    )
    axis.annotate(
        "band: median to 90th percentile",
        (0.08, curves.hard_p90[0]),
        xytext=(6, -2),
        textcoords="offset points",
        fontsize=9,
        color=INK_SECONDARY,
        va="top",
    )
    axis.set_yscale("log")
    axis.set_ylim(3, 130)
    axis.set_yticks([4, 5, 10, 20, 50, 100])
    axis.set_yticklabels(["4", "5", "10", "20", "50", "100"])
    axis.set_xlabel("true mean")
    axis.set_ylabel("width of the 95% interval, given selection")
    axis.set_title("What conditioning on a hard threshold costs, and what avoids it")
    axis.legend(loc="upper right")
    return save_figure(figure, slug="selective_width_by_procedure", output_dir=output_dir)
