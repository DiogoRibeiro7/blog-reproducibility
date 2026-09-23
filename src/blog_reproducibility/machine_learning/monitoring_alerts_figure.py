"""Figure renderers for the article on multiple comparisons in drift monitoring."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.monitoring_alerts import (
    ALPHA,
    FEATURES,
    BurstComparison,
    PowerRow,
    alert_bursts,
    power_curves,
)


def render_power_curves_figure(
    *, output_dir: Path, rows: tuple[PowerRow, ...] | None = None
) -> FigureArtifact:
    """Plot the share of drifting features flagged per day against the shift size."""
    use_house_style()
    result = rows if rows is not None else power_curves()
    shifts = [row.shift for row in result]

    figure, axis = plt.subplots()
    series = (
        ([row.uncorrected for row in result], "Uncorrected, alpha = 0.05", PALETTE[1]),
        ([row.benjamini_hochberg for row in result], "Benjamini-Hochberg, q = 0.05", PALETTE[0]),
        ([row.bonferroni for row in result], "Bonferroni", PALETTE[2]),
    )
    for values, label, colour in series:
        axis.plot(shifts, values, marker="o", color=colour, label=label)
    axis.set_xlabel("shift in the drifting features (standard deviations)")
    axis.set_ylabel("drifting features flagged per day")
    axis.set_ylim(0, 1.04)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title("Corrections cost power only where the shift is subtle")
    axis.legend(loc="lower right")

    return save_figure(figure, slug="drift_alert_power_curves", output_dir=output_dir)


def render_alert_bursts_figure(
    *, output_dir: Path, bursts: BurstComparison | None = None
) -> FigureArtifact:
    """Plot histograms of daily false-alert counts for independent and correlated features."""
    use_house_style()
    result = bursts if bursts is not None else alert_bursts()
    expected = ALPHA * FEATURES

    figure, axis = plt.subplots()
    bins = [float(edge) for edge in np.arange(0, 40, 2)]
    axis.hist(
        result.independent.counts,
        bins=bins,
        color=PALETTE[0],
        alpha=0.85,
        label=f"Independent features (sd {result.independent.sd:.1f})",
    )
    axis.hist(
        result.correlated.counts,
        bins=bins,
        color=PALETTE[1],
        alpha=0.6,
        label=f"Correlated features (sd {result.correlated.sd:.1f})",
    )
    axis.axvline(expected, color=INK_MUTED, lw=1.2)
    axis.annotate(
        "expected: 10 per day",
        xy=(expected, axis.get_ylim()[1] * 0.92),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    axis.set_xlabel("false alerts per day (no drift anywhere)")
    axis.set_ylabel("days")
    axis.set_title("Correlated features turn a steady trickle of false alerts into bursts")
    axis.legend()

    return save_figure(figure, slug="drift_alert_bursts", output_dir=output_dir)
