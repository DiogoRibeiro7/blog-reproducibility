"""Figure renderer for the article on how often to retrain a model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.retraining import FIGURE_PERIODS, schedule_cost


def render_retraining_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot total weekly cost and its two halves against the retraining interval."""
    use_house_style()
    rows = [schedule_cost(p) for p in FIGURE_PERIODS]
    periods = np.array([row.period for row in rows])
    total = np.array([row.total for row in rows])
    lost = np.array([row.quality_lost for row in rows])
    amortised = np.array([row.retraining for row in rows])
    best = int(periods[np.argmin(total)])

    figure, axis = plt.subplots()
    axis.plot(periods, total, color=PALETTE[0], lw=2.4, label="Total weekly cost")
    axis.plot(periods, lost, color=PALETTE[1], lw=2, ls="--", label="Quality foregone")
    axis.plot(periods, amortised, color=PALETTE[2], lw=2, ls=":", label="Retraining, amortised")
    axis.plot([best], [total.min()], marker="o", color=PALETTE[0], ms=8)
    axis.annotate(
        f"cheapest at {best} weeks, {total.min():,.0f} a week",
        (best, total.min()),
        color=INK_SECONDARY,
        fontsize=9,
        ha="left",
        va="top",
        xytext=(8, -6),
        textcoords="offset points",
    )
    axis.set_ylim(0, 9000)
    axis.set_xlabel("weeks between retrains")
    axis.set_ylabel("cost per week")
    axis.set_title("Too often and too rarely cost about the same")
    axis.legend(loc="upper center")

    return save_figure(figure, slug="retraining_cost_curve", output_dir=output_dir)
