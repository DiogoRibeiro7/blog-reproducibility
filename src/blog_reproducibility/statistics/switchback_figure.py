"""Figure renderer for the article on switchback experiments."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.switchback import (
    CARRY_MINUTES,
    LIFT,
    PeriodRow,
    period_rows,
)

_ERROR_TICKS = (0.02, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


def _period_label(period: int) -> str:
    return f"{period} m" if period < 60 else f"{period // 60} h"


def render_switchback_figure(
    *, output_dir: Path, rows: tuple[PeriodRow, ...] | None = None
) -> FigureArtifact:
    """Plot carryover bias and estimator spread against switchback period length."""
    use_house_style()
    result = rows if rows is not None else period_rows()
    periods = [row.period for row in result]

    figure, axis = plt.subplots()
    axis.plot(
        periods,
        [100 * row.spread for row in result],
        marker="s",
        color=PALETTE[0],
        lw=2,
        label="Spread of the estimate",
    )
    axis.plot(
        periods,
        [100 * row.spread_with_burn_in for row in result],
        marker="^",
        color=PALETTE[2],
        lw=2,
        ls="--",
        label=f"Spread with an {CARRY_MINUTES}-minute burn-in",
    )
    axis.plot(
        periods,
        [100 * abs(row.bias) for row in result],
        marker="o",
        color=PALETTE[1],
        lw=2,
        label="Carryover bias",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    # Only the chosen ticks are labelled, as on the website; log axes add minor labels otherwise.
    axis.minorticks_off()
    axis.set_xticks(periods)
    axis.set_xticklabels([_period_label(period) for period in periods])
    axis.set_yticks(_ERROR_TICKS)
    axis.set_yticklabels([f"{tick:g}" for tick in _ERROR_TICKS])
    axis.set_xlabel("length of each switchback period")
    axis.set_ylabel(f"error in percentage points, true effect {100 * LIFT:.1f}")
    axis.set_title("Long periods buy a small bias with a large variance")
    axis.legend(loc="center left")

    return save_figure(figure, slug="switchback_period_length", output_dir=output_dir)
