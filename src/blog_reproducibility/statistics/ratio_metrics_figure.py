"""Figure renderer for the article on ratio metrics and the delta method."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.ratio_metrics import (
    MILD_HETEROGENEITY,
    STRONG_HETEROGENEITY,
    FalsePositiveRow,
    RatioMetricSummary,
    example_payload,
)


def _series(
    rows: tuple[FalsePositiveRow, ...], concentration: float
) -> tuple[FalsePositiveRow, ...]:
    return tuple(row for row in rows if row.concentration == concentration)


def render_ratio_metric_figure(
    *, output_dir: Path, summary: RatioMetricSummary | None = None
) -> FigureArtifact:
    """Plot A/A false positive rates against sessions per user for both tests."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    strong = _series(result.rows, STRONG_HETEROGENEITY)
    mild = _series(result.rows, MILD_HETEROGENEITY)
    sessions = [row.mean_sessions for row in strong]

    figure, axis = plt.subplots()
    axis.plot(
        sessions,
        [row.session_level for row in strong],
        marker="o",
        color=PALETTE[1],
        label="Sessions as independent trials, strongly heterogeneous users",
    )
    axis.plot(
        [row.mean_sessions for row in mild],
        [row.session_level for row in mild],
        marker="o",
        color=PALETTE[3],
        label="Sessions as independent trials, mildly heterogeneous users",
    )
    axis.plot(
        sessions,
        [row.delta_method for row in strong],
        marker="o",
        color=PALETTE[0],
        label="Delta method, users as the unit (strongly heterogeneous)",
    )
    axis.axhline(0.05, color=PALETTE[2], lw=1, ls="--", label="Nominal 5 percent")
    axis.set_xscale("log")
    axis.set_xticks(sessions)
    axis.set_xticklabels([str(m) for m in sessions])
    axis.set_xlabel("mean sessions per user")
    axis.set_ylabel("A/A tests declared significant")
    axis.set_ylim(0, 0.32)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_title("The session-level test finds effects that are not there")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="ratio_metric_false_positives", output_dir=output_dir)
