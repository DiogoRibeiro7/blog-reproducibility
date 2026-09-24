"""Figure renderer for the article on censored labels in supervised learning."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    INK_PRIMARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.censored_labels import (
    CensoredLabelsSummary,
    example_payload,
)

# Above matplotlib's default of 2 for lines, so the truth is drawn over the models.
TRUTH_ZORDER = 3


def render_cohorts_figure(
    *, output_dir: Path, summary: CensoredLabelsSummary | None = None
) -> FigureArtifact:
    """Plot mean predicted twelve-month churn by tenure cohort for the three models."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    cohorts = result.cohorts
    positions = np.arange(len(cohorts))

    figure, axis = plt.subplots()
    # The hazard model sits on the truth, so the truth is dashed and drawn over the
    # models' lines: the hazard line shows through the gaps. Plotted first, it keeps
    # its place at the top of the legend.
    axis.plot(
        positions,
        [row.true for row in cohorts],
        color=INK_PRIMARY,
        lw=1.6,
        ls="--",
        zorder=TRUTH_ZORDER,
        label="True 12-month probability",
    )
    for values, colour, label in (
        ([row.naive for row in cohorts], PALETTE[1], "Naive: churned by extract date"),
        ([row.fixed_horizon for row in cohorts], PALETTE[2], "Fixed 12-month horizon"),
        ([row.hazard for row in cohorts], PALETTE[0], "Discrete-time hazard model"),
    ):
        axis.plot(positions, values, marker="o", color=colour, label=label)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{row.lower}-{row.upper}" for row in cohorts])
    axis.set_xlabel("months since signup at the data extract")
    axis.set_ylabel("mean predicted 12-month churn")
    axis.set_ylim(0, 0.85)
    axis.set_title("Censored labels teach the model that new customers never churn")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="censored_labels_cohorts", output_dir=output_dir)
