"""Figure renderer for the article on staggered difference in differences."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.staggered_did import StaggeredSummary, example_payload


def render_staggered_did_figure(
    *, output_dir: Path, summary: StaggeredSummary | None = None
) -> FigureArtifact:
    """Plot the event study against the true dynamic effect, with the regression's single number."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    exposures = [row.exposure for row in result.rows]

    figure, axis = plt.subplots()
    axis.plot(
        exposures,
        [row.true_effect for row in result.rows],
        color=PALETTE[0],
        lw=2.2,
        label="True effect by exposure",
    )
    axis.plot(
        exposures,
        [row.estimate for row in result.rows],
        marker="o",
        color=PALETTE[2],
        ls="--",
        label="Group-time estimates (not-yet-treated controls)",
    )
    axis.axhline(
        result.true_att,
        color=PALETTE[0],
        lw=1,
        ls=":",
        label=f"True average effect on the treated ({result.true_att:.2f})",
    )
    axis.axhline(
        result.twfe,
        color=PALETTE[1],
        lw=2,
        ls="--",
        label=f"Two-way fixed effects coefficient ({result.twfe:.2f})",
    )
    axis.set_xlabel("periods since adoption")
    axis.set_ylabel("treatment effect")
    axis.set_xticks(exposures)
    axis.set_title("A single coefficient cannot summarise a growing effect")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="staggered_did_event_study", output_dir=output_dir)
