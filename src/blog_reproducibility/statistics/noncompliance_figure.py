"""Figure renderer for the article on non-compliance in experiments."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.noncompliance import (
    ESTIMATORS,
    NoncomplianceSummary,
    example_payload,
)

COLOURS = (PALETTE[0], PALETTE[1], PALETTE[4], PALETTE[2])


def render_noncompliance_figure(
    *, output_dir: Path, summary: NoncomplianceSummary | None = None
) -> FigureArtifact:
    """Plot each estimator's mean against the compliance rate, with the true effect of use."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    rates = [row.compliance for row in result.rows]
    series = (
        [row.simulated.intention_to_treat for row in result.rows],
        [row.simulated.as_treated for row in result.rows],
        [row.simulated.per_protocol for row in result.rows],
        [row.simulated.wald for row in result.rows],
    )

    figure, axis = plt.subplots()
    axis.axhline(
        result.effect,
        color=PALETTE[3],
        lw=1,
        ls=":",
        label=f"True effect of using the feature ({result.effect:.1f})",
    )
    for name, values, colour in zip(ESTIMATORS, series, COLOURS, strict=True):
        axis.plot(rates, values, marker="o", color=colour, label=name)
    axis.set_xlabel("share of users who use the feature when assigned to it")
    axis.set_ylabel("estimated effect")
    axis.set_title("Comparing users by what they did, not what they were assigned, invents effects")
    axis.legend(loc="upper right")

    return save_figure(figure, slug="noncompliance_estimators", output_dir=output_dir)
