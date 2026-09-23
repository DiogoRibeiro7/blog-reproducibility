"""Figure renderer for the article on feature selection before cross-validation."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.machine_learning.selection_leakage import (
    SelectionLeakageSummary,
    example_payload,
)


def render_selection_leakage_figure(
    *, output_dir: Path, summary: SelectionLeakageSummary | None = None
) -> FigureArtifact:
    """Plot cross-validated accuracy on noise against the size of the feature pool."""
    use_house_style()
    result = summary if summary is not None else example_payload()
    pools = [row.candidate_features for row in result.rows]

    figure, axis = plt.subplots()
    axis.plot(
        pools,
        [row.selected_outside for row in result.rows],
        marker="o",
        color=PALETTE[1],
        lw=2,
        label="Ten features selected on all the data, then cross-validated",
    )
    axis.plot(
        pools,
        [row.selected_inside for row in result.rows],
        marker="o",
        color=PALETTE[0],
        lw=2,
        label="Ten features selected inside each training fold",
    )
    axis.axhline(0.5, color=PALETTE[3], lw=1, ls="--", label="Chance")
    axis.set_xscale("log")
    axis.set_xticks(pools)
    axis.set_xticklabels([f"{pool:,}" for pool in pools])
    axis.set_xlabel("candidate features (all noise)")
    axis.set_ylabel("cross-validated accuracy")
    axis.set_ylim(0.4, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_title("Selecting features before cross-validation manufactures accuracy")
    axis.legend(loc="upper left")

    return save_figure(figure, slug="cv_selection_leakage", output_dir=output_dir)
