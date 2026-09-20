"""Figure renderer for the adaptive baseline draft."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.health.adaptive_baseline import example_payload

FLAG_THRESHOLD = 2.5
PERSISTENT_LEVEL = 4.0


def render_baseline_figure(*, output_dir: Path) -> FigureArtifact:
    """Show the reference chasing the change, and the score fading with it."""
    use_house_style()
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    for alpha, rows in example_payload().items():
        label = "Frozen" if alpha == "0.0" else f"alpha = {alpha}"
        days = range(1, len(rows) + 1)
        axes[0].plot(days, [row.before for row in rows], label=label)
        axes[1].plot(days, [row.score for row in rows], label=label)

    axes[0].axhline(PERSISTENT_LEVEL, color=INK_MUTED, ls=":", label="Persistent observed level")
    axes[0].set(
        title="The reference follows the change",
        xlabel="Observed day after step",
        ylabel="Baseline before scoring",
    )
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)

    axes[1].axhline(FLAG_THRESHOLD, color=INK_MUTED, ls=":", label="Illustrative flag threshold")
    axes[1].set(
        title="Deviation fades without recovery",
        xlabel="Observed day after step",
        ylabel="Observation minus baseline",
    )
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)

    return save_figure(figure, slug="healthcare_adaptive_baseline", output_dir=output_dir)
