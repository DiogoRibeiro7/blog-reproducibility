"""Figure renderer for the wearable heart-alert article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.health.wearable_alerts import example_payload


def render_denominators_figure(*, output_dir: Path) -> FigureArtifact:
    """Split the alerts each prevalence produces into real ones and false ones."""
    use_house_style()
    rows = example_payload()
    figure, axis = plt.subplots(figsize=(8.5, 4.2))

    positions = list(range(len(rows)))
    true_alerts = [row.true_positive for row in rows]
    false_alerts = [row.false_positive for row in rows]

    axis.barh(positions, true_alerts, color=PALETTE[0], label="True alerts")
    axis.barh(positions, false_alerts, left=true_alerts, color=PALETTE[1], label="False alerts")
    for position, row in zip(positions, rows, strict=True):
        axis.text(
            row.alerts + 35,
            position,
            f"{row.positive_predictive_value:.1%} true",
            va="center",
        )

    axis.set_yticks(positions, labels=[f"{row.prevalence:.0%}" for row in rows])
    axis.invert_yaxis()
    axis.set(
        xlim=(0, 2800),
        xlabel="Alerts per 10,000 people",
        ylabel="Condition prevalence",
        title="Same hypothetical test; different meaning of a positive alert",
    )
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2)

    return save_figure(figure, slug="wearable_alert_denominators_2026", output_dir=output_dir)
