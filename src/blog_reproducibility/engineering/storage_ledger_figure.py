"""Figure renderer for the storage ledger draft."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.engineering.storage_ledger import DEMAND, GENERATION, example_payload


def render_storage_figure(*, output_dir: Path) -> FigureArtifact:
    """Show the day, the paths through it, and what each configuration still imports."""
    use_house_style()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    axes[0].bar(range(1, len(GENERATION) + 1), GENERATION, label="Solar")
    axes[0].axhline(DEMAND[0], color=PALETTE[1], ls="--", label="Demand")
    axes[0].set(
        title=f"Equal total energy: {sum(GENERATION):.0f} kWh",
        xlabel="Hour",
        ylabel="Power (kW)",
        xticks=range(1, len(GENERATION) + 1),
    )
    axes[0].legend()

    outcomes = example_payload()
    for index, outcome in enumerate(outcomes):
        label = f"{outcome.capacity_kwh:.0f} kWh / {outcome.power_kw:.0f} kW"
        axes[1].plot(
            range(len(outcome.rows) + 1),
            [0.0, *[row.after for row in outcome.rows]],
            marker="o",
            label=label,
        )
        axes[2].bar(index, outcome.imports_kwh, color=PALETTE[index])
        axes[2].text(index, outcome.imports_kwh + 0.07, f"{outcome.imports_kwh:.2f}", ha="center")

    axes[1].set(
        title="State depends on the whole path",
        xlabel="End of hour",
        ylabel="Stored energy (kWh)",
    )
    axes[1].legend(fontsize=8)
    axes[2].set(
        title="Imports still needed",
        ylabel="Imported energy (kWh)",
        ylim=(0, 5.5),
        xticks=range(len(outcomes)),
        xticklabels=[
            f"{outcome.capacity_kwh:.0f} kWh\n{outcome.power_kw:.0f} kW" for outcome in outcomes
        ],
    )

    return save_figure(figure, slug="environment_storage_constraints", output_dir=output_dir)
