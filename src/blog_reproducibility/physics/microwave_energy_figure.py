"""Figure renderer for the microwave article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.physics.microwave_energy import absorbed_energy_joules, example_payload

POWERS = (500.0, 1000.0)


def render_photons_and_power_figure(*, output_dir: Path) -> FigureArtifact:
    """Put one photon's energy beside a minute's worth of them."""
    use_house_style()
    summary = example_payload()
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.1), layout="constrained")

    energies = [summary.microwave_photon_ev, summary.green_photon_ev]
    axes[0].scatter(energies, [0, 1], color=list(PALETTE[:2]), s=75)
    axes[0].set(
        xscale="log",
        xlim=(1e-6, 1e2),
        ylim=(-0.6, 1.6),
        xlabel="Energy per photon (eV, logarithmic scale)",
        title="Energy of one photon",
    )
    axes[0].set_yticks([0, 1], ["Microwave\n2.45 GHz", "Green light\n550 nm"])
    labels = (f"{energies[0]:.7f} eV".rstrip("0"), f"{energies[1]:.2f} eV")
    for value, position, label in zip(energies, [0, 1], labels, strict=True):
        axes[0].annotate(
            label,
            (value, position),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    kilojoules = [absorbed_energy_joules(power, 60.0) / 1000.0 for power in POWERS]
    axes[1].bar([0, 1], kilojoules, width=0.55, color=PALETTE[0])
    axes[1].set_xticks([0, 1], [f"{int(power):,} W" for power in POWERS])
    axes[1].set(
        xlabel="Assumed absorbed microwave power",
        ylabel="Total absorbed energy (kilojoules)",
        ylim=(0, 70),
        title="Energy absorbed in 60 seconds",
    )
    for position, value in enumerate(kilojoules):
        axes[1].text(position, value + 1.5, f"{value:.0f} kJ", ha="center")

    return save_figure(figure, slug="microwave_photons_and_power_2026", output_dir=output_dir)
