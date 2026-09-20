"""Figure renderer for the article on seasons and the Earth's tilt."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.physics.seasons import OBLIQUITY, solar_geometry

LABELS = ("December\nsolstice", "Equinox", "June\nsolstice")
DECLINATIONS = (-OBLIQUITY, 0.0, OBLIQUITY)


def render_tilt_geometry_figure(*, output_dir: Path) -> FigureArtifact:
    """Put two mirrored latitudes side by side through one year."""
    use_house_style()
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    hemispheres = (
        ("45 N", [solar_geometry(45, declination) for declination in DECLINATIONS]),
        ("45 S", [solar_geometry(-45, declination) for declination in DECLINATIONS]),
    )
    for index, (name, rows) in enumerate(hemispheres):
        axes[0].plot(
            range(3),
            [row.daylight_hours for row in rows],
            marker="o",
            color=PALETTE[index],
            label=name,
        )
        axes[1].plot(
            range(3),
            [row.overhead_equivalent_hours for row in rows],
            marker="o",
            color=PALETTE[index],
            label=name,
        )

    for axis in axes:
        axis.set_xticks(range(3), LABELS)
        axis.legend()

    axes[0].set(title="Length of daylight", ylabel="Hours", ylim=(0, 24))
    axes[1].set(
        title="Daily incoming solar energy",
        ylabel="Equivalent hours of overhead sunlight",
        ylim=(0, 10),
    )
    figure.suptitle("Opposite seasons at a fixed Earth-Sun distance")

    return save_figure(figure, slug="science_seasons_tilt_geometry", output_dir=output_dir)
