"""Shared plotting utilities for reproducible blog figures."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

STYLE_PATH: Final[Path] = Path(__file__).with_name("house.mplstyle")
DEFAULT_FIGURE_OUTPUT_DIR: Final[Path] = Path("build") / "figures"

SURFACE: Final[str] = "#f7f9fa"
INK_SECONDARY: Final[str] = "#52514e"

PALETTE: Final[tuple[str, ...]] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
SEQUENTIAL: Final[tuple[str, ...]] = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#184f95",
    "#0d366b",
)
DIVERGING: Final[tuple[str, ...]] = (
    "#0d366b",
    "#2a78d6",
    "#9ec5f4",
    "#f0efec",
    "#f0a3a3",
    "#e34948",
    "#a82725",
    "#6b1614",
)


@dataclass(frozen=True, slots=True)
class FigureArtifact:
    """Metadata for a generated figure."""

    path: Path
    width: int
    height: int


def use_house_style() -> None:
    """Activate the shared blog figure style."""
    plt.style.use(str(STYLE_PATH))
    plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)


def sequential_cmap(name: str = "house_seq") -> LinearSegmentedColormap:
    """Return the shared sequential colour map."""
    return LinearSegmentedColormap.from_list(name, SEQUENTIAL)


def diverging_cmap(name: str = "house_div") -> LinearSegmentedColormap:
    """Return the shared diverging colour map."""
    return LinearSegmentedColormap.from_list(name, DIVERGING)


def save_figure(
    figure: Figure,
    *,
    slug: str,
    output_dir: Path = DEFAULT_FIGURE_OUTPUT_DIR,
) -> FigureArtifact:
    """Save a figure as PNG and return nominal pixel dimensions."""
    if not slug or Path(slug).name != slug or slug in {".", ".."}:
        raise ValueError("slug must be a non-empty file-name-safe identifier")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slug}.png"
    figure.savefig(output_path)

    size = figure.get_size_inches()
    width = int(float(size[0]) * float(figure.dpi))
    height = int(float(size[1]) * float(figure.dpi))
    plt.close(figure)

    return FigureArtifact(path=output_path, width=width, height=height)
