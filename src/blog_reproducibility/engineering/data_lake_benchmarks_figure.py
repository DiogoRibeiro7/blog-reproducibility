"""Figure renderers for the data-lake article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.engineering.data_lake_benchmarks import load_data_lake_results


def label_seconds(value: float) -> str:
    """Format a duration in whichever unit keeps it readable."""
    if value >= 1:
        return f"{value:.1f} s"
    return f"{value * 1000:.0f} ms" if value >= 0.01 else f"{value * 1000:.1f} ms"


def _count(number: int, noun: str) -> str:
    """Pluralise a count for the layout axis labels."""
    return f"{number:,} {noun}" + ("" if number == 1 else "s")


def render_formats_figure(*, output_dir: Path) -> FigureArtifact:
    """Show the file shrinking, and where the bytes move when it is sorted."""
    use_house_style()
    results = load_data_lake_results()
    sizes = results["size in MB"]
    names = list(sizes)
    values = [sizes[name] for name in names]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [5, 6]}
    )

    # One quantity, one hue: the same table under each encoding.
    left.barh(names[::-1], values[::-1], color=PALETTE[0], height=0.62)
    for row, size in enumerate(values[::-1]):
        left.text(
            size + max(values) * 0.015,
            row,
            f"{size:.0f} MB",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
    left.set_xlim(0, max(values) * 1.2)
    left.set_xlabel("size on disk, MB")
    left.set_title(f"{results['rows'] / 1e6:.0f} million events, eight columns")
    left.grid(axis="y", visible=False)

    columns = results["column sizes in MB"]
    order = sorted(columns["arrival order"], key=columns["arrival order"].get, reverse=True)
    base = list(reversed(range(len(order))))
    height = 0.38
    series = (
        ("arrival order", "rows in arrival order", PALETTE[0]),
        ("sorted by day", "rows sorted by day, then country", PALETTE[1]),
    )
    for index, (key, label, colour) in enumerate(series):
        column_values = [columns[key][column] for column in order]
        positions = [position + (0.5 - index) * height for position in base]
        right.barh(positions, column_values, height=height * 0.92, color=colour, label=label)
        for position, value in zip(positions, column_values, strict=True):
            right.text(
                value + 0.35,
                position,
                f"{value:.1f}" if value >= 0.1 else f"{value:.2f}",
                va="center",
                fontsize=8.5,
                color=INK_SECONDARY,
            )

    right.set_yticks(base)
    right.set_yticklabels(order)
    right.set_xlim(0, max(max(columns[key].values()) for key in columns) * 1.18)
    right.set_xlabel("MB of each column inside the zstd file")
    right.set_title("Sorting moves bytes between columns")
    right.legend(loc="lower right")
    right.grid(axis="y", visible=False)
    figure.set_layout_engine("tight")

    return save_figure(figure, slug="data_lake_formats_and_columns", output_dir=output_dir)


def render_layouts_figure(*, output_dir: Path) -> FigureArtifact:
    """Show four partition layouts answering the same three questions."""
    use_house_style()
    results = load_data_lake_results()
    layouts = results["layouts"]
    seconds = results["seconds"]["layouts"]
    names = list(layouts)

    series = (
        ("whole table", "sum over the whole table", PALETTE[0]),
        ("one day", "one day of 365", PALETTE[1]),
        ("one country", "one country of ten", PALETTE[2]),
        ("listing the files", "listing the files, reading nothing", INK_MUTED),
    )

    figure, axis = plt.subplots(figsize=(9.6, 4.6))
    width = 0.2
    base = list(range(len(names)))

    for index, (key, label, colour) in enumerate(series):
        values = [seconds[name][key] for name in names]
        positions = [position + (index - 1.5) * width for position in base]
        axis.bar(positions, values, width=width * 0.9, color=colour, label=label)
        for position, value in zip(positions, values, strict=True):
            axis.text(
                position,
                value * 1.18,
                label_seconds(value),
                ha="center",
                fontsize=7.5,
                color=INK_SECONDARY,
            )

    axis.set_yscale("log")
    axis.set_ylim(top=max(max(row.values()) for row in seconds.values()) * 4)
    axis.set_xticks(base)
    axis.set_xticklabels(
        [
            f"{name}\n{_count(layouts[name]['files'], 'file')} in "
            f"{_count(layouts[name]['folders'], 'folder')}\n{layouts[name]['MB']:.0f} MB"
            for name in names
        ],
        fontsize=9,
    )
    axis.set_ylabel("seconds, logarithmic scale")
    axis.set_title("The same six million rows in four layouts: the files cost more than they save")
    axis.legend(loc="upper left", ncols=2)
    axis.grid(axis="x", visible=False)
    figure.set_layout_engine("tight")

    return save_figure(figure, slug="data_lake_partition_layouts", output_dir=output_dir)


def render_data_lake_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the article."""
    return (
        render_formats_figure(output_dir=output_dir),
        render_layouts_figure(output_dir=output_dir),
    )
