"""Figure renderers for the article on databases for analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from blog_reproducibility.common.plotting import (
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.engineering.database_benchmarks import load_database_results


def label_seconds(value: float) -> str:
    """Format a duration in whichever unit keeps it readable."""
    if value >= 1:
        return f"{value:.1f} s"
    if value >= 0.001:
        return f"{value * 1000:.0f} ms" if value >= 0.01 else f"{value * 1000:.1f} ms"
    return f"{value * 1e6:.0f} µs"


def render_engines_figure(*, output_dir: Path) -> FigureArtifact:
    """Put the same five questions to both engines on one logarithmic axis."""
    use_house_style()
    results = load_database_results()
    seconds = results["seconds"]
    cpus = results["machine"]["cpus"]
    queries = list(seconds)

    series = (
        ("sqlite", "SQLite: rows, one thread", PALETTE[0]),
        ("duckdb, 1 thread", "DuckDB: columns, one thread", PALETTE[1]),
        (f"duckdb, {cpus} threads", f"DuckDB: columns, {cpus} threads", PALETTE[2]),
    )

    figure, axis = plt.subplots(figsize=(9.6, 5.2))
    height = 0.26
    base = list(reversed(range(len(queries))))

    for index, (key, label, colour) in enumerate(series):
        values = [seconds[query][key] for query in queries]
        positions = [position + (1 - index) * height for position in base]
        axis.barh(positions, values, height=height * 0.9, color=colour, label=label)
        for position, value in zip(positions, values, strict=True):
            axis.text(
                value * 1.12,
                position,
                label_seconds(value),
                va="center",
                fontsize=8.5,
                color=INK_SECONDARY,
            )

    axis.set_xscale("log")
    axis.set_xlim(2e-4, 60)
    axis.set_yticks(base)
    axis.set_yticklabels(queries)
    axis.set_xlabel("seconds, logarithmic scale")
    axis.set_title("The same five questions asked of a row store and a column store")
    # The top rows end well short of one second, so the corner is free.
    axis.legend(loc="upper right")
    axis.grid(axis="y", visible=False)

    return save_figure(figure, slug="database_rows_versus_columns", output_dir=output_dir)


def render_index_figure(*, output_dir: Path) -> FigureArtifact:
    """Show an index helping a narrow filter and hurting a broad one."""
    use_house_style()
    results = load_database_results()
    index = results["index"]

    figure, (narrow, broad_axis) = plt.subplots(
        1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [3, 2]}
    )

    steps = index["one day"]
    values = [step["seconds"] for step in steps]
    # Colour follows the access path in both panels: a scan, or a walk through an index.
    narrow.bar(
        ["no index:\nscan", "index on\nthe day", "covering\nindex"],
        values,
        color=[PALETTE[0], PALETTE[1], PALETTE[1]],
        width=0.6,
    )
    for position, value in enumerate(values):
        narrow.text(
            position,
            value * 1.25,
            label_seconds(value),
            ha="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
    narrow.set_yscale("log")
    narrow.set_ylim(2e-4, 3)
    narrow.set_ylabel("seconds, logarithmic scale")
    narrow.set_title(f"One day of 730: {index['rows matched by one day']:,} rows")
    narrow.grid(axis="x", visible=False)

    broad = index["a third of the table"]
    broad_values = [broad["seconds with the chosen plan"], broad["seconds with a forced scan"]]
    broad_axis.bar(
        ["index, as the\nplanner chose", "scan,\nforced"],
        broad_values,
        color=[PALETTE[1], PALETTE[0]],
        width=0.6,
    )
    for position, value in enumerate(broad_values):
        broad_axis.text(
            position,
            value * 1.25,
            label_seconds(value),
            ha="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
    broad_axis.set_yscale("log")
    broad_axis.set_ylim(0.1, 120)
    broad_axis.set_title("A third of the table")
    broad_axis.legend(
        handles=[
            Patch(color=PALETTE[0], label="full scan"),
            Patch(color=PALETTE[1], label="through an index"),
        ],
        loc="upper right",
    )
    broad_axis.grid(axis="x", visible=False)
    figure.set_layout_engine("tight")

    return save_figure(figure, slug="database_index_helps_and_hurts", output_dir=output_dir)


def render_database_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the article."""
    return (
        render_engines_figure(output_dir=output_dir),
        render_index_figure(output_dir=output_dir),
    )
