"""File formats and partition layouts, for the data-lake article.

Six million synthetic events are written as CSV and as Parquet in several
layouts, and the same three questions are asked of each. The article's claims
divide cleanly in two, and the model keeps them apart:

* **Facts** — sizes on disk, column sizes inside the file, how many row groups
  could hold one day, and how many files and folders each layout produces. These
  are the same on any machine and are checked as exact values.
* **Timings** — how long each question took. These depend on the machine and on
  what else it was doing, so the harness keeps several runs, refuses any taken
  while the machine was busy, and records all of them. The article quotes
  ratios, and the model reports each ratio across every kept run so a claim can
  be checked against its whole range rather than against one number.

The harness that produces the record is `scripts/benchmarks/data_lake_benchmarks.py`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from blog_reproducibility.common.validation import positive
from blog_reproducibility.engineering.benchmark_results import (
    MachineDescription,
    load_results,
    machine_description,
)

__all__ = [
    "FORMAT_ORDER",
    "DataLakeSummary",
    "LayoutShape",
    "RatioRange",
    "example_payload",
    "layout_shapes",
    "load_data_lake_results",
    "ratio_across_runs",
]

RECORD_NAME: Final[str] = "data_lake_benchmarks"

# Each step should shrink the file: columns first, then compression.
FORMAT_ORDER: Final[tuple[str, ...]] = (
    "CSV",
    "Parquet, uncompressed",
    "Parquet, snappy",
    "Parquet, zstd",
)


@dataclass(frozen=True, slots=True)
class RatioRange:
    """One timing ratio, across every run the harness kept."""

    values: tuple[float, ...]

    @property
    def lowest(self) -> float:
        """The smallest value observed."""
        return min(self.values)

    @property
    def highest(self) -> float:
        """The largest value observed."""
        return max(self.values)

    @property
    def runs(self) -> int:
        """How many runs contributed."""
        return len(self.values)


@dataclass(frozen=True, slots=True)
class LayoutShape:
    """What one partition layout costs in files, folders, and bytes."""

    name: str
    files: int
    folders: int
    megabytes: float
    rows_per_file: int
    seconds_to_write: float


@dataclass(frozen=True, slots=True)
class DataLakeSummary:
    """Every number the article reports, derived from the recorded runs."""

    machine: MachineDescription
    rows: int
    size_in_mb: dict[str, float]
    share_before_compression: float
    column_sizes: dict[str, dict[str, float]]
    row_groups_that_may_hold_one_day: dict[str, dict[str, int]]
    layouts: tuple[LayoutShape, ...]
    parquet_over_csv_narrow: RatioRange
    parquet_over_csv_wide: RatioRange
    sorting_gain_for_one_day: RatioRange
    monthly_gain_for_one_day: RatioRange
    daily_penalty_for_whole_table: RatioRange
    finest_penalty_for_whole_table: RatioRange
    listing_share_by_day: RatioRange


def load_data_lake_results() -> dict[str, Any]:
    """Load the recorded data-lake benchmark."""
    return load_results(RECORD_NAME)


def ratio_across_runs(
    results: dict[str, Any],
    of: Callable[[dict[str, Any]], float],
) -> RatioRange:
    """Evaluate one timing ratio in every run the harness kept.

    A claim about a ratio should hold in all of them, not just in whichever run
    the article happened to report.
    """
    runs = results.get("timing runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("The benchmark record has no kept timing runs")

    return RatioRange(values=tuple(float(of(run["seconds"])) for run in runs))


def layout_shapes(results: dict[str, Any] | None = None) -> tuple[LayoutShape, ...]:
    """Return what each partition layout costs, in the order the article uses."""
    record = results if results is not None else load_data_lake_results()
    return tuple(
        LayoutShape(
            name=name,
            files=int(shape["files"]),
            folders=int(shape["folders"]),
            megabytes=float(shape["MB"]),
            rows_per_file=int(shape["rows per file"]),
            seconds_to_write=float(shape["seconds to write"]),
        )
        for name, shape in record["layouts"].items()
    )


def _layout_ratio(layout: str, other: str, query: str) -> Callable[[dict[str, Any]], float]:
    """Build a ratio of the same question asked of two layouts."""

    def ratio(seconds: dict[str, Any]) -> float:
        numerator = positive(seconds["layouts"][layout][query], name=layout)
        denominator = positive(seconds["layouts"][other][query], name=other)
        return numerator / denominator

    return ratio


def example_payload() -> DataLakeSummary:
    """Return the numbers the article reports."""
    record = load_data_lake_results()
    sizes = {name: float(value) for name, value in record["size in MB"].items()}
    wide_query = "a query that needs all eight columns"

    return DataLakeSummary(
        machine=machine_description(record),
        rows=int(record["rows"]),
        size_in_mb=sizes,
        # Half the saving arrives before any compression, from the column layout alone.
        share_before_compression=(
            (sizes["CSV"] - sizes["Parquet, uncompressed"])
            / (sizes["CSV"] - sizes["Parquet, zstd"])
        ),
        column_sizes={
            order: {column: float(value) for column, value in columns.items()}
            for order, columns in record["column sizes in MB"].items()
        },
        row_groups_that_may_hold_one_day={
            order: {key: int(value) for key, value in counts.items()}
            for order, counts in record["row groups that may hold one day"].items()
        },
        layouts=layout_shapes(record),
        parquet_over_csv_narrow=ratio_across_runs(
            record,
            lambda seconds: (
                seconds["sum of one column"]["CSV"] / seconds["sum of one column"]["Parquet"]
            ),
        ),
        parquet_over_csv_wide=ratio_across_runs(
            record,
            lambda seconds: seconds[wide_query]["CSV"] / seconds[wide_query]["Parquet"],
        ),
        sorting_gain_for_one_day=ratio_across_runs(
            record,
            lambda seconds: (
                seconds["one day of 365"]["arrival order"]
                / seconds["one day of 365"]["sorted by day"]
            ),
        ),
        monthly_gain_for_one_day=ratio_across_runs(
            record, _layout_ratio("one file", "by month", "one day")
        ),
        daily_penalty_for_whole_table=ratio_across_runs(
            record, _layout_ratio("by day", "one file", "whole table")
        ),
        finest_penalty_for_whole_table=ratio_across_runs(
            record, _layout_ratio("by day and country", "one file", "whole table")
        ),
        listing_share_by_day=ratio_across_runs(
            record,
            lambda seconds: (
                seconds["layouts"]["by day"]["listing the files"]
                / seconds["layouts"]["by day"]["one day"]
            ),
        ),
    )
