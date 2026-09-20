"""Row store against column store, for the article on databases for analysis.

The benchmark loads five million synthetic orders into SQLite (rows, B-trees,
one thread) and DuckDB (columns, vectorised) and asks both the same five
questions. Four of them are analytical and the column store wins by one to two
orders of magnitude; the fifth is two thousand lookups by primary key and the
order reverses.

A second part measures what an index does: a filter matching one day of 730 gets
faster with an index and faster again with one that also carries the summed
column, while a filter matching a third of the table gets fifty times *slower*
through the index the planner chose than as a forced scan.

This module reads the recorded result and derives the ratios the article quotes.
The harness that produces the record is `scripts/benchmarks/database_benchmarks.py`.
"""

from dataclasses import dataclass
from typing import Any, Final

from blog_reproducibility.common.validation import positive
from blog_reproducibility.engineering.benchmark_results import (
    MachineDescription,
    load_results,
    machine_description,
)

__all__ = [
    "ANALYTICAL_QUERIES",
    "LOOKUP_QUERY",
    "DatabaseSummary",
    "IndexStep",
    "QuerySpeedup",
    "example_payload",
    "index_steps",
    "load_database_results",
    "speedups",
]

RECORD_NAME: Final[str] = "database_benchmarks"

# The four questions a column store is built for.
ANALYTICAL_QUERIES: Final[tuple[str, ...]] = (
    "sum of one column",
    "one day of 730",
    "group by channel",
    "star join",
)
# The one a row store is built for.
LOOKUP_QUERY: Final[str] = "2,000 lookups by key"


@dataclass(frozen=True, slots=True)
class QuerySpeedup:
    """How much faster one engine answered one question than the other."""

    query: str
    sqlite_seconds: float
    duckdb_seconds: float
    duckdb_over_sqlite: float

    @property
    def column_store_faster(self) -> bool:
        """True when the column store answered it more quickly."""
        return self.duckdb_over_sqlite > 1.0


@dataclass(frozen=True, slots=True)
class IndexStep:
    """One access path for the same one-day filter."""

    index: str
    plan: tuple[str, ...]
    seconds: float


@dataclass(frozen=True, slots=True)
class DatabaseSummary:
    """Every number the article reports, derived from the recorded run."""

    machine: MachineDescription
    rows: dict[str, int]
    analytical_speedups: tuple[QuerySpeedup, ...]
    lookup_speedup: QuerySpeedup
    index_steps: tuple[IndexStep, ...]
    broad_filter_penalty: float
    commit_per_row_penalty: float
    index_write_penalty: float
    size_in_mb: dict[str, float]


def load_database_results() -> dict[str, Any]:
    """Load the recorded database benchmark."""
    return load_results(RECORD_NAME)


def _duckdb_one_thread(timings: dict[str, float]) -> float:
    """Return the single-threaded DuckDB timing, which is the fair comparison.

    SQLite runs on one thread, so the article compares against DuckDB on one
    thread. The all-threads number is recorded too, but a speedup that mixes
    thread counts would measure the machine rather than the storage layout.
    """
    return positive(timings["duckdb, 1 thread"], name="duckdb, 1 thread")


def speedups(results: dict[str, Any] | None = None) -> tuple[QuerySpeedup, ...]:
    """Return the SQLite-over-DuckDB ratio for every recorded query."""
    record = results if results is not None else load_database_results()
    seconds = record["seconds"]

    rows: list[QuerySpeedup] = []
    for query, timings in seconds.items():
        sqlite = positive(timings["sqlite"], name="sqlite")
        duckdb = _duckdb_one_thread(timings)
        rows.append(
            QuerySpeedup(
                query=query,
                sqlite_seconds=sqlite,
                duckdb_seconds=duckdb,
                duckdb_over_sqlite=sqlite / duckdb,
            )
        )
    return tuple(rows)


def index_steps(results: dict[str, Any] | None = None) -> tuple[IndexStep, ...]:
    """Return the three access paths for the one-day filter, in order."""
    record = results if results is not None else load_database_results()
    return tuple(
        IndexStep(
            index=str(step["index"]),
            plan=tuple(str(line) for line in step["plan"]),
            seconds=positive(step["seconds"], name="seconds"),
        )
        for step in record["index"]["one day"]
    )


def example_payload() -> DatabaseSummary:
    """Return the numbers the article reports."""
    record = load_database_results()
    by_query = {row.query: row for row in speedups(record)}
    broad = record["index"]["a third of the table"]
    small_insert = record["insert 2,000 rows"]
    large_insert = record["insert 200,000 rows"]

    return DatabaseSummary(
        machine=machine_description(record),
        rows={name: int(value) for name, value in record["rows"].items()},
        analytical_speedups=tuple(by_query[query] for query in ANALYTICAL_QUERIES),
        lookup_speedup=by_query[LOOKUP_QUERY],
        index_steps=index_steps(record),
        broad_filter_penalty=(
            positive(broad["seconds with the chosen plan"], name="chosen plan")
            / positive(broad["seconds with a forced scan"], name="forced scan")
        ),
        commit_per_row_penalty=float(small_insert["ratio"]),
        index_write_penalty=(
            positive(large_insert["with two secondary indexes"], name="with indexes")
            / positive(large_insert["with none"], name="with none")
        ),
        size_in_mb={name: float(value) for name, value in record["size in MB"].items()},
    )
