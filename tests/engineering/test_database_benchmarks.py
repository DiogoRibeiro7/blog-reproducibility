"""The claims of the database article have to hold in the recorded benchmark.

The benchmark is too slow for a test suite, so its results are kept in
`data/engineering/database_benchmarks.json`. Rerunning it on another machine
changes the timings; these checks say which statements in the article depend on
them, and bound the ratios rather than the seconds.

The website version of this file compared its numbers against the rendered
article text. That article is in the other repository, so the values it quotes
are written out here instead.
"""

import pytest

from blog_reproducibility.engineering.database_benchmarks import (
    ANALYTICAL_QUERIES,
    LOOKUP_QUERY,
    example_payload,
    index_steps,
    load_database_results,
    speedups,
)

SUMMARY = example_payload()


def test_the_record_describes_the_machine_it_came_from() -> None:
    """Absolute timings mean nothing without it, so it has to be there."""
    machine = SUMMARY.machine

    assert machine.python
    assert machine.platform
    assert machine.cpus >= 1
    assert machine.duckdb is not None
    assert machine.sqlite is not None
    assert SUMMARY.rows == {"orders": 5_000_000, "customers": 200_000, "products": 5_000}


def test_column_store_wins_analytical_queries_on_one_thread() -> None:
    """Every analytical query is at least five times faster on the column store."""
    for row in SUMMARY.analytical_speedups:
        assert row.query in ANALYTICAL_QUERIES
        assert row.duckdb_over_sqlite > 5, row.query
        assert row.column_store_faster


def test_row_store_wins_point_lookups() -> None:
    """Two thousand lookups by key reverse the order, by more than five times."""
    lookup = SUMMARY.lookup_speedup

    assert lookup.query == LOOKUP_QUERY
    assert not lookup.column_store_faster
    assert lookup.sqlite_seconds * 5 < lookup.duckdb_seconds


def test_covering_index_beats_plain_index_beats_scan() -> None:
    """Each step of the access path is faster, and the last one is a covering index."""
    scan, plain, covering = index_steps()

    assert scan.seconds > plain.seconds > covering.seconds
    assert scan.index == "none"
    assert "SCAN orders" in scan.plan[0]
    assert "USING INDEX" in plain.plan[0]
    assert "COVERING INDEX" in covering.plan[0]


def test_broad_filter_is_slower_through_the_index_than_as_a_scan() -> None:
    """An index that helps a narrow filter hurts a broad one, by fifty times."""
    broad = load_database_results()["index"]["a third of the table"]

    assert "USING INDEX" in broad["plan chosen"][0]
    assert SUMMARY.broad_filter_penalty > 5
    assert SUMMARY.broad_filter_penalty == pytest.approx(51.6, abs=0.5)


def test_lookup_cost_predicts_the_broad_index_scan() -> None:
    """The broad filter costs about one lookup for each row it touches.

    This is the article's explanation, and it is checked rather than asserted:
    the measured per-lookup cost times the number of rows matched lands within a
    factor of two of the measured broad-filter time.
    """
    results = load_database_results()
    per_lookup = results["seconds"][LOOKUP_QUERY]["sqlite"] / 2000
    broad = results["index"]["a third of the table"]

    predicted = per_lookup * results["rows"]["orders"] * broad["share of rows"]
    assert 0.5 < predicted / broad["seconds with the chosen plan"] < 2


def test_commit_per_row_and_indexes_slow_writes() -> None:
    """Committing every row costs more than a thousandfold; indexes cost fortyfold."""
    assert SUMMARY.commit_per_row_penalty > 50
    assert SUMMARY.commit_per_row_penalty == 1462
    assert SUMMARY.index_write_penalty > 5
    assert SUMMARY.index_write_penalty == pytest.approx(45.7, abs=0.5)


def test_headline_ratios_quoted_in_the_article() -> None:
    """The article prints 32x, 97x and 42x; those come from this record."""
    by_query = {row.query: row for row in speedups()}

    assert round(by_query["sum of one column"].duckdb_over_sqlite) == 32
    assert round(by_query["group by channel"].duckdb_over_sqlite) == 97
    assert round(by_query["star join"].duckdb_over_sqlite) == 42


def test_the_column_file_is_the_smallest_on_disk() -> None:
    """The same data is smallest as Parquet and largest as an indexed row store."""
    sizes = SUMMARY.size_in_mb

    assert sizes["parquet, zstd"] < sizes["duckdb"] < sizes["csv"] < sizes["sqlite"]
    assert sizes["sqlite with two indexes"] > sizes["sqlite"]
    # Two secondary indexes cost more than half the table again.
    assert sizes["sqlite with two indexes"] / sizes["sqlite"] > 1.5


def test_every_recorded_query_has_both_engines() -> None:
    """A speedup needs both sides, so the record must carry both for each query."""
    rows = speedups()

    assert len(rows) == len(ANALYTICAL_QUERIES) + 1
    for row in rows:
        assert row.sqlite_seconds > 0
        assert row.duckdb_seconds > 0
