# Benchmark records

Two articles rest on benchmarks that cannot run in a test suite. This directory
holds what those benchmarks produced, so the claims can be checked without
rerunning them.

| File | Article | Harness |
| --- | --- | --- |
| `database_benchmarks.json` | [A Database for Analysis](https://diogoribeiro7.github.io/data-science/a_database_for_analysis_rows_columns_indexes/) | `scripts/benchmarks/database_benchmarks.py` |
| `data_lake_benchmarks.json` | [A Data Lake Is a Directory With Rules](https://diogoribeiro7.github.io/data-science/a_data_lake_is_a_directory_with_rules/) | `scripts/benchmarks/data_lake_benchmarks.py` |

## What is reproducible, and what is not

**Deterministic.** Both harnesses generate their data from a fixed seed
(`20260919`), so the row counts, the file sizes, the column sizes inside a
Parquet file, the number of row groups that could hold one day, and the number
of files and folders each partition layout produces are the same on any machine
with the same library versions. The tests check these as exact values.

**Not deterministic.** Every timing. They depend on the processor, the disk, the
page cache, the library versions, and on what else the machine was doing.
Rerunning either harness will produce different seconds.

**Stable anyway.** The ratios between timings — a column store against a row
store, Parquet against CSV, one layout against another. These are what both
articles quote and what the tests bound. They are bounded as ranges, not
asserted as values.

## Provenance

Both records were produced on the same machine, and each carries its own
description under `machine`:

| | |
| --- | --- |
| Platform | Windows 11 (10.0.26200), 22 logical processors |
| Python | 3.13.5 |
| DuckDB | 1.5.4 |
| SQLite | 3.49.1 (database benchmark only) |
| Generated | September 2026 |
| Input data | none — both harnesses generate synthetic data from seed 20260919 |
| External services | none; both run entirely on the local machine |

## How the data-lake record handles a busy machine

Timings taken while something else is running are not comparable. The data-lake
harness measures how many cores other processes are using immediately before and
after each run, refuses the run if another process is taking more than 1.5 cores
by itself or if too few cores are left free, and keeps every run that passed.

The record therefore carries a `timing runs` array rather than one set of
numbers, and `seconds` is one of those runs. The model reports each ratio across
all of them, and the tests require a claim to hold in every kept run. This one
holds four.

## Rerunning

The harnesses need DuckDB, pandas, NumPy, PyArrow and psutil, which are not
installed by default:

```console
poetry install --with benchmarks
poetry run python scripts/benchmarks/database_benchmarks.py     # about five minutes
poetry run python scripts/benchmarks/data_lake_benchmarks.py --phase facts
poetry run python scripts/benchmarks/data_lake_benchmarks.py --phase timings
poetry run python scripts/benchmarks/data_lake_benchmarks.py --clean
```

The data-lake harness writes about a gigabyte of temporary files and peaks near
2 GB of memory. Its two phases are separate because the facts do not depend on
how busy the machine is and the timings do.

Rerunning replaces the record in place. The exact-value tests will then fail if
a library version changed what is written to disk, which is the point: that is a
real change to the article's claims and should be looked at rather than absorbed.

## Limitations

- The record describes one machine. Another will give different seconds, and the
  article says so.
- The synthetic data is generated to be well behaved. Real event data with
  skewed keys, late arrivals, or mixed schemas would partition and compress
  differently.
- The query planner's choices are SQLite's and DuckDB's at the recorded
  versions. A later version may choose differently, and the index results in
  particular depend on that choice.
