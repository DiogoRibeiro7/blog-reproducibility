"""The claims of the data-lake article have to hold in the recorded benchmark.

The benchmark writes about a gigabyte of files and is too slow for a test suite,
so its results are kept in `data/engineering/data_lake_benchmarks.json`. The
sizes and counts are the same on any machine and are checked exactly; the
timings are not, so every ratio is checked across all the runs the harness kept
rather than against the one the article happened to print.
"""

from typing import Any

import pytest

from blog_reproducibility.engineering.data_lake_benchmarks import (
    FORMAT_ORDER,
    example_payload,
    layout_shapes,
    load_data_lake_results,
    ratio_across_runs,
)

SUMMARY = example_payload()
RESULTS = load_data_lake_results()


def test_each_step_of_the_format_shrinks_the_file() -> None:
    """Columns first, then compression, and half the saving arrives before it."""
    sizes = SUMMARY.size_in_mb

    assert all(
        sizes[first] > sizes[second]
        for first, second in zip(FORMAT_ORDER, FORMAT_ORDER[1:], strict=False)
    )
    assert 0.45 < SUMMARY.share_before_compression < 0.55


def test_a_narrow_query_touches_a_quarter_of_the_file() -> None:
    """One column of eight is a small share of the bytes, which is why it is fast."""
    assert RESULTS["share of the file in amount"] < 0.25


def test_sorting_makes_the_statistics_useful() -> None:
    """In arrival order every row group may hold the day; sorted, only one can."""
    groups = SUMMARY.row_groups_that_may_hold_one_day

    assert groups["arrival order"]["may hold it"] == groups["arrival order"]["of"]
    assert groups["sorted by day"]["may hold it"] == 1
    assert groups["sorted by day"]["of"] == groups["arrival order"]["of"]


def test_sorting_moves_bytes_between_columns_and_the_file_grows() -> None:
    """Sorting shrinks the sort keys, swells the identifier, and leaves the rest."""
    arrival = SUMMARY.column_sizes["arrival order"]
    by_day = SUMMARY.column_sizes["sorted by day"]

    assert by_day["event_day"] < 0.01 * arrival["event_day"]
    assert by_day["country"] < 0.05 * arrival["country"]
    assert by_day["event_id"] > 3 * arrival["event_id"]

    for untouched in ("user_id", "device", "product_id", "quantity", "amount"):
        assert abs(by_day[untouched] / arrival[untouched] - 1) < 0.01, untouched

    assert SUMMARY.size_in_mb["Parquet, zstd, sorted by day"] > SUMMARY.size_in_mb["Parquet, zstd"]


def test_files_multiply_and_small_files_cost_bytes() -> None:
    """A finer partition means more folders, many more files, and a larger total."""
    layouts = {shape.name: shape for shape in layout_shapes()}

    assert [shape.folders for shape in layout_shapes()] == [1, 12, 365, 3650]
    assert layouts["by day"].files > 3 * layouts["by day"].folders
    assert layouts["by day and country"].files > 3 * layouts["by day and country"].folders
    assert layouts["by day and country"].megabytes > 1.5 * layouts["one file"].megabytes
    # And it takes much longer to write.
    assert (
        layouts["by day and country"].seconds_to_write > 10 * layouts["one file"].seconds_to_write
    )


def test_no_run_was_kept_while_a_batch_job_was_running() -> None:
    """The harness refuses a timing taken on a busy machine, and this record proves it."""
    runs = RESULTS["timing runs"]
    assert len(runs) >= 3

    for run in runs:
        for moment in ("before", "after"):
            load = run["machine"][f"cores used by other processes, {moment}"]
            assert load["the busiest of them"] <= 1.5
            assert (
                load["all other processes"] + run["machine"]["threads"] + 4 <= SUMMARY.machine.cpus
            )


def test_the_reported_run_is_one_of_the_runs() -> None:
    """The timings the article shows are one of the kept runs, not an average."""
    assert RESULTS["seconds"] in [run["seconds"] for run in RESULTS["timing runs"]]


def test_parquet_beats_csv_most_on_a_narrow_query() -> None:
    """The article says 29 to 41 times on one column, four to six on all eight."""
    narrow = SUMMARY.parquet_over_csv_narrow
    wide = SUMMARY.parquet_over_csv_wide

    assert narrow.lowest > 25 and narrow.highest < 45
    assert wide.lowest > 3.5 and wide.highest < 6.5
    assert narrow.lowest > wide.highest


def test_sorted_file_answers_the_one_day_question_faster() -> None:
    """The article says between five and seven times, in every run."""
    gain = SUMMARY.sorting_gain_for_one_day

    assert gain.lowest >= 5 and gain.highest <= 7
    assert gain.runs >= 3


def test_a_coarse_partition_helps_the_query_that_uses_it() -> None:
    """Monthly folders speed up a one-day query two to three times, and cost elsewhere."""
    assert SUMMARY.monthly_gain_for_one_day.lowest > 1.8
    assert SUMMARY.monthly_gain_for_one_day.highest < 3

    for query in ("whole table", "one country"):

        def monthly_cost(seconds: dict[str, Any], query: str = query) -> float:
            layouts = seconds["layouts"]
            return float(layouts["by month"][query] / layouts["one file"][query])

        cost = ratio_across_runs(RESULTS, monthly_cost)
        assert cost.lowest > 1.15 and cost.highest < 1.75, query


def test_fine_partitions_are_slower_for_every_question() -> None:
    """Even the query the partition was built for gets slower once there are too many files."""
    one_day = ratio_across_runs(
        RESULTS,
        lambda seconds: (
            seconds["layouts"]["by day"]["one day"] / seconds["layouts"]["one file"]["one day"]
        ),
    )
    assert one_day.lowest > 4.5 and one_day.highest < 7.6

    assert SUMMARY.daily_penalty_for_whole_table.lowest > 15
    assert SUMMARY.daily_penalty_for_whole_table.highest < 27
    assert SUMMARY.finest_penalty_for_whole_table.lowest > 145
    assert SUMMARY.finest_penalty_for_whole_table.highest < 345

    country = ratio_across_runs(
        RESULTS,
        lambda seconds: (
            seconds["layouts"]["by day and country"]["one country"]
            / seconds["layouts"]["one file"]["one country"]
        ),
    )
    assert country.lowest > 45 and country.highest < 105


def test_finding_the_files_is_most_of_the_cost() -> None:
    """Listing a fine partition takes most of the time the query takes."""
    assert SUMMARY.listing_share_by_day.lowest > 0.7

    finest = ratio_across_runs(
        RESULTS,
        lambda seconds: (
            seconds["layouts"]["by day and country"]["listing the files"]
            / seconds["layouts"]["by day and country"]["one day"]
        ),
    )
    assert finest.lowest > 0.7

    # Pointed straight at the folder, the same read is 50 to 90 times cheaper.
    direct = ratio_across_runs(
        RESULTS,
        lambda seconds: (
            seconds["layouts"]["by day"]["one day"] / seconds["one day, given its folder"]
        ),
    )
    assert direct.lowest > 50 and direct.highest < 90


def test_published_sizes_and_counts() -> None:
    """The exact figures the article prints, which do not depend on the machine."""
    sizes = SUMMARY.size_in_mb
    layouts = {shape.name: shape for shape in layout_shapes()}

    assert SUMMARY.rows == 6_000_000
    assert sizes["CSV"] == 216.7
    assert sizes["Parquet, uncompressed"] == 138.9
    assert sizes["Parquet, snappy"] == 87.8
    assert sizes["Parquet, zstd"] == 57.7
    assert sizes["Parquet, zstd, sorted by day"] == 61.9

    assert layouts["by day"].files == 2190
    assert layouts["by day and country"].files == 25_557
    assert layouts["by day and country"].rows_per_file == 235
    assert layouts["by day and country"].megabytes == 97.0
    assert (
        round(100 * (layouts["by day and country"].megabytes / layouts["one file"].megabytes - 1))
        == 68
    )
    assert round(100 * RESULTS["share of the file in amount"]) == 23


def test_a_record_without_runs_is_refused() -> None:
    """A ratio has nothing to report if the harness kept no runs."""
    with pytest.raises(ValueError):
        ratio_across_runs({"timing runs": []}, lambda seconds: 1.0)
    with pytest.raises(ValueError):
        ratio_across_runs({}, lambda seconds: 1.0)
