"""Check penalised partitioning against brute force and the article's tables.

Small series are segmented by trying every set of cut points, which shares
nothing with the dynamic programme except the objective. The seeded series then
reproduce both of the article's tables exactly.
"""

from itertools import combinations
from math import log

import numpy as np
import pytest

from blog_reproducibility.time_series.optimal_partition import (
    TRUE_CHANGE_POINTS,
    example_payload,
    make_series,
    optimal_partition,
    robust_sigma,
    score,
)

SUMMARY = example_payload()


def _brute_force(series: np.ndarray, penalty: float) -> tuple[int, ...]:
    """Minimise squares plus penalty over every subset of cut points."""
    best: tuple[float, tuple[int, ...]] = (np.inf, ())
    for size in range(series.size):
        for cuts in combinations(range(1, series.size), size):
            bounds = (0, *cuts, series.size)
            cost = sum(
                float(((series[a:b] - series[a:b].mean()) ** 2).sum())
                for a, b in zip(bounds[:-1], bounds[1:], strict=True)
            )
            best = min(best, (cost + penalty * size, cuts))
    return best[1]


def test_dynamic_programme_matches_brute_force() -> None:
    """Every subset of cuts on short random series gives the same optimum."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        series = rng.normal(size=9) + np.repeat(rng.normal(0, 2, 3), 3)
        for penalty in (0.1, 1.0, 4.0):
            assert optimal_partition(series, penalty) == _brute_force(series, penalty)


def test_penalty_extremes() -> None:
    """No penalty cuts everywhere a cut helps; a huge penalty never cuts."""
    series = np.array([0.0, 0.0, 5.0, 5.0, -3.0, -3.0])

    assert optimal_partition(series, 0.0) == (2, 4)
    assert optimal_partition(series, 1e9) == ()
    assert optimal_partition([1.0], 1.0) == ()


def test_robust_sigma_ignores_shifts() -> None:
    """The first-difference MAD recovers unit noise despite four shifts in mean."""
    assert SUMMARY.robust_sigma == pytest.approx(0.96, abs=0.005)
    assert SUMMARY.sample_sd == pytest.approx(1.28, abs=0.005)
    assert robust_sigma(np.arange(10.0)) == 0.0


def test_independent_table_matches_the_article() -> None:
    """55 and 10 detections below the Schwarz scale, the true four at 2 and above."""
    rows = {row.penalty_multiple: row for row in SUMMARY.independent}

    assert len(rows[0.5].change_points) == 55
    assert len(rows[1.0].change_points) == 10
    for multiple in (2.0, 4.0, 8.0):
        assert rows[multiple].change_points == (116, 200, 350, 421)
        assert (rows[multiple].precision, rows[multiple].recall) == (1.0, 1.0)
    assert rows[0.5].precision == pytest.approx(0.13, abs=0.005)
    assert rows[1.0].precision == pytest.approx(0.40)


def test_autocorrelated_table_matches_the_article() -> None:
    """AR(1) noise at 0.6 turns the default penalty into 28 detections."""
    counts = [len(row.change_points) for row in SUMMARY.autocorrelated]
    precision = [row.precision for row in SUMMARY.autocorrelated]
    recall = [row.recall for row in SUMMARY.autocorrelated]

    assert counts == [28, 5, 5, 4]
    # The article prints precision to two decimals: 0.11, 0.80, 0.80, 0.75.
    assert precision == pytest.approx([3 / 28, 0.8, 0.8, 0.75])
    assert recall == [0.75, 1.0, 1.0, 0.75]
    assert SUMMARY.autocorrelated_robust_sigma < SUMMARY.robust_sigma


def test_count_falls_as_the_penalty_rises() -> None:
    """A dearer cut never buys more of them."""
    series, _ = make_series(np.random.default_rng(0))
    scale = log(series.size) * robust_sigma(series) ** 2
    counts = [len(optimal_partition(series, m * scale)) for m in (0.25, 0.5, 1, 2, 8, 64)]

    assert counts == sorted(counts, reverse=True)


def test_score_counts_matches_within_tolerance() -> None:
    """Detections within five samples count; nothing found has perfect precision."""
    assert score((116, 200, 350, 421), TRUE_CHANGE_POINTS) == (1.0, 1.0)
    assert score((100, 200), TRUE_CHANGE_POINTS) == (0.5, 0.25)
    assert score((), TRUE_CHANGE_POINTS) == (1.0, 0.0)


def test_series_are_deterministic_and_autocorrelated() -> None:
    """Seeds reproduce the series; the AR(1) version has lag-one correlation near 0.6."""
    first, _ = make_series(np.random.default_rng(1), phi=0.6)
    second, _ = make_series(np.random.default_rng(1), phi=0.6)
    np.testing.assert_array_equal(first, second)

    mean = np.zeros(600)
    for start, end, level in ((120, 200, 1.5), (350, 420, -1.2), (420, 600, 1.0)):
        mean[start:end] = level
    noise = first - mean
    assert np.corrcoef(noise[:-1], noise[1:])[0, 1] == pytest.approx(0.6, abs=0.1)


def test_invalid_inputs_are_rejected() -> None:
    """Empty or non-finite series, negative penalties, and bad settings are refused."""
    with pytest.raises(ValueError):
        optimal_partition([], 1.0)
    with pytest.raises(ValueError):
        optimal_partition([1.0, float("nan")], 1.0)
    with pytest.raises(ValueError):
        optimal_partition([1.0, 2.0], -1.0)
    with pytest.raises(ValueError):
        robust_sigma([1.0])
    with pytest.raises(ValueError):
        make_series(np.random.default_rng(0), phi=1.0)
    with pytest.raises(ValueError):
        score((1,), ())
