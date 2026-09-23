"""Check the retraining cost model against its closed form and reproduce the article's tables.

The search optimum is checked against the square-root rule and the continuous
lower bound over a grid of parameters, and a hand-worked case. The article's
schedule table, sensitivity grid and schedule column of the trigger comparison
all come from the same formula, so they are pinned at the printed precision.
The trigger simulations use their own seeded generators and are not modelled.
"""

from math import ceil, floor, sqrt

import pytest

from blog_reproducibility.machine_learning.retraining import (
    COST,
    DECAY,
    VALUE,
    best_period,
    example_payload,
    near_optimal_periods,
    schedule_cost,
    square_root_period,
)

SUMMARY = example_payload()


def test_cost_by_hand() -> None:
    """Every 5 weeks: 100,000 x 0.004 x 4 / 2 = 800 foregone, 15,000 / 5 = 3,000 retraining."""
    row = schedule_cost(5)

    assert row.quality_lost == pytest.approx(800.0)
    assert row.retraining == pytest.approx(3000.0)
    assert row.total == pytest.approx(3800.0)


def test_search_agrees_with_the_square_root_rule() -> None:
    """The best whole week is the floor or ceiling of T*, and never beats the continuous bound."""
    for decay in (0.0005, 0.001, 0.002, 0.004, 0.01, 0.03):
        for cost in (1_000.0, 5_000.0, 15_000.0, 50_000.0):
            best = best_period(decay=decay, cost=cost, longest=1000)
            rule = square_root_period(decay=decay, cost=cost)
            bound = sqrt(2 * cost * decay * VALUE) - decay * VALUE / 2

            assert best.period in {max(floor(rule), 1), ceil(rule)}
            assert best.total >= bound - 1e-9


def test_cost_is_convex_in_the_period() -> None:
    """Second differences are positive, so the near-optimal periods form one run."""
    totals = [schedule_cost(p).total for p in range(1, 200)]
    second = [a - 2 * b + c for a, b, c in zip(totals, totals[1:], totals[2:], strict=False)]

    assert all(value > 0 for value in second)


def test_schedule_table_matches_the_article() -> None:
    """Quality lost, retraining and total per week at 2, 4, 8, 13, 26 and 52 weeks."""
    table = [
        (row.period, round(row.quality_lost), round(row.retraining), round(row.total))
        for row in SUMMARY.schedule_table
    ]
    assert table == [
        (2, 200, 7500, 7700),
        (4, 600, 3750, 4350),
        (8, 1400, 1875, 3275),
        (13, 2400, 1154, 3554),
        (26, 5000, 577, 5577),
        (52, 10200, 288, 10488),
    ]


def test_optimum_matches_the_article() -> None:
    """Nine weeks at 3,267 a week, against the rule's 8.7 and 15,000 for weekly."""
    assert SUMMARY.best.period == 9
    assert round(SUMMARY.best.total) == 3267
    assert round(SUMMARY.square_root_period, 1) == 8.7
    assert SUMMARY.weekly.total == pytest.approx(COST)


def test_fortnightly_and_annual_cost_more_than_double() -> None:
    """Both ends of the article's table cost more than twice the optimum."""
    fortnightly, annual = SUMMARY.schedule_table[0], SUMMARY.schedule_table[-1]

    assert fortnightly.total > 2 * SUMMARY.best.total
    assert annual.total > 2 * SUMMARY.best.total


def test_curve_is_flat_between_six_and_thirteen_weeks() -> None:
    """The alt text: within ten percent of the cheapest from six to thirteen weeks only."""
    assert SUMMARY.near_optimal == (6, 13)
    assert near_optimal_periods(0.0) == (9, 9)


def test_too_often_and_too_rarely_cost_about_the_same() -> None:
    """The title: the figure's two ends, 2 and 40 weeks, are within ten percent of each other."""
    periods = [row.period for row in SUMMARY.curve]
    assert (periods[0], periods[-1]) == (1, 40)
    two, forty = SUMMARY.curve[1].total, SUMMARY.curve[-1].total

    assert abs(forty / two - 1) < 0.1
    assert min(SUMMARY.curve, key=lambda row: row.total) == SUMMARY.best


def test_sensitivity_grid_matches_the_article() -> None:
    """Best period and weekly cost for three decay rates and three retraining costs."""
    grid = [
        (row.decay, round(row.cost), row.best_period, round(row.weekly_cost))
        for row in SUMMARY.sensitivity
    ]
    assert grid == [
        (0.001, 5000, 10, 950),
        (0.001, 15000, 17, 1682),
        (0.001, 50000, 32, 3112),
        (0.004, 5000, 5, 1800),
        (0.004, 15000, 9, 3267),
        (0.004, 50000, 16, 6125),
        (0.010, 5000, 3, 2667),
        (0.010, 15000, 5, 5000),
        (0.010, 50000, 10, 9500),
    ]


def test_decay_comparison_schedules_match_the_article() -> None:
    """The schedule column of the trigger comparison: 12, 9 and 6 weeks."""
    rows = [
        (row.decay, row.best_period, round(row.weekly_cost)) for row in SUMMARY.decay_comparison
    ]
    assert rows == [(0.002, 12, 2350), (0.004, 9, 3267), (0.008, 6, 4500)]


def test_quadrupling_decay_halves_the_interval() -> None:
    """Both inputs enter under a square root: a factor of four moves T* by two."""
    assert square_root_period(decay=DECAY / 4) == pytest.approx(2 * square_root_period())
    assert square_root_period(cost=4 * COST) == pytest.approx(2 * square_root_period())


def test_invalid_inputs_are_rejected() -> None:
    """Non-positive periods, negative costs and zero decay in the rule are refused."""
    with pytest.raises(ValueError):
        schedule_cost(0)
    with pytest.raises(TypeError):
        schedule_cost(2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        schedule_cost(3, cost=-1.0)
    with pytest.raises(ValueError):
        square_root_period(decay=0.0)
    with pytest.raises(ValueError):
        best_period(longest=0)
