"""Check the novelty simulation against its closed-form expectations, the article and the figure.

The figure's twelve runs (one generator seeded at 0) are reproduced draw for
draw and a sample of points is pinned as the website's generator computes
them. Both curves are then checked against closed forms: the tenure estimate
against the true effect, and the calendar estimate against the mixture of
tenures a calendar day contains, each within four standard errors.

The article's tables use 20 runs per column, drawn in a different order, so
their simulated columns are not pinned; they are checked against the same
closed forms at their own noise level. Their true-effect columns are closed
forms and are pinned exactly.
"""

from math import sqrt

import numpy as np
import pytest

from blog_reproducibility.statistics.novelty_effects import (
    DAYS,
    LONG_RUN_EFFECT,
    NEW_PER_DAY,
    RETURN_PROBABILITY,
    RUNS,
    calendar_expectation,
    example_payload,
    novelty_effect,
    simulate_run,
    simulate_runs,
)

SUMMARY = example_payload()
ROWS = {row.day: row for row in SUMMARY.rows}

# The website generator's values in percent, by tenure and by calendar day.
WEBSITE_FIGURE = {
    1: (10.02, 10.08),
    4: (5.25, 7.80),
    8: (2.49, 5.74),
    14: (1.48, 4.34),
    21: (1.15, 3.31),
    28: (0.52, 2.65),
}


def _users(day: int, *, calendar: bool) -> float:
    """Expected active users behind one point of the calendar or tenure curve."""
    k = day - 1
    if calendar:
        return NEW_PER_DAY * (1 + RETURN_PROBABILITY * k)
    return NEW_PER_DAY * (DAYS - k) * (1.0 if k == 0 else RETURN_PROBABILITY)


def _standard_error(users: float, runs: int) -> float:
    """Standard error of a lift between two halves of ``users`` with outcome CV one half."""
    return 1 / sqrt(users * runs)


def test_the_figure_values_are_reproduced() -> None:
    """A sample of the figure's points, as the website's generator computes them."""
    assert tuple(ROWS) == tuple(range(1, DAYS + 1))
    assert SUMMARY.runs == RUNS
    for day, (tenure, calendar) in WEBSITE_FIGURE.items():
        assert round(100 * ROWS[day].by_tenure, 2) == tenure
        assert round(100 * ROWS[day].by_calendar, 2) == calendar


def test_the_true_effect_columns_match_the_article() -> None:
    """The closed-form truth at the days both of the article's tables print."""
    first_table = {1: 10.0, 4: 5.3, 8: 2.6, 14: 1.3, 21: 1.1, 28: 1.0}
    cohort_table = {1: 10.0, 3: 6.5, 5: 4.3, 7: 3.0, 9: 2.2, 11: 1.7, 13: 1.4, 15: 1.3}

    for day, printed in (first_table | cohort_table).items():
        assert round(100 * ROWS[day].true_effect, 1) == printed


def test_the_effect_settles_within_half_a_point_by_two_weeks() -> None:
    """The article: within half a point of the long-run value by tenure day 14 (from day 13)."""
    gap = novelty_effect(np.arange(DAYS)) - LONG_RUN_EFFECT

    assert np.all(gap[12:] < 0.005)
    assert gap[11] > 0.005
    assert novelty_effect(0) == pytest.approx(0.10)
    assert novelty_effect(1e6) == pytest.approx(LONG_RUN_EFFECT)


def test_the_tenure_estimate_recovers_the_truth() -> None:
    """Every tenure point lies within four standard errors of the true effect."""
    for row in SUMMARY.rows:
        error = _standard_error(_users(row.day, calendar=False), RUNS)
        assert abs(row.by_tenure - row.true_effect) < 4 * error


def test_the_calendar_estimate_tracks_the_tenure_mixture() -> None:
    """Every calendar point lies within four standard errors of its closed-form mixture."""
    for row in SUMMARY.rows:
        error = _standard_error(_users(row.day, calendar=True), RUNS)
        assert abs(row.by_calendar - row.calendar_expectation) < 4 * error


def test_the_calendar_curve_is_flatter_and_later() -> None:
    """The alt text: calendar days overstate the effect and take far longer to fall."""
    truth = np.array([row.true_effect for row in SUMMARY.rows])
    calendar = np.array([row.by_calendar for row in SUMMARY.rows])
    halfway = LONG_RUN_EFFECT + (truth[0] - LONG_RUN_EFFECT) / 2

    assert np.all(calendar[1:] > truth[1:] + 0.005)
    assert calendar[0] - calendar[-1] < truth[0] - truth[-1]
    first_below = (int(np.argmax(truth < halfway)), int(np.argmax(calendar < halfway)))
    assert first_below[1] > 2 * first_below[0]


def test_two_weeks_in_the_calendar_reports_three_times_the_current_effect() -> None:
    """The article: at two weeks the calendar grouping reports 4.3 percent against 1.3."""
    row = ROWS[14]

    assert round(100 * row.true_effect, 1) == 1.3
    assert round(100 * row.calendar_expectation, 1) == 4.2
    assert round(100 * row.by_calendar, 1) == 4.3


def test_the_article_estimates_agree_with_the_closed_forms() -> None:
    """The article's own simulated columns (20 runs each) at their noise level."""
    tenure = {1: 9.9, 4: 5.1, 8: 2.7, 14: 1.6, 21: 1.0, 28: 1.1}
    calendar = {1: 9.9, 4: 7.9, 8: 5.9, 14: 4.3, 21: 3.3, 28: 2.7}
    for day, printed in tenure.items():
        error = _standard_error(_users(day, calendar=False), 20)
        assert abs(printed / 100 - ROWS[day].true_effect) < 3 * error + 0.0005
    for day, printed in calendar.items():
        error = _standard_error(_users(day, calendar=True), 20)
        assert abs(printed / 100 - ROWS[day].calendar_expectation) < 3 * error + 0.0005


def test_calendar_expectation_closed_forms() -> None:
    """Day one is pure tenure zero; everyone returning gives a plain running mean."""
    mixture = calendar_expectation()
    effects = novelty_effect(np.arange(DAYS))

    assert mixture[0] == effects[0]
    assert mixture[1] == pytest.approx((effects[0] + 0.5 * effects[1]) / 1.5)
    np.testing.assert_allclose(
        calendar_expectation(return_probability=1.0),
        np.cumsum(effects) / np.arange(1, DAYS + 1),
    )
    # Over a long horizon the excess decays like one over the day: a geometric sum.
    q = np.exp(-1 / 4)
    long_run = 0.01 + 0.09 * (1 + 0.5 * q / (1 - q)) / (1 + 0.5 * 1999)
    assert calendar_expectation(2000)[-1] == pytest.approx(long_run, rel=1e-12)


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the curves; another seed does not."""
    first = simulate_runs(2, seed=3, days=5, new_per_day=200)
    again = simulate_runs(2, seed=3, days=5, new_per_day=200)
    other = simulate_runs(2, seed=4, days=5, new_per_day=200)

    for mine, same, different in zip(first, again, other, strict=True):
        np.testing.assert_array_equal(mine, same)
        assert not np.array_equal(mine, different)


def test_one_day_experiment_groups_identically() -> None:
    """With a single day, the calendar and tenure groupings hold the same users."""
    by_day, by_tenure = simulate_run(np.random.default_rng(1), days=1, new_per_day=500)

    np.testing.assert_array_equal(by_day, by_tenure)


def test_invalid_inputs_are_rejected() -> None:
    """Negative tenures, empty horizons, tiny cohorts and impossible rates are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        novelty_effect(-1)
    with pytest.raises(ValueError):
        novelty_effect(np.nan)
    with pytest.raises(ValueError):
        calendar_expectation(0)
    with pytest.raises(ValueError):
        calendar_expectation(return_probability=1.5)
    with pytest.raises(ValueError):
        simulate_run(rng, new_per_day=1)
    with pytest.raises(TypeError):
        simulate_run(rng, days=True)
    with pytest.raises(ValueError):
        simulate_runs(0)
