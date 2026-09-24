"""Check the switchback model against hand-worked cases, its closed form and the figure.

The figure's simulation (one generator seeded at 11 for each period length) is
reproduced draw for draw, so its values are pinned as the website's generator
prints them. The article's own simulated tables use other seeds (2026, 404, 0
and 707 plus the replication) and replication counts, so they are not pinned;
its closed-form columns are, exactly. The carryover window is checked against a
direct transcription of the article's loop, the noise-free estimate against the
exact attenuation of an alternating design with a flat season, and the
simulated bias against the closed form.

The closed-form column prints -0.124 percent at 60-minute periods. The value is
-0.12345 percent, which the article's code prints as -0.1235 and which rounds
to -0.123 at the table's three decimals; the other four entries agree.

The alt text says six-hour periods multiply the spread "by four". The article's
carryover table (seed 404, 600 replications) gives 1.513 to 6.053, a factor of
4.0, but the figure's own 120 replications give 1.328 to 6.257, a factor of 4.7.
The test checks the factor lies between four and five.
"""

import numpy as np
import pytest

from blog_reproducibility.statistics.switchback import (
    ARTICLE_PERIODS,
    CARRY_FRACTION,
    CARRY_MINUTES,
    LIFT,
    PERIODS,
    closed_form_bias,
    contaminated_share,
    daily_season,
    example_payload,
    minute_effect,
    noise_free_estimate,
    period_estimate,
    period_row,
)

SUMMARY = example_payload()
ROWS = {row.period: row for row in SUMMARY.rows}

# The website generator's values, in percentage points: bias, spread, spread with burn-in.
WEBSITE_FIGURE = {
    15: (-0.4927, 1.3284, 1.4023),
    30: (-0.2482, 1.9591, 1.9859),
    60: (-0.1225, 2.6618, 2.6836),
    120: (-0.0638, 3.6414, 3.6686),
    180: (-0.0414, 5.1746, 5.1788),
    360: (-0.0213, 6.2565, 6.2953),
}


def _article_carryover(z: np.ndarray, lift: float, carry: int, fraction: float) -> np.ndarray:
    """Run the article's slice-by-slice carryover loop."""
    effect = np.where(z, lift, 0.0)
    for j in np.flatnonzero((~z) & np.roll(z, 1)):
        effect[j : j + carry] = np.maximum(effect[j : j + carry], fraction * lift)
    return effect


def test_the_figure_values_are_reproduced() -> None:
    """Every point on the figure, as the website's generator computes it."""
    assert tuple(ROWS) == PERIODS
    for period, (bias, spread, burned) in WEBSITE_FIGURE.items():
        row = ROWS[period]
        assert round(100 * row.bias, 4) == bias
        assert round(100 * row.spread, 4) == spread
        assert round(100 * row.spread_with_burn_in, 4) == burned


def test_the_closed_form_columns_match_the_article() -> None:
    """Contaminated control time and the closed-form bias, as the article prints them."""
    contaminated = {15: 26.7, 30: 13.3, 60: 6.7, 180: 2.2, 360: 1.1}
    # The article prints -0.124 at 60 minutes: its code's four-decimal -0.1235, rounded again.
    bias = {15: -0.492, 30: -0.247, 60: -0.123, 180: -0.041, 360: -0.021}

    assert tuple(row.period for row in SUMMARY.closed_form) == ARTICLE_PERIODS
    for row in SUMMARY.closed_form:
        assert round(100 * row.contaminated_share, 1) == contaminated[row.period]
        assert round(100 * row.closed_form_bias, 3) == bias[row.period]
    sixty = next(row for row in SUMMARY.closed_form if row.period == 60)
    assert round(100 * sixty.closed_form_bias, 4) == -0.1235


def test_the_burn_in_keeps_the_article_share_of_orders() -> None:
    """An eight-minute burn-in keeps 47, 73, 87 and 96 percent of each period."""
    kept = {row.period: round(100 * row.share_kept_with_burn_in) for row in SUMMARY.closed_form}

    assert kept == {15: 47, 30: 73, 60: 87, 180: 96, 360: 98}


def test_closed_form_limiting_cases() -> None:
    """No effect or no persistence means no bias; halving a long period doubles the share."""
    assert closed_form_bias(15, lift=0.0) == 0.0
    assert closed_form_bias(15, carry_fraction=0.0) == pytest.approx(0.0, abs=1e-15)
    assert closed_form_bias(15, carry_minutes=0) == pytest.approx(0.0, abs=1e-15)
    assert contaminated_share(4) == contaminated_share(8) == 0.5
    assert contaminated_share(30) == pytest.approx(2 * contaminated_share(60))
    # To first order the bias is minus the share times gamma delta, scaled by 1 + delta.
    share = contaminated_share(360)
    first_order = -share * CARRY_FRACTION * LIFT * (1 + LIFT)
    assert closed_form_bias(360) == pytest.approx(first_order, rel=1e-3)


def test_carryover_on_a_hand_worked_assignment() -> None:
    """Two switch-offs, a wrap-around, and a window that runs into a treated minute."""
    t, f = True, False
    effect = minute_effect(
        np.array([t, t, f, f, f, t, f, f]), lift=0.1, carry_minutes=2, carry_fraction=0.5
    )
    assert effect.tolist() == pytest.approx([0.1, 0.1, 0.05, 0.05, 0.0, 0.1, 0.05, 0.05])

    wrapped = minute_effect(np.array([f, f, f, t]), lift=0.1, carry_minutes=2, carry_fraction=0.5)
    assert wrapped.tolist() == pytest.approx([0.05, 0.05, 0.0, 0.1])

    overlap = minute_effect(
        np.array([t, f, t, f, f, f]), lift=0.1, carry_minutes=3, carry_fraction=0.5
    )
    assert overlap.tolist() == pytest.approx([0.1, 0.05, 0.1, 0.05, 0.05, 0.05])

    plain = minute_effect(np.array([t, f, f]), carryover=False, lift=0.1)
    assert plain.tolist() == [0.1, 0.0, 0.0]


def test_carryover_matches_the_article_loop() -> None:
    """The vectorised windows equal the article's loop, including periods shorter than c."""
    rng = np.random.default_rng(7)
    for period in (1, 3, 8, 15, 60):
        z = np.repeat(rng.random(2880 // period) < 0.5, period)
        expected = _article_carryover(z, LIFT, CARRY_MINUTES, CARRY_FRACTION)
        np.testing.assert_array_equal(minute_effect(z), expected)


def test_noise_free_estimate_on_an_alternating_design() -> None:
    """With a flat season and strict alternation, c / L of control time is contaminated."""
    period = 20
    z = np.tile(np.repeat([True, False], period), 6)
    flat = np.ones(z.size)

    exact = (1 + LIFT) / (1 + CARRY_MINUTES / period * CARRY_FRACTION * LIFT) - 1
    assert noise_free_estimate(z, season=flat) == pytest.approx(exact, rel=1e-12)
    assert noise_free_estimate(z, carryover=False, season=flat) == pytest.approx(LIFT, rel=1e-12)


def test_period_estimate_on_a_hand_worked_trace() -> None:
    """Two periods of two minutes; the burn-in drops the first minute of each."""
    z = [True, True, False, False]
    orders = [1, 2, 1, 3]
    totals = [10.0, 30.0, 5.0, 30.0]

    assert period_estimate(z, orders, totals, period=2) == pytest.approx((40 / 3) / (35 / 4) - 1)
    assert period_estimate(z, orders, totals, period=2, burn_in=1) == pytest.approx(0.5)


def test_simulated_bias_matches_the_closed_form() -> None:
    """The paired bias is within three Monte Carlo errors of the closed form at every period."""
    for row in SUMMARY.rows:
        assert row.bias < 0
        assert abs(row.bias - closed_form_bias(row.period)) < 3 * row.bias_monte_carlo_error


def test_long_periods_trade_bias_for_spread() -> None:
    """The title: bias shrinks and spread grows as the period lengthens."""
    biases = [abs(row.bias) for row in SUMMARY.rows]
    spreads = [row.spread for row in SUMMARY.rows]

    assert biases == sorted(biases, reverse=True)
    assert spreads == sorted(spreads)


def test_the_alt_text_ratios() -> None:
    """From 15 minutes to 6 hours the bias falls more than twentyfold; the spread grows 4-5x."""
    short, long = ROWS[15], ROWS[360]

    assert short.bias / long.bias > 20
    assert 4 < long.spread / short.spread < 5


def test_the_spread_dominates_and_the_burn_in_costs_little() -> None:
    """The spread exceeds the bias everywhere; a burn-in raises it by less than 6 percent."""
    for row in SUMMARY.rows:
        assert row.spread > 2 * abs(row.bias)
        assert 1 <= row.spread_with_burn_in / row.spread < 1.06


def test_the_daily_season_averages_to_one() -> None:
    """Over whole days the season has mean one, so the average rate is three orders a minute."""
    season = daily_season()

    assert float(season.mean()) == pytest.approx(1.0, abs=1e-12)
    assert season.min() == pytest.approx(0.65, abs=1e-6)
    assert season.max() == pytest.approx(1.35, abs=1e-6)


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a row; another seed does not."""

    def small(seed: int) -> object:
        return period_row(60, seed=seed, bias_replications=3, spread_replications=4)

    assert small(5) == small(5)
    assert small(5) != small(6)


def test_invalid_inputs_are_rejected() -> None:
    """Periods that do not tile the two weeks, oversized burn-ins and bad values are refused."""
    z = [True, True, False, False]
    with pytest.raises(ValueError):
        period_row(11)
    with pytest.raises(TypeError):
        period_row(True)
    with pytest.raises(ValueError):
        period_row(60, spread_replications=1)
    with pytest.raises(ValueError):
        period_estimate(z, [1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0], period=2, burn_in=2)
    with pytest.raises(ValueError):
        period_estimate(z, [1, 1, 1], [1.0, 1.0, 1.0], period=2)
    with pytest.raises(ValueError):
        period_estimate([True] * 4, [1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0], period=2)
    with pytest.raises(ValueError):
        minute_effect(np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        minute_effect(np.array([True, False]), lift=-0.1)
    with pytest.raises(ValueError):
        minute_effect(np.array([True, False]), carry_fraction=1.5)
    with pytest.raises(ValueError):
        noise_free_estimate(np.array([True, False]), season=np.ones(3))
    with pytest.raises(ValueError):
        noise_free_estimate(np.array([True, True]))
