"""Check the marketplace model against the article's loop, its closed forms and the figure's claims.

The figure's payload (960 twenty-day experiments) takes 15 to 20 seconds: the
city design needs 192,000 arrival-order permutations to keep the generator
aligned with the article, and those draws dominate. So these tests do not
recompute it. They check instead that the port is draw-for-draw identical to a
transcription of the website's loop, pin the figure's first 60 buyer-level
experiments exactly, and test the figure's claims on a shorter run of the same
code with eight replications per design.

The expected true lift has a binomial closed form, checked against sampling.
The article's worked single day is exact arithmetic and is pinned. Its table
comes from other draws (200 replications, other inventories), so its simulated
columns are only compared with the closed form at their own noise level.
"""

from math import sqrt
from typing import cast

import numpy as np
import pytest

from blog_reproducibility.statistics.marketplace_interference import (
    BUYERS,
    CONTROL_INTENT,
    INVENTORIES,
    TREATED_INTENT,
    Design,
    InventoryRow,
    day_sales,
    expected_capped_sales,
    experiment,
    fluid_day,
    inventory_rows,
    rollout_lift,
)

REPLICATIONS = 8
SHORT = inventory_rows((320, 250, 215, 200, 195), seed=0, replications=REPLICATIONS)


def _website_experiment(
    r: np.random.Generator, inventory: int, design: str, days: int
) -> tuple[float, float]:
    """Run the website generator's experiment, transcribed with its draws in order."""
    buyers, pc, pt, n_markets = 2000, 0.10, 0.12, 20

    def sales(intents: np.ndarray) -> np.ndarray:
        order = r.permutation(len(intents))
        served = np.cumsum(intents[order]) <= inventory
        sold = np.zeros(len(intents), bool)
        sold[order] = intents[order] & served
        return sold

    tc = tt = ec = et = 0.0
    for _ in range(days):
        ic = r.random(buyers) < pc
        it = r.random(buyers) < pt
        tc += min(int(ic.sum()), inventory)
        tt += min(int(it.sum()), inventory)
        if design == "buyer":
            z = r.integers(0, 2, buyers)
            s = sales(np.where(z == 1, it, ic))
            et += float(s[z == 1].sum() / (z == 1).mean())
            ec += float(s[z == 0].sum() / (z == 0).mean())
        else:
            zm = np.repeat([0, 1], n_markets // 2)
            r.shuffle(zm)
            for m in range(n_markets):
                count = sales(r.random(buyers) < (pt if zm[m] else pc)).sum()
                if zm[m]:
                    et += float(count / zm.sum())
                else:
                    ec += float(count / (n_markets - zm.sum()))
    return et / ec - 1, tt / tc - 1


def _standard_error(spread: float) -> float:
    return spread / sqrt(REPLICATIONS)


def test_the_port_draws_exactly_as_the_website_loop() -> None:
    """Both designs, in sequence from one generator, give identical results and final state."""
    ours, theirs = np.random.default_rng(0), np.random.default_rng(0)
    for inventory, design, days in ((205, "buyer", 6), (205, "buyer", 6), (195, "market", 4)):
        assert experiment(inventory, cast(Design, design), ours, days=days) == (
            _website_experiment(theirs, inventory, design, days)
        )
    assert ours.random() == theirs.random()


def test_the_figure_first_buyer_row_is_reproduced() -> None:
    """The figure's first 60 experiments: buyer-level 20.60 and true 19.76 percent at 320 units."""
    rng = np.random.default_rng(0)
    runs = np.array([experiment(320, "buyer", rng) for _ in range(60)])

    assert float(runs[:, 0].mean()) == pytest.approx(0.2060064812955266, abs=1e-12)
    assert float(runs[:, 1].mean()) == pytest.approx(0.1975996778609539, abs=1e-12)


def test_the_worked_day_matches_the_article() -> None:
    """At 210 units, 120 and 100 intents sell about 114.5 and 95.5; the ratio stays 1.2."""
    day = fluid_day()

    assert (day.treated_intents, day.control_intents) == (120.0, 100.0)
    assert round(day.treated_sales, 1) == 114.5
    assert round(day.control_sales, 1) == 95.5
    assert day.treated_sales + day.control_sales == pytest.approx(210.0)
    assert day.buyer_level_lift == pytest.approx(0.20)
    assert day.rollout_lift == pytest.approx(0.05)


def test_the_buyer_level_ratio_ignores_the_inventory() -> None:
    """In the noise-free day the buyer-level lift is 20 percent at any stock; the truth is not."""
    rollouts = {}
    for inventory in (150, 200, 210, 230, 300):
        day = fluid_day(inventory)
        assert day.buyer_level_lift == pytest.approx(0.20)
        rollouts[inventory] = round(day.rollout_lift, 4)

    assert rollouts == {150: 0.0, 200: 0.0, 210: 0.05, 230: 0.15, 300: 0.2}


def test_capped_sales_closed_form() -> None:
    """E[min(X, c)] is the mean when stock is ample, c when scarce, and matches sampling."""
    assert expected_capped_sales(BUYERS, CONTROL_INTENT) == pytest.approx(BUYERS * CONTROL_INTENT)
    assert expected_capped_sales(0, TREATED_INTENT) == 0.0
    assert expected_capped_sales(100, TREATED_INTENT) == pytest.approx(100.0, abs=1e-9)

    draws = np.random.default_rng(2).binomial(BUYERS, CONTROL_INTENT, 400_000)
    capped = np.minimum(draws, 200)
    error = capped.std() / np.sqrt(capped.size)
    assert expected_capped_sales(200, CONTROL_INTENT) == pytest.approx(capped.mean(), abs=4 * error)


def test_the_true_lift_falls_to_zero_as_stock_binds() -> None:
    """20 percent with ample stock, falling monotonically, and nothing when both arms sell out."""
    lifts = [rollout_lift(inventory) for inventory in INVENTORIES]

    assert lifts == sorted(lifts, reverse=True)
    assert lifts[0] == pytest.approx(0.20, abs=1e-4)
    assert 0.01 < lifts[-1] < 0.02
    assert rollout_lift(100) == pytest.approx(0.0, abs=1e-9)


def test_the_article_truth_column_agrees_with_the_closed_form() -> None:
    """The article's simulated true lifts (200 replications) are within noise of the closed form."""
    printed = {300: 0.20, 230: 0.139, 210: 0.060, 200: 0.028}
    for inventory, lift in printed.items():
        assert rollout_lift(inventory) == pytest.approx(lift, abs=0.003)
    # "Beyond the 3 percent that comes from filling the days on which control demand fell short."
    assert round(100 * rollout_lift(200)) == 3


def test_a_factor_of_seven_at_control_demand() -> None:
    """At 200 units the buyer-level test's 20 percent is about seven times the true lift."""
    assert 6.5 < 0.20 / rollout_lift(200) < 8


def test_all_designs_agree_when_inventory_is_ample() -> None:
    """The alt text: with ample stock all three are about 20 percent."""
    row = SHORT[0]

    assert row.inventory == 320
    for value, spread in (
        (row.buyer_level, row.buyer_level_spread),
        (row.market_level, row.market_level_spread),
        (row.true_lift, row.true_lift_spread),
    ):
        assert abs(value - 0.20) < 4 * _standard_error(spread)


def test_the_buyer_level_estimate_stays_near_twenty_percent() -> None:
    """The alt text: the buyer-level estimate does not move as stock binds."""
    for row in SHORT:
        assert abs(row.buyer_level - 0.20) < 4 * _standard_error(row.buyer_level_spread)


def test_randomised_cities_and_the_truth_track_the_closed_form() -> None:
    """The city design is unbiased, and the simulated truth matches its expectation."""
    for row in SHORT:
        assert abs(row.market_level - row.rollout_lift) < 4 * _standard_error(
            row.market_level_spread
        )
        assert abs(row.true_lift - row.rollout_lift) < 4 * _standard_error(row.true_lift_spread)


def test_the_buyer_level_test_overstates_the_lift() -> None:
    """The title: once stock binds, the buyer-level estimate is several times the truth."""
    scarce = SHORT[-1]

    assert scarce.inventory == 195
    assert scarce.true_lift < 0.03
    assert scarce.buyer_level > 5 * scarce.true_lift


def test_day_sales_serves_intents_until_the_stock_runs_out() -> None:
    """Only buyers who want a unit get one, and exactly min(intents, stock) are sold."""
    rng = np.random.default_rng(3)
    wants = rng.random(500) < 0.3
    for stock in (0, 50, int(wants.sum()), 1000):
        sold = day_sales(wants, stock, rng)
        assert not np.any(sold & ~wants)
        assert sold.sum() == min(int(wants.sum()), stock)

    fixed = np.array([True, False, True, True])
    assert day_sales(fixed, 5, rng).tolist() == fixed.tolist()


def test_a_short_run_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces an experiment; another seed does not."""

    def short(seed: int) -> tuple[float, float]:
        return experiment(205, "market", np.random.default_rng(seed), days=2, markets=4)

    assert short(4) == short(4)
    assert short(4) != short(5)


def test_invalid_inputs_are_rejected() -> None:
    """Odd city counts, unknown designs, non-boolean intents and bad rates are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        experiment(200, "market", rng, markets=3)
    with pytest.raises(ValueError):
        experiment(200, cast(Design, "switchback"), rng)
    with pytest.raises(ValueError):
        experiment(-1, "buyer", rng)
    with pytest.raises(TypeError):
        experiment(200, "buyer", rng, days=True)
    with pytest.raises(ValueError):
        day_sales(np.array([1, 0, 1]), 2, rng)
    with pytest.raises(ValueError):
        expected_capped_sales(200, 1.5)
    with pytest.raises(ValueError):
        fluid_day(0)
    with pytest.raises(ValueError):
        inventory_rows((200,), replications=0)


def test_rows_carry_the_closed_form() -> None:
    """Each simulated row records the closed-form expectation of its true lift."""
    for row in SHORT:
        assert isinstance(row, InventoryRow)
        assert row.rollout_lift == rollout_lift(row.inventory)
