"""Check the allocation formulas against brute force, simulation and the article's tables.

Every table in the article is closed form, so all five are pinned at their
printed precision. The optimal shares are checked by minimising the variance
over a grid, and the power formula against a small simulated experiment. The
article's own simulated power check (seed 53, 40,000 users, 4,000 experiments
per split) is not reproduced: it is a separate stream of draws from the figure,
which has none.

The figure's title calls anything from thirty to seventy percent "nearly free".
The inflation there is at most 19 percent (1.19x at 30 percent), which is small
next to the edges, but the figure's own "within 10% of the even split" band
spans only about 35 to 65 percent. The tests pin the 1.19 and the band edges
rather than a stronger reading of the title.
"""

from math import sqrt

import numpy as np
import pytest

from blog_reproducibility.statistics.unequal_allocation import (
    ANNOTATED_SHARES,
    CURVE_POINTS,
    TABLE_SHARES,
    cost_optimal_share,
    cost_row,
    detectable_effect,
    example_payload,
    neyman_row,
    neyman_share,
    power_at,
    shared_control_row,
    square_root_control_share,
    users_for_power,
    variance_curve,
    variance_factor,
)

SUMMARY = example_payload()
GRID = np.linspace(0.001, 0.999, 99_801)


def test_the_split_table_matches_the_article() -> None:
    """Variance factor, power at 40,000 users and smallest detectable effect per share."""
    printed = {
        0.5: (1.00, 99.4, 0.0126),
        0.4: (1.04, 99.2, 0.0129),
        0.3: (1.19, 98.3, 0.0138),
        0.2: (1.56, 94.5, 0.0158),
        0.1: (2.78, 76.0, 0.0210),
        0.05: (5.26, 49.1, 0.0289),
    }
    assert tuple(row.share for row in SUMMARY.splits) == TABLE_SHARES
    for row in SUMMARY.splits:
        factor, power, effect = printed[row.share]
        assert round(row.variance_factor, 2) == factor
        assert round(100 * row.power, 1) == power
        assert round(row.detectable_effect, 4) == effect


def test_the_duration_table_matches_the_article() -> None:
    """Users for 80 percent power and days at 2,000 users a day."""
    printed = {
        0.5: (15_894, 7.9),
        0.25: (21_192, 10.6),
        0.10: (44_150, 22.1),
        0.05: (83_653, 41.8),
        0.01: (401_363, 200.7),
    }
    for row in SUMMARY.durations:
        users, days = printed[row.share]
        assert round(row.users) == users
        assert round(row.days, 1) == days


def test_the_neyman_table_matches_the_article() -> None:
    """Best split, variance against an even split and the saving, for four spread pairs."""
    printed = [
        (0.45, 0.45, 50, 1.000, 0.0),
        (0.60, 0.30, 67, 0.900, 10.0),
        (0.90, 0.30, 75, 0.800, 20.0),
        (0.30, 0.90, 25, 0.800, 20.0),
    ]
    for row, (s_t, s_c, share, ratio, saving) in zip(SUMMARY.neyman, printed, strict=True):
        assert (row.treatment_sd, row.control_sd) == (s_t, s_c)
        assert round(100 * row.best_share) == share
        assert round(row.variance_ratio, 3) == ratio
        assert round(100 * row.saving, 1) == saving


def test_the_cost_table_matches_the_article() -> None:
    """Best split, users affordable and variance against an even split on the same budget."""
    printed = [(1, 50, 40_000, 1.000), (2, 41, 28_284, 0.971), (4, 33, 20_000, 0.900)]
    printed.append((10, 24, 12_649, 0.787))
    for row, (ratio, share, users, variance) in zip(SUMMARY.cost, printed, strict=True):
        assert row.cost_ratio == ratio
        assert round(100 * row.best_share) == share
        assert round(row.users_affordable) == users
        assert round(row.variance_ratio, 3) == variance


def test_the_shared_control_table_matches_the_article() -> None:
    """Control share under an equal split and the square-root rule, and the reduction."""
    printed = [(1, 50, 50, 0.0), (2, 33, 41, 2.9), (3, 25, 37, 6.7), (5, 17, 31, 12.7)]
    printed.append((8, 11, 26, 18.6))
    for row, (arms, equal, rule, reduction) in zip(SUMMARY.shared_control, printed, strict=True):
        assert row.treatment_arms == arms
        assert round(100 * row.equal_control_share) == equal
        assert round(100 * row.square_root_control_share) == rule
        assert round(100 * row.variance_reduction, 1) == reduction


def test_variance_factor_matches_the_two_arm_variance() -> None:
    """0.25 / (f (1 - f)) is the ratio of 1/(fN) + 1/((1-f)N) to its value at one half."""
    for share in (0.5, 0.37, 0.1, 0.013):
        direct = (1 / share + 1 / (1 - share)) / (1 / 0.5 + 1 / 0.5)
        assert variance_factor(share) == pytest.approx(direct, rel=1e-12)
    assert variance_factor(0.5) == 1.0
    assert variance_factor(0.1) == pytest.approx(0.25 / 0.09)
    assert variance_factor(0.2) == pytest.approx(variance_factor(0.8))


def test_the_optima_minimise_the_variance() -> None:
    """Neyman, the cost rule and the square-root rule agree with a brute-force search."""
    for s_t, s_c in ((0.6, 0.3), (0.9, 0.3), (0.3, 0.9)):
        variance = s_t**2 / GRID + s_c**2 / (1 - GRID)
        assert GRID[np.argmin(variance)] == pytest.approx(neyman_share(s_t, s_c), abs=1e-4)
    for ratio in (2.0, 4.0, 10.0):
        # Variance on a fixed budget: users scale as 1 / (f c + 1 - f).
        variance = (GRID * ratio + 1 - GRID) * (1 / GRID + 1 / (1 - GRID))
        assert GRID[np.argmin(variance)] == pytest.approx(cost_optimal_share(ratio), abs=1e-4)
    for arms in (2, 3, 5, 8):
        variance = arms / (1 - GRID) + 1 / GRID
        assert GRID[np.argmin(variance)] == pytest.approx(square_root_control_share(arms), abs=1e-4)


def test_the_gains_have_closed_forms() -> None:
    """Neyman keeps (s_t + s_c)^2 / 2(s_t^2 + s_c^2); the other two (1 + sqrt r)^2 / 2(1 + r)."""
    for s_t, s_c in ((0.6, 0.3), (0.9, 0.3), (0.45, 0.45)):
        expected = (s_t + s_c) ** 2 / (2 * (s_t**2 + s_c**2))
        assert neyman_row(s_t, s_c).variance_ratio == pytest.approx(expected, rel=1e-12)
    for r in (1, 2, 4, 10):
        kept = (1 + sqrt(r)) ** 2 / (2 * (1 + r))
        assert cost_row(r).variance_ratio == pytest.approx(kept, rel=1e-12)
        assert 1 - shared_control_row(r).variance_reduction == pytest.approx(kept, rel=1e-12)


def test_power_formula_matches_a_simulation() -> None:
    """A 20 percent split of 4,000 users detects an effect of 0.045 about 72 percent of the time."""
    rng = np.random.default_rng(71)
    share, users, effect, sd, reps = 0.2, 4000, 0.045, 0.45, 2000
    treated_n = round(share * users)
    treated = rng.normal(effect, sd, (reps, treated_n))
    control = rng.normal(0.0, sd, (reps, users - treated_n))
    se = np.sqrt(
        treated.var(axis=1, ddof=1) / treated_n + control.var(axis=1, ddof=1) / (users - treated_n)
    )
    simulated = float(np.mean(np.abs(treated.mean(axis=1) - control.mean(axis=1)) / se > 1.959964))

    formula = power_at(users, share, effect, sd)
    assert formula == pytest.approx(0.716, abs=0.005)
    assert simulated == pytest.approx(formula, abs=0.035)


def test_the_sample_size_delivers_its_power() -> None:
    """At the users the formula asks for, the power formula returns 80 percent."""
    for share in (0.5, 0.25, 0.1):
        users = users_for_power(share)
        assert power_at(round(users), share) == pytest.approx(0.80, abs=0.001)
        assert detectable_effect(round(users), share) == pytest.approx(0.02, rel=1e-3)


def test_the_article_prose() -> None:
    """A 60/40 split costs four percent; eight days become twenty-two; one percent is 200 days."""
    assert round(100 * (variance_factor(0.4) - 1)) == 4
    days = {row.share: row.days for row in SUMMARY.durations}
    assert (round(days[0.5]), round(days[0.10]), round(days[0.01])) == (8, 22, 201)
    assert variance_factor(0.05) > 5


def test_the_figure_claims() -> None:
    """2.8 at ninety-ten, 5.3 at ninety-five-five, at most 1.19 between 30 and 70 percent."""
    assert round(variance_factor(0.1), 1) == 2.8
    assert round(variance_factor(0.05), 1) == 5.3
    assert ANNOTATED_SHARES == (0.5, 0.3, 0.2, 0.1, 0.05)

    shares, factors = variance_curve()
    assert shares.size == CURVE_POINTS
    assert shares[0] == 0.02 and shares[-1] == 0.98
    middle = (shares >= 0.3) & (shares <= 0.7)
    assert factors[middle].max() < 1.2
    # The shaded band (within 10 percent) spans about 35 to 65 percent.
    band = shares[factors <= 1.1]
    assert (round(band.min(), 2), round(band.max(), 2)) == (0.35, 0.65)
    # At the edges the curve leaves the plotted range (0.8 to 7).
    assert factors[0] > 12 and factors[-1] > 12


def test_invalid_inputs_are_rejected() -> None:
    """Shares outside (0, 1), non-positive spreads or costs, and bad counts are refused."""
    with pytest.raises(ValueError):
        variance_factor(0.0)
    with pytest.raises(ValueError):
        variance_factor(1.0)
    with pytest.raises(TypeError):
        variance_factor(True)
    with pytest.raises(ValueError):
        power_at(40_000, 0.5, effect=0.0)
    with pytest.raises(ValueError):
        users_for_power(0.5, power=1.0)
    with pytest.raises(ValueError):
        neyman_share(0.0, 0.3)
    with pytest.raises(ValueError):
        cost_optimal_share(-1.0)
    with pytest.raises(ValueError):
        square_root_control_share(0)
    with pytest.raises(TypeError):
        power_at(40_000.0, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        variance_curve(points=1)
