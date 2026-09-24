"""Check the regression-to-the-mean fleet against the website, the article and closed forms.

The figure's fleet (seed 42), the placebo split (seed 1) and the rerun with a
real effect (seed 42 again) are transcribed from the website and the article
and matched draw for draw. What they print splits in two:

- Numbers that depend only on the counts are pinned from the payload: the fleet
  at 3.95 and 4.00, the worst decile's 9.44 and the best's 0.52 in month one, a
  correlation of 0.54 and a prediction of 6.92, and the placebo halves' 9.84
  and 9.04 in month one.
- Numbers that depend on which tied machines enter a decile do not reproduce
  across machines. The website ranks with NumPy's default ``argsort``, which
  leaves ties to the CPU's sort kernel; the article's 7.02, 7.00 and 2.22, its
  placebo months of 7.16 and 6.88, and its -4.10 and -4.40 come from the AVX2
  kernel's order, recorded below, and all reproduce from it. The package breaks
  ties by machine index instead, which gives 6.88, 6.95, 1.86, 7.28, 6.48, -4.02
  and -4.26, all pinned. Over every possible tie-break the worst decile's month
  two lies between 6.50 and 7.46 and the best's between 1.24 and 3.10.

So one of the article's claims holds only for its own tie order: the best
decile rising "more than fourfold" (0.52 to 2.22). With ties broken by index it
rises 3.6 times, and averaged over tie-breaks 4.06 times. The title's claim
holds for both orders: the worst decile's mean lands within 0.1 of the
regression line and more than two failures below the identity line. The prose's
"almost entirely below" the identity line is four in five: 38 of the website's
50 (39 of the package's) fall strictly below it.

The closed forms are the gamma-Poisson conjugacy. They give the baseline
table's correlations of 0.50, 0.61, 0.65 and 0.68 exactly, and its true rates
from its selected baselines to within its simulation noise. The replicated
estimates, whose seeds the article does not give, are rerun from seeds 0 to 499
and land within 0.02 of the printed -4.50, -4.83 and -1.54. The winner's curse
is the exact expected maximum of ten binomials: 0.762 points, 15 percent.
"""

from math import sqrt

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from blog_reproducibility.statistics.regression_to_the_mean import (
    BASELINE_MONTHS,
    Fleet,
    baseline_row,
    compare_deciles,
    example_payload,
    expected_rate,
    extreme_deciles,
    intervention_run,
    month_to_month_correlation,
    placebo_split,
    regression_prediction,
    scatter_data,
    simulate_fleet,
    winners_curse,
)

SUMMARY = example_payload()
FLEET = simulate_fleet(np.random.default_rng(42))
M1, M2 = FLEET.month_one, FLEET.month_two

# The website's deciles in its argsort order, as NumPy's AVX2 kernel ranked month one.
WEBSITE_WORST = (
    330, 319, 187, 447, 371, 123, 149, 157, 142, 152, 229, 396, 65, 113, 94, 475, 385,
    9, 18, 488, 224, 153, 162, 355, 189, 285, 109, 68, 158, 8, 484, 288, 327, 127, 325,
    133, 185, 419, 138, 81, 424, 125, 164, 463, 331, 80, 256, 120, 343, 428,
)  # fmt: skip
WEBSITE_BEST = (
    474, 376, 86, 115, 361, 121, 101, 110, 353, 432, 118, 194, 182, 422, 159, 151, 218,
    222, 448, 272, 253, 12, 255, 289, 417, 84, 70, 498, 481, 393, 470, 472, 381, 378,
    103, 102, 369, 93, 358, 347, 434, 429, 339, 128, 354, 338, 455, 116, 147, 169,
)  # fmt: skip


def _website_scatter(worst: NDArray[np.intp]) -> tuple[NDArray[np.float64], ...]:
    """Transcribe the website generator, with the worst decile given rather than argsorted."""
    r = np.random.default_rng(42)
    n = 500
    rate = r.gamma(4.0, 1.0, n)
    m1 = r.poisson(rate)
    m2 = r.poisson(rate)
    rest = np.setdiff1d(np.arange(n), worst)

    def jit(v: NDArray[np.int64]) -> NDArray[np.float64]:
        jittered: NDArray[np.float64] = v + r.uniform(-0.3, 0.3, v.size)
        return jittered

    return jit(m1[rest]), jit(m2[rest]), jit(m1[worst]), jit(m2[worst])


def _website_intervention(worst: NDArray[np.intp]) -> tuple[float, float, float]:
    """Transcribe the article's rerun with a real effect, with the worst decile given."""
    n_units = 500
    rng = np.random.default_rng(42)
    true_rate = rng.gamma(shape=4.0, scale=1.0, size=n_units)
    m1 = rng.poisson(true_rate)
    treated = np.zeros(n_units, dtype=bool)
    treated[worst] = True
    effect = -1.5
    m2 = rng.poisson(np.clip(true_rate + effect * treated, 0.05, None))
    naive = m2[treated].mean() - m1[treated].mean()
    did = naive - (m2[~treated].mean() - m1[~treated].mean())
    x = np.column_stack([np.ones(n_units), m1, treated.astype(float)])
    ancova = np.linalg.lstsq(x, m2, rcond=None)[0][2]
    return float(naive), float(did), float(ancova)


def _website_placebo(worst: NDArray[np.intp]) -> tuple[float, float, float, float]:
    """Transcribe the article's placebo split."""
    rng2 = np.random.default_rng(1)
    treated = rng2.choice(worst, size=50 // 2, replace=False)
    control = np.setdiff1d(worst, treated)
    return (
        float(M1[treated].mean()),
        float(M2[treated].mean()),
        float(M1[control].mean()),
        float(M2[control].mean()),
    )


def _website_deciles() -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    return np.array(WEBSITE_WORST, dtype=np.intp), np.array(WEBSITE_BEST, dtype=np.intp)


def test_the_fleet_is_drawn_as_on_the_website() -> None:
    """Rates, then month one, then month two, from one generator seeded at 42."""
    r = np.random.default_rng(42)
    rate = r.gamma(4.0, 1.0, 500)
    assert np.array_equal(FLEET.rates, rate)
    assert np.array_equal(M1, r.poisson(rate))
    assert np.array_equal(M2, r.poisson(rate))


def test_the_scatter_matches_the_website_draw_for_draw() -> None:
    """The jitter follows the fleet in the website's order, whichever decile is highlighted."""
    stable, _ = extreme_deciles(M1)
    for worst in (stable, _website_deciles()[0]):
        data = scatter_data(worst=worst)
        expected = _website_scatter(worst)
        for ours, theirs in zip(
            (
                data.rest_month_one,
                data.rest_month_two,
                data.worst_month_one,
                data.worst_month_two,
            ),
            expected,
            strict=True,
        ):
            assert np.array_equal(ours, theirs)
        assert data.limits == (-0.5, float(max(M1.max(), M2.max())) + 1.0)
    assert np.array_equal(
        scatter_data().worst_month_two, scatter_data(worst=stable).worst_month_two
    )


def test_the_numbers_that_do_not_depend_on_ties() -> None:
    """3.95 and 4.00 for the fleet, 9.44 and 0.52 in month one, 0.54 and a prediction of 6.92."""
    d = SUMMARY.deciles
    assert (round(d.fleet_month_one, 2), round(d.fleet_month_two, 2)) == (3.95, 4.00)
    assert (round(d.worst_month_one, 2), round(d.best_month_one, 2)) == (9.44, 0.52)
    assert round(d.correlation, 2) == 0.54
    assert round(d.predicted_worst_month_two, 2) == 6.92
    placebo = SUMMARY.placebo
    assert (round(placebo.treated_month_one, 2), round(placebo.control_month_one, 2)) == (
        9.84,
        9.04,
    )


def test_the_website_deciles_are_valid_rankings() -> None:
    """The recorded order ranks month one: 20 of the 27 machines at eight, 26 of the 71 at one."""
    worst, best = _website_deciles()
    assert list(M1[worst]) == sorted(M1[worst])
    assert list(M1[best]) == sorted(M1[best])
    stable_worst, stable_best = extreme_deciles(M1)
    assert sorted(M1[worst]) == sorted(M1[stable_worst])
    assert sorted(M1[best]) == sorted(M1[stable_best])
    assert (int(np.sum(M1 == 8)), int(np.sum(M1[worst] == 8))) == (27, 20)
    assert (int(np.sum(M1 == 1)), int(np.sum(M1[best] == 1))) == (71, 26)
    assert set(worst) != set(stable_worst)
    assert set(best) != set(stable_best)


def test_the_article_numbers_follow_from_the_website_deciles() -> None:
    """7.02, 7.00 and 2.22; placebo halves of 7.16 and 6.88; -4.10 and -4.40 with a real effect."""
    worst, best = _website_deciles()
    d = compare_deciles(FLEET, worst=worst, best=best)
    assert (round(d.worst_month_two, 2), round(d.worst_true_rate, 2)) == (7.02, 7.00)
    assert round(d.best_month_two, 2) == 2.22
    assert round(1 - d.worst_month_two / d.worst_month_one, 2) == 0.26
    placebo, _, _ = placebo_split(np.random.default_rng(1), FLEET, worst)
    assert (placebo.treated_month_one, placebo.treated_month_two) == pytest.approx((9.84, 7.16))
    assert (placebo.control_month_one, placebo.control_month_two) == pytest.approx((9.04, 6.88))
    assert _website_placebo(worst) == (
        placebo.treated_month_one,
        placebo.treated_month_two,
        placebo.control_month_one,
        placebo.control_month_two,
    )
    run = intervention_run(np.random.default_rng(42), worst=worst)
    assert (round(run.naive, 2), round(run.difference_in_differences, 2)) == (-4.10, -4.40)
    assert (run.naive, run.difference_in_differences, run.baseline_adjusted) == (
        _website_intervention(worst)
    )
    data = scatter_data(worst=worst)
    label = f"{data.worst_mean_month_one:.1f} in month 1, {data.worst_mean_month_two:.1f}"
    assert label == "9.4 in month 1, 7.0"


def test_the_payload_breaks_ties_by_index() -> None:
    """6.88, 6.95 and 1.86; placebo halves of 7.28 and 6.48; -4.02, -4.26 and -1.49."""
    d = SUMMARY.deciles
    assert (d.worst_month_two, d.best_month_two) == pytest.approx((6.88, 1.86))
    assert round(d.worst_true_rate, 2) == 6.95
    p = SUMMARY.placebo
    assert (p.treated_month_two, p.control_month_two) == pytest.approx((7.28, 6.48))
    i = SUMMARY.intervention
    assert (i.naive, i.difference_in_differences) == pytest.approx((-4.02, -4.26))
    assert round(i.baseline_adjusted, 2) == -1.49
    stable, _ = extreme_deciles(M1)
    assert (i.naive, i.difference_in_differences, i.baseline_adjusted) == (
        _website_intervention(stable)
    )
    assert np.array_equal(stable, np.argsort(M1, kind="stable")[-50:])


def test_the_tie_break_decides_the_fourfold_claim() -> None:
    """Every tie-break keeps 9.44 and 0.52; month two ranges widely, and fourfold is not assured."""
    counts = M1
    order = np.sort(counts)
    tied_top, tied_bottom = counts == order[-50], counts == order[49]
    sure_top, sure_bottom = counts > order[-50], counts < order[49]
    need_top, need_bottom = 50 - int(sure_top.sum()), 50 - int(sure_bottom.sum())
    top_two, bottom_two = np.sort(M2[tied_top]), np.sort(M2[tied_bottom])
    worst_range = (
        (M2[sure_top].sum() + top_two[:need_top].sum()) / 50,
        (M2[sure_top].sum() + top_two[-need_top:].sum()) / 50,
    )
    best_range = (
        (M2[sure_bottom].sum() + bottom_two[:need_bottom].sum()) / 50,
        (M2[sure_bottom].sum() + bottom_two[-need_bottom:].sum()) / 50,
    )
    assert worst_range == pytest.approx((6.50, 7.46))
    assert best_range == pytest.approx((1.24, 3.10))
    for value in (7.02, SUMMARY.deciles.worst_month_two):
        assert worst_range[0] <= value <= worst_range[1]
    for value in (2.22, SUMMARY.deciles.best_month_two):
        assert best_range[0] <= value <= best_range[1]
    # Averaged over tie-breaks, and for each order.
    averaged = (M2[sure_bottom].sum() + need_bottom * M2[tied_bottom].mean()) / 50
    assert round(averaged / 0.52, 2) == 4.06
    assert 2.22 / 0.52 > 4
    assert round(SUMMARY.deciles.best_month_two / 0.52, 1) == 3.6


def test_the_worst_decile_lands_on_the_regression_line() -> None:
    """The title: within 0.1 of the prediction and more than two failures below the identity."""
    website = compare_deciles(FLEET, worst=_website_deciles()[0])
    for d in (SUMMARY.deciles, website):
        assert abs(d.worst_month_two - d.predicted_worst_month_two) < 0.11
        assert d.worst_month_one - d.worst_month_two > 2.4
        # Month two of 50 machines around their expected rate: a standard error near 0.37.
        assert abs(d.worst_month_two - d.worst_true_rate) < 0.37
    # "Almost entirely below" the identity line: about four in five.
    stable, _ = extreme_deciles(M1)
    assert int(np.sum(M2[stable] < M1[stable])) == 39
    worst = _website_deciles()[0]
    assert int(np.sum(M2[worst] < M1[worst])) == 38
    # The fleet itself does not move: within one standard deviation of its change.
    assert abs(SUMMARY.deciles.fleet_month_two - SUMMARY.deciles.fleet_month_one) < sqrt(8 / 500)
    # Half of the worst 50 by count are among the worst 50 by rate, in either order.
    assert SUMMARY.deciles.worst_also_worst_by_rate == website.worst_also_worst_by_rate == 25


def test_the_conjugate_posterior_mean() -> None:
    """Among two million machines, the mean rate given a month's count is ``(4 + y) / 2``."""
    rng = np.random.default_rng(7)
    rates = rng.gamma(4.0, 1.0, 2_000_000)
    counts = rng.poisson(rates)
    for y in (0, 4, 9):
        chosen = rates[counts == y]
        spread = float(chosen.std()) / sqrt(chosen.size)
        assert abs(float(chosen.mean()) - expected_rate(y)) < 3 * spread
    assert expected_rate(9.91) == pytest.approx(regression_prediction(9.91, 4.0, 0.5))
    assert expected_rate(8.0, 3, shape=2.0, scale=0.5) == pytest.approx((2 + 24) / (2 + 3))


def test_the_baseline_correlations_are_exact() -> None:
    """0.50, 0.61, 0.65 and 0.68: ``4 / sqrt((4 + 4 / b) 8)``."""
    rows = [baseline_row(b) for b in BASELINE_MONTHS]
    assert rows == list(SUMMARY.baselines)
    assert [round(row.correlation, 2) for row in rows] == [0.50, 0.61, 0.65, 0.68]
    for row in rows:
        assert row.correlation == pytest.approx(4 / sqrt((4 + 4 / row.months) * 8), rel=1e-12)
    assert [round(row.own_data_weight, 3) for row in rows] == [0.5, 0.75, 0.857, 0.923]
    # A simulated fleet: a three-month average against the fourth month.
    rng = np.random.default_rng(8)
    rates = rng.gamma(4.0, 1.0, 400_000)
    months = rng.poisson(rates[:, None], size=(rates.size, 4))
    simulated = float(np.corrcoef(months[:, :3].mean(axis=1), months[:, 3])[0, 1])
    assert simulated == pytest.approx(month_to_month_correlation(3), abs=0.004)


def test_the_article_baseline_table_against_the_closed_forms() -> None:
    """Its true rates are the posterior means of its selected baselines, to within its noise."""
    printed = {
        1: (9.91, 6.93, 6.95, 0.30),
        3: (8.80, 7.59, 7.59, 0.14),
        6: (8.49, 7.85, 7.85, 0.07),
        12: (8.32, 7.98, 8.03, 0.03),
    }
    for months, (baseline, truth, following, drop) in printed.items():
        # 200 replications of 50 machines: standard errors of about 0.013 and 0.028.
        assert abs(expected_rate(baseline, months) - truth) < 0.04
        assert abs(following - truth) < 0.06
        # The apparent drop agrees with its own columns to their rounding.
        assert abs(1 - following / baseline - drop) < 0.0055


def test_the_replicated_estimates() -> None:
    """500 reruns (seeds 0 to 499): about -4.50, -4.83 and -1.54, as the article prints."""
    runs = []
    for seed in range(500):
        run = intervention_run(np.random.default_rng(seed))
        runs.append((run.naive, run.difference_in_differences, run.baseline_adjusted))
    means = np.mean(runs, axis=0)
    errors = np.std(runs, axis=0) / sqrt(500)
    for mean, error, printed in zip(means, errors, (-4.50, -4.83, -1.54), strict=True):
        assert abs(mean - printed) < 3 * error
    assert means[1] < means[0] < -4
    assert abs(means[2] - (-1.5)) < 3 * errors[2]
    assert [round(float(value), 2) for value in means] == [-4.50, -4.83, -1.53]


def test_the_naive_estimate_against_its_conditional_expectation() -> None:
    """Given month one, the treated machines' change is ``effect - (y - 4) / 2`` on average."""
    stable, _ = extreme_deciles(M1)
    expected = -1.5 - (float(M1[stable].mean()) - 4) / 2
    # Month two given month one is negative binomial with variance 3 (4 + y) / 4.
    error = sqrt(3 * (4 + float(M1[stable].mean())) / 4 / 50)
    assert abs(SUMMARY.intervention.naive - expected) < 2 * error


def test_the_winners_curse_is_the_expected_maximum() -> None:
    """0.762 points, about 15 percent; the sum of ``1 - F^K`` is the mean of the maximum's law."""
    curse = SUMMARY.winners_curse
    assert round(100 * curse.expected_lift, 2) == 0.76
    assert round(curse.relative_lift, 2) == 0.15
    assert curse.lift_when_measured_again == 0.0
    support = np.arange(2001)
    cdf = stats.binom.cdf(support, 2000, 0.05)
    law = cdf**10 - np.concatenate(([0.0], cdf[:-1] ** 10))
    assert curse.expected_lift == pytest.approx(float(support @ law) / 2000 - 0.05, rel=1e-9)
    # 2,000 simulated tests: a standard error of about 0.0064 points around 0.76.
    assert abs(100 * curse.expected_lift - 0.76) < 2 * 0.0064
    assert winners_curse(1).expected_lift == pytest.approx(0.0, abs=1e-12)
    lifts = [winners_curse(k).expected_lift for k in (2, 5, 10, 20)]
    assert lifts == sorted(lifts)
    assert winners_curse(10, 8000).expected_lift == pytest.approx(curse.expected_lift / 2, rel=0.02)


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the fleet and the scatter; another seed does not."""
    first = simulate_fleet(np.random.default_rng(3), 100)
    second = simulate_fleet(np.random.default_rng(3), 100)
    other = simulate_fleet(np.random.default_rng(4), 100)
    assert np.array_equal(first.month_one, second.month_one)
    assert not np.array_equal(first.month_one, other.month_one)
    a, b, c = scatter_data(3), scatter_data(3), scatter_data(4)
    assert np.array_equal(a.worst_month_two, b.worst_month_two)
    assert np.array_equal(a.rest_month_one, b.rest_month_one)
    assert not np.array_equal(a.rest_month_one, c.rest_month_one)


def test_invalid_inputs_are_rejected() -> None:
    """Bad counts, bad deciles and out-of-range parameters are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_fleet(rng, 1)
    with pytest.raises(ValueError):
        simulate_fleet(rng, 10, shape=0.0)
    with pytest.raises(ValueError):
        extreme_deciles([1.5, 2.5, 3.5])
    with pytest.raises(ValueError):
        extreme_deciles([1, 2, 3], decile=10)
    with pytest.raises(ValueError):
        extreme_deciles([1, -2, 3, 4], decile=2)
    worst = np.array(WEBSITE_WORST)
    with pytest.raises(ValueError):
        compare_deciles(FLEET, worst=worst[:-1])
    with pytest.raises(ValueError):
        compare_deciles(FLEET, worst=np.append(worst[:-1], worst[0]))
    with pytest.raises(ValueError):
        compare_deciles(FLEET, worst=np.array(WEBSITE_BEST))
    with pytest.raises(ValueError):
        compare_deciles(FLEET, best=worst)
    with pytest.raises(ValueError):
        compare_deciles(Fleet(FLEET.rates[:10], M1, M2))
    with pytest.raises(ValueError):
        intervention_run(rng, worst=np.arange(50))
    with pytest.raises(ValueError):
        intervention_run(rng, floor=0.0)
    with pytest.raises(ValueError):
        regression_prediction(9.0, 4.0, 1.5)
    with pytest.raises(ValueError):
        expected_rate(-1.0)
    with pytest.raises(ValueError):
        month_to_month_correlation(0)
    with pytest.raises(ValueError):
        winners_curse(10, 2000, 1.0)
