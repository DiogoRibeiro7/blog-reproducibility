"""Check the recurrent-event model against the website, the article, closed forms and simulation.

The figure and the article share one generator seeded at 0 and one draw order,
so the fleet, the four curves and every number the article prints from the
fleet are reproduced and pinned: 880 failures in 800 machine-years, a quarter
never failing, 299 first failures (34 percent), rates of 0.80, 1.10 and 1.09 a
year, the mean cumulative function table, the site table, a count mean of 2.2 and
variance of 5.7, 34 percent of failures from the top tenth of machines, and 1.46
against 0.68 failures in year two. The figure's curves match a transcription of
the website's code bit for bit, and the article's own estimator (summing in
record order) agrees to the printed precision. The top-decile share sums the
largest counts after a sort, so ties between machines cannot change it; the
article's unstable ``argsort`` gives the same share.

The gamma frailty gives population values in closed form (1.12 failures a
machine-year, 26 percent never failing, a first-failure rate of 0.78, a count
variance of 6.37, and 1.46 against 0.66 in year two), which are checked against
quadrature and a large simulation, and the fleet lies within sampling noise of
them. The article's no-frailty comparison (29 percent, from a generator seeded at
1) is not pinned. It is one draw: over 2,000 fleets with the same sites and
windows and no frailty the top tenth's share averages 27.0 percent with a
standard deviation of 0.9, so the article's draw is two standard deviations high
and the frailty's excess is 7 points rather than 5. The fleet's 34 percent is
beyond the 99th percentile of that distribution.

Two statements do not hold as written, and the tests pin what is true:

* the site table's ratio of true mean rates is printed as 2.00, the design
  ratio; the table's own true rates, 1.50 and 0.77, have a ratio of 1.94;
* a budget built on first-failure rates "would be a quarter short at both kinds
  of site": it is 24 percent short at harsh sites and 20 percent at normal ones.

The figure's claims hold: the naive count equals the mean cumulative function
while every machine is observed and rises only 0.30 between two and three years,
against 1.08 for the mean cumulative function and a true rate of 1.09.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate

from blog_reproducibility.statistics.recurrent_events import (
    GRID_POINTS,
    SITES,
    TABLE_AGES,
    Fleet,
    all_events_rate,
    closed_forms,
    example_payload,
    failure_counts,
    first_failure_rate,
    first_failures,
    mean_cumulative_function,
    naive_average_count,
    simulate_fleet,
    top_share,
    year_two_history,
)

SUMMARY = example_payload()
FLEET = simulate_fleet(np.random.default_rng(0))
NUMBERS = SUMMARY.fleet
CLOSED = SUMMARY.closed_forms
NORMAL, HARSH = SUMMARY.sites
FAILED, CLEAN = SUMMARY.history


def _website_fleet() -> tuple[NDArray[Any], NDArray[np.float64], NDArray[np.float64], NDArray[Any]]:
    """Transcribe the website's draws: sites, frailties, rates, windows, then the failures."""
    r = np.random.default_rng(0)
    n = 400
    env = r.choice(["normal", "harsh"], n, p=[0.6, 0.4])
    frailty = r.gamma(2.0, 0.5, n)
    rate = 0.8 * np.where(env == "harsh", 2.0, 1.0) * frailty
    observed = r.uniform(1.0, 3.0, n)
    events = []
    for i in range(n):
        t = 0.0
        while True:
            t += r.exponential(1 / rate[i])
            if t > observed[i]:
                break
            events.append((i, t))
    return env, rate, observed, np.array(events)


ENV, RATE, OBSERVED, EVENTS = _website_fleet()


def _website_mcf(mask: NDArray[np.bool_], grid: NDArray[np.float64]) -> NDArray[np.float64]:
    ev = EVENTS[np.isin(EVENTS[:, 0], np.where(mask)[0])][:, 1]
    obs = OBSERVED[mask]
    ev = np.sort(ev)
    inc = 1.0 / np.array([np.sum(obs >= a) for a in ev])
    return np.array([inc[ev <= g].sum() for g in grid])


def _website_naive(mask: NDArray[np.bool_], grid: NDArray[np.float64]) -> NDArray[np.float64]:
    idx = np.where(mask)[0]
    return np.array(
        [np.mean([np.sum((EVENTS[:, 0] == i) & (EVENTS[:, 1] <= g)) for i in idx]) for g in grid]
    )


def test_the_fleet_is_the_websites_and_the_articles() -> None:
    """The same draws in the same order, including the article's gamma(1 / 0.5, 0.5)."""
    assert np.array_equal(FLEET.site == 1, ENV == "harsh")
    assert np.array_equal(FLEET.rate, RATE)
    assert np.array_equal(FLEET.observed, OBSERVED)
    assert np.array_equal(FLEET.machine, EVENTS[:, 0].astype(np.int64))
    assert np.array_equal(FLEET.age, EVENTS[:, 1])
    rng = np.random.default_rng(0)
    rng.choice(["normal", "harsh"], 400, p=[0.6, 0.4])
    assert np.array_equal(FLEET.frailty, rng.gamma(1 / 0.5, 0.5, 400))


def test_the_figure_curves_are_the_websites() -> None:
    """All machines, each site and the naive count on the figure's 60 ages, bit for bit."""
    curves = SUMMARY.curves
    grid = np.linspace(0.05, 3.0, 60)
    assert len(curves.ages) == GRID_POINTS
    assert np.array_equal(curves.ages, grid)
    everyone = np.ones(400, dtype=bool)
    assert np.array_equal(curves.all_machines, _website_mcf(everyone, grid))
    assert np.array_equal(curves.harsh, _website_mcf(ENV == "harsh", grid))
    assert np.array_equal(curves.normal, _website_mcf(ENV == "normal", grid))
    assert np.array_equal(curves.naive, _website_naive(everyone, grid))


def test_the_record_matches_the_article() -> None:
    """880 failures, 800 machine-years, a quarter never failing, 299 used, 0.80 against 1.10."""
    assert NUMBERS.failures == 880
    assert round(NUMBERS.machine_years) == 800
    assert round(100 * NUMBERS.never_failed_share) == 25
    assert (NUMBERS.first_failures, round(100 * NUMBERS.first_failure_share)) == (299, 34)
    assert round(NUMBERS.first_failure_rate, 2) == 0.80
    assert round(NUMBERS.all_events_rate, 2) == 1.10
    assert round(NUMBERS.true_rate, 2) == 1.09
    # 581 discarded, two thirds of the record; the first-failure rate is a quarter too low.
    assert NUMBERS.failures - NUMBERS.first_failures == 581
    assert round(3 * (1 - NUMBERS.first_failure_share)) == 2
    assert round(1 - NUMBERS.first_failure_rate / NUMBERS.all_events_rate, 2) == 0.27


def test_the_mcf_table_matches_the_article() -> None:
    """Mean cumulative function, true rate times age, and naive count at six ages."""
    printed = {
        "mcf": ("0.56", "1.11", "1.59", "2.21", "2.73", "3.29"),
        "true": ("0.54", "1.09", "1.63", "2.18", "2.72", "3.26"),
        "naive": ("0.56", "1.11", "1.52", "1.90", "2.10", "2.20"),
    }
    table = SUMMARY.table
    assert table.ages == TABLE_AGES
    for name, row in printed.items():
        assert tuple(f"{value:.2f}" for value in getattr(table, name)) == row
    # The article's estimator sums in record order; it agrees to rounding error.
    events = EVENTS[:, 1]
    article = [sum(1.0 / np.sum(a <= OBSERVED) for a in events[events <= g]) for g in TABLE_AGES]
    assert table.mcf == pytest.approx(article, abs=1e-12)
    # A third too small at three years: 2.2 against 3.3.
    assert round(1 - table.naive[-1] / table.mcf[-1], 2) == 0.33


def test_the_site_table_matches_the_article() -> None:
    """0.75, 0.60 and 0.77 at normal sites; 1.55, 1.18 and 1.50 at harsh ones."""
    assert (NORMAL.site, HARSH.site) == SITES
    shown = [
        (f"{row.all_events_rate:.2f}", f"{row.first_failure_rate:.2f}", f"{row.true_rate:.2f}")
        for row in (NORMAL, HARSH)
    ]
    assert shown == [("0.75", "0.60", "0.77"), ("1.55", "1.18", "1.50")]
    assert round(HARSH.all_events_rate / NORMAL.all_events_rate, 2) == 2.06
    assert round(HARSH.first_failure_rate / NORMAL.first_failure_rate, 2) == 1.97
    # The printed 2.00 is the design ratio; the true rates in the table give 1.94.
    assert round(HARSH.true_rate / NORMAL.true_rate, 2) == 1.94
    # First-failure budgets are 20 and 24 percent short, not a quarter at both.
    shortfall = [1 - row.first_failure_rate / row.all_events_rate for row in (NORMAL, HARSH)]
    assert [round(100 * value) for value in shortfall] == [20, 24]
    assert (NORMAL.machines, HARSH.machines) == (225, 175)


def test_the_dispersion_matches_the_article() -> None:
    """Counts with mean 2.2 and variance 5.7; the top tenth has 34 percent of failures."""
    dispersion = SUMMARY.dispersion
    assert (round(dispersion.mean_count, 1), round(dispersion.variance, 1)) == (2.2, 5.7)
    assert dispersion.variance > 2 * dispersion.mean_count
    assert round(100 * dispersion.top_decile_share) == 34
    # The article picks the top forty with an unstable argsort; ties cannot change the share.
    counts = failure_counts(FLEET)
    top = np.argsort(-counts)[:40]
    assert dispersion.top_decile_share == counts[top].sum() / counts.sum()
    assert top_share(counts) == top_share(counts[::-1])


def test_the_no_frailty_comparison() -> None:
    """Without frailty the top tenth has 27 percent on average; the article's draw is 29."""
    rng = np.random.default_rng(99)
    means = 0.8 * np.where(FLEET.site == 1, 2.0, 1.0) * FLEET.observed
    shares = np.array([top_share(rng.poisson(means)) for _ in range(2000)])
    centre, spread = float(shares.mean()), float(shares.std())
    assert (round(100 * centre), round(100 * spread, 1)) == (27, 0.9)
    # The article's single draw (seed 1) is about two standard deviations high.
    assert 1.5 * spread < 0.29 - centre < 3 * spread
    assert SUMMARY.dispersion.top_decile_share > np.quantile(shares, 0.99)


def test_the_history_table_matches_the_article() -> None:
    """107 machines that failed in year one fail 1.46 times in year two; 92 that did not, 0.68."""
    assert (FAILED.failed_in_year_one, FAILED.machines, round(FAILED.year_two_rate, 2)) == (
        True,
        107,
        1.46,
    )
    assert (CLEAN.failed_in_year_one, CLEAN.machines, round(CLEAN.year_two_rate, 2)) == (
        False,
        92,
        0.68,
    )
    assert FAILED.year_two_rate > 2 * CLEAN.year_two_rate
    assert year_two_history(FLEET) == (FAILED, CLEAN)


def test_the_closed_forms_match_quadrature() -> None:
    """No failure and first-failure exposure against direct integration, for three frailties."""
    for variance in (1.0, 0.5, 0.25):
        forms = closed_forms(frailty_variance=variance)
        shape = 1 / variance

        def survival(t: float, lam: float, s: float = shape, v: float = variance) -> float:
            return float((1 + v * lam * t) ** -s)

        never = exposure = 0.0
        for weight, lam in ((0.6, 0.8), (0.4, 1.6)):
            never += weight * integrate.quad(lambda t, m=lam: survival(t, m), 1, 3)[0] / 2
            inner = integrate.dblquad(lambda t, big_t, m=lam: survival(t, m), 1, 3, 0, lambda x: x)
            exposure += weight * inner[0] / 2
        assert forms.never_failed_share == pytest.approx(never, rel=1e-8)
        assert forms.first_failure_rate == pytest.approx((1 - never) / exposure, rel=1e-8)
        assert forms.fleet_rate == pytest.approx(1.12)
        assert forms.count_mean == pytest.approx(2.24)


def test_the_closed_forms_match_a_large_simulation() -> None:
    """Counts, first failures and the year-two update for 400,000 machines, vectorised."""
    rng = np.random.default_rng(5)
    n = 400_000
    lam = 0.8 * np.where(rng.random(n) < 0.4, 2.0, 1.0)
    frailty = rng.gamma(2.0, 0.5, n)
    window = rng.uniform(1.0, 3.0, n)
    counts = rng.poisson(lam * frailty * window)
    first = rng.exponential(1 / (lam * frailty))
    year_one = rng.poisson(lam * frailty)
    year_two = rng.poisson(lam * frailty)

    def near(simulated: float, exact: float, sd: float) -> bool:
        return bool(abs(simulated - exact) < 4 * sd / np.sqrt(n))

    assert near(float(np.mean(counts == 0)), CLOSED.never_failed_share, 0.44)
    assert near(float(counts.mean()), CLOSED.count_mean, float(counts.std()))
    assert float(counts.var()) == pytest.approx(CLOSED.count_variance, rel=0.03)
    rate = np.mean(first <= window) / np.mean(np.minimum(first, window))
    assert rate == pytest.approx(CLOSED.first_failure_rate, rel=0.01)
    failed = year_one > 0
    assert float(year_two[failed].mean()) == pytest.approx(CLOSED.year_two_after_failure, rel=0.01)
    assert float(year_two[~failed].mean()) == pytest.approx(CLOSED.year_two_after_none, rel=0.01)


def test_the_fleet_lies_near_its_population_values() -> None:
    """Never failing, the rates, and the year-two update, within sampling noise."""
    p = CLOSED.never_failed_share
    assert abs(NUMBERS.never_failed_share - p) < 3 * np.sqrt(p * (1 - p) / 400)
    assert round(CLOSED.fleet_rate, 2) == 1.12
    assert NUMBERS.all_events_rate == pytest.approx(CLOSED.fleet_rate, rel=0.1)
    assert NUMBERS.first_failure_rate == pytest.approx(CLOSED.first_failure_rate, rel=0.1)
    assert FAILED.year_two_rate == pytest.approx(CLOSED.year_two_after_failure, abs=0.2)
    assert CLEAN.year_two_rate == pytest.approx(CLOSED.year_two_after_none, abs=0.2)
    # The population variance is 6.37, more than twice the mean as well.
    assert round(CLOSED.count_variance, 2) == 6.37
    assert CLOSED.count_variance > 2 * CLOSED.count_mean


def test_the_figure_claims() -> None:
    """The naive count flattens after two years; the risk-set estimate climbs at the true rate."""
    curves = SUMMARY.curves
    ages = np.array(curves.ages)
    mcf = np.array(curves.all_machines)
    naive = np.array(curves.naive)
    # Identical while every machine is observed, and never above the MCF.
    assert np.max(np.abs(mcf - naive)[ages <= 1.0]) < 1e-12
    assert np.all(naive <= mcf + 1e-12)
    two = int(np.argmin(np.abs(ages - 2.0)))
    assert ages[two] == 2.0
    naive_rise, mcf_rise = naive[-1] - naive[two], mcf[-1] - mcf[two]
    assert (round(naive_rise, 2), round(mcf_rise, 2)) == (0.30, 1.08)
    assert mcf_rise == pytest.approx(NUMBERS.true_rate, rel=0.01)
    assert np.max(np.abs(mcf - NUMBERS.true_rate * ages)) < 0.12
    # Harsh sites accumulate about twice as many failures as normal ones.
    assert curves.harsh[-1] / curves.normal[-1] == pytest.approx(2.0, abs=0.05)
    # The naive count's expectation: the rate times the mean of min(age, window).
    for age, value in zip(ages, naive, strict=True):
        exposure = age if age <= 1 else (3 * age - age**2 / 2 - 0.5) / 2
        assert value == pytest.approx(NUMBERS.true_rate * exposure, abs=0.12)


def test_by_hand() -> None:
    """Three machines observed 1, 2 and 3 years, with four failures."""
    fleet = Fleet(
        site=np.array([0, 0, 1]),
        frailty=np.ones(3),
        rate=np.ones(3),
        observed=np.array([1.0, 2.0, 3.0]),
        machine=np.array([0, 1, 2, 2]),
        age=np.array([0.5, 1.5, 1.2, 2.5]),
    )
    ages = [0.4, 1.0, 1.3, 3.0]
    # Risk sets of 3, 2, 2 and 1 at ages 0.5, 1.2, 1.5 and 2.5.
    mcf = mean_cumulative_function(fleet.age, fleet.observed, ages)
    assert mcf == pytest.approx([0.0, 1 / 3, 1 / 3 + 1 / 2, 1 / 3 + 1 / 2 + 1 / 2 + 1])
    naive = naive_average_count(fleet.machine, fleet.age, [0, 1, 2], ages)
    assert naive.tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 4 / 3])
    failed, exposure = first_failures(fleet)
    assert failed.tolist() == [True, True, True]
    assert exposure.tolist() == [0.5, 1.5, 1.2]
    assert first_failure_rate(fleet) == pytest.approx(3 / 3.2)
    assert all_events_rate(fleet) == pytest.approx(4 / 6)
    assert all_events_rate(fleet, [False, False, True]) == pytest.approx(2 / 3)
    assert failure_counts(fleet).tolist() == [1, 1, 2]
    assert top_share([5, 0, 0, 0, 0, 0, 0, 0, 0, 5], 0.1) == 0.5


def test_limiting_cases() -> None:
    """A vanishing frailty leaves Poisson counts; no second stage means no history effect."""
    forms = closed_forms(frailty_variance=1e-7)
    windows = np.linspace(1, 3, 200_001)
    poisson_never = sum(
        w * float(np.trapezoid(np.exp(-lam * windows), windows)) / 2
        for w, lam in ((0.6, 0.8), (0.4, 1.6))
    )
    assert forms.never_failed_share == pytest.approx(poisson_never, rel=1e-5)
    mean_t2 = 13 / 3
    between = (0.6 * 0.64 + 0.4 * 2.56) * mean_t2 - 2.24**2
    assert forms.count_variance == pytest.approx(2.24 + between, rel=1e-5)
    # Within a site the year-two rate no longer depends on year one.
    one_site = closed_forms(frailty_variance=1e-7, harsh_share=0.0)
    assert one_site.year_two_after_failure == pytest.approx(0.8, rel=1e-5)
    assert one_site.year_two_after_none == pytest.approx(0.8, rel=1e-5)
    # Without frailty the first-failure rate is the all-events rate.
    assert one_site.first_failure_rate == pytest.approx(0.8, rel=1e-5)


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a fleet; another seed does not."""
    first = simulate_fleet(np.random.default_rng(4), 30)
    again = simulate_fleet(np.random.default_rng(4), 30)
    other = simulate_fleet(np.random.default_rng(5), 30)
    assert np.array_equal(first.age, again.age)
    assert not np.array_equal(first.age, other.age)


def test_invalid_inputs_are_rejected() -> None:
    """Bad fleets, windows, groups and shares are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_fleet(rng, 0)
    with pytest.raises(ValueError):
        simulate_fleet(rng, 10, min_follow_up=3.0, max_follow_up=1.0)
    with pytest.raises(ValueError):
        simulate_fleet(rng, 10, harsh_share=1.5)
    with pytest.raises(ValueError):
        simulate_fleet(rng, 10, frailty_variance=0.0)
    with pytest.raises(TypeError):
        simulate_fleet(rng, 10.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mean_cumulative_function([2.0], [1.0], [1.0])
    with pytest.raises(ValueError):
        mean_cumulative_function([0.5], [], [1.0])
    with pytest.raises(ValueError):
        naive_average_count([0], [0.5], [], [1.0])
    with pytest.raises(ValueError):
        all_events_rate(FLEET, np.zeros(400, dtype=bool))
    with pytest.raises(ValueError):
        top_share([0, 0, 0])
    with pytest.raises(ValueError):
        top_share([1, 2], 1.0)
    with pytest.raises(ValueError):
        closed_forms(max_follow_up=1.5)
