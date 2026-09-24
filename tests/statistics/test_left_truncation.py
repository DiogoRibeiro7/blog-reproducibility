"""Check the left-truncation model against the article's code, closed forms and its numbers.

The figure and the article share one generator seeded at 0, so the fleet, the
cohort, the three curves and every number the article prints are reproduced and
pinned: 7,644 units in service, 2,940 failures, medians of 7.74, 3.83 and 3.95
years, the site table and the failure rates, including those of the new units
the article draws afterwards from the same generator. Transcriptions of the
website's draws and Kaplan-Meier loop, and of the article's scalar lifetime
draws and its loop, confirm the vectorised versions bit for bit. The snapshot's
closed forms (incomplete gamma functions) agree with quadrature and with the
simulation to within three standard errors.

Some statements do not hold as written, and the tests pin what is true:

* the left-truncated median "recovers the reference to within a tenth of a
  year": it is 3.83 against 3.95, 0.12 apart;
* "the left-truncated estimates put both medians within a tenth of a year of
  the truth": normal sites are 0.09 away, harsh sites 0.14 (3.00 against 3.13);
* "the harsh sites are inflated proportionally more": the naive median is 1.88
  times the truth at normal sites and 1.79 times at harsh ones, so harsh sites
  are inflated less, in proportion and in years;
* the true median "is 3.9 years": the simulated reference is 3.95 and the
  fleet's population median 3.96, both 4.0 to one decimal.

The figure's own claims hold: counting survivors from age zero multiplies the
median by 1.96, and the left-truncated curve stays within 0.026 of the reference
at every age while the naive one is up to 0.36 above it.
"""

from math import exp, inf

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate

from blog_reproducibility.statistics.left_truncation import (
    FOLLOW_UP,
    INSTALLATION_YEARS,
    SITE_PROBABILITIES,
    SITE_SCALES,
    WEIBULL_SHAPE,
    closed_forms,
    empirical_survival,
    example_payload,
    failure_rate_from_installation,
    fleet_median,
    kaplan_meier,
    km_median,
    restricted_mean_life,
    simulate_fleet,
    simulate_new_units,
    study_cohort,
    survival_on_grid,
    weibull_median,
    weibull_survival,
)

SUMMARY = example_payload()
SNAPSHOT = SUMMARY.snapshot
MEDIANS = SUMMARY.medians
NORMAL, HARSH = SUMMARY.sites
CLOSED = SUMMARY.closed_forms
_RNG = np.random.default_rng(0)
FLEET = simulate_fleet(_RNG)
FRESH_SITE, FRESH_LIFE = simulate_new_units(_RNG)
COHORT = study_cohort(FLEET)
GRID = np.linspace(0, 10, 201)


def _article_km(
    times: NDArray[np.float64], events: NDArray[np.int64], entry: NDArray[np.float64] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transcribe the article's Kaplan-Meier loop."""
    order = np.argsort(times)
    times, events = times[order], events[order]
    entry = np.zeros_like(times) if entry is None else entry[order]
    grid = np.unique(times[events == 1])
    surv: list[float] = []
    s = 1.0
    for t in grid:
        at_risk = np.sum((entry < t) & (times >= t))
        d = np.sum((times == t) & (events == 1))
        s *= float(1 - d / at_risk)
        surv.append(s)
    return grid, np.array(surv)


def _website_km_on_grid(
    times: NDArray[np.float64],
    events: NDArray[np.bool_],
    entry_times: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Transcribe the website generator's Kaplan-Meier curve on the figure's grid."""
    entry_times = np.zeros_like(times) if entry_times is None else entry_times
    order = np.argsort(times)
    t, e, en = times[order], events[order], entry_times[order]
    ev_t = np.unique(t[e])
    s = 1.0
    surv_at: dict[float, float] = {}
    for tt in ev_t:
        at_risk = np.sum((en < tt) & (t >= tt))
        d = np.sum((t == tt) & e)
        if at_risk > 0:
            s *= float(1 - d / at_risk)
        surv_at[float(tt)] = s
    keys = np.array(list(surv_at.keys()))
    vals = np.array(list(surv_at.values()))
    idx = np.searchsorted(keys, GRID, side="right") - 1
    return np.where(idx >= 0, vals[np.clip(idx, 0, None)], 1.0)


def _mixture_survival(age: float) -> float:
    return sum(
        w * weibull_survival(age, s) for w, s in zip(SITE_PROBABILITIES, SITE_SCALES, strict=True)
    )


def _mixture_density(age: float) -> float:
    k = WEIBULL_SHAPE
    return sum(
        w * k / s * float((age / s) ** (k - 1)) * weibull_survival(age, s)
        for w, s in zip(SITE_PROBABILITIES, SITE_SCALES, strict=True)
    )


def _within(simulated: float, exact: float, standard_error: float) -> bool:
    return abs(simulated - exact) < 3 * standard_error


def test_the_draws_are_the_websites_and_the_articles() -> None:
    """Same generator, same order: the figure's arrays and the article's scalar draws."""
    r = np.random.default_rng(0)
    env = r.choice([6.0, 4.0], 20000, p=[0.6, 0.4])
    install = r.uniform(0, 12.0, 20000)
    life = r.weibull(1.5, 20000) * env
    assert np.array_equal(FLEET.scale, env)
    assert np.array_equal(FLEET.install, install)
    assert np.array_equal(FLEET.life, life)

    rng = np.random.default_rng(0)
    scale = {"normal": 6.0, "harsh": 4.0}
    names = rng.choice(["normal", "harsh"], 20000, p=[0.6, 0.4])
    rng.uniform(0, 12.0, 20000)
    article_life = np.array([rng.weibull(1.5) * scale[e] for e in names])
    assert np.array_equal(article_life, FLEET.life)
    fresh = rng.choice(["normal", "harsh"], 20000, p=[0.6, 0.4])
    fresh_life = np.array([rng.weibull(1.5) * scale[e] for e in fresh])

    ours = np.random.default_rng(0)
    simulate_fleet(ours)
    site, new_life = simulate_new_units(ours)
    assert np.array_equal(site == 1, fresh == "harsh")
    assert np.array_equal(new_life, fresh_life)

    in_service = install + life > 12.0
    entry = (12.0 - install)[in_service]
    assert np.array_equal(COHORT.entry, entry)
    assert np.array_equal(COHORT.exit, np.minimum(life[in_service], entry + 2.0))
    assert np.array_equal(COHORT.event, life[in_service] <= entry + 2.0)


def test_the_figure_curves_are_the_websites() -> None:
    """The three curves on the grid, bit for bit."""
    curves = SUMMARY.curves
    assert np.array_equal(curves.ages, GRID)
    assert np.array_equal(curves.reference, [np.mean(FLEET.life > g) for g in GRID])
    assert np.array_equal(curves.naive, _website_km_on_grid(COHORT.exit, COHORT.event))
    assert np.array_equal(
        curves.truncated, _website_km_on_grid(COHORT.exit, COHORT.event, COHORT.entry)
    )


def test_kaplan_meier_is_the_articles_loop() -> None:
    """Delayed entry, the snapshot clock and a complete sample, bit for bit."""
    events = COHORT.event.astype(int)
    for times, entry in (
        (COHORT.exit, COHORT.entry),
        (COHORT.exit - COHORT.entry, None),
    ):
        ours = kaplan_meier(times, events, entry)
        theirs = _article_km(times, events, entry)
        assert np.array_equal(ours[0], theirs[0])
        assert np.array_equal(ours[1], theirs[1])
    lives = FLEET.life[:4000]
    ours = kaplan_meier(lives, np.ones(4000, dtype=int))
    theirs = _article_km(lives, np.ones(4000, dtype=int))
    assert np.array_equal(ours[1], theirs[1])


def test_the_article_snapshot_numbers() -> None:
    """7,644 of 20,000 in service (38 percent); harsh sites 40 percent installed, 32 surviving."""
    assert SNAPSHOT.installed == 20_000
    assert SNAPSHOT.in_service == 7644
    assert round(100 * SNAPSHOT.in_service_share) == 38
    assert round(100 * SNAPSHOT.harsh_share_installed) == 40
    assert round(100 * SNAPSHOT.harsh_share_in_service) == 32
    assert round(SNAPSHOT.mean_age_in_service, 1) == 3.4
    assert round(SNAPSHOT.survivors_mean_life, 1) == 6.9
    assert round(SNAPSHOT.fleet_mean_life, 1) == 4.7
    assert SNAPSHOT.failures == 2940
    # "Half as long again", and "a third of the survivors are older than four years".
    assert SNAPSHOT.survivors_mean_life / SNAPSHOT.fleet_mean_life == pytest.approx(1.5, abs=0.05)
    assert SNAPSHOT.share_older_than_four == pytest.approx(1 / 3, abs=0.01)


def test_the_article_medians() -> None:
    """Undefined on the snapshot clock, 7.74 from age zero, 3.83 truncated, 3.95 in truth."""
    assert MEDIANS.snapshot_clock is None
    assert MEDIANS.naive is not None and round(MEDIANS.naive, 2) == 7.74
    assert MEDIANS.truncated is not None and round(MEDIANS.truncated, 2) == 3.83
    assert MEDIANS.reference is not None and round(MEDIANS.reference, 2) == 3.95
    # Fewer than half fail in two years on the snapshot clock.
    assert SNAPSHOT.failures < SNAPSHOT.in_service / 2
    # Not within a tenth of a year, as the article says, but 0.12.
    assert round(MEDIANS.reference - MEDIANS.truncated, 2) == 0.12
    # "3.9" is 3.95 truncated; to one decimal the reference and the population median are 4.0.
    assert round(MEDIANS.reference, 1) == round(CLOSED.fleet_median, 1) == 4.0


def test_the_article_site_table() -> None:
    """True medians 4.70 and 3.13, naive 8.85 and 5.60, left-truncated 4.61 and 3.00."""
    assert (NORMAL.site, HARSH.site) == ("normal", "harsh")
    assert (round(NORMAL.true_median, 2), round(HARSH.true_median, 2)) == (4.70, 3.13)
    assert NORMAL.naive_median is not None and HARSH.naive_median is not None
    assert NORMAL.truncated_median is not None and HARSH.truncated_median is not None
    assert (round(NORMAL.naive_median, 2), round(HARSH.naive_median, 2)) == (8.85, 5.60)
    assert (round(NORMAL.truncated_median, 2), round(HARSH.truncated_median, 2)) == (4.61, 3.00)
    # The gap between sites: 1.6 years in truth, 3.24 on the naive curves, twice its size.
    # The article's 3.3 is the gap between the rounded entries, 8.85 - 5.60 = 3.25.
    true_gap = NORMAL.true_median - HARSH.true_median
    naive_gap = NORMAL.naive_median - HARSH.naive_median
    assert (round(true_gap, 1), round(naive_gap, 2)) == (1.6, 3.24)
    assert naive_gap / true_gap == pytest.approx(2.0, abs=0.1)
    # Harsh sites are inflated less than normal ones, in proportion and in years.
    assert round(NORMAL.naive_median / NORMAL.true_median, 2) == 1.88
    assert round(HARSH.naive_median / HARSH.true_median, 2) == 1.79
    assert HARSH.naive_median - HARSH.true_median < NORMAL.naive_median - NORMAL.true_median
    # Normal sites are within a tenth of a year of the truth; harsh sites are not.
    assert NORMAL.true_median - NORMAL.truncated_median < 0.1
    assert round(HARSH.true_median - HARSH.truncated_median, 2) == 0.14


def test_the_article_failure_rates() -> None:
    """38.5 against 22.4 percent within two years; per unit-year 0.202 and 0.334 against half."""
    assert round(100 * SNAPSHOT.survivor_failure_probability, 1) == 38.5
    assert round(100 * SNAPSHOT.new_unit_failure_probability, 1) == 22.4
    assert (round(NORMAL.survivor_rate, 3), round(HARSH.survivor_rate, 3)) == (0.202, 0.334)
    assert (round(NORMAL.new_unit_rate, 3), round(HARSH.new_unit_rate, 3)) == (0.098, 0.167)
    for row in (NORMAL, HARSH):
        assert row.survivor_rate / row.new_unit_rate == pytest.approx(2.0, abs=0.1)


def test_the_closed_forms_agree_with_the_simulation() -> None:
    """Every snapshot quantity lies within three standard errors of its population value."""
    n, survivors = FLEET.life.size, COHORT.entry.size
    in_service = FLEET.install + FLEET.life > INSTALLATION_YEARS

    def binomial(p: float, size: int) -> float:
        return float(np.sqrt(p * (1 - p) / size))

    p = CLOSED.in_service_share
    assert _within(SNAPSHOT.in_service_share, p, binomial(p, n))
    p = CLOSED.harsh_share_in_service
    assert _within(SNAPSHOT.harsh_share_in_service, p, binomial(p, survivors))
    p = CLOSED.share_older_than_four
    assert _within(SNAPSHOT.share_older_than_four, p, binomial(p, survivors))
    p = CLOSED.survivor_failure_probability
    assert _within(SNAPSHOT.survivor_failure_probability, p, binomial(p, survivors))
    se = float(COHORT.entry.std()) / np.sqrt(survivors)
    assert _within(SNAPSHOT.mean_age_in_service, CLOSED.mean_age_in_service, se)
    se = float(FLEET.life[in_service].std()) / np.sqrt(survivors)
    assert _within(SNAPSHOT.survivors_mean_life, CLOSED.survivors_mean_life, se)
    se = float(FLEET.life.std()) / np.sqrt(n)
    assert _within(SNAPSHOT.fleet_mean_life, CLOSED.fleet_mean_life, se)
    assert MEDIANS.reference is not None
    se = 0.5 / np.sqrt(n) / _mixture_density(CLOSED.fleet_median)
    assert _within(MEDIANS.reference, CLOSED.fleet_median, se)
    # New units: failures over exposure, with Poisson noise in the failure count.
    site, life = FRESH_SITE, FRESH_LIFE
    for index, row in enumerate((NORMAL, HARSH)):
        failures = int(np.count_nonzero(life[site == index] <= FOLLOW_UP))
        se = row.new_unit_rate / np.sqrt(failures)
        assert _within(row.new_unit_rate, row.exact_new_unit_rate, se)


def test_the_closed_forms_match_quadrature() -> None:
    """The incomplete gamma expressions against direct integration of the defining integrals."""
    horizon, window = INSTALLATION_YEARS, FOLLOW_UP

    def quad(f: object, a: float, b: float) -> float:
        value, _ = integrate.quad(f, a, b, epsabs=1e-13, epsrel=1e-12, limit=200)
        return float(value)

    alive = quad(_mixture_survival, 0, horizon)
    assert CLOSED.in_service_share == pytest.approx(alive / horizon, rel=1e-9)
    harsh = 0.4 * quad(lambda a: weibull_survival(a, 4.0), 0, horizon)
    assert CLOSED.harsh_share_in_service == pytest.approx(harsh / alive, rel=1e-9)
    age = quad(lambda a: a * _mixture_survival(a), 0, horizon)
    assert CLOSED.mean_age_in_service == pytest.approx(age / alive, rel=1e-9)
    older = quad(_mixture_survival, 4.0, horizon)
    assert CLOSED.share_older_than_four == pytest.approx(older / alive, rel=1e-9)
    failing = quad(lambda a: _mixture_survival(a) - _mixture_survival(a + window), 0, horizon)
    assert CLOSED.survivor_failure_probability == pytest.approx(failing / alive, rel=1e-9)

    def life_beyond(a: float) -> float:
        """E[L; L > a] = a S(a) plus the integral of S beyond a."""
        return a * _mixture_survival(a) + quad(_mixture_survival, a, inf)

    lives = quad(life_beyond, 0, horizon)
    assert CLOSED.survivors_mean_life == pytest.approx(lives / alive, rel=1e-8)
    assert CLOSED.fleet_mean_life == pytest.approx(quad(_mixture_survival, 0, inf), rel=1e-9)
    assert _mixture_survival(CLOSED.fleet_median) == pytest.approx(0.5, abs=1e-12)
    for scale in SITE_SCALES:
        exposure = quad(lambda t, s=scale: weibull_survival(t, s), 0, window)
        assert restricted_mean_life(window, scale) == pytest.approx(exposure, rel=1e-9)
        rate = (1 - weibull_survival(window, scale)) / exposure
        assert failure_rate_from_installation(window, scale) == pytest.approx(rate, rel=1e-9)


def test_the_figure_claims() -> None:
    """Counting from age zero doubles the median; the truncated curve lies on the reference."""
    assert MEDIANS.naive is not None and MEDIANS.reference is not None
    assert MEDIANS.naive / MEDIANS.reference == pytest.approx(2.0, abs=0.05)
    curves = SUMMARY.curves
    reference = np.array(curves.reference)
    truncated_gap = np.abs(np.array(curves.truncated) - reference)
    naive_gap = np.array(curves.naive) - reference
    assert float(truncated_gap.max()) < 0.026
    assert float(naive_gap.min()) >= 0.0
    assert float(naive_gap.max()) > 0.35
    assert float(naive_gap.max()) > 10 * float(truncated_gap.max())


def test_kaplan_meier_by_hand() -> None:
    """Four units, one entering late: delayed entry changes only the risk sets."""
    entry = np.array([0.0, 1.0, 0.0, 2.5])
    exit_ = np.array([2.0, 3.0, 4.0, 5.0])
    event = np.array([True, True, False, True])
    times, survival = kaplan_meier(exit_, event, entry)
    assert times.tolist() == [2.0, 3.0, 5.0]
    assert survival == pytest.approx([2 / 3, 4 / 9, 0.0])
    _, naive = kaplan_meier(exit_, event)
    assert naive == pytest.approx([3 / 4, 1 / 2, 0.0])
    assert km_median(times, survival) == 3.0
    assert km_median(times[:1], survival[:1]) is None
    on_grid = survival_on_grid(times, survival, [0.0, 1.99, 2.0, 4.0, 6.0])
    assert on_grid == pytest.approx([1.0, 1.0, 2 / 3, 4 / 9, 0.0])
    assert survival_on_grid([], [], [1.0, 2.0]).tolist() == [1.0, 1.0]


def test_a_complete_sample_gives_the_empirical_survival() -> None:
    """Without censoring or truncation, Kaplan-Meier is one minus the empirical distribution."""
    lives = np.random.default_rng(3).weibull(1.5, 500) * 5.0
    times, survival = kaplan_meier(lives, np.ones(500, dtype=bool))
    assert survival == pytest.approx(1 - np.arange(1, 501) / 500, abs=1e-12)
    ages = np.linspace(0, 12, 97)
    assert survival_on_grid(times, survival, ages) == pytest.approx(
        empirical_survival(lives, ages), abs=1e-12
    )


def test_limiting_cases() -> None:
    """Exponential lifetimes fail at a constant rate; a single site has the Weibull median."""
    assert failure_rate_from_installation(2.0, 5.0, 1.0) == pytest.approx(1 / 5)
    assert failure_rate_from_installation(0.5, 5.0, 1.0) == pytest.approx(1 / 5)
    assert weibull_survival(weibull_median(6.0), 6.0) == pytest.approx(0.5)
    assert fleet_median((6.0,), (1.0,)) == pytest.approx(weibull_median(6.0))
    assert restricted_mean_life(500.0, 6.0) == pytest.approx(6.0 * 0.9027452929509336)
    assert restricted_mean_life(0.0, 6.0) == 0.0
    assert weibull_survival(0.0, 4.0) == 1.0
    assert weibull_survival(4.0, 4.0) == pytest.approx(exp(-1))
    # A snapshot that follows everything installed long ago: almost none remain in service.
    old = closed_forms(years=200.0, old_age=4.0)
    assert old.in_service_share < 0.03
    one_site = closed_forms(scales=(5.0,), probabilities=(1.0,), shape=1.0, years=1e-3, old_age=0.0)
    assert one_site.in_service_share == pytest.approx(1.0, abs=1e-3)


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a fleet; another seed does not."""
    first = simulate_fleet(np.random.default_rng(4), 50)
    again = simulate_fleet(np.random.default_rng(4), 50)
    other = simulate_fleet(np.random.default_rng(5), 50)
    assert np.array_equal(first.life, again.life)
    assert not np.array_equal(first.life, other.life)


def test_invalid_inputs_are_rejected() -> None:
    """Bad mixtures, parameters and survival data are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_fleet(rng, 10, scales=(6.0, 4.0), probabilities=(0.6, 0.3))
    with pytest.raises(ValueError):
        simulate_fleet(rng, 10, scales=(6.0,), probabilities=(0.6, 0.4))
    with pytest.raises(ValueError):
        simulate_fleet(rng, 0)
    with pytest.raises(ValueError):
        simulate_new_units(rng, 10, scales=())
    with pytest.raises(ValueError):
        weibull_survival(-1.0, 6.0)
    with pytest.raises(ValueError):
        weibull_median(0.0)
    with pytest.raises(ValueError):
        failure_rate_from_installation(0.0, 6.0)
    with pytest.raises(ValueError):
        closed_forms(old_age=20.0)
    with pytest.raises(ValueError):
        kaplan_meier([1.0, 2.0], [1, 2])
    with pytest.raises(ValueError):
        kaplan_meier([1.0, 2.0], [True])
    with pytest.raises(ValueError):
        kaplan_meier([1.0, 2.0], [True, True], [0.5, 2.0])
    with pytest.raises(ValueError):
        kaplan_meier([], [])
    with pytest.raises(ValueError):
        kaplan_meier([1.0, np.nan], [True, True])
    with pytest.raises(ValueError):
        km_median([1.0, 2.0], [0.4])
    with pytest.raises(ValueError):
        survival_on_grid([1.0], [0.5, 0.2], [1.0])
    with pytest.raises(ValueError):
        empirical_survival([], [1.0])
    with pytest.raises(TypeError):
        restricted_mean_life(True, 6.0)
