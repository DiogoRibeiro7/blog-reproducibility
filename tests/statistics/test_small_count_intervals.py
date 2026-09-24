"""Check small-count intervals against SciPy, exact coverage, the website's draws and the article.

The Wilson and Clopper-Pearson intervals are compared with SciPy's
``binomtest`` intervals, and the Wald interval with a hand-worked case. Coverage
is computed exactly as a binomial sum, checked against the simulation, and used
to confirm that the exact interval never covers less than 95 percent. The
figure's simulation (seed 0, 4,000 counts at each size) is pinned; it is the
website loop's, draw for draw, which a transcription of that loop confirms, and
it reproduces the published figure. Every closed-form table in the article is
pinned at its printed precision, and all of them match.

The article's coverage table comes from its own generator and design (10,000
samples at six other settings) and is not pinned. Its own code reproduces it,
and every entry is within four standard errors of the exact coverage. The one
far out is the Wald interval at 500 trials and a rate of 0.2 percent: 64.8
percent against an exact 63.2, 3.2 standard errors high, because that run drew
fewer samples with no events (35 percent against an exact 36.8). The prose that
Wald "covers 65 percent" with one expected event repeats the simulated value.

The table of upper bounds after zero events labels as the exact 95 percent
bound the upper end of the two-sided Clopper-Pearson interval,
``1 - 0.025^(1 / n)``, about ``3.69 / n``. That is a one-sided 97.5 percent bound;
the one-sided 95 percent bound the text derives from ``(1 - p)^n = 0.05`` is
``1 - 0.05^(1 / n)``, about ``3 / n``, which is 0.99 percent rather than 1.2 percent
after 300 clean runs. The rule of three is within a fifth of the tabulated
bound (at most 18.7 percent below it), but the tabulated bound is up to 23
percent above the rule of three.
"""

from math import log

import numpy as np
import pytest
from scipy import stats

from blog_reproducibility.statistics.small_count_intervals import (
    ALLOWED_FAILURES,
    CLEAN_RUN_RATES,
    FAILURE_TARGET,
    INTERVAL_EVENTS,
    METHODS,
    OBSERVATION_HOURS,
    RATE,
    SAMPLES,
    TARGET_RATES,
    TRIALS,
    ZERO_EVENT_TRIALS,
    clean_run_probability,
    clean_trials_needed,
    clopper_pearson_interval,
    exact_coverage,
    exact_coverage_row,
    example_payload,
    exposure_bound,
    interval,
    one_sided_zero_bound,
    simulated_coverage,
    trials_allowing_failures,
    wald_interval,
    wilson_interval,
)

SUMMARY = example_payload()
ARTICLE = SUMMARY.article

# Intervals containing the true rate out of 4,000 (Wald, Wilson, exact) and samples with no events.
FIGURE_COUNTS = {
    100: (1545, 3656, 3944, 2455),
    200: (2511, 3684, 3920, 1485),
    400: (3469, 3783, 3937, 525),
    800: (3609, 3718, 3825, 70),
    1600: (3594, 3821, 3876, 1),
    3200: (3786, 3798, 3836, 0),
    6400: (3800, 3846, 3873, 0),
}


def _website_coverage(
    r: np.random.Generator, ns: tuple[int, ...], p: float, samples: int
) -> list[tuple[int, int, int]]:
    """The website generator's loop, transcribed, returning counts of covering intervals."""
    z = stats.norm.ppf(0.975)

    def wald(k: int, n: int) -> tuple[float, float]:
        q = k / n
        h = z * np.sqrt(q * (1 - q) / n)
        return max(0, q - h), min(1, q + h)

    def wilson(k: int, n: int) -> tuple[float, float]:
        q = k / n
        c = (q + z**2 / (2 * n)) / (1 + z**2 / n)
        h = z * np.sqrt(q * (1 - q) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        return max(0, c - h), min(1, c + h)

    def exact(k: int, n: int) -> tuple[float, float]:
        lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
        hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
        return lo, hi

    out = []
    for n in ns:
        ks = r.binomial(n, p, samples)
        hits = []
        for fn in (wald, wilson, exact):
            covered = [fn(int(k), n) for k in ks]
            hits.append(sum(lo <= p <= hi for lo, hi in covered))
        out.append((hits[0], hits[1], hits[2]))
    return out


def test_the_figure_counts_are_reproduced() -> None:
    """Coverage of each interval and samples with no events, at each size, as published."""
    assert tuple(row.trials for row in SUMMARY.simulated) == TRIALS
    for row in SUMMARY.simulated:
        coverage = (row.coverage.wald, row.coverage.wilson, row.coverage.clopper_pearson)
        counts = (*(round(SAMPLES * share) for share in coverage), round(SAMPLES * row.zero_events))
        assert counts == FIGURE_COUNTS[row.trials]
        assert row.expected_events == pytest.approx(row.trials * RATE)


def test_the_draws_and_intervals_match_the_website() -> None:
    """Same generator, same sizes: the website loop's coverage counts on a small run."""
    rows = simulated_coverage(4, trials=(50, 300, 120), rate=0.01, samples=300)
    expected = _website_coverage(np.random.default_rng(4), (50, 300, 120), 0.01, 300)
    for row, counts in zip(rows, expected, strict=True):
        coverage = (row.coverage.wald, row.coverage.wilson, row.coverage.clopper_pearson)
        assert tuple(round(300 * share) for share in coverage) == counts


def test_intervals_match_scipy_and_a_hand_worked_case() -> None:
    """Wilson and Clopper-Pearson are SciPy's; Wald at 10 in 100 is 0.1 -+ 1.96 x 0.03."""
    for k, n in ((0, 10), (0, 300), (1, 1000), (7, 50), (50, 50), (13, 200)):
        result = stats.binomtest(k, n)
        wilson = result.proportion_ci(method="wilson")
        exact = result.proportion_ci(method="exact")
        assert wilson_interval(k, n) == pytest.approx((wilson.low, wilson.high), abs=1e-14)
        assert clopper_pearson_interval(k, n) == pytest.approx((exact.low, exact.high), abs=1e-14)
    z = stats.norm.ppf(0.975)
    assert wald_interval(10, 100) == pytest.approx((0.1 - z * 0.03, 0.1 + z * 0.03), abs=1e-15)
    assert wald_interval(0, 300) == (0.0, 0.0)
    assert wald_interval(300, 300) == (1.0, 1.0)
    assert interval("wilson", 3, 40) == wilson_interval(3, 40)


def test_the_zero_event_table_matches_the_article() -> None:
    """Upper bounds after no events: exact, rule of three, Wilson and Wald, to four decimals."""
    printed = {
        10: ("0.3085", "0.3000", "0.2775", "0.0000"),
        30: ("0.1157", "0.1000", "0.1135", "0.0000"),
        100: ("0.0362", "0.0300", "0.0370", "0.0000"),
        300: ("0.0122", "0.0100", "0.0126", "0.0000"),
        1000: ("0.0037", "0.0030", "0.0038", "0.0000"),
        3000: ("0.0012", "0.0010", "0.0013", "0.0000"),
    }
    assert tuple(row.trials for row in ARTICLE.zero_event_bounds) == ZERO_EVENT_TRIALS
    for row in ARTICLE.zero_event_bounds:
        values = (row.clopper_pearson, row.rule_of_three, row.wilson, row.wald)
        assert tuple(f"{value:.4f}" for value in values) == printed[row.trials]


def test_the_zero_event_bounds_have_closed_forms() -> None:
    """The tabulated bound is 1 - 0.025^(1/n), one-sided 97.5%; the text's is 1 - 0.05^(1/n)."""
    z = stats.norm.ppf(0.975)
    for row in ARTICLE.zero_event_bounds:
        n = row.trials
        assert row.clopper_pearson == pytest.approx(1 - 0.025 ** (1 / n), rel=1e-12)
        assert row.wilson == pytest.approx(z**2 / (n + z**2), rel=1e-12)
        assert row.one_sided == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-12)
        assert (1 - row.one_sided) ** n == pytest.approx(0.05, rel=1e-10)
        # The rule of three is within a fifth of the tabulated bound, below it.
        assert 0 < (row.clopper_pearson - row.rule_of_three) / row.clopper_pearson < 0.19
    large = ARTICLE.zero_event_bounds[-1]
    assert large.one_sided * large.trials == pytest.approx(-log(0.05), rel=1e-3)
    assert large.clopper_pearson * large.trials == pytest.approx(-log(0.025), rel=1e-3)
    # After 300 clean runs the one-sided 95 percent bound is 0.99 percent, not 1.2.
    assert round(100 * one_sided_zero_bound(300), 2) == 0.99


def test_the_clean_trials_table_matches_the_article() -> None:
    """59, 299, 2,995 and 29,956 clean trials, each the fewest that suffice."""
    assert [row.trials for row in ARTICLE.clean_trials] == [59, 299, 2995, 29_956]
    assert tuple(row.target for row in ARTICLE.clean_trials) == TARGET_RATES
    for row in ARTICLE.clean_trials:
        assert (1 - row.target) ** row.trials <= 0.05 < (1 - row.target) ** (row.trials - 1)
        assert row.trials == pytest.approx(3 / row.target, rel=0.02)


def test_the_failure_plan_table_matches_the_article() -> None:
    """2,996, 4,744, 6,296, 7,754 and 10,513 trials; 58 percent more for one, 3.5 times for five."""
    trials = {row.failures: row.trials for row in ARTICLE.failure_plans}
    assert tuple(trials) == ALLOWED_FAILURES
    assert [f"{value:,.0f}" for value in trials.values()] == [
        "2,996",
        "4,744",
        "6,296",
        "7,754",
        "10,513",
    ]
    assert round(100 * (trials[1] / trials[0] - 1)) == 58
    assert round(trials[5] / trials[0], 1) == 3.5
    for k, n in trials.items():
        # The chi-square relation is the Poisson bound: k or fewer events is 5% likely at n p.
        assert stats.poisson.cdf(k, n * FAILURE_TARGET) == pytest.approx(0.05, rel=1e-9)
        # The exact binomial one-sided bound reaches 0.1 percent within a few trials of it.
        assert stats.beta.ppf(0.95, k + 1, round(n) - k) == pytest.approx(FAILURE_TARGET, rel=2e-3)


def test_the_clean_run_table_matches_the_article() -> None:
    """300 clean runs are 74, 22, 5 and 0 percent likely at 0.1, 0.5, 1 and 2 percent."""
    assert tuple(row.rate for row in ARTICLE.clean_runs) == CLEAN_RUN_RATES
    assert [f"{row.probability:.0%}" for row in ARTICLE.clean_runs] == ["74%", "22%", "5%", "0%"]


def test_the_interval_table_matches_the_article() -> None:
    """Exact intervals after 0 to 50 events in 1,000 trials, and the upper bound as a multiple."""
    printed = {
        0: ("0.000", "0.0000", "0.0037", None),
        1: ("0.001", "0.0000", "0.0056", "5.6"),
        2: ("0.002", "0.0002", "0.0072", "3.6"),
        5: ("0.005", "0.0016", "0.0116", "2.3"),
        10: ("0.010", "0.0048", "0.0183", "1.8"),
        50: ("0.050", "0.0373", "0.0654", "1.3"),
    }
    assert tuple(row.events for row in ARTICLE.intervals) == INTERVAL_EVENTS
    for row in ARTICLE.intervals:
        multiple = None if row.upper_multiple is None else f"{row.upper_multiple:.1f}"
        shown = (f"{row.estimate:.3f}", f"{row.lower:.4f}", f"{row.upper:.4f}", multiple)
        assert shown == printed[row.events]


def test_the_exposure_table_matches_the_article() -> None:
    """3 / T per hour: 0.03 and 263 a year, 0.003 and 26, 0.0003 and 2.6."""
    assert tuple(row.hours for row in ARTICLE.exposure) == OBSERVATION_HOURS
    shown = [(f"{row.per_hour:g}", f"{row.per_year:.1f}") for row in ARTICLE.exposure]
    assert shown == [("0.03", "262.8"), ("0.003", "26.3"), ("0.0003", "2.6")]
    assert round(ARTICLE.exposure[0].per_year) == 263 and round(ARTICLE.exposure[1].per_year) == 26
    # The exact Poisson bound is -ln(0.05) / T, which three rounds.
    assert stats.poisson.cdf(0, -log(0.05)) == pytest.approx(0.05)
    # A thousand clean hours: below one failure a fortnight; a hundred: one alarm per 33 hours.
    assert round(1 / exposure_bound(1000).per_hour / 24) == 14
    assert round(1 / exposure_bound(100).per_hour) == 33
    # Fifty pumps for 200 hours each is the same exposure as one pump for 10,000.
    assert exposure_bound(50 * 200) == ARTICLE.exposure[2]


def test_the_article_coverage_table_agrees_with_exact_coverage() -> None:
    """The article's simulated coverage (10,000 samples) is within noise of the exact coverage."""
    printed = [
        (18.2, 98.2, 98.2, 82),
        (64.8, 91.9, 98.0, 35),
        (86.9, 96.4, 98.0, 1),
        (90.4, 92.9, 96.0, 2),
        (94.9, 94.4, 95.7, 0),
        (92.7, 96.7, 96.7, 0),
    ]
    for row, values in zip(ARTICLE.coverage, printed, strict=True):
        exact = (row.coverage.wald, row.coverage.wilson, row.coverage.clopper_pearson)
        for simulated, truth in zip(values[:3], exact, strict=True):
            noise = 4 * np.sqrt(truth * (1 - truth) / 10_000) + 0.0005
            assert simulated / 100 == pytest.approx(truth, abs=noise)
        noise = 4 * np.sqrt(row.zero_events * (1 - row.zero_events) / 10_000) + 0.005
        assert values[3] / 100 == pytest.approx(row.zero_events, abs=noise)
    one_event = ARTICLE.coverage[1]
    assert (one_event.trials, one_event.expected_events) == (500, 1.0)
    assert round(100 * one_event.coverage.wald, 1) == 63.2
    assert round(100 * one_event.zero_events, 1) == 36.8


def test_the_simulation_matches_exact_coverage() -> None:
    """Every simulated coverage and zero-event share is within binomial noise of the exact one."""
    for simulated, exact in zip(SUMMARY.simulated, SUMMARY.exact, strict=True):
        assert (simulated.trials, simulated.rate) == (exact.trials, exact.rate)
        pairs = [
            (simulated.coverage.wald, exact.coverage.wald),
            (simulated.coverage.wilson, exact.coverage.wilson),
            (simulated.coverage.clopper_pearson, exact.coverage.clopper_pearson),
            (simulated.zero_events, exact.zero_events),
        ]
        for share, truth in pairs:
            noise = 4 * np.sqrt(truth * (1 - truth) / SAMPLES) + 1e-3
            assert share == pytest.approx(truth, abs=noise)


def test_the_exact_interval_never_undercovers() -> None:
    """Clopper-Pearson coverage is at least 95 percent at every trial count and rate tried."""
    for n in (15, 100, 400):
        for p in np.linspace(0.001, 0.2, 25):
            assert exact_coverage("clopper_pearson", n, float(p)) >= 0.95
    # Coverage is a finite sum: Wald at 100 trials and 0.5 percent covers k = 1 to 4 only.
    covered = [
        k for k in range(101) if wald_interval(k, 100)[0] <= 0.005 <= wald_interval(k, 100)[1]
    ]
    assert covered == [1, 2, 3, 4]
    mass = stats.binom.pmf(covered, 100, 0.005).sum()
    assert exact_coverage("wald", 100, 0.005) == pytest.approx(mass, rel=1e-12)


def test_the_figure_claims() -> None:
    """Wald collapses below two expected events, Wilson stays near 95, the exact interval above."""
    for rows in (SUMMARY.simulated, SUMMARY.exact):
        for row in rows:
            if row.expected_events < 2:
                assert row.coverage.wald < 0.65
                # The Wald interval is the single point zero in these samples.
                assert row.zero_events > 0.35
            else:
                assert row.coverage.wald > 0.85
            assert abs(row.coverage.wilson - 0.95) < 0.045
    for row in SUMMARY.simulated:
        assert row.coverage.clopper_pearson > 0.95
    for row in SUMMARY.exact:
        assert row.coverage.clopper_pearson >= 0.95


def test_the_article_prose() -> None:
    """300 clean runs: rates to 1.2% remain, 0.1% passes three times in four, 0.5% one in five."""
    bound = clopper_pearson_interval(0, 300)[1]
    assert round(100 * bound, 1) == 1.2
    assert round(100_000 * bound, -2) == 1200
    assert round(4 * clean_run_probability(0.001)) == 3
    assert round(5 * clean_run_probability(0.005)) == 1
    # The Wald interval after 300 clean runs is zero to zero.
    assert wald_interval(0, 300) == (0.0, 0.0)
    # With a fifth of an event expected, Wald covers 18 percent and 82 percent of samples are empty.
    row = exact_coverage_row(100, 0.002)
    assert (round(100 * row.coverage.wald), round(100 * row.zero_events)) == (18, 82)
    # Twenty expected events bring Wald to its nominal level.
    assert round(100 * exact_coverage("wald", 2000, 0.01)) == 95
    # One event in 1,000 allows a rate five and a half times the estimate; two, three and a half.
    multiples = {row.events: row.upper_multiple for row in ARTICLE.intervals}
    assert multiples[1] == pytest.approx(5.5, abs=0.1)
    assert multiples[2] == pytest.approx(3.5, abs=0.11)


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the rows; another seed does not."""
    assert simulated_coverage(3, samples=300) == simulated_coverage(3, samples=300)
    assert simulated_coverage(3, samples=300) != simulated_coverage(4, samples=300)


def test_invalid_inputs_are_rejected() -> None:
    """More events than trials, bad counts, rates and levels, and unknown methods are refused."""
    with pytest.raises(ValueError):
        wald_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(TypeError):
        clopper_pearson_interval(1.0, 10)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        interval("agresti", 1, 10)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        wilson_interval(1, 10, significance=1.0)
    with pytest.raises(ValueError):
        exact_coverage("wald", 100, 0.0)
    with pytest.raises(ValueError):
        clean_trials_needed(0.0)
    with pytest.raises(ValueError):
        trials_allowing_failures(-1)
    with pytest.raises(ValueError):
        clean_run_probability(1.5)
    with pytest.raises(ValueError):
        exposure_bound(0)
    with pytest.raises(ValueError):
        simulated_coverage(samples=0)
    assert METHODS == ("wald", "wilson", "clopper_pearson")
