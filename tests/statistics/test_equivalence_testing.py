"""Check the equivalence tests against SciPy, exact power, the website's draws and the article.

The three tests are compared with SciPy's one-sample t-tests, and the exact
rates (noncentral t for the t-test and non-inferiority, one integral for TOST)
with the simulation and with an independent small simulation. The figure's
simulation (seed 0, 4,000 comparisons per design) runs once here and its counts
are pinned; they are the website loop's, draw for draw, which a transcription
of that loop confirms on small designs, and they reproduce the published
figure. The article's table for identical models runs the same generator in the
same order, and its rows are reproduced exactly.

The article's tables for challengers 1.5 and 0.5 points worse, and its
simulated 81 percent power at 308 cases, come from later draws of its own
generator in a different order. They are not pinned; every printed percentage
in those two tables equals the exact rate at the printed precision, and the 81
percent is within noise of the exact 79.7.

Two statements do not hold as written, and the tests pin what is true. The
article says that for a challenger 0.5 points worse the 80 percent sample size
"becomes about 1,230", four times that for identical models. The formula does
give 1,233, but with a true difference away from zero only one of the two
one-sided tests binds and ``z_{1 - beta}`` replaces ``z_{1 - beta / 2}``; the
exact TOST power reaches 80 percent at 892 cases, 2.9 times the 310 needed for
identical models, and is 90 percent at 1,233. The "about 282" cases to detect a
one-point difference is 282.6 truncated. Less strictly, an interval at 50 cases
"can never fit" inside the margin: it can, but with identical models the exact
chance of showing equivalence there is 0.004 percent.
"""

from math import sqrt

import numpy as np
import pytest
from scipy import stats

from blog_reproducibility.statistics import equivalence_testing
from blog_reproducibility.statistics.equivalence_testing import (
    REPLICATIONS,
    TEST_CASES,
    TRUE_DIFFERENCES,
    OutcomeRow,
    cases_for_equivalence,
    cases_to_detect,
    compare_paired,
    equivalence_power,
    example_payload,
    non_inferiority_power,
    outcome_rates,
    outcome_row,
    smallest_cases_for_power,
    t_test_power,
)

SUMMARY = example_payload()
ROWS = {(row.true_difference, row.test_cases): row for row in SUMMARY.rows}

# Comparisons out of 4,000: t-test significant, equivalence shown, non-inferiority shown.
FIGURE_COUNTS = {
    (0.0, 50): (211, 0, 1237),
    (0.0, 100): (199, 151, 2078),
    (0.0, 200): (200, 2065, 3033),
    (0.0, 400): (192, 3642, 3824),
    (0.0, 800): (204, 3996, 3998),
    (0.0, 1600): (200, 4000, 4000),
    (0.0, 3200): (204, 4000, 4000),
    (-1.5, 50): (1660, 0, 47),
    (-1.5, 100): (2756, 7, 27),
    (-1.5, 200): (3754, 11, 11),
    (-1.5, 400): (3997, 1, 1),
    (-1.5, 800): (4000, 0, 0),
    (-1.5, 1600): (4000, 0, 0),
    (-1.5, 3200): (4000, 0, 0),
    (-0.5, 50): (345, 0, 593),
    (-0.5, 100): (494, 120, 789),
    (-0.5, 200): (827, 1163, 1278),
    (-0.5, 400): (1542, 2015, 2018),
    (-0.5, 800): (2595, 3048, 3048),
    (-0.5, 1600): (3660, 3796, 3796),
    (-0.5, 3200): (3985, 3997, 3997),
}


def _counts(row: OutcomeRow, replications: int = REPLICATIONS) -> tuple[int, int, int]:
    return (
        round(replications * row.t_test_significant),
        round(replications * row.equivalent),
        round(replications * row.non_inferior),
    )


def _website_outcomes(
    r: np.random.Generator, n: int, delta: float, reps: int
) -> tuple[int, int, int]:
    """The website generator's inner loop, transcribed, returning counts."""
    sigma_d, margin = 6.0, 1.0
    t_sig = tost_eq = noninf = 0
    for _ in range(reps):
        d = r.normal(delta, sigma_d, n)
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
        t_sig += bool(abs(m / se) > stats.t.ppf(0.975, n - 1))
        lo = m - stats.t.ppf(0.95, n - 1) * se
        hi = m + stats.t.ppf(0.95, n - 1) * se
        tost_eq += bool((lo > -margin) and (hi < margin))
        noninf += bool(lo > -margin)
    return t_sig, tost_eq, noninf


def test_the_figure_counts_are_reproduced() -> None:
    """Every design of the figure's simulation, in the figure's order, as published."""
    assert [(row.true_difference, row.test_cases) for row in SUMMARY.rows] == [
        (delta, n) for delta in TRUE_DIFFERENCES for n in TEST_CASES
    ]
    for key, counts in FIGURE_COUNTS.items():
        assert _counts(ROWS[key]) == counts


def test_the_draws_match_the_website_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks of replications give the website's one-at-a-time counts, in sequence."""
    monkeypatch.setattr(equivalence_testing, "_DRAWS_PER_BLOCK", 70)
    ours, theirs = np.random.default_rng(3), np.random.default_rng(3)
    for n, delta in ((30, 0.0), (30, -1.5), (200, -0.5), (35, 0.0)):
        row = outcome_row(ours, n, delta, replications=41)
        assert _counts(row, 41) == _website_outcomes(theirs, n, delta, 41)


def test_the_identical_model_table_matches_the_article() -> None:
    """t-test significant, equivalence and non-inferiority shown, 50 to 800 cases."""
    printed = {
        50: ("5%", "0%", "31%"),
        100: ("5%", "4%", "52%"),
        200: ("5%", "52%", "76%"),
        400: ("5%", "91%", "96%"),
        800: ("5%", "100%", "100%"),
    }
    for n, expected in printed.items():
        row = ROWS[(0.0, n)]
        shares = (row.t_test_significant, row.equivalent, row.non_inferior)
        assert tuple(f"{share:.0%}" for share in shares) == expected


def test_the_other_tables_agree_with_the_exact_rates() -> None:
    """The 1.5-worse and 0.5-worse tables, from the article's own draws, at printed precision."""
    much_worse = {50: (59, 0, 1), 100: (30, 0, 1), 200: (6, 0, 0), 400: (0, 0, 0)}
    for n, (not_significant, equivalent, non_inferior) in much_worse.items():
        assert round(100 * (1 - t_test_power(n, -1.5))) == not_significant
        assert round(100 * equivalence_power(n, -1.5)) == equivalent
        assert round(100 * non_inferiority_power(n, -1.5)) == non_inferior
    slightly_worse = {200: (29, 32), 800: (76, 76), 1600: (95, 95), 3200: (100, 100)}
    for n, (equivalent, non_inferior) in slightly_worse.items():
        assert round(100 * equivalence_power(n, -0.5)) == equivalent
        assert round(100 * non_inferiority_power(n, -0.5)) == non_inferior


def test_compare_paired_matches_scipy() -> None:
    """The t-test is SciPy's; TOST is two one-sided t-tests at 5 percent against the margins."""
    rng = np.random.default_rng(5)
    for n, delta in ((40, 0.0), (300, 0.2), (300, -1.2), (2000, -0.3), (900, 0.0)):
        d = rng.normal(delta, 6.0, n)
        result = compare_paired(d)
        above = stats.ttest_1samp(d, -1.0, alternative="greater").pvalue < 0.05
        below = stats.ttest_1samp(d, 1.0, alternative="less").pvalue < 0.05
        assert result.t_test_significant == (stats.ttest_1samp(d, 0.0).pvalue < 0.05)
        assert result.non_inferior == above
        assert result.equivalent == (above and below)
        assert result.standard_error == pytest.approx(stats.sem(d), rel=1e-12)


def test_the_exact_rates_match_the_simulation() -> None:
    """Every simulated share is within binomial noise of its exact probability."""
    for row in SUMMARY.rows:
        pairs = (
            (row.t_test_significant, row.exact_t_test_significant),
            (row.equivalent, row.exact_equivalent),
            (row.non_inferior, row.exact_non_inferior),
        )
        for simulated, exact in pairs:
            noise = 4 * sqrt(exact * (1 - exact) / REPLICATIONS) + 1e-3
            assert simulated == pytest.approx(exact, abs=noise)


def test_equivalence_power_matches_an_independent_simulation() -> None:
    """A different design: 30 cases, unit spread, a margin of one half and a true 0.2."""
    rng = np.random.default_rng(8)
    d = rng.normal(0.2, 1.0, (40_000, 30))
    m, se = d.mean(axis=1), d.std(axis=1, ddof=1) / sqrt(30)
    t = stats.t.ppf(0.95, 29)
    shown = float(np.mean((m - t * se > -0.5) & (m + t * se < 0.5)))

    exact = equivalence_power(30, 0.2, sd=1.0, margin=0.5)
    assert exact == pytest.approx(0.466, abs=0.001)
    assert shown == pytest.approx(exact, abs=4 * sqrt(exact * (1 - exact) / 40_000))


def test_the_tests_hold_their_levels() -> None:
    """Five percent under each null, exactly for the t-test and non-inferiority."""
    for n in (50, 308, 3200):
        assert t_test_power(n, 0.0) == pytest.approx(0.05, abs=1e-12)
        assert non_inferiority_power(n, -1.0) == pytest.approx(0.05, abs=1e-12)
        assert equivalence_power(n, -1.0) <= 0.05 + 1e-9
        assert equivalence_power(n, 1.0) == pytest.approx(equivalence_power(n, -1.0), abs=1e-9)
    # TOST is exact only in the limit, when the far edge stops mattering.
    assert equivalence_power(20_000, -1.0) == pytest.approx(0.05, abs=1e-6)
    for n in (50, 200, 800):
        for delta in (0.0, -0.5, -1.5):
            assert equivalence_power(n, delta) <= non_inferiority_power(n, delta) + 1e-9


def test_the_sample_size_formulas() -> None:
    """308 cases for 80 percent power, 390 for 90, 1,233 at half a point and 282.6 to detect."""
    numbers = SUMMARY.article
    assert round(numbers.cases_for_eighty) == 308
    assert round(numbers.cases_for_ninety) == 390
    assert round(numbers.cases_for_eighty_half_point_worse) == 1233
    assert round(numbers.cases_for_eighty_half_point_worse, -1) == 1230
    assert round(numbers.cases_to_detect, 1) == 282.6
    # The article's "about 282" is truncated; a sample size would round up to 283.
    assert int(numbers.cases_to_detect) == 282
    assert numbers.cases_for_eighty_half_point_worse == pytest.approx(
        4 * numbers.cases_for_eighty, rel=1e-12
    )
    assert cases_for_equivalence(0.8, sd=3.0, margin=0.5) == pytest.approx(
        numbers.cases_for_eighty, rel=1e-12
    )


def test_the_formula_matches_the_exact_power_for_identical_models() -> None:
    """At 308 cases the exact power is 79.7 percent; the article simulated 81 percent."""
    numbers = SUMMARY.article
    assert round(100 * numbers.exact_power_at_formula_cases, 1) == 79.7
    noise = 4 * sqrt(0.2 * 0.8 / 4000)
    assert numbers.exact_power_at_formula_cases == pytest.approx(0.81, abs=noise)
    assert numbers.smallest_cases_for_eighty == 310
    assert equivalence_power(390, 0.0) == pytest.approx(0.90, abs=0.002)


def test_the_half_point_size_is_conservative() -> None:
    """Exact TOST reaches 80 percent at 892 cases, not about 1,230, and 2.9 times, not four."""
    exact = smallest_cases_for_power(0.8, true_difference=-0.5)
    assert exact == 892
    assert round(exact / SUMMARY.article.smallest_cases_for_eighty, 1) == 2.9
    assert round(100 * equivalence_power(1233, -0.5)) == 90
    # With one edge binding, z_{1 - beta} replaces z_{1 - beta / 2}.
    one_sided = 36 * (stats.norm.ppf(0.95) + stats.norm.ppf(0.80)) ** 2 / 0.5**2
    assert one_sided == pytest.approx(exact, abs=3)


def test_the_article_prose() -> None:
    """A three-point interval at 50 cases, a 5% t-test and non-inferiority easier throughout."""
    assert round(SUMMARY.article.interval_width_at_fifty, 2) == 2.85
    assert equivalence_power(50, 0.0) < 1e-4
    for row in SUMMARY.rows:
        assert row.non_inferior >= row.equivalent
        if row.true_difference == 0.0:
            assert row.exact_t_test_significant == pytest.approx(0.05)
    # The t-test on a 1.5-point deterioration is non-significant six times in ten at 50 cases.
    assert round(10 * (1 - t_test_power(50, -1.5))) == 6


def test_the_figure_claims() -> None:
    """Non-significant more than half the time at 50 cases; equivalence needs about 300."""
    assert 1 - ROWS[(-1.5, 50)].t_test_significant > 0.5
    assert ROWS[(0.0, 200)].equivalent < 0.8 < ROWS[(0.0, 400)].equivalent
    assert 290 <= SUMMARY.article.smallest_cases_for_eighty <= 320
    # A challenger outside the margin is almost never declared equivalent or non-inferior.
    for n in TEST_CASES:
        row = ROWS[(-1.5, n)]
        assert row.equivalent <= row.non_inferior <= 0.05
        assert row.exact_non_inferior <= 0.05
    # Inside the margin, the exact chance of showing equivalence grows with the test set.
    for delta in (0.0, -0.5):
        shown = [ROWS[(delta, n)].exact_equivalent for n in TEST_CASES]
        assert np.all(np.diff(shown) > -1e-9)


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the rows; another seed does not."""

    def small(seed: int) -> object:
        return outcome_rates(seed, true_differences=(-0.5,), test_cases=(200,), replications=200)

    assert small(5) == small(5)
    assert small(5) != small(6)


def test_invalid_inputs_are_rejected() -> None:
    """Short or constant samples, bad sizes, margins and powers, and targets out of reach."""
    with pytest.raises(ValueError):
        compare_paired([1.0])
    with pytest.raises(ValueError):
        compare_paired([2.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        compare_paired([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        compare_paired([1.0, float("nan"), 2.0])
    with pytest.raises(ValueError):
        compare_paired([1.0, 2.0, 3.0], margin=0.0)
    with pytest.raises(ValueError):
        t_test_power(1, 0.0)
    with pytest.raises(TypeError):
        equivalence_power(50.0, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        non_inferiority_power(50, 0.0, sd=-1.0)
    with pytest.raises(ValueError):
        cases_for_equivalence(0.8, true_difference=-1.0)
    with pytest.raises(ValueError):
        cases_for_equivalence(1.0)
    with pytest.raises(ValueError):
        cases_to_detect(0.8, difference=0.0)
    with pytest.raises(ValueError):
        smallest_cases_for_power(0.8, true_difference=-1.5, upper=1000)
    with pytest.raises(ValueError):
        outcome_rates(replications=0)
