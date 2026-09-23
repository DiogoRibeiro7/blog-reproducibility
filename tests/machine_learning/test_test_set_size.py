"""Check the test-set size model against closed forms, the article and the figure.

The figure's simulation (seed 3, 2,000 replications per size) is the one the
article prints as its comparison table, so the table is pinned at its printed
precision and the underlying counts exactly. The buffered draw loop is checked
against a direct transcription of the article's loop, and the vectorised
McNemar p-values against :func:`scipy.stats.binomtest`. The population
accuracies and disagreement rate are checked against their bivariate-normal
closed form and against sampling, and the closed-form numbers in the prose are
reproduced.

The article's population line (seed 30: 90.0 and 90.9 percent, disagreement
8.4 percent) and its power check (seed 31: 84 percent at 7,734 cases) come
from other generators and are not pinned. Its 7,734 uses the unrounded
simulated inputs; the rounded inputs it prints (d = 0.084, delta = 0.0092) give
7,790. With the exact population difference of one point the formula gives
about 6,600 cases, and the figure's own simulation puts 80 percent power
between 5,000 (65 percent) and 10,000 (93 percent) cases, consistent with the
article's "about 8,000".
"""

from math import sqrt

import numpy as np
import pytest
from scipy import stats

from blog_reproducibility.machine_learning.test_set_size import (
    REPLICATIONS,
    TEST_SIZES,
    RankingRow,
    accuracy_standard_error,
    draw_correctness,
    example_payload,
    interval_half_width,
    latent_thresholds,
    mcnemar_p_values,
    paired_comparison_size,
    population_summary,
    ranking_rates,
    required_size_for_half_width,
)

SUMMARY = example_payload()
ROWS = {row.test_size: row for row in SUMMARY.rows}

ARTICLE_TABLE = {
    200: (63, 60, 5, 0),
    500: (76, 68, 8, 0),
    1000: (84, 76, 16, 0),
    2000: (94, 86, 30, 1),
    5000: (99, 95, 65, 6),
    10000: (100, 99, 93, 30),
    20000: (100, 100, 100, 82),
}

EXACT_COUNTS = {
    200: (1268, 1205, 94, 0),
    500: (1521, 1351, 170, 1),
    1000: (1677, 1523, 317, 4),
    2000: (1874, 1721, 607, 19),
    5000: (1988, 1909, 1298, 119),
    10000: (1999, 1986, 1868, 600),
    20000: (2000, 2000, 1992, 1636),
}


def _shares(row: RankingRow) -> tuple[float, float, float, float]:
    return (row.same_set, row.separate_sets, row.paired_significant, row.intervals_disjoint)


def _article_loop(test_size: int, seed: int, replications: int) -> tuple[int, int, int, int]:
    """Run a direct transcription of the article's loop for one test-set size."""
    rng = np.random.default_rng(seed)
    thresholds = latent_thresholds()
    same = separate = paired = disjoint = 0
    for _ in range(replications):
        a, b = draw_correctness(test_size, rng, thresholds=thresholds)
        same += int(b.mean() > a.mean())
        a_fresh, _ = draw_correctness(test_size, rng, thresholds=thresholds)
        separate += int(b.mean() > a_fresh.mean())
        b_only, a_only = np.sum(~a & b), np.sum(a & ~b)
        if b_only + a_only > 0:
            paired += (
                stats.binomtest(int(b_only), int(b_only + a_only)).pvalue < 0.05 and b_only > a_only
            )
        half_a = 1.96 * np.sqrt(a.mean() * (1 - a.mean()) / test_size)
        half_b = 1.96 * np.sqrt(b.mean() * (1 - b.mean()) / test_size)
        disjoint += int((b.mean() - half_b) > (a.mean() + half_a))
    return int(same), int(separate), int(paired), int(disjoint)


def test_the_article_table_is_reproduced() -> None:
    """Every entry of the comparison table, at the article's whole-percent precision."""
    assert tuple(ROWS) == TEST_SIZES
    for size, printed in ARTICLE_TABLE.items():
        assert tuple(round(100 * share) for share in _shares(ROWS[size])) == printed


def test_the_underlying_counts_are_exact() -> None:
    """The shares are these counts out of 2,000 replications, draw for draw."""
    for size, counts in EXACT_COUNTS.items():
        assert _shares(ROWS[size]) == tuple(value / REPLICATIONS for value in counts)


def test_the_fast_loop_matches_the_article_loop() -> None:
    """Buffered draws and vectorised p-values give the article's counts on small runs."""
    for size in (200, 1000):
        (row,) = ranking_rates(11, test_sizes=(size,), replications=150)
        expected = _article_loop(size, 11, 150)
        assert tuple(round(150 * share) for share in _shares(row)) == expected


def test_mcnemar_p_values_match_binomtest() -> None:
    """The symmetric-tail formula equals scipy's exact two-sided binomial test."""
    pairs = [(k, m) for m in range(1, 60) for k in range(m + 1)]
    pairs += [(k, 400) for k in range(150, 251)]
    b_only = np.array([k for k, _ in pairs])
    discordant = np.array([m for _, m in pairs])
    expected = np.array([stats.binomtest(k, m).pvalue for k, m in pairs])
    np.testing.assert_allclose(mcnemar_p_values(b_only, discordant), expected, rtol=1e-9)
    assert mcnemar_p_values(np.array([0]), np.array([0]))[0] == 1.0


def test_population_accuracies_and_disagreement() -> None:
    """Exactly 90 and 91 percent, and a disagreement rate of 8.4 percent."""
    population = SUMMARY.population
    assert population.accuracy_a == pytest.approx(0.90, abs=1e-12)
    assert population.accuracy_b == pytest.approx(0.91, abs=1e-12)
    assert population.difference == pytest.approx(0.01, abs=1e-12)
    assert round(population.disagreement, 3) == 0.084


def test_population_matches_sampling() -> None:
    """The bivariate-normal disagreement rate agrees with a large simulated test set."""
    a, b = draw_correctness(400_000, np.random.default_rng(12))
    population = population_summary()
    assert np.mean(a != b) == pytest.approx(population.disagreement, abs=0.002)
    assert b.mean() - a.mean() == pytest.approx(population.difference, abs=0.002)


def test_less_model_noise_means_less_disagreement() -> None:
    """Models with more shared errors disagree less, so they are cheaper to compare."""
    rates = [population_summary(tau).disagreement for tau in (0.2, 0.5, 1.0)]
    assert rates == sorted(rates)


def test_the_accuracy_table_half_widths() -> None:
    """Half-widths of 5.9, 2.9, 1.9, 0.8 and 0.4 points, and a standard error of 1.5 on 400."""
    assert [(n, round(width, 1)) for n, width in SUMMARY.article.half_widths] == [
        (100, 5.9),
        (400, 2.9),
        (1000, 1.9),
        (5000, 0.8),
        (20000, 0.4),
    ]
    assert round(SUMMARY.article.standard_error_400, 1) == 1.5
    assert interval_half_width(400, 0.9) == pytest.approx(1.96 * 0.015)


def test_sizing_a_single_metric() -> None:
    """About 250 positives for +/-5 points of recall; 20 positives give +/-18; slices +/-3.7."""
    article = SUMMARY.article
    assert 240 < article.recall_positives_for_five_points < 250
    assert round(article.recall_half_width_20_positives) == 18
    assert round(article.slice_half_width_250, 1) == 3.7
    size = required_size_for_half_width(0.9, 0.01)
    assert interval_half_width(round(size), 0.9) == pytest.approx(0.01, rel=1e-3)


def test_sizing_a_paired_comparison() -> None:
    """The multiplier is 7.85; a tenth of a point at d = 0.08 needs about 630,000 cases."""
    article = SUMMARY.article
    assert round(article.paired_multiplier, 2) == 7.85
    assert round(article.paired_size_tenth_point, -4) == 630_000
    assert article.paired_size_article_inputs == pytest.approx(7734, rel=0.01)
    # A tenth of the difference needs a hundred times the cases.
    assert paired_comparison_size(0.08, 0.001) == pytest.approx(
        100 * paired_comparison_size(0.08, 0.01)
    )


def test_the_paired_formula_delivers_its_power() -> None:
    """At the formula's size for the true population, McNemar's test rejects about 80%."""
    size = round(SUMMARY.population.paired_size)
    (row,) = ranking_rates(21, test_sizes=(size,), replications=600)
    assert 0.74 < row.paired_significant < 0.88


def test_one_point_needs_thousands_of_cases() -> None:
    """The title: the paired test has little power below 2,000 cases and 80% only near 8,000."""
    assert ROWS[1000].paired_significant < 0.2
    assert ROWS[2000].paired_significant < 0.5
    assert ROWS[5000].paired_significant < 0.8 < ROWS[10000].paired_significant


def test_the_same_test_set_beats_separate_ones() -> None:
    """The alt text: sharing the test set ranks the models correctly more often at every size."""
    for row in SUMMARY.rows:
        assert row.same_set >= row.separate_sets
        if row.separate_sets < 1.0:
            assert row.same_set > row.separate_sets
    assert 0.7 < ROWS[500].same_set < 0.8


def test_the_paired_test_fires_long_before_intervals_separate() -> None:
    """The alt text: significance arrives far earlier than disjoint independent intervals."""
    for row in SUMMARY.rows:
        assert row.paired_significant >= row.intervals_disjoint
    assert ROWS[5000].paired_significant > 0.6 > 0.1 > ROWS[5000].intervals_disjoint
    assert ROWS[10000].intervals_disjoint < 0.5 < ROWS[20000].intervals_disjoint


def test_shares_are_proportions_that_grow_with_size() -> None:
    """Every series is a share in [0, 1] and, for this design, rises with the test-set size."""
    for series in zip(*(_shares(row) for row in SUMMARY.rows), strict=True):
        assert all(0.0 <= share <= 1.0 for share in series)
        assert list(series) == sorted(series)


def test_standard_error_closed_form() -> None:
    """sqrt(p (1 - p) / n), zero for a perfect model."""
    assert accuracy_standard_error(100, 0.9) == pytest.approx(0.03)
    assert accuracy_standard_error(50, 1.0) == 0.0
    assert accuracy_standard_error(400, 0.5) == pytest.approx(sqrt(0.25 / 400))


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the shares; another seed does not."""

    def small(seed: int) -> tuple[RankingRow, ...]:
        return ranking_rates(seed, test_sizes=(200, 500), replications=60)

    assert small(5) == small(5)
    assert small(5) != small(6)


def test_invalid_inputs_are_rejected() -> None:
    """Empty test sets, impossible rates, booleans and bad counts are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        draw_correctness(0, rng)
    with pytest.raises(TypeError):
        draw_correctness(True, rng)
    with pytest.raises(ValueError):
        accuracy_standard_error(100, 1.2)
    with pytest.raises(ValueError):
        required_size_for_half_width(0.9, 0.0)
    with pytest.raises(ValueError):
        paired_comparison_size(0.0, 0.01)
    with pytest.raises(ValueError):
        paired_comparison_size(0.08, -0.01)
    with pytest.raises(ValueError):
        latent_thresholds(0.5, accuracy_a=1.0)
    with pytest.raises(ValueError):
        ranking_rates(replications=0)
    with pytest.raises(ValueError):
        mcnemar_p_values(np.array([5]), np.array([3]))
    with pytest.raises(ValueError):
        mcnemar_p_values(np.array([1, 2]), np.array([3]))
