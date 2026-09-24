"""Check the cluster-randomisation model against closed forms, SciPy, the article and the figure.

The store-level test is compared with Student's t-test from SciPy and shown to
hold its level exactly on a small design; the customer-level statistic is the
Welch statistic referred to the normal. The variance of a store mean and of the
estimated effect are checked against simulation. The article's design effects,
effective sample size and table of store-mean variances are closed forms and
are pinned at its printed precision.

The figure's simulation (seed 0, 800 A/A experiments at each correlation) runs
once here and its rejection counts are pinned: they reproduce the published
figure. The article's own tables come from a different sequence of draws
(2,000 replications per design) and are not pinned. Its simulated standard
errors (a true spread of 1.04, a customer-level standard error of 0.32, a
store-level one of 1.03) agree with the closed forms here, 1.05 and 0.32.
"""

from math import sqrt

import numpy as np
import pytest
from scipy import stats

from blog_reproducibility.statistics.cluster_randomisation import (
    INTRACLASS_CORRELATIONS,
    REPLICATIONS,
    article_numbers,
    customer_level_false_positive_rate,
    customer_level_p,
    design_effect,
    draw_experiment,
    effective_sample_size,
    estimate_sd,
    example_payload,
    false_positive_rates,
    store_level_p,
    store_mean_variance,
)

SUMMARY = example_payload()
ROWS = {row.intraclass_correlation: row for row in SUMMARY.rows}

FIGURE_COUNTS = {
    0.0: (42, 30),
    0.005: (114, 39),
    0.01: (209, 38),
    0.02: (306, 39),
    0.05: (436, 42),
    0.1: (512, 31),
    0.2: (597, 47),
}


def _website_experiment(
    r: np.random.Generator, n_c: int, per: int, icc: float
) -> tuple[float, float]:
    """The website generator's draw and both tests, transcribed."""
    sd = 10.0
    sd_b, sd_w = sd * np.sqrt(icc), sd * np.sqrt(1 - icc)
    z = np.repeat([0, 1], n_c // 2)
    r.shuffle(z)
    store = r.normal(0, sd_b, n_c)
    cl = np.repeat(np.arange(n_c), per)
    y = 50 + store[cl] + r.normal(0, sd_w, len(cl))

    t = z[cl]
    a, b = y[t == 0], y[t == 1]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    customer = 2 * stats.norm.sf(abs((b.mean() - a.mean()) / se))

    m = np.array([y[cl == c].mean() for c in range(z.size)])
    a, b = m[z == 0], m[z == 1]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    store_p = 2 * stats.t.sf(abs((b.mean() - a.mean()) / se), z.size - 2)
    return float(customer), float(store_p)


def test_the_figure_counts_are_reproduced() -> None:
    """Rejections out of 800 for the customer-level and store-level tests, as published."""
    assert tuple(ROWS) == INTRACLASS_CORRELATIONS
    for icc, (customer, store) in FIGURE_COUNTS.items():
        assert ROWS[icc].customer_level == customer / REPLICATIONS
        assert ROWS[icc].store_level == store / REPLICATIONS


def test_the_draws_and_tests_match_the_website() -> None:
    """Same draws in the same order give the same two p-values in every experiment."""
    ours, theirs = np.random.default_rng(4), np.random.default_rng(4)
    for icc in (0.0, 0.05, 0.3, 0.05):
        assignment, outcomes = draw_experiment(ours, 12, 40, icc)
        expected = _website_experiment(theirs, 12, 40, icc)
        assert (customer_level_p(assignment, outcomes), store_level_p(assignment, outcomes)) == (
            expected
        )


def test_the_store_level_test_is_students_t_test() -> None:
    """Equal arms of store means: the p-value is SciPy's pooled two-sample t-test."""
    assignment, outcomes = draw_experiment(np.random.default_rng(6), 10, 25, 0.1)
    means = outcomes.mean(axis=1)

    expected = stats.ttest_ind(means[assignment == 1], means[assignment == 0]).pvalue
    assert store_level_p(assignment, outcomes) == pytest.approx(expected, rel=1e-10)


def test_the_customer_level_test_is_welchs_statistic_on_the_normal() -> None:
    """Every customer counted as an independent observation, the statistic referred to N(0, 1)."""
    assignment, outcomes = draw_experiment(np.random.default_rng(7), 10, 25, 0.1)
    treated, control = outcomes[assignment == 1].ravel(), outcomes[assignment == 0].ravel()
    statistic = stats.ttest_ind(treated, control, equal_var=False).statistic

    assert customer_level_p(assignment, outcomes) == pytest.approx(
        2 * stats.norm.sf(abs(statistic)), rel=1e-10
    )


def test_the_simulation_counts_match_the_website_loop() -> None:
    """Batched p-values give the website loop's rejection counts on a small run."""
    rows = false_positive_rates(
        13, intraclass_correlations=(0.0, 0.1), stores=8, customers_per_store=15, replications=60
    )
    theirs = np.random.default_rng(13)
    for row in rows:
        pairs = [_website_experiment(theirs, 8, 15, row.intraclass_correlation) for _ in range(60)]
        assert round(60 * row.customer_level) == sum(customer < 0.05 for customer, _ in pairs)
        assert round(60 * row.store_level) == sum(store < 0.05 for _, store in pairs)


def test_the_store_level_test_holds_its_level_on_a_small_design() -> None:
    """With 8 stores of 5 customers and strong clustering the t-test still rejects 5%."""
    (row,) = false_positive_rates(
        8, intraclass_correlations=(0.3,), stores=8, customers_per_store=5, replications=2000
    )

    assert row.store_level == pytest.approx(0.05, abs=4 * sqrt(0.05 * 0.95 / 2000))
    assert row.customer_level > 0.15


def test_store_means_and_the_estimate_have_the_closed_form_variance() -> None:
    """A store mean varies by sigma^2 [rho + (1 - rho) / m]; the estimate by 4 / stores times it."""
    rng = np.random.default_rng(9)
    means, estimates = [], []
    for _ in range(600):
        assignment, outcomes = draw_experiment(rng, 20, 200, 0.05)
        store_means = outcomes.mean(axis=1)
        means.extend(store_means.tolist())
        estimates.append(store_means[assignment == 1].mean() - store_means[assignment == 0].mean())

    assert float(np.var(means, ddof=1)) == pytest.approx(store_mean_variance(200, 0.05), rel=0.06)
    assert float(np.std(estimates, ddof=1)) == pytest.approx(estimate_sd(20, 200, 0.05), rel=0.1)


def test_the_article_design_effects() -> None:
    """1.0, 3.0, 11.0, 20.9, 26.0 and 3.0 for the six designs of the false positive table."""
    rows = article_numbers().design_effects

    assert [(row.stores, row.customers_per_store) for row in rows] == [
        (20, 200),
        (20, 200),
        (20, 200),
        (20, 200),
        (8, 500),
        (100, 40),
    ]
    assert [f"{row.design_effect:.1f}" for row in rows] == [
        "1.0",
        "3.0",
        "11.0",
        "20.9",
        "26.0",
        "3.0",
    ]


def test_the_article_standard_errors_and_effective_sample() -> None:
    """365 effective customers of 4,000; a customer-level SE of 0.32, 1/sqrt(11) of the truth."""
    numbers = article_numbers()

    assert round(numbers.effective_sample_size) == 365
    assert round(numbers.customer_level_standard_error, 2) == 0.32
    assert round(numbers.estimate_sd, 2) == 1.05
    assert numbers.standard_error_ratio == pytest.approx(1 / sqrt(10.95))
    assert round(numbers.standard_error_ratio, 2) == 0.30


def test_the_store_mean_variance_table() -> None:
    """6.90, 5.47, 5.05 and 5.00 at 50, 200, 2,000 and 100,000 customers; the floor is 5.00."""
    rows = article_numbers().store_mean_variances

    assert [row.customers_per_store for row in rows] == [50, 200, 2000, 100_000]
    assert [f"{row.variance:.2f}" for row in rows] == ["6.90", "5.47", "5.05", "5.00"]
    assert all(row.floor == 5.0 for row in rows)


def test_limiting_cases() -> None:
    """No clustering is independence; complete clustering leaves one observation per store."""
    assert design_effect(200, 0.0) == 1.0
    assert design_effect(1, 0.3) == 1.0
    assert design_effect(200, 1.0) == 200.0
    assert store_mean_variance(200, 0.0) == pytest.approx(100 / 200)
    assert store_mean_variance(200, 1.0) == pytest.approx(100.0)
    assert effective_sample_size(4000, 200, 1.0) == pytest.approx(20.0)
    assert customer_level_false_positive_rate(200, 0.0) == pytest.approx(0.05)
    rates = [customer_level_false_positive_rate(200, icc) for icc in INTRACLASS_CORRELATIONS]
    assert rates == sorted(rates)


def test_the_customer_level_test_breaks() -> None:
    """The alt text: above 5 percent at any positive correlation and above 50 at 0.05."""
    assert ROWS[0.0].customer_level == pytest.approx(0.05, abs=0.01)
    for icc in INTRACLASS_CORRELATIONS[1:]:
        assert ROWS[icc].customer_level > 0.1
    assert ROWS[0.05].customer_level > 0.5
    rates = [row.customer_level for row in SUMMARY.rows]
    assert rates == sorted(rates)


def test_the_store_level_test_stays_at_its_level() -> None:
    """The alt text: the store-level test is within binomial noise of 5 percent throughout."""
    margin = 4 * sqrt(0.05 * 0.95 / REPLICATIONS)
    for row in SUMMARY.rows:
        assert row.store_level == pytest.approx(0.05, abs=margin)


def test_the_customer_level_rate_follows_the_design_effect() -> None:
    """Simulated rejection rates are within binomial noise of 2 Phi(-1.96 / sqrt(deff))."""
    for row in SUMMARY.rows:
        p = row.predicted_customer_level
        assert row.design_effect == design_effect(200, row.intraclass_correlation)
        assert row.customer_level == pytest.approx(p, abs=4 * sqrt(p * (1 - p) / REPLICATIONS))


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the rows; another seed does not."""

    def small(seed: int) -> object:
        return false_positive_rates(
            seed, intraclass_correlations=(0.05,), stores=8, customers_per_store=20, replications=50
        )

    assert small(5) == small(5)
    assert small(5) != small(6)


def test_invalid_inputs_are_rejected() -> None:
    """Odd or too few stores, impossible correlations, malformed assignments and outcomes."""
    rng = np.random.default_rng(0)
    assignment, outcomes = draw_experiment(rng, 8, 10, 0.1)
    with pytest.raises(ValueError):
        draw_experiment(rng, 7, 10, 0.1)
    with pytest.raises(ValueError):
        draw_experiment(rng, 2, 10, 0.1)
    with pytest.raises(ValueError):
        draw_experiment(rng, 8, 1, 0.1)
    with pytest.raises(ValueError):
        draw_experiment(rng, 8, 10, 1.5)
    with pytest.raises(ValueError):
        draw_experiment(rng, 8, 10, 0.1, sd=0.0)
    with pytest.raises(ValueError):
        store_level_p(assignment, outcomes[:-1])
    with pytest.raises(ValueError):
        store_level_p(assignment * 2, outcomes)
    with pytest.raises(ValueError):
        customer_level_p(np.array([0, 1, 1, 1, 1, 1, 1, 1]), outcomes)
    with pytest.raises(ValueError):
        customer_level_p(assignment, np.where(outcomes > 50, np.nan, outcomes))
    with pytest.raises(ValueError):
        customer_level_false_positive_rate(200, 0.05, significance=0.0)
    with pytest.raises(ValueError):
        effective_sample_size(0, 200, 0.05)
    with pytest.raises(ValueError):
        false_positive_rates(replications=0)
