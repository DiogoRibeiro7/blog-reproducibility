"""Check the non-compliance estimators against the website, closed forms, article and figure.

The figure's 4,000 experiments (500 at each of eight compliance rates, one
generator seeded at 0) run once here, in about a second, and every plotted mean
is pinned as the website's generator computes it. The port is checked draw for
draw against a transcription of the website's experiment, which labels the
kinds of user with strings.

The estimators' limits are closed forms, checked against hand-worked fractions
(3.75 and 65/21 at the article's design, 1.75 and 23/21 for an inert feature,
38/13 and 2.6 with no always-takers), and their large-sample spreads are
checked against simulation. The figure's claims hold: intention to treat stays
within noise of ``2c``, as treated and per protocol sit at least 0.3 above the
truth at every rate and on their confounded limits, and the Wald estimate stays
within four standard errors of 2 throughout.

The article's tables come from 3,000 experiments per cell drawn in another
order (the 60 percent design first), so they are not pinned; run separately,
its code does reproduce every printed value. Here each printed mean and spread
is checked against the closed forms at its own noise level. Its sample sizes
(131, 294, 662 and 2,649 users per arm) are closed forms with the article's
variance of 27 and are pinned; that 27 is the variance of ``y - tau d`` at 60
percent compliance, 27.01. The article's "spread five times" at 20 against 90
percent compliance is 4.6 in the closed form (0.84 against 0.18 as printed), and
its "twenty times the users" is ``(0.9 / 0.2)^2 = 20.25``.
"""

from math import sqrt

import numpy as np
import pytest
from numpy.typing import NDArray

from blog_reproducibility.statistics.noncompliance import (
    ALWAYS_TAKERS,
    COMPLIANCE_RATES,
    REPLICATIONS,
    TRUE_EFFECT,
    USERS,
    Estimates,
    as_treated,
    draw_experiment,
    estimate_all,
    estimator_spread,
    example_payload,
    expected_estimates,
    intention_to_treat,
    per_protocol,
    residual_variance,
    simulate_estimates,
    users_per_arm,
    wald,
)

SUMMARY = example_payload()
ROWS = {row.compliance: row for row in SUMMARY.rows}
FIELDS = ("intention_to_treat", "as_treated", "per_protocol", "wald")

# The website generator's plotted means, to four decimals, at 20, 30, ..., 90 percent compliance.
WEBSITE_FIGURE = {
    "intention_to_treat": (0.3999, 0.5925, 0.7850, 1.0077, 1.1946, 1.4049, 1.5952, 1.7965),
    "as_treated": (5.2492, 4.7854, 4.4161, 4.0885, 3.7466, 3.3995, 2.9844, 2.5481),
    "per_protocol": (4.5594, 4.0642, 3.6928, 3.3920, 3.0904, 2.8270, 2.5422, 2.3000),
    "wald": (1.9801, 1.9655, 1.9589, 2.0110, 1.9914, 2.0087, 1.9901, 1.9967),
}
ARTICLE_REPLICATIONS = 3000


def _website_draw(
    r: np.random.Generator, n: int, compliers: float, always: float = 0.1, effect: float = 2.0
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Transcribe the website generator's experiment."""
    z = r.integers(0, 2, n)
    u = r.random(n)
    kind = np.where(u < compliers, "c", np.where(u < compliers + always, "a", "n"))
    d = np.where(kind == "c", z, np.where(kind == "a", 1, 0))
    base = 10 + np.where(kind == "a", 3.0, np.where(kind == "n", -2.0, 0.0))
    return z, d, base + effect * d + r.normal(0, 5, n)


def _website_estimates(
    z: NDArray[np.int64], d: NDArray[np.int64], y: NDArray[np.float64]
) -> tuple[float, float, float, float]:
    """Transcribe the website generator's four estimators."""
    itt = y[z == 1].mean() - y[z == 0].mean()
    treated = y[d == 1].mean() - y[d == 0].mean()
    protocol = y[(z == 1) & (d == 1)].mean() - y[(z == 0) & (d == 0)].mean()
    return (
        float(itt),
        float(treated),
        float(protocol),
        float(itt / (d[z == 1].mean() - d[z == 0].mean())),
    )


def _values(estimates: Estimates) -> tuple[float, ...]:
    return tuple(getattr(estimates, field) for field in FIELDS)


def _standard_errors(compliance: float, replications: int = REPLICATIONS) -> tuple[float, ...]:
    return tuple(s / sqrt(replications) for s in _values(ROWS[compliance].spread))


def test_the_figure_means_are_reproduced() -> None:
    """Every plotted mean, for every estimator and compliance rate, as the website computes it."""
    assert tuple(ROWS) == COMPLIANCE_RATES
    assert SUMMARY.effect == TRUE_EFFECT
    for row in SUMMARY.rows:
        assert row.replications == REPLICATIONS
        assert row.always_takers == ALWAYS_TAKERS
    for field, printed in WEBSITE_FIGURE.items():
        values = tuple(round(getattr(row.simulated, field), 4) for row in SUMMARY.rows)
        assert values == printed


def test_the_draws_and_estimates_match_the_website_loop() -> None:
    """Same generator, same order: identical experiments and estimates, one after another."""
    ours, theirs = np.random.default_rng(3), np.random.default_rng(3)
    for compliers in (0.2, 0.55, 0.9):
        for _ in range(3):
            z, d, y = draw_experiment(ours, USERS, compliers)
            expected = _website_draw(theirs, USERS, compliers)
            for mine, website in zip((z, d, y), expected, strict=True):
                assert np.array_equal(mine, website)
            assert _values(estimate_all(z, d, y)) == _website_estimates(*expected)
    values = simulate_estimates(np.random.default_rng(4), 0.4, replications=5)
    generator = np.random.default_rng(4)
    for row in values:
        assert tuple(row) == _website_estimates(*_website_draw(generator, USERS, 0.4))


def test_the_limits_match_hand_worked_fractions() -> None:
    """The article's design, an inert feature and one-sided non-compliance, by hand."""
    article = SUMMARY.article.expected
    assert article.intention_to_treat == pytest.approx(1.2)
    assert article.as_treated == pytest.approx(12.75 - 9.0)
    assert article.per_protocol == pytest.approx(65 / 21)
    assert article.wald == 2.0
    inert = SUMMARY.inert_feature.expected
    assert (inert.intention_to_treat, inert.wald) == (0.0, 0.0)
    assert inert.as_treated == pytest.approx(1.75)
    assert inert.per_protocol == pytest.approx(23 / 21)
    one_sided = SUMMARY.one_sided.expected
    assert one_sided.intention_to_treat == pytest.approx(1.4)
    assert one_sided.as_treated == pytest.approx(38 / 13)
    assert one_sided.per_protocol == pytest.approx(2.6)


def test_the_spreads_match_simulation() -> None:
    """2,500 experiments of 1,000 users: every spread within seven percent of its closed form."""
    users, replications = 1000, 2500
    for seed, compliers, always in ((5, 0.6, 0.1), (7, 0.7, 0.0)):
        values = simulate_estimates(
            np.random.default_rng(seed),
            compliers,
            always_takers=always,
            users=users,
            replications=replications,
        )
        spread = _values(estimator_spread(compliers, always, users=users))
        for simulated, expected in zip(values.std(axis=0), spread, strict=True):
            assert float(simulated) == pytest.approx(expected, rel=0.07)
        means = _values(expected_estimates(compliers, always))
        for simulated, expected, sd in zip(values.mean(axis=0), means, spread, strict=True):
            assert abs(float(simulated) - expected) < 4 * sd / sqrt(replications)


def test_intention_to_treat_falls_in_proportion_to_compliance() -> None:
    """The alt text: the intention-to-treat mean is within noise of 2c at every rate."""
    for compliance, row in ROWS.items():
        assert row.expected.intention_to_treat == pytest.approx(TRUE_EFFECT * compliance)
        error = _standard_errors(compliance)[0]
        assert abs(row.simulated.intention_to_treat - TRUE_EFFECT * compliance) < 4 * error


def test_as_treated_and_per_protocol_sit_above_the_truth() -> None:
    """The title: comparing by what users did overstates the effect at every compliance rate."""
    for compliance, row in ROWS.items():
        _, treated_error, protocol_error, _ = _standard_errors(compliance)
        for simulated, limit, error in (
            (row.simulated.as_treated, row.expected.as_treated, treated_error),
            (row.simulated.per_protocol, row.expected.per_protocol, protocol_error),
        ):
            assert simulated > TRUE_EFFECT + 0.29
            assert limit > TRUE_EFFECT + 0.29
            assert abs(simulated - limit) < 4 * error
    # The bias shrinks as compliers crowd out the other kinds, but never reaches zero.
    treated = [row.expected.as_treated for row in SUMMARY.rows]
    assert treated == sorted(treated, reverse=True)


def test_the_wald_estimate_recovers_the_complier_effect() -> None:
    """The alt text: the Wald mean is within four standard errors of 2 at every rate."""
    for compliance, row in ROWS.items():
        assert row.expected.wald == TRUE_EFFECT
        assert abs(row.simulated.wald - TRUE_EFFECT) < 4 * _standard_errors(compliance)[3]


def test_the_article_tables_agree_with_the_closed_forms() -> None:
    """Each printed mean and spread lies within its noise of the closed form."""

    def check(expected: Estimates, spread: Estimates, printed: dict[str, float]) -> None:
        for field, value in printed.items():
            error = getattr(spread, field) / sqrt(ARTICLE_REPLICATIONS)
            assert abs(value - getattr(expected, field)) < 0.005 + 4 * error

    def check_spread(spread: Estimates, printed: dict[str, float]) -> None:
        for field, value in printed.items():
            closed = getattr(spread, field)
            assert abs(value - closed) < 0.005 + 4 * closed / sqrt(2 * ARTICLE_REPLICATIONS)

    def printed(*values: float) -> dict[str, float]:
        return dict(zip(FIELDS, values, strict=True))

    article = SUMMARY.article
    check(article.expected, article.spread, printed(1.20, 3.75, 3.09, 1.99))
    check_spread(article.spread, printed(0.17, 0.16, 0.17, 0.26))
    inert = SUMMARY.inert_feature
    check(inert.expected, inert.spread, printed(-0.00, 1.75, 1.09, -0.00))
    one_sided = SUMMARY.one_sided
    check(one_sided.expected, one_sided.spread, printed(1.40, 2.92, 2.60, 2.00))

    compliance_table = {
        0.9: (1.80, 0.16, 2.00, 0.18),
        0.6: (1.20, 0.17, 2.00, 0.27),
        0.4: (0.81, 0.17, 2.01, 0.41),
        0.2: (0.39, 0.17, 1.95, 0.84),
    }
    for compliance, (itt, itt_sd, iv, iv_sd) in compliance_table.items():
        expected = expected_estimates(compliance)
        spread = estimator_spread(compliance)
        check(expected, spread, {"intention_to_treat": itt, "wald": iv})
        check_spread(spread, {"intention_to_treat": itt_sd})
        # The delta method understates the Wald spread by about two percent at 20 percent.
        assert iv_sd == pytest.approx(spread.wald, rel=0.05)


def test_the_article_sample_sizes() -> None:
    """131, 294, 662 and 2,649 users per arm, growing as one over the compliance squared."""
    printed = {0.9: 131, 0.6: 294, 0.4: 662, 0.2: 2649}
    assert {row.compliance: round(row.users_per_arm) for row in SUMMARY.sample_sizes} == printed
    assert users_per_arm(0.2) / users_per_arm(0.9) == pytest.approx(20.25)
    # The article's 27 is the variance of y - tau d at its 60 percent design.
    assert residual_variance(0.6) == pytest.approx(27.01)
    for row in SUMMARY.sample_sizes:
        assert row.residual_variance == pytest.approx(27.0, rel=0.05)
    spreads = [estimator_spread(c).wald for c in (0.2, 0.9)]
    assert round(spreads[0] / spreads[1], 1) == 4.6


def test_full_compliance_makes_every_estimator_the_same() -> None:
    """Without always-takers or never-takers, use is assignment and all four contrasts coincide."""
    z, d, y = draw_experiment(np.random.default_rng(8), 500, 1.0, always_takers=0.0)
    assert np.array_equal(z, d)
    estimates = estimate_all(z, d, y)
    assert estimates.as_treated == estimates.intention_to_treat
    assert estimates.per_protocol == estimates.intention_to_treat
    assert estimates.wald == pytest.approx(estimates.intention_to_treat)
    limits = expected_estimates(1.0, 0.0)
    assert _values(limits) == pytest.approx((2.0, 2.0, 2.0, 2.0))
    assert residual_variance(1.0, 0.0) == pytest.approx(25.0)


def test_the_public_estimators_match_the_bundle() -> None:
    """Each estimator on its own gives the value estimate_all reports."""
    z, d, y = draw_experiment(np.random.default_rng(9), 800, 0.5)
    bundle = estimate_all(z, d, y)
    assert intention_to_treat(z, d, y) == bundle.intention_to_treat
    assert as_treated(z, d, y) == bundle.as_treated
    assert per_protocol(z, d, y) == bundle.per_protocol
    assert wald(z, d, y) == bundle.wald


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the estimates; another seed does not."""

    def small(seed: int) -> NDArray[np.float64]:
        return simulate_estimates(np.random.default_rng(seed), 0.5, users=400, replications=10)

    assert np.array_equal(small(11), small(11))
    assert not np.array_equal(small(11), small(12))


def test_invalid_inputs_are_rejected() -> None:
    """Impossible shares, empty groups, bad arrays and bad design parameters are refused."""
    rng = np.random.default_rng(0)
    z, d, y = draw_experiment(rng, 200, 0.5)
    with pytest.raises(ValueError):
        draw_experiment(rng, 200, 0.95, always_takers=0.1)
    with pytest.raises(ValueError):
        draw_experiment(rng, 0, 0.5)
    with pytest.raises(TypeError):
        draw_experiment(rng, 200, True)
    with pytest.raises(ValueError):
        estimate_all(z, d, y[:-1])
    with pytest.raises(ValueError):
        estimate_all(z, d + 1, y)
    with pytest.raises(ValueError):
        estimate_all(z, d, np.append(y[:-1], np.nan))
    with pytest.raises(ValueError):
        intention_to_treat(np.ones(4), np.ones(4), np.arange(4.0))
    with pytest.raises(ValueError):
        wald(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]), np.arange(4.0))
    with pytest.raises(ValueError):
        expected_estimates(0.0)
    with pytest.raises(ValueError):
        estimator_spread(0.5, users=0)
    with pytest.raises(ValueError):
        users_per_arm(0.0)
    with pytest.raises(ValueError):
        users_per_arm(0.5, power=1.0)
    with pytest.raises(ValueError):
        simulate_estimates(rng, 0.5, replications=0)
