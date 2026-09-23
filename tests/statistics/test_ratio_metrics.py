"""Check the ratio-metric tests against closed forms, hand-worked cases and the figure.

The delta-method variance is checked against its definition as the variance of
the linearised per-user values, against the binomial variance when every user
has one session, and against the Monte Carlo spread of the ratio. The
closed-form variance ratio ``1 + (m - 1/m) / (kappa + 1)`` is checked against
large samples and gives the article's 1.40 design effect and its 1.53 variance
ratio; the article prints the 1.53 from one sample of 20,000 users, and 1.533
is that sample quantity's large-sample value.

The full figure run (12,000 A/A tests) takes 7 seconds or more, so it is not
repeated here. Its first point (61 of 1,200 tests for both methods) is pinned,
the simulation is checked against a direct transcription of the website's loop,
and the figure's claims are checked with the closed form and a reduced run of
the same design. The full run gives 59, 112, 130, 174 and 330 session-level
rejections of 1,200 for kappa 4, against closed-form rates of 5.0, 8.6, 11.4,
16.2 and 25.6 percent; the delta method gives 59, 56, 52, 50 and 52.

The article's own tables (2,000 replications per cell, kappa 1,000 to 1, a user
bootstrap, power at 20,000 users) run from a different sequence of draws and
are not pinned.
"""

from math import sqrt

import numpy as np
import pytest

from blog_reproducibility.statistics.ratio_metrics import (
    REPLICATIONS,
    article_numbers,
    delta_method_variance,
    delta_method_z,
    design_effect,
    draw_arm,
    false_positive_rates,
    intraclass_correlation,
    propensity_sd,
    session_level_false_positive_rate,
    session_level_z,
    variance_ratio,
)


def _website_loop(
    seed: int, kappas: tuple[float, ...], sessions: tuple[int, ...], users: int, reps: int
) -> list[tuple[int, int]]:
    """The website generator's loop, transcribed, returning rejection counts per cell."""
    r = np.random.default_rng(seed)

    def experiment(n_users: int, kappa: float, mean_sessions: int) -> list[tuple[np.ndarray, ...]]:
        out: list[tuple[np.ndarray, ...]] = []
        for _ in range(2):
            p = r.beta(0.10 * kappa, 0.90 * kappa, n_users)
            s = 1 + r.poisson(mean_sessions - 1, n_users)
            out.append((s, r.binomial(s, p)))
        return out

    def naive_z(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> float:
        (sa, ca), (sb, cb) = a, b
        pa, pb = ca.sum() / sa.sum(), cb.sum() / sb.sum()
        pooled = (ca.sum() + cb.sum()) / (sa.sum() + sb.sum())
        return float((pb - pa) / np.sqrt(pooled * (1 - pooled) * (1 / sa.sum() + 1 / sb.sum())))

    def delta_var(s: np.ndarray, c: np.ndarray) -> float:
        n, mx, my = len(s), s.mean(), c.mean()
        vx, vy, cxy = s.var(ddof=1), c.var(ddof=1), np.cov(s, c, ddof=1)[0, 1]
        return float((vy - 2 * (my / mx) * cxy + (my / mx) ** 2 * vx) / (mx**2 * n))

    def delta_z(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> float:
        (sa, ca), (sb, cb) = a, b
        return float(
            (cb.sum() / sb.sum() - ca.sum() / sa.sum())
            / np.sqrt(delta_var(sa, ca) + delta_var(sb, cb))
        )

    counts = []
    for kappa in kappas:
        for ms in sessions:
            hits = np.zeros(2)
            for _ in range(reps):
                a, b = experiment(users, kappa, ms)
                hits += np.abs([naive_z(a, b), delta_z(a, b)]) > 1.96
            counts.append((int(hits[0]), int(hits[1])))
    return counts


def test_the_simulation_matches_the_website_loop() -> None:
    """Same draws in the same order, and the same rejections, on a small run."""
    rows = false_positive_rates(
        11, concentrations=(20.0, 4.0), mean_sessions=(1, 3), users=300, replications=40
    )
    expected = _website_loop(11, (20.0, 4.0), (1, 3), 300, 40)

    assert [(round(40 * row.session_level), round(40 * row.delta_method)) for row in rows] == (
        expected
    )


def test_the_figure_starts_from_the_published_first_point() -> None:
    """Seed 0, kappa 20 and one session per user: 61 of 1,200 A/A tests reject for both."""
    (row,) = false_positive_rates(concentrations=(20.0,), mean_sessions=(1,))

    assert row.session_level == row.delta_method == 61 / REPLICATIONS


def test_delta_variance_is_the_variance_of_the_linearised_values() -> None:
    """Var(Y - R X) / (n mean(X)^2), with R the ratio of means."""
    rng = np.random.default_rng(3)
    sessions = 1 + rng.poisson(4, 500)
    conversions = rng.binomial(sessions, rng.beta(0.5, 4.5, 500))
    ratio = conversions.mean() / sessions.mean()
    residual = conversions - ratio * sessions
    expected = residual.var(ddof=1) / (500 * sessions.mean() ** 2)

    assert delta_method_variance(sessions, conversions) == pytest.approx(expected, rel=1e-10)


def test_one_session_per_user_gives_the_binomial_variance() -> None:
    """With X = 1 the delta variance is the sample variance of 0/1 outcomes over n."""
    conversions = np.array([1] * 30 + [0] * 170)
    rate = 30 / 200

    variance = delta_method_variance(np.ones(200), conversions)

    assert variance == pytest.approx(rate * (1 - rate) / 199, rel=1e-12)


def test_a_hand_worked_session_level_z() -> None:
    """10 of 100 sessions against 20 of 100: pooled 0.15, z = 0.1 / sqrt(0.15 0.85 0.02)."""
    control = (np.array([50, 50]), np.array([5, 5]))
    treatment = (np.array([60, 40]), np.array([12, 8]))

    assert session_level_z(control, treatment) == pytest.approx(0.1 / sqrt(0.15 * 0.85 * 0.02))
    assert session_level_z(treatment, control) == pytest.approx(
        -session_level_z(control, treatment)
    )


def test_delta_method_z_uses_both_arms_variances() -> None:
    """The statistic is the difference in ratios over the root of the summed variances."""
    rng = np.random.default_rng(8)
    control = draw_arm(rng, 400, 4.0, 3)
    treatment = draw_arm(rng, 400, 4.0, 3)
    difference = treatment[1].sum() / treatment[0].sum() - control[1].sum() / control[0].sum()
    variance = delta_method_variance(*control) + delta_method_variance(*treatment)

    assert delta_method_z(control, treatment) == pytest.approx(difference / sqrt(variance))


def test_delta_variance_matches_the_monte_carlo_spread() -> None:
    """Over 1,500 arms of 400 users, the delta method and the closed form match the spread."""
    rng = np.random.default_rng(5)
    ratios, estimates = [], []
    for _ in range(1500):
        sessions, conversions = draw_arm(rng, 400, 4.0, 3)
        ratios.append(conversions.sum() / sessions.sum())
        estimates.append(delta_method_variance(sessions, conversions))
    spread = float(np.var(ratios, ddof=1))
    closed_form = 0.1 * 0.9 * variance_ratio(3, 4.0) / (400 * 3)

    assert float(np.mean(estimates)) == pytest.approx(spread, rel=0.1)
    assert closed_form == pytest.approx(spread, rel=0.1)


def test_the_variance_ratio_matches_a_large_sample() -> None:
    """With Poisson session counts the variance ratio tends to 1 + (m - 1/m) rho."""
    rng = np.random.default_rng(9)
    for kappa, m in ((4.0, 3), (4.0, 10), (20.0, 5)):
        sessions, conversions = draw_arm(rng, 200_000, kappa, m)
        rate = conversions.sum() / sessions.sum()
        observed = delta_method_variance(sessions, conversions) / (
            rate * (1 - rate) / sessions.sum()
        )
        assert observed == pytest.approx(variance_ratio(m, kappa), rel=0.02)


def test_fixed_session_counts_give_the_design_effect() -> None:
    """Exactly m sessions per user: the inflation is the textbook 1 + (m - 1) rho."""
    rng = np.random.default_rng(10)
    sessions = np.full(200_000, 5)
    conversions = rng.binomial(sessions, rng.beta(0.4, 3.6, 200_000))
    rate = conversions.sum() / sessions.sum()
    observed = delta_method_variance(sessions, conversions) / (rate * (1 - rate) / sessions.sum())

    assert observed == pytest.approx(design_effect(5, 4.0), rel=0.02)
    assert design_effect(5, 4.0) == pytest.approx(1.8)


def test_the_article_closed_forms() -> None:
    """rho 0.2, a spread of 13 points, design effect 1.40, ratio 1.53, SEs about 20% too small."""
    numbers = article_numbers()
    assert numbers.intraclass_correlation == pytest.approx(0.2)
    assert round(100 * numbers.propensity_sd) == 13
    assert round(numbers.design_effect, 2) == 1.40
    assert round(numbers.variance_ratio, 2) == 1.53
    assert round(numbers.standard_error_shortfall, 1) == 0.2
    # A standard error 20 percent too small turns a 5 percent test into roughly a 10 percent one.
    assert round(session_level_false_positive_rate(1 / 0.8**2), 1) == 0.1


def test_limiting_cases() -> None:
    """One session per user, or identical users, leave nothing for the session test to miss."""
    assert variance_ratio(1, 4.0) == 1.0
    assert design_effect(1, 4.0) == 1.0
    assert variance_ratio(10, 1e12) == pytest.approx(1.0)
    assert intraclass_correlation(1.0) == 0.5
    assert propensity_sd(1e12) == pytest.approx(0.0, abs=1e-6)
    assert session_level_false_positive_rate(1.0) == pytest.approx(0.05, abs=1e-4)
    # Poisson session counts add variance beyond the fixed-m design effect.
    for m in (2, 3, 5, 10):
        assert variance_ratio(m, 4.0) > design_effect(m, 4.0)


def test_the_closed_form_supports_the_figure_claims() -> None:
    """5 percent at one session, above 20 at ten, worse with more heterogeneity."""
    sessions = (1, 2, 3, 5, 10)
    strong = [session_level_false_positive_rate(variance_ratio(m, 4.0)) for m in sessions]
    mild = [session_level_false_positive_rate(variance_ratio(m, 20.0)) for m in sessions]

    assert strong[0] == mild[0] == pytest.approx(0.05, abs=1e-4)
    assert strong == sorted(strong) and mild == sorted(mild)
    assert strong[-1] > 0.2 > mild[-1]
    assert all(s > m for s, m in zip(strong[1:], mild[1:], strict=True))


def test_a_reduced_run_supports_the_figure_claims() -> None:
    """At ten sessions and kappa 4 the session test rejects over 20%; the delta method about 5%."""
    (row,) = false_positive_rates(21, concentrations=(4.0,), mean_sessions=(10,), replications=400)
    margin = 4 * sqrt(row.predicted_session_level * (1 - row.predicted_session_level) / 400)

    assert row.session_level > 0.2
    assert row.session_level == pytest.approx(row.predicted_session_level, abs=margin)
    assert 0.02 < row.delta_method < 0.08


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the rows; another seed does not."""

    def small(seed: int) -> object:
        return false_positive_rates(
            seed, concentrations=(4.0,), mean_sessions=(3,), users=200, replications=60
        )

    assert small(5) == small(5)
    assert small(5) != small(6)


def test_invalid_inputs_are_rejected() -> None:
    """Short or mismatched arms, negative counts, degenerate rates and bad parameters."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        delta_method_variance([1.0], [0.0])
    with pytest.raises(ValueError):
        delta_method_variance([1.0, 2.0], [0.0])
    with pytest.raises(ValueError):
        delta_method_variance([1.0, -2.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        delta_method_variance([0.0, 0.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        session_level_z((np.ones(3), np.zeros(3)), (np.ones(3), np.zeros(3)))
    with pytest.raises(ValueError):
        delta_method_z((np.ones(3), np.ones(3)), (np.ones(3), np.ones(3)))
    with pytest.raises(ValueError):
        draw_arm(rng, 1, 4.0, 3)
    with pytest.raises(ValueError):
        draw_arm(rng, 100, 0.0, 3)
    with pytest.raises(ValueError):
        draw_arm(rng, 100, 4.0, 0)
    with pytest.raises(TypeError):
        draw_arm(rng, 100, 4.0, 2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        design_effect(0.5, 4.0)
    with pytest.raises(ValueError):
        variance_ratio(0.5, 4.0)
    with pytest.raises(ValueError):
        session_level_false_positive_rate(0.0)
    with pytest.raises(ValueError):
        propensity_sd(4.0, base_rate=1.0)
    with pytest.raises(ValueError):
        false_positive_rates(replications=0)
