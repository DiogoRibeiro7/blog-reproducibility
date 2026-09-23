"""Check the CUPED simulation against hand-worked cases, theory and the figure's claims.

The block-vectorised simulation is checked against a direct transcription of
the figure's one-experiment-at-a-time loop across a block boundary, and the
estimators on cases whose answer is known exactly. The figure's standard
errors are then pinned (they are the figure's own values; the article prints
none of them) and checked against ``sigma sqrt(2 / n) sqrt(1 - rho^2)``.

The article's tables come from a different design (correlations 0 to 0.85,
4,000 replications, a generator shared across its code blocks) and are not
pinned. Its closed-form sample-size table is.

The alt text says that at a correlation of 0.85 CUPED's standard error is
about half the unadjusted one, "which is the same precision as four times the
users". The ratio is ``sqrt(1 - 0.85^2) = 0.53``, and the equivalent sample is
``1 / (1 - 0.85^2) = 3.6`` times larger: about half and about four times, not
exactly. The figure's grid has no point at 0.85, so the simulated ratio there
is interpolated between 0.8 and 0.9.
"""

import numpy as np
import pytest

from blog_reproducibility.statistics.cuped import (
    CORRELATIONS,
    REPLICATIONS,
    cuped_effect,
    difference_in_means,
    draw_experiment,
    example_payload,
    sample_size_rows,
    simulate_estimates,
    standard_error_rows,
    stratified_effect,
    theoretical_standard_error,
    users_per_arm_for_power,
)

SUMMARY = example_payload()
ROWS = SUMMARY.rows

FIGURE_VALUES = {
    0.0: (0.3100, 0.3101, 0.3103),
    0.1: (0.3202, 0.3186, 0.3191),
    0.2: (0.3221, 0.3174, 0.3187),
    0.3: (0.3214, 0.3088, 0.3110),
    0.4: (0.3169, 0.2872, 0.2940),
    0.5: (0.3113, 0.2767, 0.2824),
    0.6: (0.3192, 0.2550, 0.2642),
    0.7: (0.3174, 0.2311, 0.2442),
    0.8: (0.3013, 0.1902, 0.2073),
    0.9: (0.3102, 0.1324, 0.1712),
}


def _figure_loop(correlations: list[float], replications: int, seed: int = 0) -> list[np.ndarray]:
    """Run a direct transcription of the figure's generator."""
    rng = np.random.default_rng(seed)
    n, sd = 2000, 10.0
    out = []
    for rho in correlations:
        estimates = []
        for _ in range(replications):
            t = np.repeat([0, 1], n)
            x = rng.normal(50, sd, 2 * n)
            y = 50 + rho * (x - 50) + rng.normal(0, sd * np.sqrt(1 - rho**2), 2 * n)
            estimates.append(
                (difference_in_means(t, y), cuped_effect(t, x, y), stratified_effect(t, x, y))
            )
        out.append(np.array(estimates).T)
    return out


def test_the_block_simulation_matches_the_figure_loop() -> None:
    """Same data draw for draw, across a block boundary and a change of correlation."""
    grid = np.linspace(0, 0.9, 10)
    expected = _figure_loop([grid[3], grid[7]], 205)
    rng = np.random.default_rng(0)
    for rho, direct in zip((CORRELATIONS[3], CORRELATIONS[7]), expected, strict=True):
        fast = np.array(simulate_estimates(rho, rng, replications=205))
        np.testing.assert_allclose(fast, direct, rtol=0, atol=1e-11)


def test_draw_experiment_is_the_figure_draw() -> None:
    """One experiment from the public draw equals the figure's first experiment."""
    t, x, y = draw_experiment(0.5, np.random.default_rng(0))
    (direct,) = _figure_loop([0.5], 1)

    assert t.tolist() == [0] * 2000 + [1] * 2000
    assert difference_in_means(t, y) == direct[0, 0]
    assert cuped_effect(t, x, y) == direct[1, 0]


def test_the_figure_standard_errors_are_reproduced() -> None:
    """The figure's three series at four decimals, from 1,200 experiments per correlation."""
    assert len(ROWS) == len(CORRELATIONS) == 10
    for row in ROWS:
        printed = FIGURE_VALUES[round(row.correlation, 1)]
        assert (round(row.difference_in_means, 4), round(row.cuped, 4)) == printed[:2]
        assert round(row.stratified, 4) == printed[2]


def test_cuped_follows_the_square_root_rule() -> None:
    """The simulated spread stays within Monte Carlo noise of sigma sqrt(2/n) sqrt(1 - rho^2)."""
    # The spread of 1,200 draws has a relative standard error of about 2 percent.
    for row in ROWS:
        assert row.cuped == pytest.approx(row.theory, rel=0.05)
        assert row.difference_in_means == pytest.approx(ROWS[0].theory, rel=0.05)
    assert ROWS[0].theory == pytest.approx(10 * np.sqrt(2 / 2000))


def test_about_half_at_a_correlation_of_085() -> None:
    """Interpolated at 0.85, CUPED's standard error is about half the unadjusted one."""
    lower, upper = ROWS[8], ROWS[9]
    cuped = (lower.cuped + upper.cuped) / 2
    unadjusted = (lower.difference_in_means + upper.difference_in_means) / 2
    assert 0.45 < cuped / unadjusted < 0.58
    assert theoretical_standard_error(0.85) / theoretical_standard_error(0.0) == pytest.approx(
        0.527, abs=0.001
    )
    # The equivalent sample is 3.6 times larger: "four times the users" is a round-up.
    variance_ratio = (theoretical_standard_error(0.85) / theoretical_standard_error(0.0)) ** 2
    assert 1 / variance_ratio == pytest.approx(3.6, abs=0.01)


def test_stratification_captures_most_but_not_all() -> None:
    """Quartile strata sit between CUPED and no adjustment once the covariate is informative."""
    for row in ROWS[1:]:
        assert row.cuped <= row.stratified <= row.difference_in_means
    assert ROWS[-1].stratified > 1.2 * ROWS[-1].cuped
    # An uncorrelated covariate costs almost nothing.
    first = ROWS[0]
    assert first.cuped == pytest.approx(first.difference_in_means, rel=0.005)
    assert first.stratified == pytest.approx(first.difference_in_means, rel=0.005)


def test_regression_adjustment_gives_the_same() -> None:
    """The legend's claim: OLS on treatment and the centred covariate matches CUPED."""
    rng = np.random.default_rng(8)
    for _ in range(5):
        t, x, y = draw_experiment(0.7, rng)
        design = np.column_stack([np.ones(t.size), t, x - x.mean()])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        assert beta[1] == pytest.approx(cuped_effect(t, x, y), abs=1e-3)


def test_estimators_on_hand_worked_data() -> None:
    """With the same covariates in both arms and y = 2x + 3t, every estimator returns 3."""
    x_arm = np.arange(1.0, 41.0)
    t = np.repeat([0, 1], 40)
    x = np.concatenate([x_arm, x_arm])
    y = 2 * x + 3 * t

    assert difference_in_means(t, y) == pytest.approx(3.0)
    assert cuped_effect(t, x, y) == pytest.approx(3.0, abs=1e-12)
    assert stratified_effect(t, x, y) == pytest.approx(3.0, abs=1e-12)
    # A covariate imbalance of 5 moves the raw difference to 13. CUPED removes nearly
    # all of it: the pooled theta is 2 + 3 cov(x, t) / var(x) = 2 + 3.75 / 139.5.
    x_shifted = x + 5.0 * t
    shifted = 2 * x_shifted + 3 * t
    assert difference_in_means(t, shifted) == pytest.approx(13.0)
    assert cuped_effect(t, x_shifted, shifted) == pytest.approx(3 - 18.75 / 139.5, abs=1e-12)


def test_the_draw_has_the_stated_correlation() -> None:
    """Covariate and outcome both have spread 10 and correlation rho."""
    t, x, y = draw_experiment(0.7, np.random.default_rng(2), users_per_arm=100_000)
    assert np.corrcoef(x, y)[0, 1] == pytest.approx(0.7, abs=0.005)
    assert x.std() == pytest.approx(10, rel=0.01)
    assert y.std() == pytest.approx(10, rel=0.01)


def test_the_sample_size_table_matches_the_article() -> None:
    """1,570, 1,428, 1,177, 801 and 436 users per arm; the saving is exactly rho^2."""
    printed = {0.0: (1570, 0), 0.3: (1428, 9), 0.5: (1177, 25), 0.7: (801, 49), 0.85: (436, 72)}
    for row in sample_size_rows():
        users, saving = printed[row.correlation]
        assert round(row.users_per_arm) == users
        assert round(100 * row.saving) == saving
        assert row.saving == pytest.approx(row.correlation**2, abs=1e-12)


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the rows; another seed does not."""

    def small(seed: int) -> tuple[object, ...]:
        return standard_error_rows(seed, correlations=(0.5,), replications=20, users_per_arm=200)

    assert small(4) == small(4)
    assert small(4) != small(5)
    assert REPLICATIONS == 1200


def test_invalid_inputs_are_rejected() -> None:
    """Impossible correlations, tiny arms, booleans and bad counts are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        draw_experiment(1.0, rng)
    with pytest.raises(ValueError):
        draw_experiment(0.5, rng, users_per_arm=1)
    with pytest.raises(TypeError):
        draw_experiment(True, rng)
    with pytest.raises(ValueError):
        simulate_estimates(0.5, rng, replications=0)
    with pytest.raises(ValueError):
        theoretical_standard_error(0.5, sd=0.0)
    with pytest.raises(ValueError):
        users_per_arm_for_power(0.5, effect=0.0)
    with pytest.raises(ValueError):
        users_per_arm_for_power(0.5, power=1.0)
    with pytest.raises(ValueError):
        stratified_effect(np.array([0, 1]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0)
