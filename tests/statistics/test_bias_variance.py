"""Check the complexity curve and the lasso paths against closed forms and each figure's claim.

The article prints no numbers from either simulation, so nothing is pinned to
it. Both simulations draw from generators seeded at 20260816 instead of the
site's shared global generator; the bias-variance draws therefore differ from
the published image.

The polynomial fits have a fixed design, so the expected training and test
errors, their gap (the optimism 2 sigma^2 p / n) and the spread of a single
replication are closed forms, and every simulated mean is checked against them.
The lasso path is checked against its optimality conditions, against
least squares at the smallest penalty, and against an independent algorithm
(least angle regression in its lasso form), whose knots fix the order and the
penalties at which the coefficients leave zero.

One claim does not hold in the simulation. The bias-variance alt text says
"training error falls monotonically". Its expectation does, at every degree,
but the simulated mean rises once, from degree 5 to 6 by 0.0006, about a third
of its standard error: each degree draws fresh noise, and the expected fall
there is only sigma^2 / n = 0.0026. The published image appears to rise once too
(degree 7 to 8). The title's weaker "keeps falling" holds overall.
"""

import numpy as np
import pytest
from sklearn.linear_model import lars_path

from blog_reproducibility.statistics.bias_variance import (
    COEFFICIENTS,
    DEGREES,
    DESIGN_POINTS,
    LABEL_THRESHOLD,
    NOISE_SD,
    PENALTIES,
    PENALTY_RATIO,
    REPLICATIONS,
    best_degree,
    design_points,
    example_payload,
    expected_errors,
    lasso_coefficient_path,
    penalty_grid,
    simulate_complexity_curve,
    simulate_lasso_data,
    true_curve,
)

SUMMARY = example_payload()
CURVE = SUMMARY.complexity.curve
EXPECTED = SUMMARY.complexity.expected
DATA = simulate_lasso_data()
PATH = lasso_coefficient_path(DATA)
LASSO = SUMMARY.lasso


def test_expected_errors_match_a_direct_fit_of_the_truth() -> None:
    """The squared bias is the misfit to the noiseless curve; the gap is 2 sigma^2 p / n."""
    x = design_points()
    for row in EXPECTED:
        residual = true_curve(x) - np.polyval(np.polyfit(x, true_curve(x), row.degree), x)
        assert row.squared_bias == pytest.approx(float(np.mean(residual**2)), abs=1e-12)
        p = row.degree + 1
        assert row.training == pytest.approx(row.squared_bias + NOISE_SD**2 * (1 - p / 40))
        assert row.test - row.training == pytest.approx(row.optimism, rel=1e-12)
        assert row.optimism == pytest.approx(2 * NOISE_SD**2 * p / DESIGN_POINTS)


def test_expected_errors_in_limiting_cases() -> None:
    """Interpolation leaves no training error and doubles the test error; no noise, no gap."""
    interpolating = expected_errors(15, points=16)
    assert interpolating.training == pytest.approx(0.0, abs=1e-12)
    assert interpolating.test == pytest.approx(2 * NOISE_SD**2, rel=1e-9)
    noiseless = expected_errors(3, noise_sd=0.0)
    assert noiseless.training == noiseless.test == noiseless.squared_bias
    assert noiseless.training_sd == noiseless.test_sd == 0.0
    # cos is even and the design symmetric, so degrees 2k and 2k + 1 have the same bias.
    assert expected_errors(3).squared_bias == pytest.approx(expected_errors(2).squared_bias)


def test_the_replication_spread_matches_a_simulation() -> None:
    """One replication's training and test error vary as the closed-form quadratic forms say."""
    rng = np.random.default_rng(5)
    x = design_points()
    truth = true_curve(x)
    for degree in (1, 4):
        row = expected_errors(degree)
        noise = rng.normal(0, NOISE_SD, (4000, 2, x.size))
        basis, _ = np.linalg.qr(np.vander(x, degree + 1))
        train = truth + noise[:, 0]
        fitted = train @ basis @ basis.T
        training = np.mean((fitted - train) ** 2, axis=1)
        test = np.mean((fitted - (truth + noise[:, 1])) ** 2, axis=1)
        assert float(np.mean(training)) == pytest.approx(row.training, rel=0.02)
        assert float(np.mean(test)) == pytest.approx(row.test, rel=0.02)
        assert float(np.std(training)) == pytest.approx(row.training_sd, rel=0.05)
        assert float(np.std(test)) == pytest.approx(row.test_sd, rel=0.05)


def test_simulated_errors_match_their_expectations() -> None:
    """Every simulated mean lies within four standard errors of its closed form."""
    assert CURVE.degrees == DEGREES
    for row, training, test in zip(EXPECTED, CURVE.training_error, CURVE.test_error, strict=True):
        assert abs(training - row.training) < 4 * row.training_sd / np.sqrt(REPLICATIONS)
        assert abs(test - row.test) < 4 * row.test_sd / np.sqrt(REPLICATIONS)


def test_the_gap_is_the_overfitting_penalty() -> None:
    """Summed over degrees, the simulated test-training gap is the optimism 2 sigma^2 p / n."""
    gaps = np.subtract(CURVE.test_error, CURVE.training_error)
    optimism = np.array([row.optimism for row in EXPECTED])
    assert float(gaps.sum()) == pytest.approx(float(optimism.sum()), rel=0.05)
    assert np.all(gaps > 0)
    assert np.corrcoef(gaps, optimism)[0, 1] > 0.98


def test_training_error_keeps_falling() -> None:
    """Its expectation falls at every degree; the simulated mean falls at all but one."""
    expected = np.array([row.training for row in EXPECTED])
    assert np.all(np.diff(expected) < 0)
    steps = np.diff(CURVE.training_error)
    rises = np.flatnonzero(steps > 0)
    assert rises.tolist() == [4]  # degree 5 to 6
    rise = float(steps[4])
    assert rise < 0.5 * EXPECTED[5].training_sd / np.sqrt(REPLICATIONS)
    assert CURVE.training_error[-1] < CURVE.training_error[1] < CURVE.training_error[0]


def test_test_error_turns_upward() -> None:
    """Test error bottoms out at degree 2, as its expectation does, then climbs."""
    assert SUMMARY.complexity.best_degree == 2
    assert SUMMARY.complexity.expected_best_degree == 2
    assert best_degree(CURVE) == 2
    expected = np.array([row.test for row in EXPECTED])
    # Once the bias is nearly gone (degree 4 on), each extra coefficient adds about sigma^2 / n.
    assert np.diff(expected[3:]) == pytest.approx(NOISE_SD**2 / DESIGN_POINTS, rel=0.005)
    assert CURVE.test_error[-1] - min(CURVE.test_error) > 0.03
    assert CURVE.test_error[-1] > CURVE.test_error[1] + 0.025


def test_the_complexity_curve_is_deterministic() -> None:
    """The same seed gives the same curve; another seed does not."""
    first = simulate_complexity_curve(replications=4, degrees=(1, 3))
    assert first == simulate_complexity_curve(replications=4, degrees=(1, 3))
    other = simulate_complexity_curve(replications=4, degrees=(1, 3), seed=1)
    assert first.training_error != other.training_error


def test_the_penalty_grid_starts_where_every_coefficient_is_zero() -> None:
    """120 geometric penalties from max |X^T y| / n down to a thousandth of it."""
    grid = penalty_grid(DATA)
    largest = float(np.max(np.abs(DATA.features.T @ DATA.response))) / DATA.response.size
    assert grid.size == PENALTIES
    assert grid[0] == pytest.approx(largest)
    assert grid[-1] == pytest.approx(largest * PENALTY_RATIO)
    assert np.allclose(grid[1:] / grid[:-1], PENALTY_RATIO ** (1 / (PENALTIES - 1)))
    assert np.array_equal(PATH.penalties, grid)
    assert np.all(PATH.coefficients[:, 0] == 0.0)
    assert np.count_nonzero(PATH.coefficients[:, 1]) == 1


def test_the_features_are_standardised() -> None:
    """Zero mean and unit (population) variance, as StandardScaler leaves them."""
    assert np.allclose(DATA.features.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(DATA.features.std(axis=0), 1.0)
    assert DATA.features.shape == (120, len(COEFFICIENTS))


def test_the_path_satisfies_the_lasso_conditions() -> None:
    """At every penalty X^T r / n is alpha sign(beta) on the support and within alpha off it."""
    n = DATA.response.size
    for alpha, beta in zip(PATH.penalties, PATH.coefficients.T, strict=True):
        gradient = DATA.features.T @ (DATA.response - DATA.features @ beta) / n
        support = beta != 0
        assert np.allclose(gradient[support], alpha * np.sign(beta[support]), atol=2e-4)
        assert np.all(np.abs(gradient[~support]) <= alpha + 2e-4)


def test_the_smallest_penalty_is_close_to_least_squares() -> None:
    """At a thousandth of the largest penalty the coefficients are within 0.004 of OLS."""
    final = np.array(LASSO.final_coefficients)
    ols = np.array(LASSO.least_squares)
    assert np.allclose(final, ols, atol=0.004)
    assert np.allclose(final, PATH.coefficients[:, -1])


def test_coefficients_leave_zero_one_by_one() -> None:
    """The support grows by one feature at a time and never loses one, as LARS confirms."""
    active = PATH.coefficients != 0
    sizes = active.sum(axis=0)
    assert set(np.diff(sizes).tolist()) <= {0, 1}
    assert sizes[0] == 0 and sizes[-1] == len(COEFFICIENTS)
    assert np.all(active[:, 1:] >= active[:, :-1])
    assert len(set(LASSO.entry_penalties)) == len(COEFFICIENTS)

    knots, order, _ = lars_path(DATA.features, DATA.response, method="lasso")
    assert tuple(int(j) + 1 for j in order) == LASSO.entry_order == (1, 2, 6, 3, 8, 7, 5, 4)
    grid = list(PATH.penalties)
    for knot, entered in zip(knots[:-1], LASSO.entry_penalties, strict=True):
        index = grid.index(entered)
        assert entered < knot <= grid[index - 1] + 1e-12


def test_the_labelled_survivors_are_the_true_signals() -> None:
    """Only the five non-zero true coefficients end above 0.25 in size."""
    true_signals = tuple(j + 1 for j, beta in enumerate(COEFFICIENTS) if beta != 0.0)
    assert LASSO.labelled == true_signals == (1, 2, 3, 6, 8)
    for j in (4, 5, 7):
        assert abs(LASSO.final_coefficients[j - 1]) < LABEL_THRESHOLD
    for j in true_signals:
        assert np.sign(LASSO.final_coefficients[j - 1]) == np.sign(COEFFICIENTS[j - 1])


def test_invalid_inputs_are_rejected() -> None:
    """Degrees that interpolate or exceed the design, empty lists, bad counts and ratios."""
    with pytest.raises(ValueError, match="fewer coefficients"):
        expected_errors(40)
    with pytest.raises(ValueError, match="empty"):
        simulate_complexity_curve(degrees=())
    with pytest.raises(ValueError):
        simulate_complexity_curve(replications=0)
    with pytest.raises(ValueError):
        expected_errors(2, noise_sd=-0.1)
    with pytest.raises(TypeError):
        expected_errors(2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="below 1"):
        penalty_grid(DATA, ratio=1.0)
    with pytest.raises(ValueError):
        design_points(1)
