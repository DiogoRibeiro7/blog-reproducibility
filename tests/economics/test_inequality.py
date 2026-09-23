"""Check the Gini coefficient against the Lorenz area and known distributions.

The rank-weighted formula the model evaluates is compared with a direct
trapezoidal integration of the Lorenz curve, and the simulated sample with the
closed-form lognormal Gini, neither of which uses that formula.
"""

import numpy as np
import pytest

from blog_reproducibility.economics.inequality import (
    LOG_SIGMA,
    example_payload,
    gini_coefficient,
    lognormal_gini,
    lorenz_curve,
    simulate_incomes,
)

SUMMARY = example_payload()


def test_gini_equals_twice_the_area_between_equality_and_lorenz() -> None:
    """The figure's title: Gini is twice the shaded area, computed by trapezoids."""
    incomes = simulate_incomes(size=500, seed=3)
    population, income = lorenz_curve(incomes)
    area = float(np.trapezoid(population - income, population))

    assert gini_coefficient(incomes) == pytest.approx(2 * area, abs=1e-12)


def test_lorenz_curve_shape() -> None:
    """The curve runs from the origin to (1, 1), rises, is convex, and lies below equality."""
    population, income = lorenz_curve([4.0, 1.0, 3.0, 2.0])

    assert population.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert income.tolist() == pytest.approx([0.0, 0.1, 0.3, 0.6, 1.0])
    assert np.all(np.diff(income) >= 0)
    assert np.all(np.diff(income, n=2) >= 0)
    assert np.all(income <= population)


def test_gini_boundaries_and_invariance() -> None:
    """Equal incomes give zero; one holder of everything gives (n - 1) / n."""
    assert gini_coefficient([5.0] * 10) == pytest.approx(0.0, abs=1e-15)
    assert gini_coefficient([0.0] * 9 + [1.0]) == pytest.approx(0.9)

    incomes = simulate_incomes(size=200, seed=8)
    gini = gini_coefficient(incomes)
    assert gini_coefficient(incomes * 37.5) == pytest.approx(gini)
    assert gini_coefficient(incomes[::-1]) == pytest.approx(gini)


def test_gini_matches_hand_computed_example() -> None:
    """Mean absolute difference over twice the mean, from every ordered pair."""
    incomes = [1.0, 2.0, 3.0, 10.0]
    pairs = sum(abs(left - right) for left in incomes for right in incomes)
    expected = pairs / (2 * len(incomes) ** 2 * (sum(incomes) / len(incomes)))

    assert gini_coefficient(incomes) == pytest.approx(expected)


def test_lognormal_gini_closed_form() -> None:
    """erf(sigma / 2) agrees with 2 * Phi(sigma / sqrt(2)) - 1 and rises with spread."""
    assert lognormal_gini(LOG_SIGMA) == pytest.approx(0.4521871641994016)
    assert lognormal_gini(0.2) < lognormal_gini(0.85) < lognormal_gini(2.0) < 1


def test_simulated_gini_matches_the_lognormal_population() -> None:
    """Twenty thousand draws land within sampling error of the population Gini."""
    assert SUMMARY.sample_gini == pytest.approx(SUMMARY.lognormal_gini, abs=0.01)
    assert round(SUMMARY.sample_gini, 2) == 0.45
    assert SUMMARY.sample_gini == pytest.approx(0.4484639039052685, rel=1e-9)


def test_simulation_is_deterministic() -> None:
    """The same seed reproduces the same incomes; a different one does not."""
    np.testing.assert_array_equal(simulate_incomes(seed=1), simulate_incomes(seed=1))
    assert not np.array_equal(simulate_incomes(seed=1), simulate_incomes(seed=2))


@pytest.mark.parametrize(
    "incomes",
    [[], [[1.0, 2.0]], [1.0, float("nan")], [1.0, -1.0], [0.0, 0.0]],
)
def test_invalid_incomes_are_rejected(incomes: object) -> None:
    """Empty, nested, non-finite, negative, and all-zero inputs are refused."""
    with pytest.raises(ValueError):
        gini_coefficient(incomes)  # type: ignore[arg-type]


def test_invalid_simulation_settings_are_rejected() -> None:
    """Sizes, spreads, and seeds are validated before drawing."""
    with pytest.raises(ValueError):
        simulate_incomes(size=0)
    with pytest.raises(ValueError):
        simulate_incomes(log_sigma=0.0)
    with pytest.raises(ValueError):
        lognormal_gini(-1.0)
    with pytest.raises(TypeError):
        simulate_incomes(seed=1.5)  # type: ignore[arg-type]
