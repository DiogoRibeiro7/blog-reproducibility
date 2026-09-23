"""Check proxy selection against the normal-theory closed form and the figure's claims.

For jointly normal proxy and goal, selecting on the proxy moves the goal by
(1 - cost) / 2 times the proxy's gain, and keeping the top share f of the proxy
raises it by sqrt(2) phi(z) / f. The simulation is checked against both on ten
replications per cell; the full figure payload (150 replications) takes about
eight seconds.

The article's goal-gain table uses 200 replications from a shared generator, so
its numbers are not the figure's; they are checked against the closed form
instead, to two decimals. The figure's own simulation gives 0.56, 0.90, 1.24,
1.46 and 1.88 at cost 0, and within 0.004 of zero at cost 1.
"""

from math import pi, sqrt

import numpy as np
import pytest

from blog_reproducibility.machine_learning.proxy_selection import (
    COSTS,
    SHARES,
    draw_pool,
    expected_goal_gain,
    expected_proxy_gain,
    selection_gain,
    simulate_selection,
)

FEW_REPLICATIONS = simulate_selection(replications=10)


def test_selection_gain_on_a_hand_worked_pool() -> None:
    """Keeping the top half by proxy keeps values 1 and 2, whose mean is 1 below 2.5."""
    assert selection_gain([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], 0.5) == -1.0
    assert selection_gain([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 0.25) == 1.5


def test_proxy_gain_closed_form_at_the_median() -> None:
    """Above the median of N(0, 2) the mean is sqrt(2) * sqrt(2 / pi) = 2 / sqrt(pi)."""
    assert expected_proxy_gain(0.5) == pytest.approx(2 / sqrt(pi))
    assert expected_proxy_gain(0.5, 1.0) == pytest.approx(sqrt(2 / pi))
    assert expected_proxy_gain(0.999) < 0.01


def test_goal_gain_is_half_the_proxy_gain_scaled_by_the_cost() -> None:
    """beta = (1 - c) / 2: half at zero cost, nothing at one, reversed at two."""
    proxy = expected_proxy_gain(0.1)

    assert expected_goal_gain(0.1, 0.0) == pytest.approx(proxy / 2)
    assert expected_goal_gain(0.1, 1.0) == 0.0
    assert expected_goal_gain(0.1, 2.0) == pytest.approx(-proxy / 2)


def test_pool_regression_slope_matches_the_cost() -> None:
    """In a large pool the least-squares slope of goal on proxy is (1 - c) / 2."""
    for cost in COSTS:
        proxy, goal = draw_pool(200_000, cost, np.random.default_rng(3))
        slope = np.cov(proxy, goal)[0, 1] / proxy.var(ddof=1)

        assert slope == pytest.approx((1 - cost) / 2, abs=0.01)


def test_article_goal_gain_table_follows_the_closed_form() -> None:
    """The article's table (200 replications, its own generator) sits on the theory."""
    published = {
        0.5: (0.56, 0.28, -0.00, -0.56),
        0.25: (0.90, 0.45, 0.00, -0.90),
        0.1: (1.24, 0.62, 0.00, -1.24),
        0.01: (1.89, 0.95, -0.01, -1.88),
    }
    for share, row in published.items():
        for cost, value in zip(COSTS, row, strict=True):
            assert expected_goal_gain(share, cost) == pytest.approx(value, abs=0.02)
    assert expected_proxy_gain(0.01) == pytest.approx(3.77, abs=0.01)


def test_simulation_matches_the_closed_form() -> None:
    """Ten pools of 20,000 per cell land within about three standard errors of the theory.

    The goal's spread grows with the cost, so its tolerance does too: at cost 2 and
    the top 1 percent, ten means of 200 items have a standard error near 0.045.
    """
    for share, gain in zip(SHARES, FEW_REPLICATIONS.proxy_gain, strict=True):
        assert gain == pytest.approx(expected_proxy_gain(share), abs=0.05)
    for cost, row in zip(COSTS, FEW_REPLICATIONS.goal_gain, strict=True):
        for share, gain in zip(SHARES, row, strict=True):
            assert gain == pytest.approx(expected_goal_gain(share, cost), abs=0.05 + 0.05 * cost)


def test_figure_claims_hold() -> None:
    """Goal rises at half the proxy's rate at cost 0, is flat at 1, and falls at 2."""
    proxy = np.array(FEW_REPLICATIONS.proxy_gain)
    harmless, mild, break_even, costly = (np.array(row) for row in FEW_REPLICATIONS.goal_gain)

    assert np.all(np.diff(proxy) > 0)
    np.testing.assert_allclose(harmless / proxy, 0.5, atol=0.03)
    assert np.all(np.diff(mild) > 0)
    assert np.max(np.abs(break_even)) < 0.05
    assert np.all(np.diff(costly) < 0)
    assert costly[-1] < -1.5 < 3.5 < proxy[-1]


def test_simulation_is_deterministic_under_the_seed() -> None:
    """The same seed gives the same gains; another seed gives different ones."""
    small = {"shares": (0.5, 0.1), "costs": (0.0,), "pool_size": 1_000, "replications": 3}
    first = simulate_selection(**small)  # type: ignore[arg-type]

    assert simulate_selection(**small) == first  # type: ignore[arg-type]
    assert simulate_selection(seed=1, **small) != first  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: selection_gain([1.0, 2.0], [1.0], 0.5), ValueError),
        (lambda: selection_gain([1.0, 2.0], [1.0, 2.0], 0.2), ValueError),
        (lambda: selection_gain([1.0, 2.0], [1.0, 2.0], 1.0), ValueError),
        (lambda: expected_proxy_gain(0.0), ValueError),
        (lambda: expected_proxy_gain(0.5, -1.0), ValueError),
        (lambda: expected_goal_gain(0.5, float("nan")), ValueError),
        (lambda: draw_pool(0, 0.0, np.random.default_rng(0)), ValueError),
        (lambda: simulate_selection(replications=0), ValueError),
        (lambda: simulate_selection(seed=True), TypeError),
    ],
)
def test_invalid_inputs_are_rejected(call: object, error: type[Exception]) -> None:
    """Shares lie strictly between 0 and 1 and must keep a candidate; sizes are positive."""
    assert callable(call)
    with pytest.raises(error):
        call()
