"""Check the bandit regret simulation against exact accounting and the figure's claims.

The even split's regret is deterministic, half the gap per user, so it is
checked exactly. The quarter split is checked against its own recomputed
choice, and Thompson sampling's first decision is replayed by hand from the
same generator. The full figure payload (100 runs of 20,000 users) takes about
25 seconds, so the claims in the figure's alt text are checked on five runs.

The article's 19, 77 and 301 lost conversions are not pinned: they come from
300 replications drawn from a shared generator and count realised rather than
expected losses. The figure's own simulation ends at 23.7, 75.0 and 300.0.
"""

import numpy as np
import pytest

from blog_reproducibility.machine_learning.bandit_regret import (
    GRID_STEP,
    HORIZON,
    RATES,
    even_split_regret,
    simulate_regret,
    thompson_regret,
)

GAP = RATES[1] - RATES[0]
FEW_RUNS = simulate_regret(runs=5)


def test_whole_test_even_split_loses_half_the_gap_per_user() -> None:
    """Alternating arms loses the gap on every second user, whatever the conversions."""
    curve = even_split_regret(RATES, 1_000, 1.0, np.random.default_rng(11))

    steps = np.diff(curve, prepend=0.0)

    np.testing.assert_allclose(steps[0::2], GAP, atol=1e-12)
    np.testing.assert_allclose(steps[1::2], 0.0, atol=1e-12)
    assert curve[-1] == pytest.approx(15.0)


def test_quarter_split_exploits_the_arm_it_observed_best() -> None:
    """After the test the slope is zero or the gap, as the recomputed choice says."""
    rates = (0.3, 0.32)
    curve = even_split_regret(rates, 400, 0.25, np.random.default_rng(5))

    replay = np.random.default_rng(5)
    arms = np.tile([0, 1], 51)[:100]
    converted = replay.random(100) < np.asarray(rates)[arms]
    chosen = int(np.argmax([converted[arms == arm].mean() for arm in (0, 1)]))

    assert curve[99] == pytest.approx(50 * 0.02)
    np.testing.assert_allclose(np.diff(curve[99:]), 0.02 * (chosen == 0), atol=1e-12)


def test_thompson_first_choice_replays_by_hand() -> None:
    """With flat Beta(1, 1) priors the first user goes to the larger of two uniform draws."""
    curve = thompson_regret(RATES, 1, np.random.default_rng(9))
    first = int(np.argmax(np.random.default_rng(9).beta([1.0, 1.0], [1.0, 1.0])))

    assert curve.tolist() == [pytest.approx(RATES[1] - RATES[first])]


def test_thompson_loses_nothing_between_identical_arms() -> None:
    """Regret is zero when no arm is worse."""
    curve = thompson_regret((0.2, 0.2, 0.2), 500, np.random.default_rng(1))

    assert np.all(curve == 0.0)


def test_thompson_increments_are_zero_or_the_gap() -> None:
    """Each user costs either nothing or the full gap, so the curve never falls."""
    steps = np.diff(thompson_regret(RATES, 2_000, np.random.default_rng(2)), prepend=0.0)

    assert np.all(np.isclose(steps, 0.0) | np.isclose(steps, GAP))


def test_thompson_stops_exploring_a_hopeless_arm() -> None:
    """An arm that never converts is abandoned after a handful of users."""
    curve = thompson_regret((0.0, 1.0), 2_000, np.random.default_rng(3))

    assert curve[-1] < 10
    assert curve[-1] == curve[-1000]


def test_simulation_is_deterministic_under_the_seed() -> None:
    """The same seed gives the same curves; another seed gives different ones."""
    first = simulate_regret(horizon=400, runs=3, grid_step=100)

    assert simulate_regret(horizon=400, runs=3, grid_step=100) == first
    assert simulate_regret(seed=1, horizon=400, runs=3, grid_step=100).thompson != first.thompson


def test_curves_start_at_zero_on_the_figure_grid() -> None:
    """The figure plots every 500 users from zero to the horizon."""
    assert FEW_RUNS.users == tuple(range(0, HORIZON + 1, GRID_STEP))
    assert FEW_RUNS.thompson[0] == FEW_RUNS.quarter_then_exploit[0] == FEW_RUNS.even_split[0] == 0


def test_fixed_splits_lose_300_and_75_conversions() -> None:
    """The whole-test split loses 0.015 per user; the quarter split stops at 75."""
    np.testing.assert_allclose(FEW_RUNS.even_split, [0.015 * user for user in FEW_RUNS.users])
    assert FEW_RUNS.quarter_then_exploit[-1] == pytest.approx(75.0)
    assert FEW_RUNS.quarter_then_exploit[FEW_RUNS.users.index(5_000)] == pytest.approx(75.0)


def test_thompson_loses_quickly_at_first_and_then_almost_stops() -> None:
    """The figure's claim: most of the bandit's loss is in its first 2,000 users."""
    thompson = FEW_RUNS.thompson
    early = thompson[FEW_RUNS.users.index(2_000)]
    late_half = thompson[-1] - thompson[FEW_RUNS.users.index(10_000)]

    assert early > thompson[-1] / 2
    assert late_half < 0.05 * 150
    assert thompson[-1] < FEW_RUNS.quarter_then_exploit[-1] < FEW_RUNS.even_split[-1]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rates": (0.1,)}, ValueError),
        ({"rates": (0.1, 1.5)}, ValueError),
        ({"horizon": 0}, ValueError),
        ({"runs": 0}, ValueError),
        ({"seed": True}, TypeError),
        ({"share": 1.2}, ValueError),
        ({"horizon": 4, "share": 0.25}, ValueError),
        ({"grid_step": 0}, ValueError),
    ],
)
def test_invalid_inputs_are_rejected(kwargs: dict[str, object], error: type[Exception]) -> None:
    """Rates must be probabilities for two or more arms, and the test must see every arm."""
    with pytest.raises(error):
        simulate_regret(**{"horizon": 40, "runs": 1, **kwargs})  # type: ignore[arg-type]
