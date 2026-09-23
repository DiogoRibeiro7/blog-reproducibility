"""Cumulative regret of Thompson sampling against fixed splits, for the bandits article.

Two variants convert at 10 and 13 percent. Regret is the expected number of
conversions lost against always showing the better variant: each user shown
the worse arm costs the gap between the rates, here 0.03, whatever that user
then does. Using the expected loss rather than the realised one removes the
conversion noise from the curves, so they show only the allocation.

Three policies share a horizon of 20,000 users:

* Thompson sampling keeps a Beta(1 + successes, 1 + failures) posterior on each
  arm, draws once from each, and shows the arm with the largest draw.
* An even split for the whole test alternates the arms, so it loses half the
  gap on every user: 0.015 per user, 300 conversions over the horizon.
* An even split for the first quarter, then everyone to the arm with the higher
  observed rate, loses 75 conversions in the test and nothing afterwards unless
  it picks the wrong arm.

Each curve is the mean over 100 runs. The simulation is the figure's, seed 0,
reproduced draw for draw: all Thompson runs first, then the quarter-split runs,
then the whole-test runs. The article's tables come from a different design
(300 replications from a shared generator, realised rather than expected
losses), so its 19, 77 and 301 lost conversions are close to, but not the same
numbers as, this simulation's.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from blog_reproducibility.common.validation import count, probability

__all__ = [
    "GRID_STEP",
    "HORIZON",
    "QUARTER",
    "RATES",
    "RUNS",
    "SEED",
    "BanditRegretSummary",
    "even_split_regret",
    "example_payload",
    "simulate_regret",
    "thompson_regret",
]

SEED: Final[int] = 0
RATES: Final[tuple[float, float]] = (0.10, 0.13)
HORIZON: Final[int] = 20_000
RUNS: Final[int] = 100
QUARTER: Final[float] = 0.25
# The figure plots the curves every 500 users.
GRID_STEP: Final[int] = 500


@dataclass(frozen=True, slots=True)
class BanditRegretSummary:
    """Mean cumulative regret of each policy at every grid point, starting at zero users."""

    users: tuple[int, ...]
    thompson: tuple[float, ...]
    quarter_then_exploit: tuple[float, ...]
    even_split: tuple[float, ...]


def _rates(rates: tuple[float, ...]) -> NDArray[np.float64]:
    if len(rates) < 2:
        raise ValueError("rates must name at least two arms")
    return np.array([probability(rate, name="rate") for rate in rates], dtype=np.float64)


def thompson_regret(
    rates: tuple[float, ...], horizon: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Cumulative expected regret after each user of one Thompson sampling run."""
    p = _rates(rates)
    users = count(horizon, name="horizon", minimum=1)
    best = p.max()
    shown = np.zeros(p.size)
    successes = np.zeros(p.size)
    lost = np.zeros(users)
    for user in range(users):
        arm = int(np.argmax(rng.beta(1 + successes, 1 + shown - successes)))
        shown[arm] += 1
        successes[arm] += rng.random() < p[arm]
        lost[user] = best - p[arm]
    return np.cumsum(lost)


def even_split_regret(
    rates: tuple[float, ...], horizon: int, share: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Cumulative expected regret of an even split for ``share`` of the horizon, then exploit.

    The test alternates the arms in turn; afterwards every user goes to the arm
    with the highest observed conversion rate, the first on a tie.
    """
    p = _rates(rates)
    users = count(horizon, name="horizon", minimum=1)
    tested = int(users * probability(share, name="share"))
    if tested < p.size:
        raise ValueError("the test must show every arm at least once")
    arms = np.tile(np.arange(p.size), tested // p.size + 1)[:tested]
    converted = rng.random(tested) < p[arms]
    observed = [converted[arms == arm].mean() for arm in range(p.size)]
    chosen = int(np.argmax(observed))
    lost = np.concatenate([p.max() - p[arms], np.full(users - tested, p.max() - p[chosen])])
    return np.cumsum(lost)


def simulate_regret(
    seed: int = SEED,
    *,
    rates: tuple[float, ...] = RATES,
    horizon: int = HORIZON,
    runs: int = RUNS,
    share: float = QUARTER,
    grid_step: int = GRID_STEP,
) -> BanditRegretSummary:
    """Average each policy's regret curve over ``runs`` runs, in the figure's draw order."""
    rng = np.random.default_rng(count(seed, name="seed"))
    replications = count(runs, name="runs", minimum=1)
    step = count(grid_step, name="grid_step", minimum=1)
    thompson = np.mean([thompson_regret(rates, horizon, rng) for _ in range(replications)], axis=0)
    quarter = np.mean(
        [even_split_regret(rates, horizon, share, rng) for _ in range(replications)], axis=0
    )
    even = np.mean(
        [even_split_regret(rates, horizon, 1.0, rng) for _ in range(replications)], axis=0
    )
    grid = np.arange(0, horizon + 1, step)

    def on_grid(curve: NDArray[np.float64]) -> tuple[float, ...]:
        return (0.0, *(float(value) for value in curve[grid[1:] - 1]))

    return BanditRegretSummary(
        users=tuple(int(user) for user in grid),
        thompson=on_grid(thompson),
        quarter_then_exploit=on_grid(quarter),
        even_split=on_grid(even),
    )


def example_payload() -> BanditRegretSummary:
    """Return the three regret curves the figure plots."""
    return simulate_regret()
