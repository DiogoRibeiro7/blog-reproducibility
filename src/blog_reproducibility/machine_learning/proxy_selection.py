"""Selection on a proxy metric, for the article on Goodhart's law and optimisation pressure.

The proxy is the sum of quality and a manipulable component, and the goal
rewards quality but pays a cost ``c`` per unit of the manipulable part:

    P = q + m,    Y = q - c m + e,

with q and m independent standard normals and e normal noise of standard
deviation 0.5. Optimisation is modelled as keeping the top share of a pool of
20,000 candidates by proxy; the gain in a metric is the mean of the kept
candidates minus the mean of the pool.

Because (P, Y) is jointly normal, E[Y | P] = beta P with
beta = cov(Y, P) / var(P) = (1 - c) / 2. Selecting on P therefore moves the goal
by beta times whatever it moves the proxy: half the reported gain at c = 0,
nothing at c = 1, and the reported gain with its sign reversed at c = 2. For an
infinite pool, keeping the top share f raises the proxy by
sd(P) phi(z) / f with z the upper-f quantile of the standard normal.

The simulation is the figure's, seed 0, reproduced draw for draw: 150 pools per
share and cost, looping over costs, then shares, then replications, and then
150 more pools per share for the proxy's own gain. The article's goal-gain table
runs 200 replications from a shared generator, so its numbers are not this
simulation's; both sit on the closed form above.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from blog_reproducibility.common.validation import count, non_negative, probability, real

__all__ = [
    "COSTS",
    "NOISE_SD",
    "POOL_SIZE",
    "REPLICATIONS",
    "SEED",
    "SHARES",
    "ProxySelectionSummary",
    "draw_pool",
    "example_payload",
    "expected_goal_gain",
    "expected_proxy_gain",
    "selection_gain",
    "simulate_selection",
]

SEED: Final[int] = 0
POOL_SIZE: Final[int] = 20_000
REPLICATIONS: Final[int] = 150
NOISE_SD: Final[float] = 0.5
# Share of candidates kept; smaller is harder optimisation.
SHARES: Final[tuple[float, ...]] = (0.5, 0.25, 0.1, 0.05, 0.01)
COSTS: Final[tuple[float, ...]] = (0.0, 0.5, 1.0, 2.0)


@dataclass(frozen=True, slots=True)
class ProxySelectionSummary:
    """Mean gains at each share kept: the proxy's, and the goal's at each cost."""

    shares: tuple[float, ...]
    costs: tuple[float, ...]
    proxy_gain: tuple[float, ...]
    goal_gain: tuple[tuple[float, ...], ...]


def draw_pool(
    size: int, cost: float, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Proxy and goal values for one candidate pool, drawing q, m, then the goal noise."""
    n = count(size, name="size", minimum=1)
    penalty = real(cost, name="cost")
    quality = rng.normal(0, 1, n)
    manipulable = rng.normal(0, 1, n)
    return quality + manipulable, quality - penalty * manipulable + rng.normal(0, NOISE_SD, n)


def selection_gain(values: ArrayLike, proxy: ArrayLike, share: float) -> float:
    """Mean of ``values`` over the top ``share`` of candidates by proxy, minus the overall mean."""
    metric = np.asarray(values, dtype=np.float64)
    ranking = np.asarray(proxy, dtype=np.float64)
    if metric.ndim != 1 or metric.shape != ranking.shape:
        raise ValueError("values and proxy must be one-dimensional and the same length")
    kept = int(metric.size * probability(share, name="share", inclusive=False))
    if kept < 1:
        raise ValueError("share must keep at least one candidate")
    selected = np.argsort(ranking)[-kept:]
    return float(metric[selected].mean() - metric.mean())


def expected_proxy_gain(share: float, proxy_variance: float = 2.0) -> float:
    """Proxy gain from keeping the top ``share`` of an infinite normal pool."""
    kept = probability(share, name="share", inclusive=False)
    spread = sqrt(non_negative(proxy_variance, name="proxy_variance"))
    return float(spread * norm.pdf(norm.isf(kept)) / kept)


def expected_goal_gain(share: float, cost: float) -> float:
    """Goal gain from keeping the top ``share`` by proxy: (1 - cost) / 2 times the proxy gain."""
    return (1 - real(cost, name="cost")) / 2 * expected_proxy_gain(share)


def simulate_selection(
    seed: int = SEED,
    *,
    shares: tuple[float, ...] = SHARES,
    costs: tuple[float, ...] = COSTS,
    pool_size: int = POOL_SIZE,
    replications: int = REPLICATIONS,
) -> ProxySelectionSummary:
    """Average proxy and goal gains over ``replications`` pools, in the figure's draw order."""
    rng = np.random.default_rng(count(seed, name="seed"))
    reps = count(replications, name="replications", minimum=1)

    goal_gain = []
    for cost in costs:
        row = []
        for share in shares:
            gains = []
            for _ in range(reps):
                proxy, goal = draw_pool(pool_size, cost, rng)
                gains.append(selection_gain(goal, proxy, share))
            row.append(float(np.mean(gains)))
        goal_gain.append(tuple(row))

    proxy_gain = []
    for share in shares:
        gains = []
        for _ in range(reps):
            proxy, _ = draw_pool(pool_size, 0.0, rng)
            gains.append(selection_gain(proxy, proxy, share))
        proxy_gain.append(float(np.mean(gains)))

    return ProxySelectionSummary(
        shares=tuple(shares),
        costs=tuple(costs),
        proxy_gain=tuple(proxy_gain),
        goal_gain=tuple(goal_gain),
    )


def example_payload() -> ProxySelectionSummary:
    """Return the proxy and goal gains the figure plots."""
    return simulate_selection()
