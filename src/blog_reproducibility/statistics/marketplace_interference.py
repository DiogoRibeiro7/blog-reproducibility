"""Interference through shared inventory, for the article on marketplace experiments.

A buyer-level A/B test assumes one buyer's treatment does not change another
buyer's outcome. In a market with fixed stock that fails: a treated buyer who
is persuaded to purchase takes a unit a control buyer would have bought. The
article's market has 2,000 buyers a day, arriving in random order, each buying
with probability 10 percent under the old ranking and 12 percent under the new
one if a unit is left. Inventory is fixed per day, so the day's sales are the
first ``inventory`` purchase intents in arrival order.

The true lift of rolling the treatment out is the ratio of capped sales with
everyone treated to capped sales with no one treated. Its expectation is a
closed form in the binomial distribution,

    E[min(X_t, c)] / E[min(X_c, c)] - 1,   X_t ~ Bin(2000, 0.12), X_c ~ Bin(2000, 0.10).

A buyer-level split does not see the cap: arrival order is random, so a
stock-out removes the same fraction of each arm's intents and the ratio of
the arms' sales stays near 1.2 whatever the inventory. Randomising separate
cities, each with its own buyers and stock, reproduces the rollout in each
city and is unbiased.

The figure runs 60 twenty-day experiments of each design at each of eight
inventory levels, from one generator seeded at 0, and takes the true lift from
the buyer-level runs. This is reproduced draw for draw. In the city design
only the number of units sold matters, which is exactly
``min(intents, inventory)``; the arrival order is still drawn, so the
generator advances as in the article, but it is not used. The article's table
(200 replications, other inventories, and a switchback design) comes from a
different sequence of draws and is not reproduced; its worked single-day
example is exact arithmetic and is.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, probability

__all__ = [
    "BUYERS",
    "CONTROL_INTENT",
    "DAYS",
    "INVENTORIES",
    "MARKETS",
    "REPLICATIONS",
    "SEED",
    "TREATED_INTENT",
    "Design",
    "FluidDay",
    "InventoryRow",
    "MarketplaceSummary",
    "day_sales",
    "example_payload",
    "expected_capped_sales",
    "experiment",
    "fluid_day",
    "inventory_row",
    "inventory_rows",
    "rollout_lift",
]

Design = Literal["buyer", "market"]

SEED: Final[int] = 0
BUYERS: Final[int] = 2000
CONTROL_INTENT: Final[float] = 0.10
TREATED_INTENT: Final[float] = 0.12
DAYS: Final[int] = 20
MARKETS: Final[int] = 20
REPLICATIONS: Final[int] = 60
INVENTORIES: Final[tuple[int, ...]] = (320, 280, 250, 230, 215, 205, 200, 195)
WORKED_INVENTORY: Final[int] = 210


@dataclass(frozen=True, slots=True)
class InventoryRow:
    """Mean estimated and true lifts over the replications at one daily inventory.

    ``buyer_level`` and ``market_level`` are the two designs' mean estimates and
    the ``*_spread`` fields their standard deviations across replications.
    ``true_lift`` is the mean rollout lift in the buyer-level replications and
    ``rollout_lift`` its closed-form expectation.
    """

    inventory: int
    buyer_level: float
    buyer_level_spread: float
    market_level: float
    market_level_spread: float
    true_lift: float
    true_lift_spread: float
    rollout_lift: float


@dataclass(frozen=True, slots=True)
class FluidDay:
    """The article's single day with expected intents in place of random ones."""

    inventory: int
    treated_intents: float
    control_intents: float
    treated_sales: float
    control_sales: float
    buyer_level_lift: float
    rollout_lift: float


@dataclass(frozen=True, slots=True)
class MarketplaceSummary:
    """The figure's rows and the article's worked single day."""

    rows: tuple[InventoryRow, ...]
    worked_day: FluidDay


def day_sales(intents: ArrayLike, inventory: int, rng: np.random.Generator) -> NDArray[np.bool_]:
    """Which buyers buy: intents served in a random arrival order until the stock runs out."""
    wants = np.asarray(intents)
    if wants.ndim != 1 or wants.dtype != np.bool_:
        raise ValueError("intents must be a boolean vector")
    stock = count(inventory, name="inventory")
    order = rng.permutation(wants.size)
    served = np.cumsum(wants[order]) <= stock
    sold = np.zeros(wants.size, dtype=np.bool_)
    sold[order] = wants[order] & served
    return sold


def experiment(
    inventory: int,
    design: Design,
    rng: np.random.Generator,
    *,
    days: int = DAYS,
    markets: int = MARKETS,
    buyers: int = BUYERS,
) -> tuple[float, float]:
    """Estimated and true lift from one experiment, in the article's draw order.

    Each day draws every buyer's intent under control and under treatment, which
    give the true rollout sales. The buyer design then splits one market's buyers
    at random; the market design treats half of ``markets`` separate cities.
    """
    stock = count(inventory, name="inventory")
    horizon = count(days, name="days", minimum=1)
    size = count(buyers, name="buyers", minimum=2)
    cities = count(markets, name="markets", minimum=2)
    if cities % 2:
        raise ValueError("markets must be even so that half can be treated")
    if design not in ("buyer", "market"):
        raise ValueError("design must be 'buyer' or 'market'")

    arrival = np.arange(size)
    truth_c = truth_t = est_c = est_t = 0.0
    for _ in range(horizon):
        intent_c = rng.random(size) < CONTROL_INTENT
        intent_t = rng.random(size) < TREATED_INTENT
        truth_c += min(int(np.count_nonzero(intent_c)), stock)
        truth_t += min(int(np.count_nonzero(intent_t)), stock)
        if design == "buyer":
            treated = rng.integers(0, 2, size) == 1
            order = rng.permutation(size)
            wants = np.where(treated, intent_t, intent_c)[order]
            served = wants & (np.cumsum(wants) <= stock)
            sold = int(np.count_nonzero(served))
            sold_t = int(np.count_nonzero(served & treated[order]))
            est_t += sold_t / treated.mean()
            est_c += (sold - sold_t) / (~treated).mean()
        else:
            assignment = np.repeat([0, 1], cities // 2)
            rng.shuffle(assignment)
            n_treated = int(assignment.sum())
            for city in assignment.tolist():
                rate = TREATED_INTENT if city else CONTROL_INTENT
                wanting = int(np.count_nonzero(rng.random(size) < rate))
                # The article draws an arrival order here; sales do not depend on it.
                rng.shuffle(arrival)
                sold = min(wanting, stock)
                if city:
                    est_t += sold / n_treated
                else:
                    est_c += sold / (cities - n_treated)
    return est_t / est_c - 1, truth_t / truth_c - 1


def expected_capped_sales(inventory: int, intent: float, *, buyers: int = BUYERS) -> float:
    """Expected ``min(X, inventory)`` for ``X ~ Binomial(buyers, intent)``."""
    stock = count(inventory, name="inventory")
    rate = probability(intent, name="intent")
    k = np.arange(count(buyers, name="buyers", minimum=1) + 1)
    return float(np.sum(np.minimum(k, stock) * stats.binom.pmf(k, buyers, rate)))


def rollout_lift(inventory: int, *, buyers: int = BUYERS) -> float:
    """Expected true lift: capped sales with everyone treated over those with no one."""
    treated = expected_capped_sales(inventory, TREATED_INTENT, buyers=buyers)
    return treated / expected_capped_sales(inventory, CONTROL_INTENT, buyers=buyers) - 1


def fluid_day(inventory: int = WORKED_INVENTORY, *, buyers: int = BUYERS) -> FluidDay:
    """A day with each arm's intents at their expectations, for a 50/50 buyer split."""
    stock = count(inventory, name="inventory", minimum=1)
    half = count(buyers, name="buyers", minimum=2) / 2
    treated_intents, control_intents = half * TREATED_INTENT, half * CONTROL_INTENT
    served = min(1.0, stock / (treated_intents + control_intents))
    treated_sales, control_sales = treated_intents * served, control_intents * served
    rollout = min(buyers * TREATED_INTENT, stock) / min(buyers * CONTROL_INTENT, stock) - 1
    return FluidDay(
        inventory=stock,
        treated_intents=treated_intents,
        control_intents=control_intents,
        treated_sales=treated_sales,
        control_sales=control_sales,
        buyer_level_lift=treated_sales / control_sales - 1,
        rollout_lift=rollout,
    )


def inventory_row(
    inventory: int, rng: np.random.Generator, *, replications: int = REPLICATIONS
) -> InventoryRow:
    """Run the buyer-level replications, then the city replications, at one inventory."""
    reps = count(replications, name="replications", minimum=1)
    buyer = np.array([experiment(inventory, "buyer", rng) for _ in range(reps)])
    market = np.array([experiment(inventory, "market", rng) for _ in range(reps)])
    return InventoryRow(
        inventory=inventory,
        buyer_level=float(buyer[:, 0].mean()),
        buyer_level_spread=float(buyer[:, 0].std()),
        market_level=float(market[:, 0].mean()),
        market_level_spread=float(market[:, 0].std()),
        true_lift=float(buyer[:, 1].mean()),
        true_lift_spread=float(buyer[:, 1].std()),
        rollout_lift=rollout_lift(inventory),
    )


def inventory_rows(
    inventories: tuple[int, ...] = INVENTORIES,
    *,
    seed: int = SEED,
    replications: int = REPLICATIONS,
) -> tuple[InventoryRow, ...]:
    """The figure's rows, every inventory drawing from one generator in turn."""
    rng = np.random.default_rng(count(seed, name="seed"))
    return tuple(inventory_row(stock, rng, replications=replications) for stock in inventories)


def example_payload() -> MarketplaceSummary:
    """Return the figure's rows and the article's worked single day."""
    return MarketplaceSummary(rows=inventory_rows(), worked_day=fluid_day())
