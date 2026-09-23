"""The weekly cost of a retraining schedule, for the article on how often to retrain.

Quality falls linearly with the age of the model, by ``decay`` per week, and a
unit of quality is worth ``value`` a week. Retraining every ``T`` weeks, the
model spends ages ``0, 1, ..., T - 1``, so the quality foregone averages
``value * decay * (T - 1) / 2`` a week, and a retraining costing ``cost`` adds
``cost / T`` a week. The total is convex in ``T``. Setting the derivative of
its continuous version to zero gives the square-root rule

    T* = sqrt(2 cost / (decay * value)),

and the best whole number of weeks is the floor or the ceiling of ``T*``.

With the article's decay of 0.004, value of 100,000 and retraining cost of
15,000 the rule gives 8.7 weeks, the cheapest whole schedule is nine weeks at
3,267 a week, and every interval from six to thirteen weeks is within ten
percent of it. The model is deterministic. The article's trigger simulations
use separately seeded generators and do not enter the figure, so they are not
reproduced here.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

from blog_reproducibility.common.validation import count, non_negative, positive

__all__ = [
    "COST",
    "DECAY",
    "FIGURE_PERIODS",
    "SCHEDULE_PERIODS",
    "VALUE",
    "CostBreakdown",
    "RetrainingSummary",
    "SensitivityRow",
    "best_period",
    "example_payload",
    "near_optimal_periods",
    "schedule_cost",
    "square_root_period",
]

DECAY: Final[float] = 0.004
VALUE: Final[float] = 100_000.0
COST: Final[float] = 15_000.0
# Intervals drawn in the figure, one to forty weeks.
FIGURE_PERIODS: Final[tuple[int, ...]] = tuple(range(1, 41))
# The article's first table, and its search limits (``range(1, 105)`` and ``range(1, 205)``).
SCHEDULE_PERIODS: Final[tuple[int, ...]] = (2, 4, 8, 13, 26, 52)
HEADLINE_SEARCH: Final[int] = 104
GRID_SEARCH: Final[int] = 204
SENSITIVITY_DECAYS: Final[tuple[float, ...]] = (0.001, 0.004, 0.010)
SENSITIVITY_COSTS: Final[tuple[float, ...]] = (5_000.0, 15_000.0, 50_000.0)
COMPARISON_DECAYS: Final[tuple[float, ...]] = (0.002, 0.004, 0.008)
# "Close to optimal": within this share of the cheapest weekly cost.
FLAT_TOLERANCE: Final[float] = 0.10


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Weekly cost of retraining every ``period`` weeks, split into its two halves."""

    period: int
    quality_lost: float
    retraining: float
    total: float


@dataclass(frozen=True, slots=True)
class SensitivityRow:
    """The cheapest schedule for one decay rate and retraining cost."""

    decay: float
    cost: float
    best_period: int
    weekly_cost: float


@dataclass(frozen=True, slots=True)
class RetrainingSummary:
    """Every number the article derives from the schedule-cost formula."""

    schedule_table: tuple[CostBreakdown, ...]
    weekly: CostBreakdown
    best: CostBreakdown
    square_root_period: float
    near_optimal: tuple[int, int]
    sensitivity: tuple[SensitivityRow, ...]
    decay_comparison: tuple[SensitivityRow, ...]
    curve: tuple[CostBreakdown, ...]


def schedule_cost(
    period: int, *, decay: float = DECAY, value: float = VALUE, cost: float = COST
) -> CostBreakdown:
    """Average weekly cost of retraining every ``period`` weeks."""
    weeks = count(period, name="period", minimum=1)
    rate = non_negative(decay, name="decay")
    worth = non_negative(value, name="value")
    retrain = non_negative(cost, name="cost")
    lost = worth * rate * (weeks - 1) / 2
    amortised = retrain / weeks
    return CostBreakdown(weeks, lost, amortised, lost + amortised)


def best_period(
    *,
    decay: float = DECAY,
    value: float = VALUE,
    cost: float = COST,
    longest: int = GRID_SEARCH,
) -> CostBreakdown:
    """The cheapest whole-week schedule from one to ``longest`` weeks, by search."""
    limit = count(longest, name="longest", minimum=1)
    return min(
        (schedule_cost(p, decay=decay, value=value, cost=cost) for p in range(1, limit + 1)),
        key=lambda row: row.total,
    )


def square_root_period(*, decay: float = DECAY, value: float = VALUE, cost: float = COST) -> float:
    """The continuous optimum ``sqrt(2 cost / (decay * value))``."""
    rate = positive(decay, name="decay")
    worth = positive(value, name="value")
    return sqrt(2 * non_negative(cost, name="cost") / (rate * worth))


def near_optimal_periods(
    tolerance: float = FLAT_TOLERANCE,
    *,
    decay: float = DECAY,
    value: float = VALUE,
    cost: float = COST,
    longest: int = GRID_SEARCH,
) -> tuple[int, int]:
    """Shortest and longest interval whose cost is within ``tolerance`` of the cheapest.

    The cost is convex in the period, so the qualifying periods form one run.
    """
    slack = non_negative(tolerance, name="tolerance")
    limit = count(longest, name="longest", minimum=1)
    cheapest = best_period(decay=decay, value=value, cost=cost, longest=limit).total
    close = [
        p
        for p in range(1, limit + 1)
        if schedule_cost(p, decay=decay, value=value, cost=cost).total <= (1 + slack) * cheapest
    ]
    return close[0], close[-1]


def _row(decay: float, cost: float) -> SensitivityRow:
    best = best_period(decay=decay, cost=cost)
    return SensitivityRow(decay, cost, best.period, best.total)


def example_payload() -> RetrainingSummary:
    """Return the article's schedule table, optimum, sensitivity grid and figure curve."""
    return RetrainingSummary(
        schedule_table=tuple(schedule_cost(p) for p in SCHEDULE_PERIODS),
        weekly=schedule_cost(1),
        best=best_period(longest=HEADLINE_SEARCH),
        square_root_period=square_root_period(),
        near_optimal=near_optimal_periods(),
        sensitivity=tuple(
            _row(decay, cost) for decay in SENSITIVITY_DECAYS for cost in SENSITIVITY_COSTS
        ),
        decay_comparison=tuple(_row(decay, COST) for decay in COMPARISON_DECAYS),
        curve=tuple(schedule_cost(p) for p in FIGURE_PERIODS),
    )
