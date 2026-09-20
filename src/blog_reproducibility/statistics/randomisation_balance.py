"""Covariate balance under complete randomisation, for a research draft.

Randomising does not make the arms match on every covariate; it makes the
*distribution* of mismatches known. With twenty units of which ten carry a
binary covariate, split ten and ten, the allocation is hypergeometric, and the
exact distribution says perfect balance happens about a third of the time and a
forty-point imbalance about one time in six.

A second calculation covers adjustment: subtracting a fixed multiple of the
covariate from the outcome before comparing arms. The multiple is chosen in
advance rather than fitted, so the randomisation variance is exact. It helps
when the multiple is near the covariate's real contribution and hurts when it is
not; at twice the true value it is no better than not adjusting at all.

Both calculations are exact, and both are standard library.
"""

from dataclasses import dataclass
from math import comb, sqrt
from statistics import variance

from blog_reproducibility.common.validation import count, real

__all__ = [
    "AllocationRow",
    "BalanceSummary",
    "adjusted_sd",
    "balance_probabilities",
    "example_payload",
]


@dataclass(frozen=True, slots=True)
class AllocationRow:
    """One possible allocation: how many high-covariate units the treated arm got."""

    high_treated: int
    difference: float
    probability: float


@dataclass(frozen=True, slots=True)
class BalanceSummary:
    """Every number the draft reports."""

    allocations: tuple[AllocationRow, ...]
    perfect_balance_probability: float
    imbalance_at_least_40_points: float
    adjustment_sd: dict[str, float]


def balance_probabilities(
    size: int = 20,
    high: int = 10,
    treated: int = 10,
) -> tuple[AllocationRow, ...]:
    """Exact binary-covariate imbalance under complete randomisation.

    ``difference`` is the share of high-covariate units in the treated arm minus
    the share in the control arm, so zero is perfect balance.
    """
    population = count(size, name="size", minimum=1)
    carriers = count(high, name="high", minimum=0)
    assigned = count(treated, name="treated", minimum=1)

    if carriers > population:
        raise ValueError("high must not exceed size")
    if assigned >= population:
        raise ValueError("treated must leave a non-empty control arm")

    denominator = comb(population, assigned)
    lowest = max(0, assigned - (population - carriers))
    highest = min(carriers, assigned)

    return tuple(
        AllocationRow(
            high_treated=taken,
            difference=taken / assigned - (carriers - taken) / (population - assigned),
            probability=comb(carriers, taken)
            * comb(population - carriers, assigned - taken)
            / denominator,
        )
        for taken in range(lowest, highest + 1)
    )


def adjusted_sd(
    outcomes: tuple[float, ...],
    covariates: tuple[float, ...],
    treated: int,
    coefficient: float,
) -> float:
    """Randomisation standard deviation with a fixed adjustment coefficient.

    ``outcomes`` are the fixed untreated potential outcomes. A constant additive
    treatment effect shifts every treated outcome equally and does not change
    this variance, which is why the effect never appears here. The coefficient
    is chosen in advance and not fitted, so no estimation uncertainty enters.
    """
    if len(outcomes) != len(covariates):
        raise ValueError("outcomes and covariates must be the same length")
    assigned = count(treated, name="treated", minimum=1)
    if assigned >= len(outcomes):
        raise ValueError("treated must leave a non-empty control arm")
    slope = real(coefficient, name="coefficient")

    residuals = [
        outcome - slope * covariate for outcome, covariate in zip(outcomes, covariates, strict=True)
    ]
    scale = 1 / assigned + 1 / (len(outcomes) - assigned)
    return sqrt(scale * variance(residuals))


def _example_population() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The draft's twenty units: ten without the covariate, ten with it."""
    covariates = tuple([0.0] * 10 + [1.0] * 10)
    noise = [-2.0, -1.0, 0.0, 1.0, 2.0] * 4
    outcomes = tuple(
        10 + 4 * covariate + error for covariate, error in zip(covariates, noise, strict=True)
    )
    return outcomes, covariates


def example_payload() -> BalanceSummary:
    """Return the numbers the draft reports."""
    allocations = balance_probabilities()
    outcomes, covariates = _example_population()

    return BalanceSummary(
        allocations=allocations,
        # Ten of twenty high-covariate units in the treated arm is perfect balance.
        perfect_balance_probability=next(
            row.probability for row in allocations if row.high_treated == 5
        ),
        imbalance_at_least_40_points=sum(
            row.probability for row in allocations if abs(row.high_treated - 5) >= 2
        ),
        adjustment_sd={
            str(slope): adjusted_sd(outcomes, covariates, 10, slope) for slope in (0, 2, 4, 8)
        },
    )


def adjustment_curve(points: int = 81) -> tuple[tuple[float, float], ...]:
    """Return the coefficient and standard deviation pairs the figure draws."""
    steps = count(points, name="points", minimum=2)
    outcomes, covariates = _example_population()
    return tuple(
        (index / 10, adjusted_sd(outcomes, covariates, 10, index / 10)) for index in range(steps)
    )
