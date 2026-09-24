"""The cost of an uneven traffic split, for the article on unequal allocation.

With ``N`` users of whom a share ``f`` go to the treatment arm, the difference in
means has variance

    sigma^2 (1 / (f N) + 1 / ((1 - f) N)) = sigma^2 / (N f (1 - f)),

so relative to an even split the variance, and with it the number of users
needed for a fixed power, is inflated by ``0.25 / (f (1 - f))``. The article
turns that factor into power at 40,000 users, the smallest detectable effect
and the days of traffic needed, for an effect of 0.02 on an outcome with
standard deviation 0.45.

It then gives the three cases where an uneven split is better. Arms with
different spreads are best split in proportion to their standard deviations
(Neyman allocation, ``f = sigma_t / (sigma_t + sigma_c)``). An arm that costs
``c`` times as much per user deserves a share ``1 / (1 + sqrt(c))`` of a fixed
budget. And ``k`` treatment arms sharing one control are compared most
precisely when the control gets ``sqrt(k) / (sqrt(k) + k)`` of the traffic.
The last two optima both cut the variance to ``(1 + sqrt(r))^2 / (2 (1 + r))``
of the even split, with ``r`` the cost ratio or the number of arms.

Everything here is closed form, and the figure draws the inflation factor
itself, so the article's five closed-form tables are reproduced exactly. Its
simulated power check (a generator seeded at 53, 4,000 experiments of 40,000
users at each of three splits) is not reproduced; the power formula is checked
against a smaller simulation in the tests instead.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "ANNOTATED_SHARES",
    "COST_RATIOS",
    "CURVE_POINTS",
    "CURVE_START",
    "CURVE_STOP",
    "DAILY_USERS",
    "DURATION_SHARES",
    "EFFECT",
    "OUTCOME_SD",
    "POWER",
    "SIGNIFICANCE",
    "SPREADS",
    "TABLE_SHARES",
    "TOTAL_USERS",
    "TREATMENT_ARMS",
    "AllocationSummary",
    "CostRow",
    "DurationRow",
    "NeymanRow",
    "SharedControlRow",
    "SplitRow",
    "cost_optimal_share",
    "cost_row",
    "detectable_effect",
    "duration_row",
    "example_payload",
    "neyman_row",
    "neyman_share",
    "power_at",
    "shared_control_row",
    "square_root_control_share",
    "split_row",
    "users_for_power",
    "variance_curve",
    "variance_factor",
]

EFFECT: Final[float] = 0.02
OUTCOME_SD: Final[float] = 0.45
TOTAL_USERS: Final[int] = 40_000
DAILY_USERS: Final[int] = 2000
SIGNIFICANCE: Final[float] = 0.05
POWER: Final[float] = 0.80
TABLE_SHARES: Final[tuple[float, ...]] = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
ANNOTATED_SHARES: Final[tuple[float, ...]] = (0.5, 0.3, 0.2, 0.1, 0.05)
DURATION_SHARES: Final[tuple[float, ...]] = (0.5, 0.25, 0.10, 0.05, 0.01)
SPREADS: Final[tuple[tuple[float, float], ...]] = (
    (0.45, 0.45),
    (0.60, 0.30),
    (0.90, 0.30),
    (0.30, 0.90),
)
COST_RATIOS: Final[tuple[int, ...]] = (1, 2, 4, 10)
TREATMENT_ARMS: Final[tuple[int, ...]] = (1, 2, 3, 5, 8)
# The figure's curve: 400 shares from 2 to 98 percent.
CURVE_START: Final[float] = 0.02
CURVE_STOP: Final[float] = 0.98
CURVE_POINTS: Final[int] = 400


@dataclass(frozen=True, slots=True)
class SplitRow:
    """Variance inflation, power and detectable effect at one treatment share."""

    share: float
    variance_factor: float
    power: float
    detectable_effect: float


@dataclass(frozen=True, slots=True)
class DurationRow:
    """Users needed for 80 percent power at one share, and the days they take."""

    share: float
    users: float
    days: float


@dataclass(frozen=True, slots=True)
class NeymanRow:
    """Neyman allocation for one pair of arm spreads, against an even split."""

    treatment_sd: float
    control_sd: float
    best_share: float
    variance_ratio: float
    saving: float


@dataclass(frozen=True, slots=True)
class CostRow:
    """Budget-optimal allocation when a treated user costs ``cost_ratio`` times as much."""

    cost_ratio: float
    best_share: float
    users_affordable: float
    variance_ratio: float


@dataclass(frozen=True, slots=True)
class SharedControlRow:
    """Control share under an equal split and under the square-root rule."""

    treatment_arms: int
    equal_control_share: float
    square_root_control_share: float
    variance_reduction: float


@dataclass(frozen=True, slots=True)
class AllocationSummary:
    """The article's five closed-form tables; the figure's labelled points are ``splits``."""

    splits: tuple[SplitRow, ...]
    durations: tuple[DurationRow, ...]
    neyman: tuple[NeymanRow, ...]
    cost: tuple[CostRow, ...]
    shared_control: tuple[SharedControlRow, ...]


def _share(value: float) -> float:
    return probability(value, name="share", inclusive=False)


def _z_values(significance: float, power: float) -> tuple[float, float]:
    alpha = probability(significance, name="significance", inclusive=False)
    beta = probability(power, name="power", inclusive=False)
    return float(stats.norm.ppf(1 - alpha / 2)), float(stats.norm.ppf(beta))


def variance_factor(share: float) -> float:
    """Variance of the difference in means relative to an even split."""
    f = _share(share)
    return 0.25 / (f * (1 - f))


def variance_curve(
    start: float = CURVE_START, stop: float = CURVE_STOP, points: int = CURVE_POINTS
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the figure's curve: evenly spaced shares and their variance factors."""
    shares = np.linspace(_share(start), _share(stop), count(points, name="points", minimum=2))
    return shares, 0.25 / (shares * (1 - shares))


def _standard_error(total_users: int, share: float, sd: float) -> float:
    n = count(total_users, name="total_users", minimum=2)
    f = _share(share)
    return positive(sd, name="sd") * sqrt(1 / (f * n) + 1 / ((1 - f) * n))


def power_at(
    total_users: int,
    share: float,
    effect: float = EFFECT,
    sd: float = OUTCOME_SD,
    *,
    significance: float = SIGNIFICANCE,
) -> float:
    """Two-sided power of the normal test for a difference in means."""
    z_crit, _ = _z_values(significance, POWER)
    z = positive(effect, name="effect") / _standard_error(total_users, share, sd)
    return float(stats.norm.cdf(z - z_crit) + stats.norm.cdf(-z_crit - z))


def detectable_effect(
    total_users: int,
    share: float,
    sd: float = OUTCOME_SD,
    *,
    significance: float = SIGNIFICANCE,
    power: float = POWER,
) -> float:
    """Smallest effect detected with the given power: ``(z_a + z_b)`` standard errors."""
    z_crit, z_power = _z_values(significance, power)
    return (z_crit + z_power) * _standard_error(total_users, share, sd)


def users_for_power(
    share: float,
    effect: float = EFFECT,
    sd: float = OUTCOME_SD,
    *,
    significance: float = SIGNIFICANCE,
    power: float = POWER,
) -> float:
    """Total users needed for the given power: ``4 (z_a + z_b)^2 sigma^2 / delta^2`` inflated."""
    z_crit, z_power = _z_values(significance, power)
    spread = positive(sd, name="sd")
    delta = positive(effect, name="effect")
    return 4 * (z_crit + z_power) ** 2 * spread**2 / delta**2 * variance_factor(share)


def neyman_share(treatment_sd: float, control_sd: float) -> float:
    """Neyman allocation: the treatment share proportional to its standard deviation."""
    s_t = positive(treatment_sd, name="treatment_sd")
    s_c = positive(control_sd, name="control_sd")
    return s_t / (s_t + s_c)


def cost_optimal_share(cost_ratio: float) -> float:
    """Treatment share minimising variance for a fixed budget when it costs ``c`` times more."""
    return 1 / (1 + sqrt(positive(cost_ratio, name="cost_ratio")))


def square_root_control_share(treatment_arms: int) -> float:
    """Control share under the square-root rule for ``k`` treatments sharing one control."""
    k = count(treatment_arms, name="treatment_arms", minimum=1)
    return sqrt(k) / (sqrt(k) + k)


def split_row(share: float, total_users: int = TOTAL_USERS) -> SplitRow:
    """One row of the article's first table."""
    return SplitRow(
        share=share,
        variance_factor=variance_factor(share),
        power=power_at(total_users, share),
        detectable_effect=detectable_effect(total_users, share),
    )


def duration_row(share: float, daily_users: int = DAILY_USERS) -> DurationRow:
    """Users for 80 percent power at one share, and the days of traffic they take."""
    users = users_for_power(share)
    return DurationRow(
        share=share, users=users, days=users / count(daily_users, name="daily_users", minimum=1)
    )


def neyman_row(treatment_sd: float, control_sd: float, total_users: int = TOTAL_USERS) -> NeymanRow:
    """Variance of the Neyman split against an even split with the same users."""
    n = count(total_users, name="total_users", minimum=2)
    best = neyman_share(treatment_sd, control_sd)
    even = treatment_sd**2 / (0.5 * n) + control_sd**2 / (0.5 * n)
    optimal = treatment_sd**2 / (best * n) + control_sd**2 / ((1 - best) * n)
    return NeymanRow(
        treatment_sd=treatment_sd,
        control_sd=control_sd,
        best_share=best,
        variance_ratio=optimal / even,
        saving=1 - optimal / even,
    )


def cost_row(cost_ratio: float, total_users: int = TOTAL_USERS) -> CostRow:
    """Spend the budget of an even split at unit cost at the optimal share instead."""
    c = positive(cost_ratio, name="cost_ratio")
    budget = float(count(total_users, name="total_users", minimum=2))
    best = cost_optimal_share(c)
    users = budget / (best * c + (1 - best))
    optimal = OUTCOME_SD**2 * (1 / (best * users) + 1 / ((1 - best) * users))
    even_users = budget / (0.5 * c + 0.5)
    even = OUTCOME_SD**2 * (1 / (0.5 * even_users) + 1 / (0.5 * even_users))
    return CostRow(
        cost_ratio=c, best_share=best, users_affordable=users, variance_ratio=optimal / even
    )


def shared_control_row(treatment_arms: int, total_users: int = TOTAL_USERS) -> SharedControlRow:
    """Variance of one treatment-control comparison, equal split against the square-root rule."""
    k = count(treatment_arms, name="treatment_arms", minimum=1)
    n = count(total_users, name="total_users", minimum=2)

    def comparison_variance(control: float) -> float:
        treated = (1 - control) / k
        return OUTCOME_SD**2 * (1 / (treated * n) + 1 / (control * n))

    equal = 1 / (k + 1)
    rule = square_root_control_share(k)
    return SharedControlRow(
        treatment_arms=k,
        equal_control_share=equal,
        square_root_control_share=rule,
        variance_reduction=1 - comparison_variance(rule) / comparison_variance(equal),
    )


def example_payload() -> AllocationSummary:
    """Return the article's five closed-form tables."""
    return AllocationSummary(
        splits=tuple(split_row(share) for share in TABLE_SHARES),
        durations=tuple(duration_row(share) for share in DURATION_SHARES),
        neyman=tuple(neyman_row(s_t, s_c) for s_t, s_c in SPREADS),
        cost=tuple(cost_row(ratio) for ratio in COST_RATIOS),
        shared_control=tuple(shared_control_row(arms) for arms in TREATMENT_ARMS),
    )
