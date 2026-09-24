"""Switchback period length, for the article on randomising time instead of users.

A switchback experiment switches a shared system between treatment and control
in periods of ``L`` minutes, flipping a fair coin for each period. The article
simulates two weeks of minute-level demand: a daily season
``s(t) = 1 + 0.35 sin(2 pi t / 1440 - 1.2)``, an AR(1) state with persistence
0.98 and a stationary standard deviation of 5 percent, about three orders a
minute with values around 20, and a treatment that raises order value by
``delta = 3`` percent. The effect keeps acting for ``c = 8`` minutes after the
system is switched off, at ``gamma = 60`` percent strength, so the first
minutes of a control period that follows a treated one are partly treated.

Each period is one observation. The estimate is the ratio of the mean order
value over treated periods to that over control periods, minus one. Carryover
inflates the control mean: half the control periods follow a treated one, so a
share ``c / (2 L)`` of control time is contaminated and, to first order in the
weights, the estimate is attenuated by

    bias = (1 + delta) / (1 + c gamma delta / (2 L)) - 1 - delta.

The figure isolates that bias by running each assignment twice, with and
without carryover, on the noise-free demand-weighted means, and measures the
spread of the full simulated estimator with and without a burn-in that drops
the first ``c`` minutes of every period. Each period length uses its own
generator seeded at 11: 60 paired assignments first, then 120 full
simulations (AR innovations, assignment, Poisson order counts, normal order
totals). This is reproduced draw for draw. The article's own tables come from
other seeds and replication counts and are not reproduced; its closed-form
bias and contamination columns, and the share of orders a burn-in keeps, are.
The simulated two weeks wrap around, as in the article: a control first
minute that follows a treated last minute counts as a switch-off.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter

from blog_reproducibility.common.validation import count, non_negative, probability

__all__ = [
    "ARTICLE_PERIODS",
    "BIAS_REPLICATIONS",
    "CARRY_FRACTION",
    "CARRY_MINUTES",
    "LIFT",
    "MINUTES",
    "PERIODS",
    "SEED",
    "SPREAD_REPLICATIONS",
    "CarryoverRow",
    "PeriodRow",
    "SwitchbackSummary",
    "carryover_table",
    "closed_form_bias",
    "contaminated_share",
    "daily_season",
    "example_payload",
    "minute_effect",
    "noise_free_estimate",
    "period_estimate",
    "period_row",
    "period_rows",
]

MINUTES: Final[int] = 14 * 1440
MINUTES_PER_DAY: Final[int] = 1440
RATE: Final[float] = 3.0
VALUE: Final[float] = 20.0
SD_ORDER: Final[float] = 8.0
RHO: Final[float] = 0.98
SD_STATE: Final[float] = 0.05
SEASON_AMPLITUDE: Final[float] = 0.35
SEASON_PHASE: Final[float] = 1.2
CARRY_MINUTES: Final[int] = 8
CARRY_FRACTION: Final[float] = 0.6
LIFT: Final[float] = 0.03
SEED: Final[int] = 11
BIAS_REPLICATIONS: Final[int] = 60
SPREAD_REPLICATIONS: Final[int] = 120
PERIODS: Final[tuple[int, ...]] = (15, 30, 60, 120, 180, 360)
ARTICLE_PERIODS: Final[tuple[int, ...]] = (15, 30, 60, 180, 360)


@dataclass(frozen=True, slots=True)
class PeriodRow:
    """Carryover bias and estimator spread at one period length, as fractions of order value.

    ``bias`` is the mean paired difference between the noise-free estimate with
    and without carryover, and ``bias_monte_carlo_error`` its standard error over
    the paired assignments. ``spread`` is the standard deviation of the full
    simulated estimate, and ``spread_with_burn_in`` the same after dropping the
    first ``CARRY_MINUTES`` of every period.
    """

    period: int
    bias: float
    bias_monte_carlo_error: float
    spread: float
    spread_with_burn_in: float


@dataclass(frozen=True, slots=True)
class CarryoverRow:
    """The article's closed forms at one period length."""

    period: int
    contaminated_share: float
    closed_form_bias: float
    share_kept_with_burn_in: float


@dataclass(frozen=True, slots=True)
class SwitchbackSummary:
    """The figure's simulated rows and the article's closed-form carryover table."""

    rows: tuple[PeriodRow, ...]
    closed_form: tuple[CarryoverRow, ...]


def _period_length(period: int, minutes: int = MINUTES) -> int:
    length = count(period, name="period", minimum=1)
    if minutes % length:
        raise ValueError(f"period must divide the {minutes} simulated minutes")
    return length


def daily_season(minutes: int = MINUTES) -> NDArray[np.float64]:
    """Multiplicative daily season of demand and order value at each minute."""
    t = np.arange(count(minutes, name="minutes", minimum=1))
    return 1 + SEASON_AMPLITUDE * np.sin(2 * np.pi * t / MINUTES_PER_DAY - SEASON_PHASE)


def minute_effect(
    assignment: ArrayLike,
    *,
    carryover: bool = True,
    lift: float = LIFT,
    carry_minutes: int = CARRY_MINUTES,
    carry_fraction: float = CARRY_FRACTION,
) -> NDArray[np.float64]:
    """Proportional effect on order value at each minute of a treated/control assignment.

    Treated minutes get ``lift``. With carryover, the ``carry_minutes`` minutes
    from each switch-off get at least ``carry_fraction * lift``. A switch-off is
    a control minute whose predecessor, cyclically, is treated.
    """
    z = np.asarray(assignment)
    if z.ndim != 1 or z.size == 0 or z.dtype != np.bool_:
        raise ValueError("assignment must be a non-empty boolean vector")
    delta = non_negative(lift, name="lift")
    effect = np.where(z, delta, 0.0)
    width = count(carry_minutes, name="carry_minutes")
    fraction = probability(carry_fraction, name="carry_fraction")
    if carryover and width:
        starts = np.flatnonzero(~z & np.roll(z, 1))
        window = (starts[:, None] + np.arange(width)).ravel()
        # Overlapping windows take the larger effect, as the article's slice-by-slice loop does.
        np.maximum.at(effect, window[window < z.size], fraction * delta)
    return effect


def noise_free_estimate(
    assignment: ArrayLike,
    *,
    carryover: bool = True,
    season: ArrayLike | None = None,
) -> float:
    """Relative effect from demand-weighted mean order values, with no sampling noise."""
    z = np.asarray(assignment)
    profile = daily_season(z.size) if season is None else np.asarray(season, dtype=np.float64)
    if profile.shape != z.shape:
        raise ValueError("season must have one value per minute of the assignment")
    if z.all() or not z.any():
        raise ValueError("assignment must contain treated and control minutes")
    value = VALUE * profile * (1 + minute_effect(z, carryover=carryover))
    weights = RATE * profile
    treated = np.average(value[z], weights=weights[z])
    control = np.average(value[~z], weights=weights[~z])
    return float(treated / control - 1)


def period_estimate(
    assignment: ArrayLike,
    orders: ArrayLike,
    totals: ArrayLike,
    *,
    period: int,
    burn_in: int = 0,
) -> float:
    """Ratio of mean treated to mean control period order value, minus one.

    Each period's mean is its order total over its order count, both summed over
    the minutes after the first ``burn_in`` of the period.
    """
    z = np.asarray(assignment, dtype=np.bool_)
    n = np.asarray(orders, dtype=np.float64)
    total = np.asarray(totals, dtype=np.float64)
    if z.ndim != 1 or z.shape != n.shape or z.shape != total.shape or z.size == 0:
        raise ValueError("assignment, orders and totals must be equal-length vectors")
    length = _period_length(period, z.size)
    burn = count(burn_in, name="burn_in")
    if burn >= length:
        raise ValueError("burn_in must be shorter than the period")
    periods = z.size // length
    keep = (np.arange(z.size) % length) >= burn
    label = np.repeat(np.arange(periods), length)
    period_orders = np.bincount(label[keep], weights=n[keep], minlength=periods)
    period_totals = np.bincount(label[keep], weights=total[keep], minlength=periods)
    means = period_totals / np.maximum(period_orders, 1e-9)
    treated = z[::length]
    if treated.all() or not treated.any():
        raise ValueError("assignment must contain treated and control periods")
    return float(means[treated].mean() / means[~treated].mean() - 1)


def period_row(
    period: int,
    *,
    seed: int = SEED,
    bias_replications: int = BIAS_REPLICATIONS,
    spread_replications: int = SPREAD_REPLICATIONS,
) -> PeriodRow:
    """Simulate the figure's bias and spreads at one period length, in the article's draw order."""
    length = _period_length(period)
    rng = np.random.default_rng(count(seed, name="seed"))
    pairs = count(bias_replications, name="bias_replications", minimum=1)
    draws = count(spread_replications, name="spread_replications", minimum=2)
    periods = MINUTES // length
    season = daily_season()
    demand = RATE * season

    # Bias: the same assignments with and without carryover, no sampling noise.
    paired = []
    for _ in range(pairs):
        z = np.repeat(rng.random(periods) < 0.5, length)
        paired.append(
            noise_free_estimate(z, carryover=True, season=season)
            - noise_free_estimate(z, carryover=False, season=season)
        )

    # Spread: the full simulation, analysed with and without a burn-in.
    innovation = SD_STATE * sqrt(1 - RHO**2)
    estimates, burned = [], []
    for _ in range(draws):
        noise = rng.normal(0, innovation, MINUTES)
        state = np.asarray(lfilter([1.0], [1.0, -RHO], noise), dtype=np.float64)
        z = np.repeat(rng.random(periods) < 0.5, length)
        value = VALUE * season * (1 + state) * (1 + minute_effect(z))
        orders = rng.poisson(demand)
        totals = rng.normal(orders * value, SD_ORDER * np.sqrt(np.maximum(orders, 1e-9)))
        estimates.append(period_estimate(z, orders, totals, period=length))
        burned.append(period_estimate(z, orders, totals, period=length, burn_in=CARRY_MINUTES))

    return PeriodRow(
        period=length,
        bias=float(np.mean(paired)),
        bias_monte_carlo_error=float(np.std(paired)) / sqrt(pairs),
        spread=float(np.std(estimates)),
        spread_with_burn_in=float(np.std(burned)),
    )


def period_rows(periods: tuple[int, ...] = PERIODS, *, seed: int = SEED) -> tuple[PeriodRow, ...]:
    """The figure's rows, each period length from its own generator seeded at ``seed``."""
    return tuple(period_row(period, seed=seed) for period in periods)


def contaminated_share(period: int, *, carry_minutes: int = CARRY_MINUTES) -> float:
    """Expected share of control time inside a carryover window: ``min(1, c / L) / 2``."""
    length = count(period, name="period", minimum=1)
    return min(1.0, count(carry_minutes, name="carry_minutes") / length) * 0.5


def closed_form_bias(
    period: int,
    *,
    lift: float = LIFT,
    carry_minutes: int = CARRY_MINUTES,
    carry_fraction: float = CARRY_FRACTION,
) -> float:
    """Attenuation of the relative effect by carryover, from the contaminated share."""
    delta = non_negative(lift, name="lift")
    gamma = probability(carry_fraction, name="carry_fraction")
    share = contaminated_share(period, carry_minutes=carry_minutes)
    return (1 + delta) / (1 + share * gamma * delta) - 1 - delta


def carryover_table(periods: tuple[int, ...] = ARTICLE_PERIODS) -> tuple[CarryoverRow, ...]:
    """The article's closed-form bias, contaminated share and share of orders a burn-in keeps."""
    return tuple(
        CarryoverRow(
            period=period,
            contaminated_share=contaminated_share(period),
            closed_form_bias=closed_form_bias(period),
            share_kept_with_burn_in=max(0.0, 1 - CARRY_MINUTES / period),
        )
        for period in periods
    )


def example_payload() -> SwitchbackSummary:
    """Return the figure's simulated rows and the article's closed-form table."""
    return SwitchbackSummary(rows=period_rows(), closed_form=carryover_table())
