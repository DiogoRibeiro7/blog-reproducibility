"""Recurrent failures and the mean cumulative function, for the article on repeated failures.

A fleet of 400 machines is 60 percent at normal sites and 40 percent at harsh
ones, where the failure rate doubles. Each machine carries a gamma frailty ``Z``
of mean one and variance 0.5 (shape 2, scale 0.5), fails as a Poisson process
at rate ``lambda_e Z`` with ``lambda_e`` 0.8 or 1.6 a year, is repaired at once,
and is observed from age zero to a uniform age between one and three years.

The mean cumulative function (MCF), the expected failures per machine by age
``t``, is estimated by the Nelson-Aalen sum of ``1 / n(a)`` over failure ages
``a <= t``, where ``n(a)`` counts the machines still observed at ``a``. Averaging
raw counts over every machine instead (the naive count) credits machines with
zeros at ages they never reached, so it agrees with the MCF while every machine
is observed, up to one year, and falls behind after.

The gamma frailty gives closed forms. Given its site and window ``T`` a machine
fails ``N`` times with ``N`` negative binomial: ``P(N = 0) = (1 + lambda T / 2)^-2``
and ``Var N = lambda T + (lambda T)^2 / 2``. A machine's first failure comes at
age ``t`` with survival ``(1 + lambda t / 2)^-2``, so its expected exposure to a
first failure is ``T / (1 + lambda T / 2)``. And a year-one count of zero or more
updates the frailty by conjugacy: over two observed years the expected year-two
count is ``lambda / (1 + lambda / 2)`` after a clean first year and
``lambda [1 - (1 + lambda / 2)^-3] / [1 - (1 + lambda / 2)^-2]`` after a failure.
Averaging over sites and windows gives the fleet's population values.

The figure and the article are one computation from a generator seeded at 0:
sites, frailties and windows for all machines, then each machine's failures in
turn, one exponential gap at a time until its window closes. The figure draws the
MCF for all machines and for each site, and the naive count, on 60 ages from 0.05
to 3 years; every number the article prints from the fleet is reproduced here. Its
no-frailty comparison draws Poisson counts from a generator seeded at 1 and is not
reproduced; the tests compare it with the distribution it comes from.
"""

from dataclasses import dataclass
from math import log
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "BASE_RATE",
    "FRAILTY_VARIANCE",
    "GRID_POINTS",
    "GRID_START",
    "GRID_STOP",
    "HARSH_RATIO",
    "MACHINES",
    "MAX_FOLLOW_UP",
    "MIN_FOLLOW_UP",
    "SEED",
    "SITES",
    "SITE_PROBABILITIES",
    "TABLE_AGES",
    "TOP_SHARE",
    "ClosedForms",
    "Dispersion",
    "Fleet",
    "FleetNumbers",
    "HistoryRow",
    "McfCurves",
    "McfTable",
    "RecurrentEventsSummary",
    "SiteRow",
    "all_events_rate",
    "closed_forms",
    "example_payload",
    "failure_counts",
    "first_failures",
    "first_failure_rate",
    "mean_cumulative_function",
    "naive_average_count",
    "simulate_fleet",
    "top_share",
    "year_two_history",
]

SEED: Final[int] = 0
MACHINES: Final[int] = 400
BASE_RATE: Final[float] = 0.8
HARSH_RATIO: Final[float] = 2.0
SITES: Final[tuple[str, ...]] = ("normal", "harsh")
SITE_PROBABILITIES: Final[tuple[float, ...]] = (0.6, 0.4)
FRAILTY_VARIANCE: Final[float] = 0.5
# Each machine is observed from age zero to a uniform age in this range.
MIN_FOLLOW_UP: Final[float] = 1.0
MAX_FOLLOW_UP: Final[float] = 3.0
# The figure's ages, and the article's table.
GRID_START: Final[float] = 0.05
GRID_STOP: Final[float] = 3.0
GRID_POINTS: Final[int] = 60
TABLE_AGES: Final[tuple[float, ...]] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
TOP_SHARE: Final[float] = 0.1


@dataclass(frozen=True, slots=True)
class Fleet:
    """Each machine's site (an index into ``SITES``), frailty, rate and window, and the failures.

    Failures are in the order they were drawn: machine by machine, by age within
    a machine.
    """

    site: NDArray[np.int64]
    frailty: NDArray[np.float64]
    rate: NDArray[np.float64]
    observed: NDArray[np.float64]
    machine: NDArray[np.int64]
    age: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class McfCurves:
    """The figure: the MCF for all machines and by site, and the naive count, by age."""

    ages: tuple[float, ...]
    all_machines: tuple[float, ...]
    harsh: tuple[float, ...]
    normal: tuple[float, ...]
    naive: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FleetNumbers:
    """What the record holds, and what the first-failure analysis keeps of it."""

    failures: int
    machine_years: float
    never_failed_share: float
    first_failures: int
    first_failure_share: float
    first_failure_rate: float
    all_events_rate: float
    true_rate: float


@dataclass(frozen=True, slots=True)
class McfTable:
    """The article's table: MCF, the true rate times age, and the naive count."""

    ages: tuple[float, ...]
    mcf: tuple[float, ...]
    true: tuple[float, ...]
    naive: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SiteRow:
    """Rates at one kind of site: from all events, from first failures, and the machines' mean."""

    site: str
    machines: int
    all_events_rate: float
    first_failure_rate: float
    true_rate: float


@dataclass(frozen=True, slots=True)
class Dispersion:
    """Failure counts per machine: mean, variance, and the top decile's share of failures."""

    mean_count: float
    variance: float
    top_decile_share: float


@dataclass(frozen=True, slots=True)
class HistoryRow:
    """Machines observed two years, by whether they failed in year one: year-two failures."""

    failed_in_year_one: bool
    machines: int
    year_two_rate: float


@dataclass(frozen=True, slots=True)
class ClosedForms:
    """Population values of the fleet, from the gamma frailty and the uniform window."""

    fleet_rate: float
    never_failed_share: float
    first_failure_rate: float
    count_mean: float
    count_variance: float
    year_two_after_failure: float
    year_two_after_none: float


@dataclass(frozen=True, slots=True)
class RecurrentEventsSummary:
    """The figure's curves and every number the article prints from the fleet."""

    curves: McfCurves
    fleet: FleetNumbers
    table: McfTable
    sites: tuple[SiteRow, ...]
    dispersion: Dispersion
    history: tuple[HistoryRow, ...]
    closed_forms: ClosedForms


def _window(low: float, high: float) -> tuple[float, float]:
    first = positive(low, name="min_follow_up")
    last = positive(high, name="max_follow_up")
    if last <= first:
        raise ValueError("max_follow_up must exceed min_follow_up")
    return first, last


def _site_rates(base_rate: float, harsh_ratio: float) -> NDArray[np.float64]:
    return np.array([1.0, positive(harsh_ratio, name="harsh_ratio")]) * positive(
        base_rate, name="base_rate"
    )


def simulate_fleet(
    rng: np.random.Generator,
    machines: int = MACHINES,
    *,
    base_rate: float = BASE_RATE,
    harsh_ratio: float = HARSH_RATIO,
    harsh_share: float = SITE_PROBABILITIES[1],
    frailty_variance: float = FRAILTY_VARIANCE,
    min_follow_up: float = MIN_FOLLOW_UP,
    max_follow_up: float = MAX_FOLLOW_UP,
) -> Fleet:
    """Draw sites, frailties and windows, then each machine's failures one gap at a time."""
    n = count(machines, name="machines", minimum=1)
    harsh = probability(harsh_share, name="harsh_share")
    variance = positive(frailty_variance, name="frailty_variance")
    low, high = _window(min_follow_up, max_follow_up)
    base = positive(base_rate, name="base_rate")
    ratio = positive(harsh_ratio, name="harsh_ratio")

    site = rng.choice(2, n, p=[1 - harsh, harsh]).astype(np.int64)
    frailty = rng.gamma(1 / variance, variance, n)
    rate = base * np.where(site == 1, ratio, 1.0) * frailty
    observed = rng.uniform(low, high, n)
    machine: list[int] = []
    age: list[float] = []
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / rate[i])
            if t > observed[i]:
                break
            machine.append(i)
            age.append(t)
    return Fleet(
        site=site,
        frailty=frailty,
        rate=rate,
        observed=observed,
        machine=np.array(machine, dtype=np.int64),
        age=np.array(age, dtype=np.float64),
    )


def mean_cumulative_function(
    event_ages: ArrayLike, observed: ArrayLike, ages: ArrayLike
) -> NDArray[np.float64]:
    """Nelson-Aalen MCF: the sum of ``1 / n(a)`` over failure ages ``a <= t``, at each ``t``.

    ``n(a)`` counts windows reaching ``a``. The increments are summed in age
    order, each age's sum over the prefix of failures up to it, as the figure does.
    """
    events = np.sort(np.asarray(event_ages, dtype=np.float64))
    windows = np.sort(np.asarray(observed, dtype=np.float64))
    grid = np.asarray(ages, dtype=np.float64)
    if windows.ndim != 1 or windows.size == 0 or events.ndim != 1 or grid.ndim != 1:
        raise ValueError("event ages, windows and ages must be 1-D, with at least one window")
    at_risk = windows.size - np.searchsorted(windows, events, side="left")
    if np.any(at_risk == 0):
        raise ValueError("every failure must fall inside some observation window")
    increments = 1.0 / at_risk
    ends = np.searchsorted(events, grid, side="right")
    return np.array([increments[:end].sum() for end in ends], dtype=np.float64)


def naive_average_count(
    machine: ArrayLike, event_ages: ArrayLike, members: ArrayLike, ages: ArrayLike
) -> NDArray[np.float64]:
    """Failures by each age among the member machines, divided by the number of members."""
    ids = np.asarray(machine, dtype=np.int64)
    times = np.asarray(event_ages, dtype=np.float64)
    group = np.unique(np.asarray(members, dtype=np.int64))
    if group.size == 0 or ids.shape != times.shape:
        raise ValueError("members must be non-empty and machines must match event ages")
    inside = np.sort(times[np.isin(ids, group)])
    counts = np.searchsorted(inside, np.asarray(ages, dtype=np.float64), side="right")
    naive: NDArray[np.float64] = counts / group.size
    return naive


def failure_counts(fleet: Fleet) -> NDArray[np.int64]:
    """Failures per machine."""
    counts: NDArray[np.int64] = np.bincount(fleet.machine, minlength=fleet.observed.size)
    return counts


def first_failures(fleet: Fleet) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Whether each machine failed, and its age at first failure or at the end of its window."""
    failed = failure_counts(fleet) > 0
    first = np.full(fleet.observed.size, np.inf)
    np.minimum.at(first, fleet.machine, fleet.age)
    return failed, np.where(failed, first, fleet.observed)


def _members(fleet: Fleet, mask: ArrayLike | None) -> NDArray[np.bool_]:
    if mask is None:
        return np.ones(fleet.observed.size, dtype=bool)
    chosen = np.asarray(mask, dtype=bool)
    if chosen.shape != fleet.observed.shape or not chosen.any():
        raise ValueError("mask must select at least one machine")
    return chosen


def all_events_rate(fleet: Fleet, mask: ArrayLike | None = None) -> float:
    """Failures per machine-year observed, over the selected machines."""
    chosen = _members(fleet, mask)
    return float(np.sum(chosen[fleet.machine]) / fleet.observed[chosen].sum())


def first_failure_rate(fleet: Fleet, mask: ArrayLike | None = None) -> float:
    """First failures per machine-year at risk of a first failure, over the selected machines."""
    chosen = _members(fleet, mask)
    failed, exposure = first_failures(fleet)
    return float(failed[chosen].sum() / exposure[chosen].sum())


def top_share(counts: ArrayLike, share: float = TOP_SHARE) -> float:
    """The share of all failures from the machines with the most, the top ``share`` of machines.

    The largest counts are summed after a sort, so ties between machines cannot
    change the result.
    """
    values = np.sort(np.asarray(counts, dtype=np.int64))[::-1]
    top = int(values.size * probability(share, name="share", inclusive=False))
    if values.sum() == 0:
        raise ValueError("counts must include at least one failure")
    return float(values[:top].sum() / values.sum())


def year_two_history(fleet: Fleet, *, year: float = 1.0) -> tuple[HistoryRow, HistoryRow]:
    """Year-two failures per machine observed two years, after a failure in year one or none."""
    length = positive(year, name="year")
    n = fleet.observed.size
    first_year = np.bincount(fleet.machine[fleet.age <= length], minlength=n) > 0
    in_second = (fleet.age > length) & (fleet.age <= 2 * length)
    second_year = np.bincount(fleet.machine[in_second], minlength=n)
    watched = fleet.observed >= 2 * length
    rows = []
    for failed in (True, False):
        group = watched & (first_year == failed)
        rows.append(HistoryRow(failed, int(group.sum()), float(np.mean(second_year[group]))))
    return rows[0], rows[1]


def closed_forms(
    *,
    base_rate: float = BASE_RATE,
    harsh_ratio: float = HARSH_RATIO,
    harsh_share: float = SITE_PROBABILITIES[1],
    frailty_variance: float = FRAILTY_VARIANCE,
    min_follow_up: float = MIN_FOLLOW_UP,
    max_follow_up: float = MAX_FOLLOW_UP,
    year: float = 1.0,
) -> ClosedForms:
    """Population values for a gamma frailty of mean one and a uniform window.

    With ``v`` the frailty variance and ``a = 1 / v`` its shape, a machine with
    site rate ``lambda`` has no failure by age ``t`` with probability
    ``(1 + c t)^-a``, ``c = v lambda``, and ``E[Z exp(-s Z)] = (1 + v s)^-(a + 1)``.
    Both are integrated over the uniform window in closed form.
    """
    harsh = probability(harsh_share, name="harsh_share")
    v = positive(frailty_variance, name="frailty_variance")
    low, high = _window(min_follow_up, max_follow_up)
    length = positive(year, name="year")
    if 2 * length > high:
        raise ValueError("two years of the given length must fit in the longest window")
    weights = (1 - harsh, harsh)
    rates = tuple(float(lam) for lam in _site_rates(base_rate, harsh_ratio))
    a = 1 / v
    width = high - low

    def power_integral(c: float, exponent: float) -> float:
        """The integral of ``(1 + c T)^exponent`` over the window."""
        if exponent == -1:
            return log((1 + c * high) / (1 + c * low)) / c
        rise = (1 + c * high) ** (exponent + 1) - (1 + c * low) ** (exponent + 1)
        return float(rise / (c * (exponent + 1)))

    def no_failure(lam: float) -> float:
        """The chance of no failure in the window, averaged over windows."""
        return power_integral(v * lam, -a) / width

    def exposure(lam: float) -> float:
        """The mean time at risk of a first failure, averaged over windows."""
        c = v * lam
        if a == 1:
            upper, lower = 1 + c * high, 1 + c * low
            return (upper * log(upper) - upper - lower * log(lower) + lower) / (c * c * width)
        return (width - power_integral(c, 1 - a)) / (c * (a - 1) * width)

    mean_t = (low + high) / 2
    mean_t2 = (low**2 + low * high + high**2) / 3
    mean_rate = sum(w * lam for w, lam in zip(weights, rates, strict=True))
    second_moment = sum(w * lam**2 for w, lam in zip(weights, rates, strict=True))
    never = sum(w * no_failure(lam) for w, lam in zip(weights, rates, strict=True))
    at_risk = sum(w * exposure(lam) for w, lam in zip(weights, rates, strict=True))
    count_mean = mean_rate * mean_t
    after_failure = after_none = failed = clean = 0.0
    for w, lam in zip(weights, rates, strict=True):
        m = lam * length
        none = (1 + v * m) ** -a
        # E[Z; no failure in year one], so year two expects m E[Z | history].
        frail_if_none = (1 + v * m) ** -(a + 1)
        after_none += w * m * frail_if_none
        after_failure += w * m * (1 - frail_if_none)
        clean += w * none
        failed += w * (1 - none)
    return ClosedForms(
        fleet_rate=mean_rate,
        never_failed_share=never,
        first_failure_rate=(1 - never) / at_risk,
        count_mean=count_mean,
        count_variance=count_mean + (1 + v) * second_moment * mean_t2 - count_mean**2,
        year_two_after_failure=after_failure / failed,
        year_two_after_none=after_none / clean,
    )


def example_payload() -> RecurrentEventsSummary:
    """Return the figure's curves, the article's numbers and the population values."""
    fleet = simulate_fleet(np.random.default_rng(SEED))
    harsh = fleet.site == SITES.index("harsh")
    everyone = np.ones(fleet.observed.size, dtype=bool)
    grid = np.linspace(GRID_START, GRID_STOP, GRID_POINTS)
    table_ages = np.array(TABLE_AGES)

    def mcf(mask: NDArray[np.bool_], ages: NDArray[np.float64]) -> tuple[float, ...]:
        curve = mean_cumulative_function(fleet.age[mask[fleet.machine]], fleet.observed[mask], ages)
        return tuple(float(value) for value in curve)

    def naive(ages: NDArray[np.float64]) -> tuple[float, ...]:
        curve = naive_average_count(fleet.machine, fleet.age, np.arange(fleet.observed.size), ages)
        return tuple(float(value) for value in curve)

    failed, _ = first_failures(fleet)
    counts = failure_counts(fleet)
    true_rate = float(fleet.rate.mean())
    sites = []
    for index, name in enumerate(SITES):
        mask = fleet.site == index
        sites.append(
            SiteRow(
                site=name,
                machines=int(mask.sum()),
                all_events_rate=all_events_rate(fleet, mask),
                first_failure_rate=first_failure_rate(fleet, mask),
                true_rate=float(fleet.rate[mask].mean()),
            )
        )
    return RecurrentEventsSummary(
        curves=McfCurves(
            ages=tuple(float(age) for age in grid),
            all_machines=mcf(everyone, grid),
            harsh=mcf(harsh, grid),
            normal=mcf(~harsh, grid),
            naive=naive(grid),
        ),
        fleet=FleetNumbers(
            failures=int(fleet.age.size),
            machine_years=float(fleet.observed.sum()),
            never_failed_share=float(np.mean(counts == 0)),
            first_failures=int(failed.sum()),
            first_failure_share=float(failed.sum() / fleet.age.size),
            first_failure_rate=first_failure_rate(fleet),
            all_events_rate=all_events_rate(fleet),
            true_rate=true_rate,
        ),
        table=McfTable(
            ages=TABLE_AGES,
            mcf=mcf(everyone, table_ages),
            true=tuple(float(true_rate * age) for age in table_ages),
            naive=naive(table_ages),
        ),
        sites=tuple(sites),
        dispersion=Dispersion(
            mean_count=float(counts.mean()),
            variance=float(counts.var()),
            top_decile_share=top_share(counts),
        ),
        history=year_two_history(fleet),
        closed_forms=closed_forms(),
    )
