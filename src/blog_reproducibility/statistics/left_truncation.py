"""Survivorship bias and left truncation, for the article on units missing from operational data.

A fleet is installed at uniform times over ``T = 12`` years, with Weibull
lifetimes of shape ``k = 1.5`` (a rising hazard) and characteristic life six
years at normal sites and four at harsh ones, 40 percent of installations being
harsh. At year twelve the analyst takes the units still in service and follows
them for two years. A unit installed ``a`` years before the snapshot is in the
table only if its lifetime exceeds ``a``, so the snapshot over-represents long
lives and under-represents harsh sites.

Each survivor enters observation at its age ``a`` and leaves at failure or at
``a + 2``. The Kaplan-Meier estimator multiplies ``1 - d_t / n_t`` over the
distinct failure ages ``t``, with ``d_t`` failures among the ``n_t`` units at risk.
Counting every survivor at risk from age zero (the naive curve) credits it with
early years in which it could not have been seen to fail. Left truncation, or
delayed entry, puts a unit in the risk set at age ``t`` only if it entered before
``t`` and had not left: ``n_t = #{entry < t} - #{exit < t}``, since every unit
leaves after it enters. That count is what is computed here, with a cumulative
product; it gives the article's loop bit for bit in a fraction of the time.

With ``S_j(t) = exp(-(t / s_j)^k)`` at site ``j`` and ``R_j(h)``, ``M_j(h)`` the
integrals of ``S_j(t)`` and ``t S_j(t)`` over ``[0, h]`` (incomplete gamma
functions), the snapshot has closed forms: a share ``sum_j w_j R_j(T) / T`` in
service, a survivor age with density proportional to the mixture survival on
``[0, T]``, survivors' mean lifetime ``sum_j w_j [2 M_j(T) + T (mu_j - R_j(T))] /
sum_j w_j R_j(T)``, and a two-year failure probability among survivors of
``sum_j w_j [R_j(T) - R_j(T + 2) + R_j(2)] / sum_j w_j R_j(T)``. A new unit fails
within two years with probability ``1 - S_j(2)`` at a rate of ``(1 - S_j(2)) /
R_j(2)`` failures per unit-year.

The figure and the article are one computation from a generator seeded at 0:
site, installation time and lifetime for 20,000 units, drawn in that order (the
article's lifetimes are one scalar draw per unit, the same stream), then, for the
rates of new units, the sites and lifetimes of 20,000 more from the same
generator. Every number the article prints is reproduced and pinned.
"""

from dataclasses import dataclass
from math import exp, gamma, log
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special

from blog_reproducibility.common.validation import count, non_negative, positive, probability

__all__ = [
    "FOLLOW_UP",
    "FRESH_UNITS",
    "GRID_POINTS",
    "GRID_STOP",
    "INSTALLATION_YEARS",
    "OLD_AGE",
    "SEED",
    "SITES",
    "SITE_PROBABILITIES",
    "SITE_SCALES",
    "UNITS",
    "WEIBULL_SHAPE",
    "ClosedForms",
    "Cohort",
    "Fleet",
    "LeftTruncationSummary",
    "MedianEstimates",
    "SiteRow",
    "SnapshotNumbers",
    "SurvivalCurves",
    "closed_forms",
    "empirical_survival",
    "example_payload",
    "failure_rate_from_installation",
    "fleet_median",
    "kaplan_meier",
    "km_median",
    "restricted_mean_life",
    "simulate_fleet",
    "simulate_new_units",
    "study_cohort",
    "survival_on_grid",
    "weibull_median",
    "weibull_survival",
]

SEED: Final[int] = 0
UNITS: Final[int] = 20_000
WEIBULL_SHAPE: Final[float] = 1.5
# Characteristic life in years at each kind of site, and the share of installations there.
SITES: Final[tuple[str, ...]] = ("normal", "harsh")
SITE_SCALES: Final[tuple[float, ...]] = (6.0, 4.0)
SITE_PROBABILITIES: Final[tuple[float, ...]] = (0.6, 0.4)
INSTALLATION_YEARS: Final[float] = 12.0
FOLLOW_UP: Final[float] = 2.0
# The figure's age grid: 201 points from 0 to 10 years.
GRID_STOP: Final[float] = 10.0
GRID_POINTS: Final[int] = 201
# The article's new units, drawn after the fleet from the same generator.
FRESH_UNITS: Final[int] = 20_000
# "A third of the survivors are older than four years."
OLD_AGE: Final[float] = 4.0


@dataclass(frozen=True, slots=True)
class Fleet:
    """Every unit ever installed: its site (an index into ``SITES``), scale, install time, life."""

    site: NDArray[np.int64]
    scale: NDArray[np.float64]
    install: NDArray[np.float64]
    life: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Cohort:
    """The units in service at the snapshot, on the age scale: entry, exit and failure."""

    site: NDArray[np.int64]
    entry: NDArray[np.float64]
    exit: NDArray[np.float64]
    event: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SurvivalCurves:
    """The figure's three curves on its age grid."""

    ages: tuple[float, ...]
    reference: tuple[float, ...]
    naive: tuple[float, ...]
    truncated: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MedianEstimates:
    """Median lifetime four ways; ``None`` where fewer than half the units fail."""

    snapshot_clock: float | None
    naive: float | None
    truncated: float | None
    reference: float | None


@dataclass(frozen=True, slots=True)
class SiteRow:
    """Medians and failure rates at one kind of site."""

    site: str
    true_median: float
    naive_median: float | None
    truncated_median: float | None
    survivor_rate: float
    new_unit_rate: float
    exact_new_unit_rate: float


@dataclass(frozen=True, slots=True)
class SnapshotNumbers:
    """What the snapshot selects, as the article counts it."""

    installed: int
    in_service: int
    in_service_share: float
    harsh_share_installed: float
    harsh_share_in_service: float
    mean_age_in_service: float
    survivors_mean_life: float
    fleet_mean_life: float
    failures: int
    share_older_than_four: float
    survivor_failure_probability: float
    new_unit_failure_probability: float


@dataclass(frozen=True, slots=True)
class ClosedForms:
    """The snapshot's population values, for a fleet of unlimited size."""

    in_service_share: float
    harsh_share_in_service: float
    mean_age_in_service: float
    survivors_mean_life: float
    fleet_mean_life: float
    fleet_median: float
    share_older_than_four: float
    survivor_failure_probability: float


@dataclass(frozen=True, slots=True)
class LeftTruncationSummary:
    """The figure's curves and every number the article prints."""

    curves: SurvivalCurves
    snapshot: SnapshotNumbers
    medians: MedianEstimates
    sites: tuple[SiteRow, ...]
    closed_forms: ClosedForms


def _mixture(
    scales: tuple[float, ...], probabilities: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not scales or len(scales) != len(probabilities):
        raise ValueError("scales and probabilities must be non-empty and of equal length")
    lives = tuple(positive(value, name="scale") for value in scales)
    weights = tuple(probability(value, name="probability") for value in probabilities)
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to one")
    return lives, weights


def weibull_survival(age: float, scale: float, shape: float = WEIBULL_SHAPE) -> float:
    """Chance a unit outlives ``age``: ``exp(-(age / scale)^shape)``."""
    t = non_negative(age, name="age")
    return exp(-((t / positive(scale, name="scale")) ** positive(shape, name="shape")))


def weibull_median(scale: float, shape: float = WEIBULL_SHAPE) -> float:
    """Median lifetime: ``scale (log 2)^(1 / shape)``."""
    return positive(scale, name="scale") * float(log(2) ** (1 / positive(shape, name="shape")))


def _weibull_mean(scale: float, shape: float) -> float:
    return scale * gamma(1 + 1 / shape)


def restricted_mean_life(horizon: float, scale: float, shape: float = WEIBULL_SHAPE) -> float:
    """Expected years in service up to ``horizon``: the integral of the survival function."""
    h = non_negative(horizon, name="horizon")
    s = positive(scale, name="scale")
    k = positive(shape, name="shape")
    return float(s / k * special.gamma(1 / k) * special.gammainc(1 / k, (h / s) ** k))


def _age_moment(horizon: float, scale: float, shape: float) -> float:
    """The integral of ``t S(t)`` over ``[0, horizon]``."""
    return float(
        scale**2
        / shape
        * special.gamma(2 / shape)
        * special.gammainc(2 / shape, (horizon / scale) ** shape)
    )


def failure_rate_from_installation(
    horizon: float, scale: float, shape: float = WEIBULL_SHAPE
) -> float:
    """Failures per unit-year for new units followed to ``horizon``: ``F(h) / R(h)``."""
    h = positive(horizon, name="horizon")
    return (1 - weibull_survival(h, scale, shape)) / restricted_mean_life(h, scale, shape)


def fleet_median(
    scales: tuple[float, ...] = SITE_SCALES,
    probabilities: tuple[float, ...] = SITE_PROBABILITIES,
    shape: float = WEIBULL_SHAPE,
) -> float:
    """Median lifetime of the whole fleet: where the mixture survival crosses one half."""
    lives, weights = _mixture(scales, probabilities)
    k = positive(shape, name="shape")

    def excess(age: float) -> float:
        return (
            sum(w * weibull_survival(age, s, k) for w, s in zip(weights, lives, strict=True)) - 0.5
        )

    upper = 2 * max(weibull_median(s, k) for s in lives)
    return float(optimize.brentq(excess, 0.0, upper, xtol=1e-14))


def closed_forms(
    *,
    scales: tuple[float, ...] = SITE_SCALES,
    probabilities: tuple[float, ...] = SITE_PROBABILITIES,
    shape: float = WEIBULL_SHAPE,
    years: float = INSTALLATION_YEARS,
    follow_up: float = FOLLOW_UP,
    old_age: float = OLD_AGE,
) -> ClosedForms:
    """Population values of the snapshot, from incomplete gamma functions.

    Installation times are uniform on ``[0, T]``, so the age at the snapshot is
    uniform and a unit of age ``a`` survives to it with probability ``S(a)``.
    """
    lives, weights = _mixture(scales, probabilities)
    k = positive(shape, name="shape")
    horizon = positive(years, name="years")
    window = positive(follow_up, name="follow_up")
    old = non_negative(old_age, name="old_age")
    if old > horizon:
        raise ValueError("old_age cannot exceed years")

    def mix(values: list[float]) -> float:
        return sum(w * v for w, v in zip(weights, values, strict=True))

    in_service = [restricted_mean_life(horizon, s, k) for s in lives]
    moments = [_age_moment(horizon, s, k) for s in lives]
    means = [_weibull_mean(s, k) for s in lives]
    alive = mix(in_service)
    return ClosedForms(
        in_service_share=alive / horizon,
        harsh_share_in_service=weights[-1] * in_service[-1] / alive,
        mean_age_in_service=mix(moments) / alive,
        survivors_mean_life=mix(
            [
                2 * moment + horizon * (mean - service)
                for moment, mean, service in zip(moments, means, in_service, strict=True)
            ]
        )
        / alive,
        fleet_mean_life=mix(means),
        fleet_median=fleet_median(lives, weights, k),
        share_older_than_four=mix(
            [
                service - restricted_mean_life(old, s, k)
                for service, s in zip(in_service, lives, strict=True)
            ]
        )
        / alive,
        survivor_failure_probability=mix(
            [
                service
                - restricted_mean_life(horizon + window, s, k)
                + restricted_mean_life(window, s, k)
                for service, s in zip(in_service, lives, strict=True)
            ]
        )
        / alive,
    )


def simulate_fleet(
    rng: np.random.Generator,
    units: int = UNITS,
    *,
    scales: tuple[float, ...] = SITE_SCALES,
    probabilities: tuple[float, ...] = SITE_PROBABILITIES,
    shape: float = WEIBULL_SHAPE,
    years: float = INSTALLATION_YEARS,
) -> Fleet:
    """Draw each unit's site, then its installation time, then its lifetime."""
    n = count(units, name="units", minimum=1)
    lives, weights = _mixture(scales, probabilities)
    k = positive(shape, name="shape")
    site = rng.choice(len(lives), n, p=list(weights)).astype(np.int64)
    install = rng.uniform(0, positive(years, name="years"), n)
    scale = np.asarray(lives, dtype=np.float64)[site]
    life = rng.weibull(k, n) * scale
    return Fleet(site=site, scale=scale, install=install, life=life)


def simulate_new_units(
    rng: np.random.Generator,
    units: int = FRESH_UNITS,
    *,
    scales: tuple[float, ...] = SITE_SCALES,
    probabilities: tuple[float, ...] = SITE_PROBABILITIES,
    shape: float = WEIBULL_SHAPE,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Draw new units' sites, then their lifetimes, followed from installation."""
    n = count(units, name="units", minimum=1)
    lives, weights = _mixture(scales, probabilities)
    k = positive(shape, name="shape")
    site = rng.choice(len(lives), n, p=list(weights)).astype(np.int64)
    life = rng.weibull(k, n) * np.asarray(lives, dtype=np.float64)[site]
    return site, life


def study_cohort(
    fleet: Fleet, *, years: float = INSTALLATION_YEARS, follow_up: float = FOLLOW_UP
) -> Cohort:
    """Units in service at ``years``, entering at their age then and followed for ``follow_up``."""
    snapshot = positive(years, name="years")
    window = positive(follow_up, name="follow_up")
    in_service = fleet.install + fleet.life > snapshot
    entry = (snapshot - fleet.install)[in_service]
    life = fleet.life[in_service]
    return Cohort(
        site=fleet.site[in_service],
        entry=entry,
        exit=np.minimum(life, entry + window),
        event=life <= entry + window,
    )


def _survival_data(
    times: ArrayLike, events: ArrayLike, entry: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("times must be a non-empty, finite 1-D array")
    raw = np.asarray(events)
    if raw.shape != t.shape:
        raise ValueError("events must match times")
    if raw.dtype != np.bool_ and not np.all((raw == 0) | (raw == 1)):
        raise ValueError("events must be booleans or zeros and ones")
    e = raw.astype(bool)
    en = np.zeros_like(t) if entry is None else np.asarray(entry, dtype=np.float64)
    if en.shape != t.shape or not np.all(np.isfinite(en)):
        raise ValueError("entry must be finite and match times")
    if np.any(en >= t):
        raise ValueError("every unit must leave after it enters")
    return t, e, en


def kaplan_meier(
    times: ArrayLike, events: ArrayLike, entry: ArrayLike | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Product-limit survival at each distinct failure time, with optional delayed entry.

    A unit is at risk at ``t`` if it entered before ``t`` and has not left, which,
    since every unit leaves after it enters, is ``#{entry < t} - #{time < t}``.
    """
    t, e, en = _survival_data(times, events, entry)
    event_times, deaths = np.unique(t[e], return_counts=True)
    at_risk = np.searchsorted(np.sort(en), event_times, side="left") - np.searchsorted(
        np.sort(t), event_times, side="left"
    )
    survival: NDArray[np.float64] = np.cumprod(1 - deaths / at_risk)
    return event_times, survival


def km_median(event_times: ArrayLike, survival: ArrayLike) -> float | None:
    """First failure time at which survival is one half or less, as the article reads it."""
    times = np.asarray(event_times, dtype=np.float64)
    values = np.asarray(survival, dtype=np.float64)
    if times.shape != values.shape or times.ndim != 1:
        raise ValueError("event_times and survival must be matching 1-D arrays")
    below = np.flatnonzero(values <= 0.5)
    return float(times[below[0]]) if below.size else None


def survival_on_grid(
    event_times: ArrayLike, survival: ArrayLike, grid: ArrayLike
) -> NDArray[np.float64]:
    """The Kaplan-Meier step function read at each grid age, one before the first failure."""
    times = np.asarray(event_times, dtype=np.float64)
    values = np.asarray(survival, dtype=np.float64)
    ages = np.asarray(grid, dtype=np.float64)
    if times.shape != values.shape or times.ndim != 1:
        raise ValueError("event_times and survival must be matching 1-D arrays")
    if times.size == 0:
        return np.ones_like(ages)
    idx = np.searchsorted(times, ages, side="right") - 1
    curve: NDArray[np.float64] = np.where(idx >= 0, values[np.clip(idx, 0, None)], 1.0)
    return curve


def empirical_survival(lifetimes: ArrayLike, grid: ArrayLike) -> NDArray[np.float64]:
    """Share of lifetimes longer than each grid age: the figure's reference curve."""
    life = np.sort(np.asarray(lifetimes, dtype=np.float64))
    if life.ndim != 1 or life.size == 0:
        raise ValueError("lifetimes must be a non-empty 1-D array")
    longer = life.size - np.searchsorted(life, np.asarray(grid, dtype=np.float64), side="right")
    shares: NDArray[np.float64] = longer / life.size
    return shares


def _unit_rate(failures: NDArray[np.bool_], exposure: NDArray[np.float64]) -> float:
    return float(failures.sum() / exposure.sum())


def example_payload() -> LeftTruncationSummary:
    """Return the figure's curves, the article's numbers and the closed forms beside them."""
    rng = np.random.default_rng(SEED)
    fleet = simulate_fleet(rng)
    cohort = study_cohort(fleet)
    harsh = SITES.index("harsh")

    naive = kaplan_meier(cohort.exit, cohort.event)
    truncated = kaplan_meier(cohort.exit, cohort.event, cohort.entry)
    reference = kaplan_meier(fleet.life, np.ones(fleet.life.size, dtype=bool))
    snapshot_clock = kaplan_meier(cohort.exit - cohort.entry, cohort.event)
    grid = np.linspace(0, GRID_STOP, GRID_POINTS)

    # The article's new units come from the same generator, after the fleet.
    fresh_site, fresh_life = simulate_new_units(rng)
    fresh_exposure = np.minimum(fresh_life, FOLLOW_UP)

    sites = []
    for index, (name, scale) in enumerate(zip(SITES, SITE_SCALES, strict=True)):
        mask = cohort.site == index
        new = fresh_site == index
        sites.append(
            SiteRow(
                site=name,
                true_median=weibull_median(scale),
                naive_median=km_median(*kaplan_meier(cohort.exit[mask], cohort.event[mask])),
                truncated_median=km_median(
                    *kaplan_meier(cohort.exit[mask], cohort.event[mask], cohort.entry[mask])
                ),
                survivor_rate=_unit_rate(
                    cohort.event[mask], cohort.exit[mask] - cohort.entry[mask]
                ),
                new_unit_rate=_unit_rate(fresh_life[new] <= FOLLOW_UP, fresh_exposure[new]),
                exact_new_unit_rate=failure_rate_from_installation(FOLLOW_UP, scale),
            )
        )

    in_service = fleet.install + fleet.life > INSTALLATION_YEARS
    return LeftTruncationSummary(
        curves=SurvivalCurves(
            ages=tuple(float(age) for age in grid),
            reference=tuple(float(value) for value in empirical_survival(fleet.life, grid)),
            naive=tuple(float(value) for value in survival_on_grid(*naive, grid)),
            truncated=tuple(float(value) for value in survival_on_grid(*truncated, grid)),
        ),
        snapshot=SnapshotNumbers(
            installed=int(fleet.life.size),
            in_service=int(in_service.sum()),
            in_service_share=float(in_service.mean()),
            harsh_share_installed=float(np.mean(fleet.site == harsh)),
            harsh_share_in_service=float(np.mean(cohort.site == harsh)),
            mean_age_in_service=float(cohort.entry.mean()),
            survivors_mean_life=float(fleet.life[in_service].mean()),
            fleet_mean_life=float(fleet.life.mean()),
            failures=int(cohort.event.sum()),
            share_older_than_four=float(np.mean(cohort.entry > OLD_AGE)),
            survivor_failure_probability=float(cohort.event.mean()),
            new_unit_failure_probability=sum(
                weight * (1 - weibull_survival(FOLLOW_UP, scale))
                for weight, scale in zip(SITE_PROBABILITIES, SITE_SCALES, strict=True)
            ),
        ),
        medians=MedianEstimates(
            snapshot_clock=km_median(*snapshot_clock),
            naive=km_median(*naive),
            truncated=km_median(*truncated),
            reference=km_median(*reference),
        ),
        sites=tuple(sites),
        closed_forms=closed_forms(),
    )
