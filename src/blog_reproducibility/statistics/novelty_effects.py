"""Novelty effects and experiment duration, for the article on tenure versus calendar time.

When a treatment's effect depends on how long a user has been exposed, the
effect is a curve, not a number. The article's novelty pattern gives a user on
their ``k``-th exposed day (``k = 0`` first) a relative lift of

    effect(k) = 0.01 + 0.09 exp(-k / 4),

which starts at 10 percent and decays to a long-run 1 percent. Four thousand
users enter on each of 28 days, split evenly at random, are active on their
first day and then on each later day with probability one half, and produce an
outcome of mean ``20 (1 + effect(k))`` for treated users and 20 for control,
with standard deviation 10.

Grouping outcomes by tenure estimates ``effect(k)`` directly. Grouping by
calendar day mixes every cohort that has entered so far: on day ``d`` the
cohort at tenure ``k`` contributes in proportion to its expected active users,
one for ``k = 0`` and one half after that, so the calendar estimate tracks

    sum_k w_k effect(k) / sum_k w_k,   w_0 = 1, w_k = 1/2 for 0 < k <= d,

a curve that is flatter and later than the truth.

The figure averages 12 runs from one generator seeded at 0, drawing each
cohort's arms and then, day by day, its activity and outcomes. This is
reproduced draw for draw. The article's tables use 20 runs per column in a
different order, so their simulated columns are not reproduced; their
true-effect columns are closed forms and are.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count, probability

__all__ = [
    "DAYS",
    "LONG_RUN_EFFECT",
    "NEW_PER_DAY",
    "RETURN_PROBABILITY",
    "RUNS",
    "SEED",
    "NoveltyRow",
    "NoveltySummary",
    "calendar_expectation",
    "example_payload",
    "novelty_effect",
    "simulate_run",
    "simulate_runs",
]

SEED: Final[int] = 0
RUNS: Final[int] = 12
DAYS: Final[int] = 28
NEW_PER_DAY: Final[int] = 4000
RETURN_PROBABILITY: Final[float] = 0.5
BASE: Final[float] = 20.0
SD: Final[float] = 10.0
LONG_RUN_EFFECT: Final[float] = 0.01
NOVELTY_EXCESS: Final[float] = 0.09
DECAY_DAYS: Final[float] = 4.0


@dataclass(frozen=True, slots=True)
class NoveltyRow:
    """The true effect and both estimates on one day, as relative lifts.

    ``day`` counts from one. ``true_effect`` is the effect at tenure
    ``day - 1``, ``by_tenure`` the estimate for users at that tenure,
    ``by_calendar`` the estimate from all users active on that calendar day,
    and ``calendar_expectation`` the mixture the calendar estimate tracks.
    """

    day: int
    true_effect: float
    by_tenure: float
    by_calendar: float
    calendar_expectation: float


@dataclass(frozen=True, slots=True)
class NoveltySummary:
    """The figure's curves, averaged over ``runs`` simulated experiments."""

    runs: int
    rows: tuple[NoveltyRow, ...]


def novelty_effect(tenure: ArrayLike) -> NDArray[np.float64]:
    """Relative lift for a user on their ``tenure``-th exposed day, counting from zero."""
    k = np.asarray(tenure, dtype=np.float64)
    if np.any(k < 0) or not np.all(np.isfinite(k)):
        raise ValueError("tenure must be finite and non-negative")
    return np.asarray(LONG_RUN_EFFECT + NOVELTY_EXCESS * np.exp(-k / DECAY_DAYS))


def calendar_expectation(
    days: int = DAYS, *, return_probability: float = RETURN_PROBABILITY
) -> NDArray[np.float64]:
    """Effect a calendar-day grouping estimates: tenures weighted by expected active users."""
    horizon = count(days, name="days", minimum=1)
    ret = probability(return_probability, name="return_probability")
    weights = np.where(np.arange(horizon) == 0, 1.0, ret)
    effects = novelty_effect(np.arange(horizon))
    return np.asarray(np.cumsum(weights * effects) / np.cumsum(weights))


def simulate_run(
    rng: np.random.Generator,
    *,
    days: int = DAYS,
    new_per_day: int = NEW_PER_DAY,
    return_probability: float = RETURN_PROBABILITY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One experiment's lift by calendar day and by tenure, in the article's draw order."""
    horizon = count(days, name="days", minimum=1)
    size = count(new_per_day, name="new_per_day", minimum=2)
    ret = probability(return_probability, name="return_probability")
    day_t, day_c, day_nt, day_nc = (np.zeros(horizon) for _ in range(4))
    ten_t, ten_c, ten_nt, ten_nc = (np.zeros(horizon) for _ in range(4))
    for entry in range(horizon):
        arm = rng.integers(0, 2, size)
        for day in range(entry, horizon):
            k = day - entry
            active = np.ones(size, bool) if k == 0 else (rng.random(size) < ret)
            treated = arm[active]
            y = BASE * (1 + novelty_effect(k) * treated) + rng.normal(0, SD, int(active.sum()))
            on, off = treated == 1, treated == 0
            sum_t, sum_c = y[on].sum(), y[off].sum()
            count_t, count_c = on.sum(), off.sum()
            day_t[day] += sum_t
            day_nt[day] += count_t
            day_c[day] += sum_c
            day_nc[day] += count_c
            ten_t[k] += sum_t
            ten_nt[k] += count_t
            ten_c[k] += sum_c
            ten_nc[k] += count_c
    by_day = (day_t / day_nt) / (day_c / day_nc) - 1
    by_tenure = (ten_t / ten_nt) / (ten_c / ten_nc) - 1
    return by_day, by_tenure


def simulate_runs(
    runs: int = RUNS, *, seed: int = SEED, days: int = DAYS, new_per_day: int = NEW_PER_DAY
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean calendar-day and tenure curves over ``runs`` experiments from one generator."""
    rng = np.random.default_rng(count(seed, name="seed"))
    results = [
        simulate_run(rng, days=days, new_per_day=new_per_day)
        for _ in range(count(runs, name="runs", minimum=1))
    ]
    by_day = np.mean([day for day, _ in results], axis=0)
    by_tenure = np.mean([tenure for _, tenure in results], axis=0)
    return by_day, by_tenure


def example_payload() -> NoveltySummary:
    """Return the figure's true, tenure and calendar curves."""
    by_day, by_tenure = simulate_runs()
    truth = novelty_effect(np.arange(DAYS))
    mixture = calendar_expectation()
    return NoveltySummary(
        runs=RUNS,
        rows=tuple(
            NoveltyRow(
                day=day + 1,
                true_effect=float(truth[day]),
                by_tenure=float(by_tenure[day]),
                by_calendar=float(by_day[day]),
                calendar_expectation=float(mixture[day]),
            )
            for day in range(DAYS)
        ),
    )
