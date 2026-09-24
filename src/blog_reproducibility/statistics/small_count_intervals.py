"""Intervals for small counts, for the article on zero failures and the rule of three.

After ``k`` events in ``n`` independent trials, three 95 percent intervals for
the rate are compared. Wald's is ``p -+ z sqrt(p (1 - p) / n)`` with
``p = k / n``, which is the single point zero when ``k = 0``. Wilson's inverts the
score test,

    (p + z^2 / 2n) / (1 + z^2 / n) -+ z sqrt(p (1 - p) / n + z^2 / 4n^2) / (1 + z^2 / n),

and the exact Clopper-Pearson interval takes its ends from Beta quantiles,
``B(alpha / 2; k, n - k + 1)`` and ``B(1 - alpha / 2; k + 1, n - k)``. With no events
its upper end is ``1 - (alpha / 2)^(1 / n)``. The one-sided bound the article
derives from ``(1 - p)^n = alpha`` is ``1 - alpha^(1 / n)``, about ``3 / n``: the rule of
three. Coverage, the chance an interval contains the true rate, is a finite sum
over the binomial distribution of ``k``, so it is computed exactly here as well
as simulated.

The figure draws 4,000 binomial counts at each of seven sample sizes with a
true rate of half a percent, from one generator seeded at 0, and is reproduced
draw for draw. The article's coverage table runs a different design (10,000
samples, six other settings) from its own generator and is not reproduced; the
tests check it against the exact coverage. Its other tables (the bound after
zero events, the clean trials needed for a target rate, the chi-square plan
sizes, the chance of a clean run, exact intervals at 1,000 trials and the bound
for events in time) are closed forms and are reproduced exactly.
"""

from dataclasses import dataclass
from math import ceil, log
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, probability

__all__ = [
    "ALLOWED_FAILURES",
    "CLEAN_RUN_RATES",
    "CLEAN_RUN_TRIALS",
    "COVERAGE_SETTINGS",
    "FAILURE_TARGET",
    "HOURS_PER_YEAR",
    "INTERVAL_EVENTS",
    "INTERVAL_TRIALS",
    "METHODS",
    "OBSERVATION_HOURS",
    "RATE",
    "SAMPLES",
    "SEED",
    "SIGNIFICANCE",
    "TARGET_RATES",
    "TRIALS",
    "ZERO_EVENT_TRIALS",
    "ArticleNumbers",
    "CleanRunRow",
    "CleanTrialsRow",
    "Coverage",
    "CoverageRow",
    "ExposureRow",
    "FailurePlanRow",
    "IntervalRow",
    "Method",
    "SmallCountSummary",
    "ZeroEventRow",
    "article_numbers",
    "clean_run_probability",
    "clean_trials_needed",
    "clopper_pearson_interval",
    "exact_coverage",
    "exact_coverage_row",
    "example_payload",
    "exposure_bound",
    "interval",
    "one_sided_zero_bound",
    "simulated_coverage",
    "trials_allowing_failures",
    "wald_interval",
    "wilson_interval",
]

Method = Literal["wald", "wilson", "clopper_pearson"]

SEED: Final[int] = 0
SIGNIFICANCE: Final[float] = 0.05
METHODS: Final[tuple[Method, ...]] = ("wald", "wilson", "clopper_pearson")
# The figure: 4,000 samples at each size with a true rate of half a percent.
RATE: Final[float] = 0.005
TRIALS: Final[tuple[int, ...]] = (100, 200, 400, 800, 1600, 3200, 6400)
SAMPLES: Final[int] = 4000
# The article's closed-form tables.
ZERO_EVENT_TRIALS: Final[tuple[int, ...]] = (10, 30, 100, 300, 1000, 3000)
TARGET_RATES: Final[tuple[float, ...]] = (0.05, 0.01, 0.001, 0.0001)
ALLOWED_FAILURES: Final[tuple[int, ...]] = (0, 1, 2, 3, 5)
FAILURE_TARGET: Final[float] = 0.001
CLEAN_RUN_TRIALS: Final[int] = 300
CLEAN_RUN_RATES: Final[tuple[float, ...]] = (0.001, 0.005, 0.01, 0.02)
INTERVAL_TRIALS: Final[int] = 1000
INTERVAL_EVENTS: Final[tuple[int, ...]] = (0, 1, 2, 5, 10, 50)
OBSERVATION_HOURS: Final[tuple[int, ...]] = (100, 1000, 10_000)
HOURS_PER_YEAR: Final[int] = 8760
# The article's coverage table: trials and true rate.
COVERAGE_SETTINGS: Final[tuple[tuple[int, float], ...]] = (
    (100, 0.002),
    (500, 0.002),
    (500, 0.01),
    (2000, 0.002),
    (2000, 0.01),
    (200, 0.05),
)


@dataclass(frozen=True, slots=True)
class Coverage:
    """Share of intervals containing the true rate, for each of the three methods."""

    wald: float
    wilson: float
    clopper_pearson: float


@dataclass(frozen=True, slots=True)
class CoverageRow:
    """Coverage at one number of trials and true rate, simulated or exact."""

    trials: int
    rate: float
    expected_events: float
    coverage: Coverage
    zero_events: float


@dataclass(frozen=True, slots=True)
class ZeroEventRow:
    """Upper ends of the three intervals after no events, with ``3 / n`` and the one-sided bound."""

    trials: int
    clopper_pearson: float
    rule_of_three: float
    wilson: float
    wald: float
    one_sided: float


@dataclass(frozen=True, slots=True)
class CleanTrialsRow:
    """Failure-free trials needed for the one-sided bound to reach a target rate."""

    target: float
    trials: int


@dataclass(frozen=True, slots=True)
class FailurePlanRow:
    """Trials for the 95 percent bound to reach the target when some failures are allowed."""

    failures: int
    trials: float


@dataclass(frozen=True, slots=True)
class CleanRunRow:
    """Chance of a run with no failures at a true failure rate."""

    rate: float
    probability: float


@dataclass(frozen=True, slots=True)
class IntervalRow:
    """The exact interval after ``events`` in the article's 1,000 trials."""

    events: int
    estimate: float
    lower: float
    upper: float
    upper_multiple: float | None


@dataclass(frozen=True, slots=True)
class ExposureRow:
    """Rule-of-three bound on an event rate after hours of observation with no event."""

    hours: int
    per_hour: float
    per_year: float


@dataclass(frozen=True, slots=True)
class ArticleNumbers:
    """The article's closed-form tables and the exact coverage behind its simulated one."""

    zero_event_bounds: tuple[ZeroEventRow, ...]
    clean_trials: tuple[CleanTrialsRow, ...]
    failure_plans: tuple[FailurePlanRow, ...]
    clean_runs: tuple[CleanRunRow, ...]
    intervals: tuple[IntervalRow, ...]
    exposure: tuple[ExposureRow, ...]
    coverage: tuple[CoverageRow, ...]


@dataclass(frozen=True, slots=True)
class SmallCountSummary:
    """The figure's simulated coverage, its exact coverage and the article's numbers."""

    simulated: tuple[CoverageRow, ...]
    exact: tuple[CoverageRow, ...]
    article: ArticleNumbers


def _alpha(significance: float) -> float:
    return probability(significance, name="significance", inclusive=False)


def _checked_counts(events: int, trials: int) -> tuple[int, int]:
    n = count(trials, name="trials", minimum=1)
    k = count(events, name="events")
    if k > n:
        raise ValueError("events cannot exceed trials")
    return k, n


def _bounds(
    method: Method, events: NDArray[np.int64], n: int, significance: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interval ends for an array of event counts, written as the article writes them."""
    alpha = _alpha(significance)
    k = events.astype(np.float64)
    if method == "clopper_pearson":
        lower, upper = np.zeros(k.size), np.ones(k.size)
        some, short = events > 0, events < n
        lower[some] = stats.beta.ppf(alpha / 2, k[some], n - k[some] + 1)
        upper[short] = stats.beta.ppf(1 - alpha / 2, k[short] + 1, n - k[short])
        return lower, upper
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = k / n
    if method == "wald":
        h = z * np.sqrt(p * (1 - p) / n)
        return np.maximum(0.0, p - h), np.minimum(1.0, p + h)
    if method == "wilson":
        c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        return np.maximum(0.0, c - h), np.minimum(1.0, c + h)
    raise ValueError(f"unknown interval method {method!r}")


def interval(
    method: Method, events: int, trials: int, *, significance: float = SIGNIFICANCE
) -> tuple[float, float]:
    """The ``1 - significance`` interval for the rate after ``events`` in ``trials``."""
    k, n = _checked_counts(events, trials)
    lower, upper = _bounds(method, np.array([k], dtype=np.int64), n, significance)
    return float(lower[0]), float(upper[0])


def wald_interval(
    events: int, trials: int, *, significance: float = SIGNIFICANCE
) -> tuple[float, float]:
    """The normal-approximation interval, clipped to [0, 1]."""
    return interval("wald", events, trials, significance=significance)


def wilson_interval(
    events: int, trials: int, *, significance: float = SIGNIFICANCE
) -> tuple[float, float]:
    """The Wilson score interval, clipped to [0, 1]."""
    return interval("wilson", events, trials, significance=significance)


def clopper_pearson_interval(
    events: int, trials: int, *, significance: float = SIGNIFICANCE
) -> tuple[float, float]:
    """The exact Clopper-Pearson interval from Beta quantiles."""
    return interval("clopper_pearson", events, trials, significance=significance)


def _covered(
    method: Method, events: NDArray[np.int64], n: int, rate: float, significance: float
) -> NDArray[np.bool_]:
    lower, upper = _bounds(method, events, n, significance)
    return (lower <= rate) & (rate <= upper)


def exact_coverage(
    method: Method, trials: int, rate: float, *, significance: float = SIGNIFICANCE
) -> float:
    """Chance the interval contains ``rate``: the binomial mass of the counts it covers."""
    n = count(trials, name="trials", minimum=1)
    p = probability(rate, name="rate", inclusive=False)
    events = np.arange(n + 1, dtype=np.int64)
    mass = stats.binom.pmf(events, n, p)
    return float(np.sum(mass[_covered(method, events, n, p, significance)]))


def _coverage_row(n: int, p: float, events: NDArray[np.int64], significance: float) -> CoverageRow:
    """Coverage of each method over a sample of event counts, one interval per distinct count."""
    distinct, tally = np.unique(events, return_counts=True)
    shares = [
        int(tally[_covered(method, distinct, n, p, significance)].sum()) / events.size
        for method in METHODS
    ]
    return CoverageRow(
        trials=n,
        rate=p,
        expected_events=n * p,
        coverage=Coverage(*shares),
        zero_events=int(np.count_nonzero(events == 0)) / events.size,
    )


def simulated_coverage(
    seed: int = SEED,
    *,
    trials: tuple[int, ...] = TRIALS,
    rate: float = RATE,
    samples: int = SAMPLES,
    significance: float = SIGNIFICANCE,
) -> tuple[CoverageRow, ...]:
    """Draw ``samples`` counts at each size in turn from one generator and score each interval."""
    rng = np.random.default_rng(count(seed, name="seed"))
    p = probability(rate, name="rate", inclusive=False)
    draws = count(samples, name="samples", minimum=1)
    rows = []
    for n in (count(size, name="trials", minimum=1) for size in trials):
        events = rng.binomial(n, p, draws).astype(np.int64)
        rows.append(_coverage_row(n, p, events, significance))
    return tuple(rows)


def exact_coverage_row(
    trials: int, rate: float, *, significance: float = SIGNIFICANCE
) -> CoverageRow:
    """Exact coverage of each method, and the exact chance of no events."""
    n = count(trials, name="trials", minimum=1)
    p = probability(rate, name="rate", inclusive=False)
    return CoverageRow(
        trials=n,
        rate=p,
        expected_events=n * p,
        coverage=Coverage(
            *(exact_coverage(method, n, p, significance=significance) for method in METHODS)
        ),
        zero_events=clean_run_probability(p, n),
    )


def one_sided_zero_bound(trials: int, *, significance: float = SIGNIFICANCE) -> float:
    """Largest rate with a clean run at least ``alpha`` likely: ``1 - alpha^(1 / n)``."""
    n = count(trials, name="trials", minimum=1)
    return float(1 - _alpha(significance) ** (1 / n))


def clean_trials_needed(target: float, *, significance: float = SIGNIFICANCE) -> int:
    """Fewest failure-free trials that put the one-sided bound below ``target``."""
    p = probability(target, name="target", inclusive=False)
    return ceil(log(_alpha(significance)) / log(1 - p))


def trials_allowing_failures(
    failures: int, target: float = FAILURE_TARGET, *, significance: float = SIGNIFICANCE
) -> float:
    """Trials for the Poisson bound to reach ``target`` after ``k`` failures: ``chi2 / 2p``."""
    k = count(failures, name="failures")
    p = probability(target, name="target", inclusive=False)
    return float(stats.chi2.ppf(1 - _alpha(significance), 2 * k + 2)) / (2 * p)


def clean_run_probability(rate: float, trials: int = CLEAN_RUN_TRIALS) -> float:
    """Chance of ``n`` trials without a failure: ``(1 - p)^n``."""
    p = probability(rate, name="rate")
    return float((1 - p) ** count(trials, name="trials"))


def exposure_bound(hours: int) -> ExposureRow:
    """Rule-of-three bound on an event rate after ``hours`` without an event, ``3 / T``.

    The exact Poisson bound is ``-ln(alpha) / T``, 2.996 / T at 95 percent; the
    article rounds it to three, as the rule of three does.
    """
    exposure = count(hours, name="hours", minimum=1)
    per_hour = 3.0 / exposure
    return ExposureRow(hours=exposure, per_hour=per_hour, per_year=per_hour * HOURS_PER_YEAR)


def _zero_event_row(n: int) -> ZeroEventRow:
    return ZeroEventRow(
        trials=n,
        clopper_pearson=clopper_pearson_interval(0, n)[1],
        rule_of_three=3 / n,
        wilson=wilson_interval(0, n)[1],
        wald=wald_interval(0, n)[1],
        one_sided=one_sided_zero_bound(n),
    )


def _interval_row(k: int) -> IntervalRow:
    lower, upper = clopper_pearson_interval(k, INTERVAL_TRIALS)
    estimate = k / INTERVAL_TRIALS
    return IntervalRow(
        events=k,
        estimate=estimate,
        lower=lower,
        upper=upper,
        upper_multiple=upper / estimate if k > 0 else None,
    )


def article_numbers() -> ArticleNumbers:
    """The article's closed-form tables, and exact coverage for its simulated settings."""
    return ArticleNumbers(
        zero_event_bounds=tuple(_zero_event_row(n) for n in ZERO_EVENT_TRIALS),
        clean_trials=tuple(CleanTrialsRow(p, clean_trials_needed(p)) for p in TARGET_RATES),
        failure_plans=tuple(
            FailurePlanRow(k, trials_allowing_failures(k)) for k in ALLOWED_FAILURES
        ),
        clean_runs=tuple(CleanRunRow(p, clean_run_probability(p)) for p in CLEAN_RUN_RATES),
        intervals=tuple(_interval_row(k) for k in INTERVAL_EVENTS),
        exposure=tuple(exposure_bound(t) for t in OBSERVATION_HOURS),
        coverage=tuple(exact_coverage_row(n, p) for n, p in COVERAGE_SETTINGS),
    )


def example_payload() -> SmallCountSummary:
    """Return the figure's simulated and exact coverage and the article's closed forms."""
    return SmallCountSummary(
        simulated=simulated_coverage(),
        exact=tuple(exact_coverage_row(n, RATE) for n in TRIALS),
        article=article_numbers(),
    )
