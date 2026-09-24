"""Acceptance sampling plans, for the article on what a clean sample of fifty proves.

A single sampling plan inspects ``n`` items from a lot and accepts it when at
most ``c`` are defective. With a defect rate ``p`` and a lot large enough for
the draws to be independent, the number found is binomial, and the plan accepts
with probability

    OC(p) = P(Bin(n, p) <= c),

its operating characteristic. A clean sample (``c = 0``) passes with probability
``(1 - p)^n``, so the highest rate that still passes five percent of the time is
``1 - 0.05^(1 / n)``, close to ``3 / n`` for large ``n`` (the rule of three,
``-log 0.05 = 2.996``).

A pair of risks fixes a plan: the acceptable quality level (1 percent) must pass
at least 95 percent of the time and the tolerance level (5 percent) at most 10
percent of the time. For fixed ``c`` the operating characteristic falls with
``n`` at every rate, so the first condition bounds ``n`` above and the second
below, and the smallest plan meeting both is the first ``n`` that meets the
second, provided it still meets the first. The article's search walks ``n``
upward from ``c + 1``; here the same search is vectorised.

A two-stage plan inspects ``n1``, accepts at ``c1`` or fewer, rejects at ``r1``
or more, and otherwise inspects ``n2`` more and accepts when the total is at most
``c2``. Its acceptance probability and average number inspected are exact sums
over the first count:

    P(accept) = P(X1 <= c1) + sum_{k = c1 + 1}^{r1 - 1} P(X1 = k) P(X2 <= c2 - k),
    average inspected = n1 + n2 P(c1 < X1 < r1).

A lot of finite size ``N`` with ``D`` defective gives a hypergeometric count
instead of a binomial one, and a data audit of ``n`` rows finds nothing with
probability ``(1 - p)^n``.

The figure is deterministic: the operating characteristic of four plans (inspect
50 allowing 0, 100 allowing 1, 200 allowing 3, 500 allowing 8) over 250 rates from
0.01 to 8 percent. Every table in the article that is the same closed form is
reproduced here: the clean-sample bounds, the operating characteristic at five
rates, the plan search, the zero-tolerance comparison, the single plan in the
two-stage comparison and the finite-lot table. The two-stage plan's acceptance and
average inspected, and the audit's chance of finding nothing, are simulated in the
article with generators seeded at 3 and 5 that the figure does not use; they are
computed here exactly, and the tests check the article's simulated values against
them.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, probability

__all__ = [
    "ACCEPTABLE_QUALITY",
    "AUDIT_ERROR_RATES",
    "AUDIT_SAMPLE_SIZES",
    "AUDIT_TABLE_ROWS",
    "CLEAN_PASS_RATE",
    "CLEAN_SAMPLE_SIZES",
    "CONSUMER_RISK",
    "DOUBLE_PLAN",
    "FIGURE_PLANS",
    "LOT_ACCEPT_NUMBER",
    "LOT_DEFECT_RATE",
    "LOT_SAMPLE_SIZE",
    "LOT_SIZES",
    "LOT_TOLERANCE",
    "MAX_ACCEPT_NUMBER",
    "PRODUCER_RISK",
    "RATE_POINTS",
    "RATE_START",
    "RATE_STOP",
    "SEARCH_LIMIT",
    "SINGLE_PLAN",
    "TABLE_RATES",
    "TWO_STAGE_RATES",
    "ZERO_TOLERANCE_PLANS",
    "AcceptanceSamplingSummary",
    "AuditRow",
    "CleanSampleRow",
    "DoublePlan",
    "FiniteLotRow",
    "OperatingCurve",
    "OperatingRow",
    "PlanSearchRow",
    "TwoStageRow",
    "ZeroToleranceRow",
    "acceptance_probability",
    "audit_row",
    "clean_sample_bound",
    "clean_sample_row",
    "double_sampling",
    "example_payload",
    "finite_lot_acceptance",
    "finite_lot_row",
    "operating_characteristic",
    "operating_curve",
    "plan_search_row",
    "smallest_sample",
    "two_stage_row",
    "zero_tolerance_row",
]


@dataclass(frozen=True, slots=True)
class DoublePlan:
    """A two-stage plan: inspect a first sample, and a second only when the first is unclear.

    Accept at ``first_accept`` defects or fewer in the first ``first_size``, reject
    at ``first_reject`` or more, and otherwise inspect ``second_size`` more and
    accept when the total is at most ``second_accept``.
    """

    first_size: int
    first_accept: int
    first_reject: int
    second_size: int
    second_accept: int


# The figure: four plans as (inspect, allow), over 250 rates from 0.01 to 8 percent.
FIGURE_PLANS: Final[tuple[tuple[int, int], ...]] = ((50, 0), (100, 1), (200, 3), (500, 8))
RATE_START: Final[float] = 0.0001
RATE_STOP: Final[float] = 0.08
RATE_POINTS: Final[int] = 250
# The two rates that matter: acceptable quality, and the rate that must be caught.
ACCEPTABLE_QUALITY: Final[float] = 0.01
LOT_TOLERANCE: Final[float] = 0.05
PRODUCER_RISK: Final[float] = 0.05
CONSUMER_RISK: Final[float] = 0.10
# What a clean sample rules out: the highest rate that still passes 5 percent of the time.
CLEAN_SAMPLE_SIZES: Final[tuple[int, ...]] = (20, 50, 100, 300, 1000)
CLEAN_PASS_RATE: Final[float] = 0.05
TABLE_RATES: Final[tuple[float, ...]] = (0.001, 0.005, 0.01, 0.02, 0.05)
# The article's search over accept numbers 0 to 11, with samples below 5,000.
MAX_ACCEPT_NUMBER: Final[int] = 11
SEARCH_LIMIT: Final[int] = 5000
ZERO_TOLERANCE_PLANS: Final[tuple[tuple[int, int], ...]] = ((45, 0), (77, 1), (105, 2), (132, 3))
# The two-stage comparison.
SINGLE_PLAN: Final[tuple[int, int]] = (132, 3)
DOUBLE_PLAN: Final[DoublePlan] = DoublePlan(80, 1, 4, 80, 4)
TWO_STAGE_RATES: Final[tuple[float, ...]] = (0.005, 0.01, 0.02, 0.05)
# The finite lot: inspect 50, allow none, in lots 2 percent defective.
LOT_SIZES: Final[tuple[int, ...]] = (200, 500, 2000, 100_000)
LOT_SAMPLE_SIZE: Final[int] = 50
LOT_ACCEPT_NUMBER: Final[int] = 0
LOT_DEFECT_RATE: Final[float] = 0.02
# The data audit: a table of two million rows.
AUDIT_TABLE_ROWS: Final[int] = 2_000_000
AUDIT_SAMPLE_SIZES: Final[tuple[int, ...]] = (200, 1000, 5000)
AUDIT_ERROR_RATES: Final[tuple[float, ...]] = (0.001, 0.005)


@dataclass(frozen=True, slots=True)
class OperatingCurve:
    """The figure: one plan's acceptance probability over its grid of defect rates."""

    sample_size: int
    accept_number: int
    rates: tuple[float, ...]
    acceptance: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CleanSampleRow:
    """What a sample with no defects rules out, and how often a lot at half that rate passes."""

    sample_size: int
    rate_bound: float
    half_bound_acceptance: float


@dataclass(frozen=True, slots=True)
class OperatingRow:
    """One plan's acceptance probability at each of the table's rates."""

    sample_size: int
    accept_number: int
    acceptance: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PlanSearchRow:
    """The smallest sample meeting both risks at one accept number; ``None`` if there is none."""

    accept_number: int
    sample_size: int | None
    acceptable_acceptance: float | None
    tolerance_acceptance: float | None


@dataclass(frozen=True, slots=True)
class ZeroToleranceRow:
    """A plan's acceptance at the two rates, and the share of acceptable lots it rejects."""

    sample_size: int
    accept_number: int
    acceptable_acceptance: float
    tolerance_acceptance: float
    producer_risk: float


@dataclass(frozen=True, slots=True)
class TwoStageRow:
    """The single and two-stage plans at one defect rate: acceptance and items inspected."""

    rate: float
    single_acceptance: float
    single_inspected: int
    double_acceptance: float
    double_average_inspected: float


@dataclass(frozen=True, slots=True)
class FiniteLotRow:
    """Exact hypergeometric acceptance in a finite lot against the binomial approximation."""

    lot_size: int
    defects: int
    exact: float
    binomial: float
    difference: float


@dataclass(frozen=True, slots=True)
class AuditRow:
    """A data audit: the chance it finds nothing, and the errors expected in sample and table."""

    sample_size: int
    error_rate: float
    clean_probability: float
    expected_in_sample: float
    errors_in_table: float


@dataclass(frozen=True, slots=True)
class AcceptanceSamplingSummary:
    """The figure's curves and the article's closed-form tables."""

    curves: tuple[OperatingCurve, ...]
    clean_samples: tuple[CleanSampleRow, ...]
    table_rates: tuple[float, ...]
    operating: tuple[OperatingRow, ...]
    plan_search: tuple[PlanSearchRow, ...]
    zero_tolerance: tuple[ZeroToleranceRow, ...]
    two_stage: tuple[TwoStageRow, ...]
    finite_lot: tuple[FiniteLotRow, ...]
    audit: tuple[AuditRow, ...]


def _plan(sample_size: int, accept_number: int) -> tuple[int, int]:
    n = count(sample_size, name="sample_size", minimum=1)
    c = count(accept_number, name="accept_number")
    if c > n:
        raise ValueError("accept_number cannot exceed sample_size")
    return n, c


def acceptance_probability(sample_size: int, accept_number: int, rate: float) -> float:
    """Chance a lot with defect rate ``rate`` passes: ``P(Bin(n, rate) <= c)``."""
    n, c = _plan(sample_size, accept_number)
    return float(stats.binom.cdf(c, n, probability(rate, name="rate")))


def operating_characteristic(
    sample_size: int, accept_number: int, rates: ArrayLike
) -> NDArray[np.float64]:
    """The acceptance probability at each of an array of defect rates."""
    n, c = _plan(sample_size, accept_number)
    p = np.asarray(rates, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("rates must be a non-empty 1-D array of probabilities")
    curve: NDArray[np.float64] = stats.binom.cdf(c, n, p)
    return curve


def operating_curve(
    sample_size: int,
    accept_number: int,
    *,
    start: float = RATE_START,
    stop: float = RATE_STOP,
    points: int = RATE_POINTS,
) -> OperatingCurve:
    """One plan's operating characteristic over evenly spaced rates, as the figure draws it."""
    first = probability(start, name="start")
    last = probability(stop, name="stop")
    if last <= first:
        raise ValueError("stop must exceed start")
    rates = np.linspace(first, last, count(points, name="points", minimum=2))
    return OperatingCurve(
        sample_size=sample_size,
        accept_number=accept_number,
        rates=tuple(float(p) for p in rates),
        acceptance=tuple(
            float(a) for a in operating_characteristic(sample_size, accept_number, rates)
        ),
    )


def clean_sample_bound(sample_size: int, *, pass_rate: float = CLEAN_PASS_RATE) -> float:
    """The highest defect rate that still yields a clean sample ``pass_rate`` of the time."""
    n = count(sample_size, name="sample_size", minimum=1)
    alpha = probability(pass_rate, name="pass_rate", inclusive=False)
    return float(1 - alpha ** (1 / n))


def clean_sample_row(sample_size: int, *, pass_rate: float = CLEAN_PASS_RATE) -> CleanSampleRow:
    """One row of the article's first table: the bound, and a lot at half of it."""
    bound = clean_sample_bound(sample_size, pass_rate=pass_rate)
    return CleanSampleRow(
        sample_size=sample_size,
        rate_bound=bound,
        half_bound_acceptance=acceptance_probability(sample_size, 0, bound / 2),
    )


def _risks(
    acceptable: float, tolerance: float, producer_risk: float, consumer_risk: float
) -> tuple[float, float, float, float]:
    good = probability(acceptable, name="acceptable", inclusive=False)
    bad = probability(tolerance, name="tolerance", inclusive=False)
    if bad <= good:
        raise ValueError("tolerance must exceed acceptable")
    alpha = probability(producer_risk, name="producer_risk", inclusive=False)
    beta = probability(consumer_risk, name="consumer_risk", inclusive=False)
    return good, bad, alpha, beta


def smallest_sample(
    accept_number: int,
    *,
    acceptable: float = ACCEPTABLE_QUALITY,
    tolerance: float = LOT_TOLERANCE,
    producer_risk: float = PRODUCER_RISK,
    consumer_risk: float = CONSUMER_RISK,
    limit: int = SEARCH_LIMIT,
) -> int | None:
    """The smallest ``n`` from ``c + 1`` up to ``limit - 1`` that meets both risks.

    A lot at ``acceptable`` must pass with probability at least ``1 - producer_risk``
    and one at ``tolerance`` with probability at most ``consumer_risk``. ``None``
    when no such sample exists below ``limit``.
    """
    c = count(accept_number, name="accept_number")
    good, bad, alpha, beta = _risks(acceptable, tolerance, producer_risk, consumer_risk)
    n = np.arange(c + 1, count(limit, name="limit", minimum=1))
    meets = (stats.binom.cdf(c, n, good) >= 1 - alpha) & (stats.binom.cdf(c, n, bad) <= beta)
    first = np.flatnonzero(meets)
    return int(n[first[0]]) if first.size else None


def plan_search_row(
    accept_number: int,
    *,
    acceptable: float = ACCEPTABLE_QUALITY,
    tolerance: float = LOT_TOLERANCE,
    producer_risk: float = PRODUCER_RISK,
    consumer_risk: float = CONSUMER_RISK,
    limit: int = SEARCH_LIMIT,
) -> PlanSearchRow:
    """One accept number of the article's plan search."""
    n = smallest_sample(
        accept_number,
        acceptable=acceptable,
        tolerance=tolerance,
        producer_risk=producer_risk,
        consumer_risk=consumer_risk,
        limit=limit,
    )
    if n is None:
        return PlanSearchRow(accept_number, None, None, None)
    return PlanSearchRow(
        accept_number=accept_number,
        sample_size=n,
        acceptable_acceptance=acceptance_probability(n, accept_number, acceptable),
        tolerance_acceptance=acceptance_probability(n, accept_number, tolerance),
    )


def zero_tolerance_row(
    sample_size: int,
    accept_number: int,
    *,
    acceptable: float = ACCEPTABLE_QUALITY,
    tolerance: float = LOT_TOLERANCE,
) -> ZeroToleranceRow:
    """A plan's acceptance at the two rates and its producer's risk at the acceptable one."""
    good = acceptance_probability(sample_size, accept_number, acceptable)
    return ZeroToleranceRow(
        sample_size=sample_size,
        accept_number=accept_number,
        acceptable_acceptance=good,
        tolerance_acceptance=acceptance_probability(sample_size, accept_number, tolerance),
        producer_risk=1 - good,
    )


def double_sampling(rate: float, plan: DoublePlan = DOUBLE_PLAN) -> tuple[float, float]:
    """Exact acceptance probability and average number inspected for a two-stage plan."""
    p = probability(rate, name="rate")
    n1 = count(plan.first_size, name="first_size", minimum=1)
    n2 = count(plan.second_size, name="second_size", minimum=1)
    c1 = count(plan.first_accept, name="first_accept")
    r1 = count(plan.first_reject, name="first_reject")
    c2 = count(plan.second_accept, name="second_accept")
    if not c1 < r1 <= n1:
        raise ValueError("the plan needs first_accept < first_reject <= first_size")
    first = stats.binom.pmf(np.arange(n1 + 1), n1, p)
    undecided = np.arange(c1 + 1, r1)
    second = stats.binom.cdf(c2 - undecided, n2, p)
    accept = float(first[: c1 + 1].sum()) + float(np.dot(first[undecided], second))
    return accept, n1 + n2 * float(first[undecided].sum())


def two_stage_row(
    rate: float, *, single: tuple[int, int] = SINGLE_PLAN, plan: DoublePlan = DOUBLE_PLAN
) -> TwoStageRow:
    """The single plan against the two-stage plan at one defect rate."""
    n, c = _plan(*single)
    accept, inspected = double_sampling(rate, plan)
    return TwoStageRow(
        rate=rate,
        single_acceptance=acceptance_probability(n, c, rate),
        single_inspected=n,
        double_acceptance=accept,
        double_average_inspected=inspected,
    )


def finite_lot_acceptance(
    lot_size: int, defects: int, sample_size: int, accept_number: int
) -> float:
    """Chance a lot of ``lot_size`` with ``defects`` defective passes: a hypergeometric count."""
    lot = count(lot_size, name="lot_size", minimum=1)
    bad = count(defects, name="defects")
    n, c = _plan(sample_size, accept_number)
    if bad > lot or n > lot:
        raise ValueError("defects and sample_size cannot exceed lot_size")
    return float(stats.hypergeom.cdf(c, lot, bad, n))


def finite_lot_row(
    lot_size: int,
    *,
    rate: float = LOT_DEFECT_RATE,
    sample_size: int = LOT_SAMPLE_SIZE,
    accept_number: int = LOT_ACCEPT_NUMBER,
) -> FiniteLotRow:
    """The exact finite-lot acceptance against the binomial, with ``round(rate N)`` defective."""
    lot = count(lot_size, name="lot_size", minimum=1)
    p = probability(rate, name="rate")
    defects = int(round(p * lot))
    exact = finite_lot_acceptance(lot, defects, sample_size, accept_number)
    binomial = acceptance_probability(sample_size, accept_number, p)
    return FiniteLotRow(lot, defects, exact, binomial, exact - binomial)


def audit_row(sample_size: int, error_rate: float, *, rows: int = AUDIT_TABLE_ROWS) -> AuditRow:
    """An audit of ``sample_size`` rows finds nothing with probability ``(1 - rate)^n``."""
    n = count(sample_size, name="sample_size", minimum=1)
    p = probability(error_rate, name="error_rate")
    return AuditRow(
        sample_size=n,
        error_rate=p,
        clean_probability=acceptance_probability(n, 0, p),
        expected_in_sample=n * p,
        errors_in_table=count(rows, name="rows", minimum=1) * p,
    )


def example_payload() -> AcceptanceSamplingSummary:
    """Return the figure's operating characteristics and the article's closed-form tables."""
    return AcceptanceSamplingSummary(
        curves=tuple(operating_curve(n, c) for n, c in FIGURE_PLANS),
        clean_samples=tuple(clean_sample_row(n) for n in CLEAN_SAMPLE_SIZES),
        table_rates=TABLE_RATES,
        operating=tuple(
            OperatingRow(n, c, tuple(acceptance_probability(n, c, p) for p in TABLE_RATES))
            for n, c in FIGURE_PLANS
        ),
        plan_search=tuple(plan_search_row(c) for c in range(MAX_ACCEPT_NUMBER + 1)),
        zero_tolerance=tuple(zero_tolerance_row(n, c) for n, c in ZERO_TOLERANCE_PLANS),
        two_stage=tuple(two_stage_row(p) for p in TWO_STAGE_RATES),
        finite_lot=tuple(finite_lot_row(lot) for lot in LOT_SIZES),
        audit=tuple(audit_row(n, p) for n in AUDIT_SAMPLE_SIZES for p in AUDIT_ERROR_RATES),
    )
