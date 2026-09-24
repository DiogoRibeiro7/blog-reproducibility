"""Equivalence and non-inferiority tests, for the article on proving a model is no worse.

Two models are scored on the same ``n`` test cases and each case gives a paired
difference in loss, normal with mean ``Delta`` (negative when the challenger is
worse) and standard deviation ``sigma_d``. Three tests read the mean difference
``m`` and its standard error ``se = s / sqrt(n)``:

* the two-sided paired t-test is significant when ``|m / se| > t_{0.975, n-1}``;
* the two one-sided tests (TOST) show equivalence within a margin ``delta``
  when the 90 percent interval ``m -+ t_{0.95, n-1} se`` lies inside
  ``(-delta, delta)``;
* non-inferiority holds when the lower end of that interval is above ``-delta``.

The t-test and non-inferiority rates are noncentral t probabilities with
noncentrality ``Delta sqrt(n) / sigma_d`` and ``(Delta + delta) sqrt(n) / sigma_d``.
TOST has no single-distribution form, but given the sample standard deviation
the mean is normal, so its power is one integral over the chi-square law of
``(n - 1) s^2 / sigma_d^2``. For identical models the article sizes the test set
with ``n = sigma_d^2 (z_{1 - alpha} + z_{1 - beta / 2})^2 / delta^2``.

The figure runs 4,000 paired comparisons with ``sigma_d = 6`` and a one-point
margin at seven test-set sizes, for identical models, then a challenger 1.5
points worse, then one 0.5 points worse, from one generator seeded at 0, and is
reproduced draw for draw. The article's table for identical models runs the
same generator in the same order over the first six sizes, so its rows are
reproduced exactly too. Its other two tables and its simulated power at 308
cases come from later draws in a different order and are not reproduced; the
tests check them against the exact probabilities instead.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, stats

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "DIFFERENCE_SD",
    "MARGIN",
    "REPLICATIONS",
    "SEED",
    "SIGNIFICANCE",
    "TEST_CASES",
    "TRUE_DIFFERENCES",
    "ArticleNumbers",
    "Conclusions",
    "EquivalenceSummary",
    "OutcomeRow",
    "article_numbers",
    "cases_for_equivalence",
    "cases_to_detect",
    "compare_paired",
    "equivalence_power",
    "example_payload",
    "non_inferiority_power",
    "outcome_rates",
    "outcome_row",
    "smallest_cases_for_power",
    "t_test_power",
]

SEED: Final[int] = 0
DIFFERENCE_SD: Final[float] = 6.0
MARGIN: Final[float] = 1.0
SIGNIFICANCE: Final[float] = 0.05
REPLICATIONS: Final[int] = 4000
TEST_CASES: Final[tuple[int, ...]] = (50, 100, 200, 400, 800, 1600, 3200)
# Identical models, a challenger 1.5 points worse, and one 0.5 points worse: the figure's order.
TRUE_DIFFERENCES: Final[tuple[float, ...]] = (0.0, -1.5, -0.5)
# Draws are made a block of replications at a time, in the same order as one at a time.
_DRAWS_PER_BLOCK: Final[int] = 2_000_000


@dataclass(frozen=True, slots=True)
class Conclusions:
    """What the three tests conclude from one set of paired differences."""

    mean: float
    standard_error: float
    t_test_significant: bool
    equivalent: bool
    non_inferior: bool


@dataclass(frozen=True, slots=True)
class OutcomeRow:
    """Share of comparisons reaching each conclusion, simulated and exact, at one design."""

    true_difference: float
    test_cases: int
    t_test_significant: float
    equivalent: float
    non_inferior: float
    exact_t_test_significant: float
    exact_equivalent: float
    exact_non_inferior: float


@dataclass(frozen=True, slots=True)
class ArticleNumbers:
    """The article's closed-form sizes and the exact rates behind its prose."""

    interval_width_at_fifty: float
    cases_for_eighty: float
    cases_for_ninety: float
    cases_for_eighty_half_point_worse: float
    cases_to_detect: float
    exact_power_at_formula_cases: float
    smallest_cases_for_eighty: int


@dataclass(frozen=True, slots=True)
class EquivalenceSummary:
    """The figure's simulated rates, with exact rates beside them, and the article's numbers."""

    rows: tuple[OutcomeRow, ...]
    article: ArticleNumbers


def _cases(test_cases: int) -> int:
    return count(test_cases, name="test_cases", minimum=2)


def _critical_values(n: int, significance: float) -> tuple[float, float]:
    """Two-sided and one-sided t critical values with ``n - 1`` degrees of freedom."""
    alpha = probability(significance, name="significance", inclusive=False)
    return float(stats.t.ppf(1 - alpha / 2, n - 1)), float(stats.t.ppf(1 - alpha, n - 1))


def _decide(
    means: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
    n: int,
    margin: float,
    significance: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    two_sided, one_sided = _critical_values(n, significance)
    significant = np.abs(means / standard_errors) > two_sided
    lower = means - one_sided * standard_errors
    upper = means + one_sided * standard_errors
    non_inferior = lower > -margin
    return significant, non_inferior & (upper < margin), non_inferior


def compare_paired(
    differences: ArrayLike, *, margin: float = MARGIN, significance: float = SIGNIFICANCE
) -> Conclusions:
    """Run the paired t-test, TOST and the non-inferiority test on challenger-minus-incumbent."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("differences must be a one-dimensional sample of at least two values")
    if not np.all(np.isfinite(values)):
        raise ValueError("differences must be finite")
    n = values.size
    mean = values.mean(keepdims=True)
    se = values.std(ddof=1, keepdims=True) / np.sqrt(n)
    if se[0] <= 0:
        raise ValueError("differences must not all be equal")
    significant, equivalent, non_inferior = _decide(
        mean, se, n, positive(margin, name="margin"), significance
    )
    return Conclusions(
        mean=float(mean[0]),
        standard_error=float(se[0]),
        t_test_significant=bool(significant[0]),
        equivalent=bool(equivalent[0]),
        non_inferior=bool(non_inferior[0]),
    )


def _simulate(
    rng: np.random.Generator,
    n: int,
    true_difference: float,
    replications: int,
    *,
    sd: float,
    margin: float,
) -> tuple[float, float, float]:
    """Share of ``replications`` comparisons with each conclusion, one sample after another."""
    hits = np.zeros(3, dtype=np.int64)
    block = max(1, _DRAWS_PER_BLOCK // n)
    done = 0
    while done < replications:
        rows = min(block, replications - done)
        draws = rng.normal(true_difference, sd, (rows, n))
        means = draws.mean(axis=1)
        standard_errors = draws.std(axis=1, ddof=1) / np.sqrt(n)
        decisions = _decide(means, standard_errors, n, margin, SIGNIFICANCE)
        hits += [int(np.count_nonzero(decision)) for decision in decisions]
        done += rows
    shares = hits / replications
    return float(shares[0]), float(shares[1]), float(shares[2])


def t_test_power(
    test_cases: int,
    true_difference: float,
    *,
    sd: float = DIFFERENCE_SD,
    significance: float = SIGNIFICANCE,
) -> float:
    """Exact chance the two-sided paired t-test is significant: a noncentral t probability."""
    n = _cases(test_cases)
    two_sided, _ = _critical_values(n, significance)
    shift = real(true_difference, name="true_difference") * sqrt(n) / positive(sd, name="sd")
    return float(stats.nct.sf(two_sided, n - 1, shift) + stats.nct.cdf(-two_sided, n - 1, shift))


def non_inferiority_power(
    test_cases: int,
    true_difference: float,
    *,
    sd: float = DIFFERENCE_SD,
    margin: float = MARGIN,
    significance: float = SIGNIFICANCE,
) -> float:
    """Exact chance the lower end of the interval clears ``-margin``: one noncentral t tail."""
    n = _cases(test_cases)
    _, one_sided = _critical_values(n, significance)
    distance = real(true_difference, name="true_difference") + positive(margin, name="margin")
    shift = distance * sqrt(n) / positive(sd, name="sd")
    return float(stats.nct.sf(one_sided, n - 1, shift))


def equivalence_power(
    test_cases: int,
    true_difference: float,
    *,
    sd: float = DIFFERENCE_SD,
    margin: float = MARGIN,
    significance: float = SIGNIFICANCE,
) -> float:
    """Exact chance TOST shows equivalence, integrating over the sample standard deviation.

    With ``k = sqrt(n) / sigma_d``, the standardised error of the mean
    ``Z = (m - Delta) k`` is standard normal and independent of
    ``u = (n - 1) s^2 / sigma_d^2``, which is chi-square with ``n - 1`` degrees of
    freedom. Given ``u``, with ``w = t_{1 - alpha} sqrt(u / (n - 1))``, equivalence
    needs ``(-delta - Delta) k + w < Z < (delta - Delta) k - w``, a window that is
    empty once ``w`` reaches ``delta k``.
    """
    n = _cases(test_cases)
    _, one_sided = _critical_values(n, significance)
    delta = real(true_difference, name="true_difference")
    edge = positive(margin, name="margin")
    k = sqrt(n) / positive(sd, name="sd")
    df = n - 1

    def window(u: float) -> float:
        w = one_sided * sqrt(u / df)
        upper = (edge - delta) * k - w
        lower = (-edge - delta) * k + w
        return float(stats.norm.cdf(upper) - stats.norm.cdf(lower)) * float(stats.chi2.pdf(u, df))

    widest = df * (edge * k / one_sided) ** 2
    lo = float(stats.chi2.ppf(1e-13, df))
    hi = min(widest, float(stats.chi2.isf(1e-13, df)))
    if hi <= lo:
        return 0.0
    value, _ = integrate.quad(window, lo, hi, limit=200, epsabs=1e-12)
    return max(0.0, float(value))


def outcome_row(
    rng: np.random.Generator,
    test_cases: int,
    true_difference: float,
    *,
    replications: int = REPLICATIONS,
    sd: float = DIFFERENCE_SD,
    margin: float = MARGIN,
) -> OutcomeRow:
    """Simulate one design from ``rng`` and put the exact rates beside the simulated ones.

    Each replication draws ``test_cases`` paired differences in turn, as the
    website does one ``rng.normal(delta, sd, n)`` call at a time.
    """
    n = _cases(test_cases)
    delta = real(true_difference, name="true_difference")
    spread = positive(sd, name="sd")
    edge = positive(margin, name="margin")
    reps = count(replications, name="replications", minimum=1)
    significant, equivalent, non_inferior = _simulate(rng, n, delta, reps, sd=spread, margin=edge)
    return OutcomeRow(
        true_difference=delta,
        test_cases=n,
        t_test_significant=significant,
        equivalent=equivalent,
        non_inferior=non_inferior,
        exact_t_test_significant=t_test_power(n, delta, sd=spread),
        exact_equivalent=equivalence_power(n, delta, sd=spread, margin=edge),
        exact_non_inferior=non_inferiority_power(n, delta, sd=spread, margin=edge),
    )


def outcome_rates(
    seed: int = SEED,
    *,
    true_differences: tuple[float, ...] = TRUE_DIFFERENCES,
    test_cases: tuple[int, ...] = TEST_CASES,
    replications: int = REPLICATIONS,
) -> tuple[OutcomeRow, ...]:
    """Simulate every size for each true difference in turn, from one generator."""
    rng = np.random.default_rng(count(seed, name="seed"))
    return tuple(
        outcome_row(rng, n, delta, replications=replications)
        for delta in true_differences
        for n in test_cases
    )


def cases_for_equivalence(
    power: float,
    *,
    true_difference: float = 0.0,
    sd: float = DIFFERENCE_SD,
    margin: float = MARGIN,
    significance: float = SIGNIFICANCE,
) -> float:
    """The article's sample size, ``sigma_d^2 (z_{1-a} + z_{1-b/2})^2 / (delta - |Delta|)^2``.

    For identical models this is the usual TOST approximation. Away from zero
    only one edge binds, and the exact size is smaller (``z_{1-b}`` in place of
    ``z_{1-b/2}``); the article uses this form for both, as reproduced here.
    """
    alpha = probability(significance, name="significance", inclusive=False)
    beta = 1 - probability(power, name="power", inclusive=False)
    room = positive(margin, name="margin") - abs(real(true_difference, name="true_difference"))
    if room <= 0:
        raise ValueError("the true difference must lie inside the margin")
    z = float(stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - beta / 2))
    return positive(sd, name="sd") ** 2 * z**2 / room**2


def cases_to_detect(
    power: float,
    *,
    difference: float = MARGIN,
    sd: float = DIFFERENCE_SD,
    significance: float = SIGNIFICANCE,
) -> float:
    """Cases for a two-sided test to detect ``d``: ``sigma_d^2 (z_{1-a/2} + z_b)^2 / d^2``."""
    alpha = probability(significance, name="significance", inclusive=False)
    beta = probability(power, name="power", inclusive=False)
    z = float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(beta))
    return positive(sd, name="sd") ** 2 * z**2 / positive(difference, name="difference") ** 2


def smallest_cases_for_power(
    power: float, *, true_difference: float = 0.0, upper: int = 100_000
) -> int:
    """Fewest cases at which the exact TOST power reaches ``power``, by bisection on ``n``."""
    goal = probability(power, name="power", inclusive=False)
    lo, hi = 2, count(upper, name="upper", minimum=3)
    if equivalence_power(hi, true_difference) < goal:
        raise ValueError("the target power is not reached within the upper bound")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if equivalence_power(mid, true_difference) >= goal:
            hi = mid
        else:
            lo = mid
    return hi


def article_numbers() -> ArticleNumbers:
    """The interval width at 50 cases, the article's sample sizes and the exact power at 308."""
    _, one_sided = _critical_values(50, SIGNIFICANCE)
    formula_cases = round(cases_for_equivalence(0.80))
    return ArticleNumbers(
        interval_width_at_fifty=2 * one_sided * DIFFERENCE_SD / sqrt(50),
        cases_for_eighty=cases_for_equivalence(0.80),
        cases_for_ninety=cases_for_equivalence(0.90),
        cases_for_eighty_half_point_worse=cases_for_equivalence(0.80, true_difference=-0.5),
        cases_to_detect=cases_to_detect(0.80),
        exact_power_at_formula_cases=equivalence_power(formula_cases, 0.0),
        smallest_cases_for_eighty=smallest_cases_for_power(0.80),
    )


def example_payload() -> EquivalenceSummary:
    """Return the figure's simulated rates beside the exact ones, and the article's numbers."""
    return EquivalenceSummary(rows=outcome_rates(), article=article_numbers())
