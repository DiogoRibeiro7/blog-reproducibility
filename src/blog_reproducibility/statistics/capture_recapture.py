"""Capture-recapture under uneven detectability, for the article on estimating what was missed.

A release holds ``N = 500`` defects, and independent review passes find each one
with probability ``q_i = min(p_i e, 1)``, where the ease ``e`` of a defect is
lognormal with mean one and log standard deviation ``s``, the spread. With
``n_1`` and ``n_2`` found by two passes and ``m`` by both, Lincoln-Petersen
estimates ``n_1 n_2 / m`` and Chapman ``(n_1 + 1)(n_2 + 1) / (m + 1) - 1``.
With three passes, Chao's estimator adds ``f_1^2 / (2 f_2)`` to the number seen,
from the defects seen exactly once and exactly twice.

Defects are independent, so every count has a closed form in the moments of the
detection probabilities over the ease, one integral over a standard normal
(computed here by adaptive quadrature, with the kinks where ``p_i e = 1``
marked). A defect is missed by every pass with probability
``P_0 = E[prod_i (1 - q_i)]``, so the number missed is binomial with that
probability, and the expected shares seen exactly ``k`` times are the
coefficients of ``E[prod_i (1 - q_i + q_i x)]``. As ``N`` grows the two-pass
estimate tends to ``N E[q_1] E[q_2] / E[q_1 q_2]``, which is ``N`` when every
defect is equally easy and falls below it as soon as the ease varies, since
the overlap ``E[q_1 q_2]`` then exceeds ``E[q_1] E[q_2]``. The correlation this
induces between the passes is ``(E[q_1 q_2] - E[q_1] E[q_2]) / sqrt(E[q_1](1 -
E[q_1]) E[q_2](1 - E[q_2]))``.

Chao's estimator tends to ``N (1 - P_0 + P_1^2 / (2 P_2))``. When each defect
is equally likely to be found on every one of ``t`` passes, Cauchy-Schwarz puts
the expected number missed at no less than ``(t - 1) / t`` times
``E[f_1]^2 / (2 E[f_2])``, the bound of Chao (1987). The figure and the article
use the form without that factor, the limit for many passes; with three passes
it overstates the correction by half, so with every defect equally easy it is
543 rather than a bound near 500.

The figure reseeds a generator at 17 for each of 15 spreads from 0 to 1.4 and
runs 200 replications of three passes with detection 0.5, 0.45 and 0.4, drawing
the ease (except at zero spread) and then the three passes; it plots the median
two-pass Chapman estimate from the first two passes, the median of Chao's
estimate from all three, and the mean number found by at least one pass. It is
reproduced draw for draw. The article's tables use other seeds and designs, so
they are compared with the closed forms rather than reproduced.
"""

from dataclasses import dataclass
from itertools import pairwise
from math import exp, inf, log, nan, pi, sqrt
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

from blog_reproducibility.common.validation import count, non_negative, probability

__all__ = [
    "ARTICLE_SPREADS",
    "DETECTION",
    "HOMOGENEOUS_DETECTIONS",
    "REPLICATIONS",
    "SEED",
    "SPREAD_POINTS",
    "SPREAD_START",
    "SPREAD_STOP",
    "TRUE_POPULATION",
    "TWO_PASS_DETECTION",
    "CaptureRecaptureSummary",
    "ExpectedCapture",
    "HeterogeneityCurve",
    "HomogeneousRow",
    "capture_frequencies",
    "chao_estimate",
    "chapman",
    "chapman_standard_error",
    "draw_passes",
    "example_payload",
    "expected_capture",
    "heterogeneity_curve",
    "homogeneous_row",
    "lincoln_petersen",
    "overlap",
    "two_pass_standard_deviation",
]

SEED: Final[int] = 17
TRUE_POPULATION: Final[int] = 500
# The figure: three passes, 15 spreads from 0 to 1.4, 200 replications at each.
DETECTION: Final[tuple[float, ...]] = (0.5, 0.45, 0.4)
SPREAD_START: Final[float] = 0.0
SPREAD_STOP: Final[float] = 1.4
SPREAD_POINTS: Final[int] = 15
REPLICATIONS: Final[int] = 200
# The article's two-pass designs and the spreads of its tables.
TWO_PASS_DETECTION: Final[tuple[float, float]] = (0.60, 0.45)
HOMOGENEOUS_DETECTIONS: Final[tuple[tuple[float, float], ...]] = (
    (0.60, 0.45),
    (0.40, 0.30),
    (0.25, 0.20),
)
ARTICLE_SPREADS: Final[tuple[float, ...]] = (0.0, 0.4, 0.8, 1.2)


@dataclass(frozen=True, slots=True)
class ExpectedCapture:
    """Closed forms for one design: shares seen ``k`` times and the estimators' limits."""

    spread: float
    detection: tuple[float, ...]
    frequencies: tuple[float, ...]
    seen: float
    missed_sd: float
    two_pass_limit: float
    pass_correlation: float
    chao_limit: float | None
    chao_bound: float | None


@dataclass(frozen=True, slots=True)
class HeterogeneityCurve:
    """The figure: medians of the two estimators and the mean number seen, at each spread."""

    spreads: tuple[float, ...]
    two_pass: tuple[float, ...]
    chao: tuple[float, ...]
    seen: tuple[float, ...]
    two_pass_sd: tuple[float, ...]
    chao_sd: tuple[float, ...]
    smallest_twice: int


@dataclass(frozen=True, slots=True)
class HomogeneousRow:
    """Two passes over equally easy defects: seen, missed, and the spread of the estimate."""

    detection: tuple[float, float]
    seen: float
    missed: float
    missed_sd: float
    estimate_sd: float


@dataclass(frozen=True, slots=True)
class CaptureRecaptureSummary:
    """The figure's curves, the closed forms beside them, and those behind the article's tables."""

    curve: HeterogeneityCurve
    expected: tuple[ExpectedCapture, ...]
    homogeneous: tuple[HomogeneousRow, ...]
    two_pass: tuple[ExpectedCapture, ...]
    three_pass: tuple[ExpectedCapture, ...]


def _detection(values: tuple[float, ...], minimum: int = 1) -> tuple[float, ...]:
    if len(values) < minimum:
        raise ValueError(f"detection needs at least {minimum} passes")
    return tuple(probability(p, name="detection") for p in values)


def draw_passes(
    rng: np.random.Generator,
    detection: tuple[float, ...] = DETECTION,
    spread: float = 0.0,
    population: int = TRUE_POPULATION,
) -> tuple[NDArray[np.bool_], ...]:
    """Draw each item's ease (unless the spread is zero), then each pass, as the website."""
    n = count(population, name="population", minimum=1)
    ps = _detection(detection)
    s = non_negative(spread, name="spread")
    ease = rng.lognormal(-(s**2) / 2, s, n) if s else np.ones(n)
    return tuple(rng.random(n) < np.clip(p * ease, 0, 1) for p in ps)


def overlap(first: NDArray[np.bool_], second: NDArray[np.bool_]) -> tuple[int, int, int]:
    """Found by the first pass, by the second, and by both."""
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("passes must be one-dimensional and of equal length")
    return int(first.sum()), int(second.sum()), int((first & second).sum())


def _counts(first: int, second: int, both: int) -> tuple[int, int, int]:
    n1, n2 = count(first, name="first"), count(second, name="second")
    m = count(both, name="both")
    if m > min(n1, n2):
        raise ValueError("both cannot exceed either pass")
    return n1, n2, m


def lincoln_petersen(first: int, second: int, both: int) -> float:
    """``n_1 n_2 / m``, infinite when the passes share nothing."""
    n1, n2, m = _counts(first, second, both)
    return inf if m == 0 else n1 * n2 / m


def chapman(first: int, second: int, both: int) -> float:
    """Chapman's estimate ``(n_1 + 1)(n_2 + 1) / (m + 1) - 1``, always finite."""
    n1, n2, m = _counts(first, second, both)
    return (n1 + 1) * (n2 + 1) / (m + 1) - 1


def chapman_standard_error(first: int, second: int, both: int) -> float:
    """Seber's standard error of Chapman's estimate."""
    n1, n2, m = _counts(first, second, both)
    variance = (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    return sqrt(variance)


def capture_frequencies(passes: tuple[NDArray[np.bool_], ...]) -> tuple[int, ...]:
    """Number of items seen exactly ``k`` times, for ``k`` from zero to the number of passes."""
    if not passes:
        raise ValueError("passes must not be empty")
    times = sum(p.astype(int) for p in passes)
    return tuple(int(np.count_nonzero(times == k)) for k in range(len(passes) + 1))


def chao_estimate(seen: int, once: int, twice: int, *, occasions: int | None = None) -> float:
    """Items seen plus ``f_1^2 / (2 f_2)``, times ``(t - 1) / t`` for ``t`` occasions if given.

    Without ``occasions`` this is the website's form; it is not a number when
    nothing was seen twice, as there.
    """
    d = count(seen, name="seen")
    f1, f2 = count(once, name="once"), count(twice, name="twice")
    if f1 + f2 > d:
        raise ValueError("once and twice cannot exceed seen")
    factor = 1.0
    if occasions is not None:
        t = count(occasions, name="occasions", minimum=2)
        factor = (t - 1) / t
    return d + factor * f1**2 / (2 * f2) if f2 else nan


def two_pass_standard_deviation(
    first: float, second: float, population: int = TRUE_POPULATION
) -> float:
    """Large-sample spread of the two-pass estimate: ``sqrt(N (1 - p_1)(1 - p_2) / (p_1 p_2))``."""
    p1 = probability(first, name="first", inclusive=False)
    p2 = probability(second, name="second", inclusive=False)
    return sqrt(count(population, name="population", minimum=1) * (1 - p1) * (1 - p2) / (p1 * p2))


def _moments(detection: tuple[float, ...], spread: float) -> NDArray[np.float64]:
    """``E[q_1], E[q_2], E[q_1 q_2]`` and the chances of ``0 .. t`` captures, over the ease."""

    def integrand_at(ease: float) -> NDArray[np.float64]:
        q = np.clip(np.asarray(detection) * ease, 0, 1)
        captures = np.array([1.0])
        for qi in q:
            captures = np.convolve(captures, [1 - qi, qi])
        second = q[1] if q.size > 1 else 0.0
        return np.concatenate([[q[0], second, q[0] * second], captures])

    if spread == 0:
        return integrand_at(1.0)

    def integrand(z: float) -> NDArray[np.float64]:
        weighted: NDArray[np.float64] = integrand_at(exp(-spread * spread / 2 + spread * z)) * (
            exp(-z * z / 2) / sqrt(2 * pi)
        )
        return weighted

    kinks = sorted((log(1 / p) + spread * spread / 2) / spread for p in detection if p > 0)
    bounds = [-12.0, *(k for k in kinks if -12 < k < 12), 12.0]
    total = np.zeros(len(detection) + 4)
    for low, high in pairwise(bounds):
        value, _ = integrate.quad_vec(integrand, low, high, epsabs=1e-14, epsrel=1e-12)
        total += value
    return total


def expected_capture(
    spread: float,
    detection: tuple[float, ...] = DETECTION,
    population: int = TRUE_POPULATION,
) -> ExpectedCapture:
    """Closed forms of the counts and the estimators' large-population limits."""
    s = non_negative(spread, name="spread")
    ps = _detection(detection, minimum=2)
    n = count(population, name="population", minimum=1)
    moments = _moments(ps, s)
    e1, e2, e12 = moments[:3]
    shares = moments[3:]
    missed = float(shares[0])
    correlation = (e12 - e1 * e2) / sqrt(e1 * (1 - e1) * e2 * (1 - e2))
    chao: float | None = None
    bound: float | None = None
    if len(ps) >= 3:
        correction = float(shares[1] ** 2 / (2 * shares[2]))
        chao = n * (1 - missed + correction)
        bound = n * (1 - missed + (len(ps) - 1) / len(ps) * correction)
    return ExpectedCapture(
        spread=s,
        detection=ps,
        frequencies=tuple(float(f) for f in shares),
        seen=n * (1 - missed),
        missed_sd=sqrt(n * missed * (1 - missed)),
        two_pass_limit=float(n * e1 * e2 / e12),
        pass_correlation=float(correlation),
        chao_limit=chao,
        chao_bound=bound,
    )


def homogeneous_row(
    detection: tuple[float, float], population: int = TRUE_POPULATION
) -> HomogeneousRow:
    """Two passes over equally easy items: seen, missed, and the spread of the estimate."""
    p1, p2 = _detection(detection, minimum=2)
    n = count(population, name="population", minimum=1)
    missed = (1 - p1) * (1 - p2)
    return HomogeneousRow(
        detection=(p1, p2),
        seen=n * (1 - missed),
        missed=n * missed,
        missed_sd=sqrt(n * missed * (1 - missed)),
        estimate_sd=two_pass_standard_deviation(p1, p2, n),
    )


def heterogeneity_curve(
    seed: int = SEED,
    *,
    replications: int = REPLICATIONS,
    population: int = TRUE_POPULATION,
    detection: tuple[float, ...] = DETECTION,
    spreads: tuple[float, ...] | None = None,
) -> HeterogeneityCurve:
    """The figure: at each spread, a generator reseeded and three passes replicated."""
    rng_seed = count(seed, name="seed")
    runs = count(replications, name="replications", minimum=1)
    ps = _detection(detection, minimum=3)
    grid = (
        tuple(float(s) for s in np.linspace(SPREAD_START, SPREAD_STOP, SPREAD_POINTS))
        if spreads is None
        else tuple(non_negative(s, name="spread") for s in spreads)
    )
    if not grid:
        raise ValueError("spreads must not be empty")
    two_pass, chao, seen, two_sd, chao_sd = [], [], [], [], []
    smallest_twice = population
    for spread in grid:
        rng = np.random.default_rng(rng_seed)
        pair, three, union = [], [], []
        for _ in range(runs):
            passes = draw_passes(rng, ps, spread, population)
            pair.append(chapman(*overlap(passes[0], passes[1])))
            _, once, twice, *_ = capture_frequencies(passes)
            union.append(int(np.logical_or.reduce(passes).sum()))
            three.append(chao_estimate(union[-1], once, twice))
            smallest_twice = min(smallest_twice, twice)
        two_pass.append(float(np.median(pair)))
        chao.append(float(np.median(three)))
        seen.append(float(np.mean(union)))
        two_sd.append(float(np.std(pair)))
        chao_sd.append(float(np.std(three)))
    return HeterogeneityCurve(
        spreads=grid,
        two_pass=tuple(two_pass),
        chao=tuple(chao),
        seen=tuple(seen),
        two_pass_sd=tuple(two_sd),
        chao_sd=tuple(chao_sd),
        smallest_twice=smallest_twice,
    )


def example_payload() -> CaptureRecaptureSummary:
    """Return the figure's curves with their closed forms, and those of the article's tables."""
    curve = heterogeneity_curve()
    return CaptureRecaptureSummary(
        curve=curve,
        expected=tuple(expected_capture(s) for s in curve.spreads),
        homogeneous=tuple(homogeneous_row(pair) for pair in HOMOGENEOUS_DETECTIONS),
        two_pass=tuple(expected_capture(s, TWO_PASS_DETECTION) for s in ARTICLE_SPREADS),
        three_pass=tuple(expected_capture(s) for s in ARTICLE_SPREADS),
    )
