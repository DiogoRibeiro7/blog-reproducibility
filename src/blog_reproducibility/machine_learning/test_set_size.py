"""Test-set size and the ranking of two classifiers, for the article on test sets.

A test-set accuracy is a sample proportion, so on ``n`` cases it has standard
error ``sqrt(p (1 - p) / n)`` and a 95 percent interval of half-width
``1.96 sqrt(p (1 - p) / n)``. Sizing a test set for a target half-width ``E``
inverts that: ``n = 1.96^2 p (1 - p) / E^2``, counted in the units the metric is
built from (cases for accuracy, positives for recall).

Comparing two models is a different problem. Two classifiers scored on the same
cases share their hard cases, so the difference in their accuracies depends
only on the cases where they disagree. The article models this with a shared
item difficulty ``D ~ N(0, 1)`` and independent model noise of scale ``tau``:
model A gets an item right when ``D + tau e_A < q_A`` and model B when
``D + tau e_B < q_B``, with thresholds chosen so the accuracies are exactly 90
and 91 percent. The disagreement rate follows from the bivariate normal
distribution of the two latent scores, whose correlation is ``1 / (1 + tau^2)``.
McNemar's exact test uses only the discordant cases, and under the alternative
its statistic has mean ``sqrt(n) delta / sqrt(d)``, which gives the sizing rule
``n = (z_{0.975} + z_{0.80})^2 d / delta^2``, about ``7.85 d / delta^2``.

The figure replicates 2,000 test sets of each size from one generator seeded at
3, recording how often model B measures higher on a shared test set and on
separate test sets, how often McNemar's test is significant in B's favour, and
how often two independent 95 percent intervals are disjoint. The article prints
that same simulation as a table, and it is reproduced draw for draw. The
article's population figures (seed 30) and its power check (seed 31) come from
other generators and are not reproduced; the closed-form numbers it derives
are.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from scipy import stats

from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "ACCURACY_A",
    "ACCURACY_B",
    "NOISE_SCALE",
    "REPLICATIONS",
    "SEED",
    "SIGNIFICANCE",
    "TEST_SIZES",
    "Z_95",
    "ArticleNumbers",
    "PopulationSummary",
    "RankingRow",
    "TestSetSizeSummary",
    "accuracy_standard_error",
    "article_numbers",
    "draw_correctness",
    "example_payload",
    "interval_half_width",
    "latent_thresholds",
    "mcnemar_p_values",
    "paired_comparison_size",
    "population_summary",
    "ranking_rates",
    "required_size_for_half_width",
]

SEED: Final[int] = 3
REPLICATIONS: Final[int] = 2000
TEST_SIZES: Final[tuple[int, ...]] = (200, 500, 1000, 2000, 5000, 10000, 20000)
NOISE_SCALE: Final[float] = 0.5
ACCURACY_A: Final[float] = 0.90
ACCURACY_B: Final[float] = 0.91
SIGNIFICANCE: Final[float] = 0.05
Z_95: Final[float] = 1.96
POWER: Final[float] = 0.80


@dataclass(frozen=True, slots=True)
class RankingRow:
    """Shares of simulated test sets of one size, for each of the figure's four series."""

    test_size: int
    same_set: float
    separate_sets: float
    paired_significant: float
    intervals_disjoint: float


@dataclass(frozen=True, slots=True)
class PopulationSummary:
    """Exact population accuracies and disagreement rate of the two classifiers."""

    accuracy_a: float
    accuracy_b: float
    difference: float
    disagreement: float
    paired_size: float


@dataclass(frozen=True, slots=True)
class ArticleNumbers:
    """Closed-form numbers the article derives, in accuracy points or cases."""

    half_widths: tuple[tuple[int, float], ...]
    standard_error_400: float
    recall_positives_for_five_points: float
    recall_half_width_20_positives: float
    slice_half_width_250: float
    paired_multiplier: float
    paired_size_article_inputs: float
    paired_size_tenth_point: float


@dataclass(frozen=True, slots=True)
class TestSetSizeSummary:
    """The figure's simulated shares, the population and the article's closed forms."""

    rows: tuple[RankingRow, ...]
    population: PopulationSummary
    article: ArticleNumbers


def accuracy_standard_error(test_size: int, accuracy: float) -> float:
    """Binomial standard error of an accuracy measured on ``test_size`` cases."""
    n = count(test_size, name="test_size", minimum=1)
    p = probability(accuracy, name="accuracy")
    return sqrt(p * (1 - p) / n)


def interval_half_width(test_size: int, accuracy: float, *, z: float = Z_95) -> float:
    """Half-width of the normal-approximation interval for an accuracy."""
    return positive(z, name="z") * accuracy_standard_error(test_size, accuracy)


def required_size_for_half_width(accuracy: float, half_width: float, *, z: float = Z_95) -> float:
    """Cases (or positives) needed for an interval of the given half-width."""
    p = probability(accuracy, name="accuracy")
    width = positive(half_width, name="half_width")
    return positive(z, name="z") ** 2 * p * (1 - p) / width**2


def paired_comparison_size(
    disagreement: float,
    difference: float,
    *,
    significance: float = SIGNIFICANCE,
    power: float = POWER,
) -> float:
    """Test-set size for McNemar's test to reach ``power`` at a two-sided ``significance``."""
    d = probability(disagreement, name="disagreement", inclusive=False)
    delta = positive(difference, name="difference")
    alpha = probability(significance, name="significance", inclusive=False)
    beta = probability(power, name="power", inclusive=False)
    multiplier = float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(beta)) ** 2
    return multiplier * d / delta**2


def latent_thresholds(
    noise_scale: float = NOISE_SCALE,
    accuracy_a: float = ACCURACY_A,
    accuracy_b: float = ACCURACY_B,
) -> tuple[float, float]:
    """Thresholds on ``D + tau e`` that give each model its target accuracy."""
    tau = positive(noise_scale, name="noise_scale")
    p_a = probability(accuracy_a, name="accuracy_a", inclusive=False)
    p_b = probability(accuracy_b, name="accuracy_b", inclusive=False)
    scale = sqrt(1 + tau**2)
    return float(stats.norm.ppf(p_a)) * scale, float(stats.norm.ppf(p_b)) * scale


def draw_correctness(
    test_size: int,
    rng: np.random.Generator,
    *,
    noise_scale: float = NOISE_SCALE,
    thresholds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one test set and return which cases models A and B get right.

    The draws are the article's: item difficulty, then model A's noise, then
    model B's noise, each a vector of ``test_size`` normals.
    """
    n = count(test_size, name="test_size", minimum=1)
    tau = positive(noise_scale, name="noise_scale")
    q_a, q_b = thresholds if thresholds is not None else latent_thresholds(tau)
    difficulty = rng.normal(size=n)
    correct_a = difficulty + rng.normal(scale=tau, size=n) < q_a
    correct_b = difficulty + rng.normal(scale=tau, size=n) < q_b
    return correct_a, correct_b


def _draw_counts(
    rng: np.random.Generator,
    difficulty: np.ndarray,
    noise: np.ndarray,
    tau: float,
    thresholds: tuple[float, float],
) -> tuple[int, int, int]:
    """Counts of cases A gets right, B gets right, and both get right.

    This consumes the generator exactly as :func:`draw_correctness` does and
    gives the same comparisons, since ``normal(scale=tau)`` is ``tau`` times a
    standard normal, but it reuses two buffers to keep the 2,000-replication
    loops fast.
    """
    q_a, q_b = thresholds
    rng.standard_normal(out=difficulty)
    rng.standard_normal(out=noise)
    noise *= tau
    noise += difficulty
    correct_a = noise < q_a
    rng.standard_normal(out=noise)
    noise *= tau
    noise += difficulty
    correct_b = noise < q_b
    both = int(np.count_nonzero(correct_a & correct_b))
    return int(np.count_nonzero(correct_a)), int(np.count_nonzero(correct_b)), both


def mcnemar_p_values(b_only: np.ndarray, discordant: np.ndarray) -> np.ndarray:
    """Exact two-sided McNemar p-values: binomial tests of ``b_only`` of ``discordant``.

    Under the null the count is Binomial(``discordant``, 1/2), which is symmetric,
    so the two-sided exact p-value is ``min(1, 2 P(X >= max(k, m - k)))``. This
    is the value :func:`scipy.stats.binomtest` returns, computed for many tests
    at once. Rows with no discordant cases get a p-value of one.
    """
    k = np.asarray(b_only, dtype=np.int64)
    m = np.asarray(discordant, dtype=np.int64)
    if k.shape != m.shape:
        raise ValueError("b_only and discordant must have the same shape")
    if np.any(k < 0) or np.any(k > m):
        raise ValueError("b_only must lie between zero and discordant")
    extreme = np.maximum(k, m - k)
    tail = stats.binom.sf(extreme - 1, m, 0.5)
    values = np.minimum(1.0, 2 * tail)
    return np.where(m > 0, values, 1.0)


def _ranking_row(
    test_size: int,
    rng: np.random.Generator,
    replications: int,
    thresholds: tuple[float, float],
) -> RankingRow:
    n = test_size
    difficulty = np.empty(n)
    noise = np.empty(n)
    same = separate = disjoint = 0
    b_counts = np.zeros(replications, dtype=np.int64)
    a_counts = np.zeros(replications, dtype=np.int64)
    for rep in range(replications):
        right_a, right_b, both = _draw_counts(rng, difficulty, noise, NOISE_SCALE, thresholds)
        acc_a, acc_b = right_a / n, right_b / n
        same += acc_b > acc_a
        fresh_a, _, _ = _draw_counts(rng, difficulty, noise, NOISE_SCALE, thresholds)
        separate += acc_b > fresh_a / n
        b_counts[rep] = right_b - both
        a_counts[rep] = right_a - both
        half_a = Z_95 * sqrt(acc_a * (1 - acc_a) / n)
        half_b = Z_95 * sqrt(acc_b * (1 - acc_b) / n)
        disjoint += (acc_b - half_b) > (acc_a + half_a)
    p_values = mcnemar_p_values(b_counts, b_counts + a_counts)
    paired = int(np.sum((p_values < SIGNIFICANCE) & (b_counts > a_counts)))
    return RankingRow(
        test_size=n,
        same_set=same / replications,
        separate_sets=separate / replications,
        paired_significant=paired / replications,
        intervals_disjoint=disjoint / replications,
    )


def ranking_rates(
    seed: int = SEED,
    *,
    test_sizes: tuple[int, ...] = TEST_SIZES,
    replications: int = REPLICATIONS,
) -> tuple[RankingRow, ...]:
    """Simulate the figure's four shares for every test-set size, in the article's draw order."""
    rng = np.random.default_rng(count(seed, name="seed"))
    reps = count(replications, name="replications", minimum=1)
    sizes = tuple(count(n, name="test_size", minimum=1) for n in test_sizes)
    thresholds = latent_thresholds()
    return tuple(_ranking_row(n, rng, reps, thresholds) for n in sizes)


def population_summary(noise_scale: float = NOISE_SCALE) -> PopulationSummary:
    """Exact accuracies and disagreement rate from the bivariate normal latent scores."""
    tau = positive(noise_scale, name="noise_scale")
    q_a, q_b = latent_thresholds(tau)
    variance = 1 + tau**2
    both_right = float(
        stats.multivariate_normal(mean=[0.0, 0.0], cov=[[variance, 1.0], [1.0, variance]]).cdf(
            [q_a, q_b]
        )
    )
    accuracy_a = float(stats.norm.cdf(q_a / sqrt(variance)))
    accuracy_b = float(stats.norm.cdf(q_b / sqrt(variance)))
    disagreement = accuracy_a + accuracy_b - 2 * both_right
    difference = accuracy_b - accuracy_a
    return PopulationSummary(
        accuracy_a=accuracy_a,
        accuracy_b=accuracy_b,
        difference=difference,
        disagreement=disagreement,
        paired_size=paired_comparison_size(disagreement, difference),
    )


def article_numbers() -> ArticleNumbers:
    """Closed-form numbers from the article's prose and first table."""
    return ArticleNumbers(
        half_widths=tuple(
            (n, 100 * interval_half_width(n, 0.9)) for n in (100, 400, 1000, 5000, 20000)
        ),
        standard_error_400=100 * accuracy_standard_error(400, 0.9),
        recall_positives_for_five_points=required_size_for_half_width(0.8, 0.05),
        recall_half_width_20_positives=100 * interval_half_width(20, 0.8),
        slice_half_width_250=100 * interval_half_width(250, 0.9),
        paired_multiplier=paired_comparison_size(0.5, 1.0) / 0.5,
        paired_size_article_inputs=paired_comparison_size(0.084, 0.0092),
        paired_size_tenth_point=paired_comparison_size(0.08, 0.001),
    )


def example_payload() -> TestSetSizeSummary:
    """Return the figure's simulated shares, the exact population and the closed forms."""
    return TestSetSizeSummary(
        rows=ranking_rates(),
        population=population_summary(),
        article=article_numbers(),
    )
