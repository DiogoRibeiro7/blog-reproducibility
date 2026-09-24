"""Cluster-randomised experiments, for the article on randomising stores and analysing customers.

When stores are randomised and customers measured, part of each customer's
outcome belongs to the store. With total outcome variance ``sigma^2``, a share
``rho`` of it between stores (the intraclass correlation) and ``m`` customers
per store, the mean of one store has variance ``sigma^2 [rho + (1 - rho) / m]``,
which is ``sigma^2 / m`` times the design effect ``1 + (m - 1) rho``. A test
that treats customers as independent uses standard errors too small by the
square root of the design effect, so under the null its z statistic has
variance close to the design effect and it rejects at about
``2 Phi(-1.96 / sqrt(1 + (m - 1) rho))``. A t-test on the store means, with the
number of stores minus two degrees of freedom, is exact for this model when the
arms have equal numbers of equal-sized stores.

The figure runs 800 A/A experiments with 20 stores of 200 customers at each of
seven intraclass correlations, from one generator seeded at 0, and is
reproduced draw for draw. The article's tables run their own simulations
(2,000 replications, other designs) from their own generator and are not
reproduced; the design effects, the effective sample size and the variance of a
store mean it derives in closed form are.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "CUSTOMERS_PER_STORE",
    "INTRACLASS_CORRELATIONS",
    "OUTCOME_MEAN",
    "OUTCOME_SD",
    "REPLICATIONS",
    "SEED",
    "SIGNIFICANCE",
    "STORES",
    "ArticleNumbers",
    "ClusterSummary",
    "DesignEffectRow",
    "FalsePositiveRow",
    "StoreMeanVariance",
    "article_numbers",
    "customer_level_false_positive_rate",
    "customer_level_p",
    "design_effect",
    "draw_experiment",
    "effective_sample_size",
    "estimate_sd",
    "example_payload",
    "false_positive_rates",
    "store_level_p",
    "store_mean_variance",
]

SEED: Final[int] = 0
STORES: Final[int] = 20
CUSTOMERS_PER_STORE: Final[int] = 200
OUTCOME_MEAN: Final[float] = 50.0
OUTCOME_SD: Final[float] = 10.0
INTRACLASS_CORRELATIONS: Final[tuple[float, ...]] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
REPLICATIONS: Final[int] = 800
SIGNIFICANCE: Final[float] = 0.05
# The article's false positive table: stores, customers per store, intraclass correlation.
ARTICLE_DESIGNS: Final[tuple[tuple[int, int, float], ...]] = (
    (20, 200, 0.0),
    (20, 200, 0.01),
    (20, 200, 0.05),
    (20, 200, 0.10),
    (8, 500, 0.05),
    (100, 40, 0.05),
)
ARTICLE_STORE_SIZES: Final[tuple[int, ...]] = (50, 200, 2000, 100_000)
ARTICLE_ICC: Final[float] = 0.05


@dataclass(frozen=True, slots=True)
class FalsePositiveRow:
    """Share of A/A experiments declared significant at one intraclass correlation."""

    intraclass_correlation: float
    design_effect: float
    predicted_customer_level: float
    customer_level: float
    store_level: float


@dataclass(frozen=True, slots=True)
class DesignEffectRow:
    """Design effect of one design in the article's false positive table."""

    stores: int
    customers_per_store: int
    intraclass_correlation: float
    design_effect: float


@dataclass(frozen=True, slots=True)
class StoreMeanVariance:
    """Variance of one store's mean outcome, and the between-store floor it cannot pass."""

    customers_per_store: int
    variance: float
    floor: float


@dataclass(frozen=True, slots=True)
class ArticleNumbers:
    """Closed-form numbers from the article, for 20 stores of 200 at a correlation of 0.05."""

    design_effects: tuple[DesignEffectRow, ...]
    effective_sample_size: float
    estimate_sd: float
    customer_level_standard_error: float
    standard_error_ratio: float
    store_mean_variances: tuple[StoreMeanVariance, ...]


@dataclass(frozen=True, slots=True)
class ClusterSummary:
    """The figure's false positive rates and the article's closed-form numbers."""

    rows: tuple[FalsePositiveRow, ...]
    article: ArticleNumbers


def _icc(value: float) -> float:
    return probability(value, name="intraclass_correlation", inclusive=True)


def design_effect(customers_per_store: int, intraclass_correlation: float) -> float:
    """Variance of a clustered mean over an independent one of the same size."""
    m = count(customers_per_store, name="customers_per_store", minimum=1)
    return 1 + (m - 1) * _icc(intraclass_correlation)


def effective_sample_size(
    customers: int, customers_per_store: int, intraclass_correlation: float
) -> float:
    """Independent observations carrying the same information as the clustered sample."""
    total = count(customers, name="customers", minimum=1)
    return total / design_effect(customers_per_store, intraclass_correlation)


def store_mean_variance(
    customers_per_store: int, intraclass_correlation: float, *, sd: float = OUTCOME_SD
) -> float:
    """Variance of one store's mean: the between-store part plus the within part over ``m``."""
    m = count(customers_per_store, name="customers_per_store", minimum=1)
    rho = _icc(intraclass_correlation)
    total = positive(sd, name="sd") ** 2
    return total * rho + total * (1 - rho) / m


def estimate_sd(
    stores: int,
    customers_per_store: int,
    intraclass_correlation: float,
    *,
    sd: float = OUTCOME_SD,
) -> float:
    """Standard deviation of the difference in arm means with half the stores in each arm."""
    per_arm = count(stores, name="stores", minimum=2) / 2
    variance = store_mean_variance(customers_per_store, intraclass_correlation, sd=sd)
    return sqrt(2 * variance / per_arm)


def customer_level_false_positive_rate(
    customers_per_store: int,
    intraclass_correlation: float,
    *,
    significance: float = SIGNIFICANCE,
) -> float:
    """Large-sample A/A rejection rate of the customer-level test: ``2 Phi(-z / sqrt(deff))``."""
    alpha = probability(significance, name="significance", inclusive=False)
    critical = float(stats.norm.isf(alpha / 2))
    deff = design_effect(customers_per_store, intraclass_correlation)
    return float(2 * stats.norm.sf(critical / sqrt(deff)))


def draw_experiment(
    rng: np.random.Generator,
    stores: int,
    customers_per_store: int,
    intraclass_correlation: float,
    *,
    sd: float = OUTCOME_SD,
    mean: float = OUTCOME_MEAN,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Randomise stores 50/50 and draw outcomes, one row of customers per store.

    The draws are the article's: a shuffle of the assignment, one normal store
    effect per store, then one normal noise term per customer, store by store.
    """
    n_stores = count(stores, name="stores", minimum=4)
    if n_stores % 2:
        raise ValueError("stores must be even so that the arms are the same size")
    m = count(customers_per_store, name="customers_per_store", minimum=2)
    rho = _icc(intraclass_correlation)
    scale = positive(sd, name="sd")
    between, within = scale * np.sqrt(rho), scale * np.sqrt(1 - rho)
    assignment = np.repeat([0, 1], n_stores // 2)
    rng.shuffle(assignment)
    store_effects = rng.normal(0, between, n_stores)
    baseline = real(mean, name="mean") + np.repeat(store_effects, m)
    outcomes = baseline + rng.normal(0, within, n_stores * m)
    return assignment, outcomes.reshape(n_stores, m)


def _checked(
    assignment: ArrayLike, outcomes: ArrayLike
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    arms = np.asarray(assignment)
    values = np.asarray(outcomes, dtype=np.float64)
    if arms.ndim != 1 or values.ndim != 2 or values.shape[0] != arms.size:
        raise ValueError("outcomes must have one row of customers per store in the assignment")
    if not np.all((arms == 0) | (arms == 1)):
        raise ValueError("assignment must be 0 for control and 1 for treatment")
    treated = arms == 1
    if min(int(treated.sum()), int((~treated).sum())) < 2 or values.shape[1] < 1:
        raise ValueError("each arm needs at least two stores with customers")
    if not np.all(np.isfinite(values)):
        raise ValueError("outcomes must be finite")
    return treated, values


def _two_sample_statistic(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Difference in means over the unpooled standard error."""
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((b.mean() - a.mean()) / se)


def _customer_statistic(treated: NDArray[np.bool_], values: NDArray[np.float64]) -> float:
    return _two_sample_statistic(values[~treated].ravel(), values[treated].ravel())


def _store_statistic(treated: NDArray[np.bool_], values: NDArray[np.float64]) -> float:
    means = values.mean(axis=1)
    return _two_sample_statistic(means[~treated], means[treated])


def customer_level_p(assignment: ArrayLike, outcomes: ArrayLike) -> float:
    """Two-sided p-value of a two-sample z-test that treats every customer as independent."""
    treated, values = _checked(assignment, outcomes)
    return float(2 * stats.norm.sf(abs(_customer_statistic(treated, values))))


def store_level_p(assignment: ArrayLike, outcomes: ArrayLike) -> float:
    """Two-sided p-value of a t-test on the store means, stores minus two degrees of freedom."""
    treated, values = _checked(assignment, outcomes)
    statistic = _store_statistic(treated, values)
    return float(2 * stats.t.sf(abs(statistic), values.shape[0] - 2))


def false_positive_rates(
    seed: int = SEED,
    *,
    intraclass_correlations: tuple[float, ...] = INTRACLASS_CORRELATIONS,
    stores: int = STORES,
    customers_per_store: int = CUSTOMERS_PER_STORE,
    replications: int = REPLICATIONS,
) -> tuple[FalsePositiveRow, ...]:
    """Run A/A experiments at each intraclass correlation in turn from one generator.

    The statistics are those of :func:`customer_level_p` and :func:`store_level_p`;
    their p-values are computed together after each batch, which gives the same
    values without a SciPy call per experiment.
    """
    rng = np.random.default_rng(count(seed, name="seed"))
    reps = count(replications, name="replications", minimum=1)
    rows = []
    for icc in (_icc(value) for value in intraclass_correlations):
        customer_z, store_t = np.empty(reps), np.empty(reps)
        for rep in range(reps):
            assignment, outcomes = draw_experiment(rng, stores, customers_per_store, icc)
            treated = assignment == 1
            customer_z[rep] = _customer_statistic(treated, outcomes)
            store_t[rep] = _store_statistic(treated, outcomes)
        customer_p = 2 * stats.norm.sf(np.abs(customer_z))
        store_p = 2 * stats.t.sf(np.abs(store_t), stores - 2)
        rows.append(
            FalsePositiveRow(
                intraclass_correlation=icc,
                design_effect=design_effect(customers_per_store, icc),
                predicted_customer_level=customer_level_false_positive_rate(
                    customers_per_store, icc
                ),
                customer_level=int(np.count_nonzero(customer_p < SIGNIFICANCE)) / reps,
                store_level=int(np.count_nonzero(store_p < SIGNIFICANCE)) / reps,
            )
        )
    return tuple(rows)


def article_numbers() -> ArticleNumbers:
    """Design effects, effective sample size, standard errors and store-mean variances."""
    customers = STORES * CUSTOMERS_PER_STORE
    truth = estimate_sd(STORES, CUSTOMERS_PER_STORE, ARTICLE_ICC)
    naive = OUTCOME_SD * sqrt(2 / (customers / 2))
    return ArticleNumbers(
        design_effects=tuple(
            DesignEffectRow(n, m, icc, design_effect(m, icc)) for n, m, icc in ARTICLE_DESIGNS
        ),
        effective_sample_size=effective_sample_size(customers, CUSTOMERS_PER_STORE, ARTICLE_ICC),
        estimate_sd=truth,
        customer_level_standard_error=naive,
        standard_error_ratio=naive / truth,
        store_mean_variances=tuple(
            StoreMeanVariance(m, store_mean_variance(m, ARTICLE_ICC), OUTCOME_SD**2 * ARTICLE_ICC)
            for m in ARTICLE_STORE_SIZES
        ),
    )


def example_payload() -> ClusterSummary:
    """Return the figure's A/A false positive rates and the article's closed forms."""
    return ClusterSummary(rows=false_positive_rates(), article=article_numbers())
