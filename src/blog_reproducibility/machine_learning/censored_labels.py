"""Censored snapshot labels, for the article on censored labels in supervised learning.

A churn label read from a database extract says whether a customer had churned
by the extract date. For a customer who signed up last month that is a
one-month outcome, and for a customer who signed up three years ago it is a
three-year one, so the label mixes "will not churn" with "has not been observed
for long enough". A model given tenure at the extract as a feature learns the
extract date rather than the risk.

The simulation has 20,000 customers signing up uniformly over 36 months, with
the extract at month 36. Two standard normal features set a Weibull scale
``18 exp(-(0.8 x1 - 0.6 x2))`` months with shape 1.3, so each customer's true
twelve-month churn probability has the closed form

    P(T <= 12) = 1 - exp(-(12 / scale) ** shape),

and does not depend on signup date. A quarter of the customers are held out.
Three models are trained on the rest:

* naive: gradient boosting on the snapshot label, with tenure as a feature;
* fixed horizon: gradient boosting on churn within twelve months, restricted to
  customers observed for at least twelve months;
* discrete-time hazard: logistic regression on one row per customer-month up to
  churn, censoring, or month twelve, with a dummy per month; the twelve-month
  probability is one minus the product of the monthly survival probabilities.

The article's code and the figure generator share the design and one generator
seeded at 0, drawn in the same order (signup, both features, Weibull times,
test split), and both boosting models use ``random_state=0``, so the figure and
every number the article prints come from this one simulation. The customer-
month rows are built with array operations rather than the article's Python
loop, in the same order, so the hazard fit is unchanged.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from blog_reproducibility.common.validation import count, non_negative, positive, probability

__all__ = [
    "BASE_SCALE",
    "COHORTS",
    "CUSTOMERS",
    "EXTRACT_MONTH",
    "HORIZON",
    "SEED",
    "WEIBULL_SHAPE",
    "CensoredLabelsSummary",
    "ChurnRates",
    "CohortRow",
    "CustomerBase",
    "ModelPredictions",
    "ModelScore",
    "PersonMonths",
    "cohort_table",
    "example_payload",
    "fit_models",
    "hazard_design",
    "horizon_probability",
    "life_table_probability",
    "person_months",
    "simulate_customers",
    "weibull_scale",
]

SEED: Final[int] = 0
CUSTOMERS: Final[int] = 20_000
EXTRACT_MONTH: Final[float] = 36.0
HORIZON: Final[int] = 12
BASE_SCALE: Final[float] = 18.0
WEIBULL_SHAPE: Final[float] = 1.3
FEATURE_EFFECTS: Final[tuple[float, float]] = (0.8, -0.6)
TEST_SHARE: Final[float] = 0.25
COHORTS: Final[tuple[tuple[int, int], ...]] = ((0, 3), (3, 6), (6, 12), (12, 24), (24, 36))
BOOSTING_ITERATIONS: Final[int] = 200
BOOSTING_LEARNING_RATE: Final[float] = 0.05
HAZARD_C: Final[float] = 10.0
HAZARD_MAX_ITER: Final[int] = 2000
_FOLLOWUP_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True, slots=True, eq=False)
class CustomerBase:
    """One simulated extract: features, churn times, follow-up, and the test split."""

    x1: NDArray[np.float64]
    x2: NDArray[np.float64]
    churn_time: NDArray[np.float64]
    followup: NDArray[np.float64]
    scale: NDArray[np.float64]
    test: NDArray[np.bool_]
    horizon: int


@dataclass(frozen=True, slots=True, eq=False)
class PersonMonths:
    """Customer-month rows: features, month index, and churn-in-that-month label."""

    x1: NDArray[np.float64]
    x2: NDArray[np.float64]
    month: NDArray[np.int64]
    churned: NDArray[np.int64]


@dataclass(frozen=True, slots=True, eq=False)
class ModelPredictions:
    """Twelve-month churn predictions for the held-out customers from each model."""

    naive: NDArray[np.float64]
    fixed_horizon: NDArray[np.float64]
    hazard: NDArray[np.float64]
    naive_new_customer: float
    person_month_rows: int


@dataclass(frozen=True, slots=True)
class CohortRow:
    """Mean true and predicted twelve-month churn for one tenure cohort."""

    lower: int
    upper: int
    customers: int
    true: float
    naive: float
    fixed_horizon: float
    hazard: float


@dataclass(frozen=True, slots=True)
class ModelScore:
    """AUC against the true twelve-month outcome, and mean absolute probability error."""

    model: str
    auc: float
    mean_absolute_error: float


@dataclass(frozen=True, slots=True)
class ChurnRates:
    """Twelve-month churn rate among test customers, true and as the snapshot counts it."""

    true: float
    observed_all: float
    observed_full_followup: float


@dataclass(frozen=True, slots=True)
class CensoredLabelsSummary:
    """Every number the article prints from its simulation."""

    training_customers: int
    short_followup_training: int
    person_month_rows: int
    cohorts: tuple[CohortRow, ...]
    scores: tuple[ModelScore, ...]
    naive_own_label_auc: float
    naive_new_customer: float
    true_new_customer: float
    test_rates: ChurnRates


def weibull_scale(
    x1: NDArray[np.float64], x2: NDArray[np.float64], *, base: float = BASE_SCALE
) -> NDArray[np.float64]:
    """Weibull scale in months: ``base exp(-(0.8 x1 - 0.6 x2))``."""
    first, second = FEATURE_EFFECTS
    return positive(base, name="base") * np.exp(-(first * x1 + second * x2))


def horizon_probability(
    scale: NDArray[np.float64] | float,
    *,
    horizon: float = HORIZON,
    shape: float = WEIBULL_SHAPE,
) -> NDArray[np.float64]:
    """Closed-form probability of churning within ``horizon``: ``1 - exp(-(h / scale)^k)``."""
    scales = np.asarray(scale, dtype=np.float64)
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("scale must be finite and positive")
    ratio = non_negative(horizon, name="horizon") / scales
    return 1.0 - np.exp(-(ratio ** positive(shape, name="shape")))


def simulate_customers(
    customers: int = CUSTOMERS, *, horizon: int = HORIZON, seed: int = SEED
) -> CustomerBase:
    """Draw one extract in the article's order: signup, features, churn times, test split."""
    size = count(customers, name="customers", minimum=2)
    rng = np.random.default_rng(count(seed, name="seed"))
    signup = rng.uniform(0, EXTRACT_MONTH, size)
    x1, x2 = rng.normal(size=size), rng.normal(size=size)
    scale = weibull_scale(x1, x2)
    churn_time = scale * rng.weibull(WEIBULL_SHAPE, size)
    test = rng.uniform(size=size) < probability(TEST_SHARE, name="test share")
    return CustomerBase(
        x1=x1,
        x2=x2,
        churn_time=churn_time,
        followup=EXTRACT_MONTH - signup,
        scale=scale,
        test=test,
        horizon=count(horizon, name="horizon", minimum=1),
    )


def person_months(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    churn_time: NDArray[np.float64],
    followup: NDArray[np.float64],
    *,
    horizon: int = HORIZON,
) -> PersonMonths:
    """One row per customer per month up to ``ceil(min(churn, follow-up, horizon))``.

    Each customer contributes at least one row. Month ``m`` is labelled 1 when
    churn falls in ``(m - 1, m]`` and the customer was observed through month
    ``m``; a customer censored part-way through a month is counted as having
    survived it, as in the article's code. Rows are ordered customer by customer,
    months ascending.
    """
    limit = count(horizon, name="horizon", minimum=1)
    if not (x1.shape == x2.shape == churn_time.shape == followup.shape) or x1.ndim != 1:
        raise ValueError("x1, x2, churn_time and followup must be vectors of one length")
    if np.any(churn_time < 0) or np.any(followup < 0):
        raise ValueError("churn times and follow-up must be non-negative")
    last = np.ceil(np.minimum(np.minimum(churn_time, followup), limit)).astype(np.int64)
    months_per_customer = np.maximum(last, 1)
    owner = np.repeat(np.arange(x1.size), months_per_customer)
    starts = np.cumsum(months_per_customer) - months_per_customer
    month = np.arange(owner.size) - np.repeat(starts, months_per_customer) + 1
    time = churn_time[owner]
    churned = (
        (time <= month) & (time > month - 1) & (month <= followup[owner] + _FOLLOWUP_TOLERANCE)
    )
    return PersonMonths(
        x1=x1[owner],
        x2=x2[owner],
        month=month.astype(np.int64),
        churned=churned.astype(np.int64),
    )


def hazard_design(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    month: NDArray[np.int64],
    *,
    horizon: int = HORIZON,
) -> NDArray[np.float64]:
    """Features followed by one dummy column per month, as in the article."""
    limit = count(horizon, name="horizon", minimum=1)
    if np.any(month < 1) or np.any(month > limit):
        raise ValueError("months must lie between 1 and the horizon")
    return np.column_stack([x1, x2, np.eye(limit)[month - 1]])


def life_table_probability(rows: PersonMonths, *, horizon: int = HORIZON) -> float:
    """Covariate-free discrete-time estimate: ``1 - prod_m (1 - events_m / at_risk_m)``."""
    limit = count(horizon, name="horizon", minimum=1)
    survival = 1.0
    for month in range(1, limit + 1):
        at_risk = rows.month == month
        if not np.any(at_risk):
            break
        survival *= 1.0 - float(np.mean(rows.churned[at_risk]))
    return 1.0 - survival


def _boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=BOOSTING_ITERATIONS, learning_rate=BOOSTING_LEARNING_RATE, random_state=0
    )


def fit_models(base: CustomerBase) -> ModelPredictions:
    """Fit the naive, fixed-horizon and hazard models; predict for held-out customers."""
    horizon = base.horizon
    train, test = ~base.test, base.test
    churned_by_extract = base.churn_time <= base.followup
    event = base.churn_time <= horizon
    full = base.followup >= horizon

    naive_features = np.column_stack([base.x1, base.x2, base.followup])
    naive = _boosting().fit(naive_features[train], churned_by_extract[train])
    fixed_features = np.column_stack([base.x1, base.x2])
    fixed = _boosting().fit(fixed_features[train & full], event[train & full])

    rows = person_months(
        base.x1[train],
        base.x2[train],
        base.churn_time[train],
        base.followup[train],
        horizon=horizon,
    )
    hazard = LogisticRegression(C=HAZARD_C, max_iter=HAZARD_MAX_ITER).fit(
        hazard_design(rows.x1, rows.x2, rows.month, horizon=horizon), rows.churned
    )
    x1, x2 = base.x1[test], base.x2[test]
    survival = np.ones(x1.size)
    for month in range(1, horizon + 1):
        design = hazard_design(x1, x2, np.full(x1.size, month, dtype=np.int64), horizon=horizon)
        survival *= 1.0 - hazard.predict_proba(design)[:, 1]

    return ModelPredictions(
        naive=naive.predict_proba(naive_features[test])[:, 1],
        fixed_horizon=fixed.predict_proba(fixed_features[test])[:, 1],
        hazard=1.0 - survival,
        naive_new_customer=float(naive.predict_proba([[0.0, 0.0, 0.0]])[0, 1]),
        person_month_rows=int(rows.month.size),
    )


def cohort_table(
    followup: NDArray[np.float64],
    true: NDArray[np.float64],
    predictions: ModelPredictions,
    *,
    cohorts: tuple[tuple[int, int], ...] = COHORTS,
) -> tuple[CohortRow, ...]:
    """Mean true and predicted probability for test customers in each tenure band."""
    rows = []
    for lower, upper in cohorts:
        members = (followup >= lower) & (followup < upper)
        if not np.any(members):
            raise ValueError(f"cohort {lower}-{upper} has no customers")
        rows.append(
            CohortRow(
                lower=lower,
                upper=upper,
                customers=int(members.sum()),
                true=float(np.mean(true[members])),
                naive=float(np.mean(predictions.naive[members])),
                fixed_horizon=float(np.mean(predictions.fixed_horizon[members])),
                hazard=float(np.mean(predictions.hazard[members])),
            )
        )
    return tuple(rows)


def example_payload() -> CensoredLabelsSummary:
    """Run the article's simulation and return every number it prints."""
    base = simulate_customers()
    predictions = fit_models(base)
    test, train = base.test, ~base.test
    horizon = base.horizon
    true = horizon_probability(base.scale, horizon=horizon)[test]
    followup = base.followup[test]
    event = base.churn_time[test] <= horizon
    churned_by_extract = base.churn_time[test] <= followup
    full = followup >= horizon

    scores = tuple(
        ModelScore(
            model=name,
            auc=float(roc_auc_score(event, prediction)),
            mean_absolute_error=float(np.mean(np.abs(prediction - true))),
        )
        for name, prediction in (
            ("naive", predictions.naive),
            ("fixed horizon", predictions.fixed_horizon),
            ("hazard", predictions.hazard),
        )
    )
    return CensoredLabelsSummary(
        training_customers=int(train.sum()),
        short_followup_training=int(np.sum(train & (base.followup < horizon))),
        person_month_rows=predictions.person_month_rows,
        cohorts=cohort_table(followup, true, predictions),
        scores=scores,
        naive_own_label_auc=float(roc_auc_score(churned_by_extract, predictions.naive)),
        naive_new_customer=predictions.naive_new_customer,
        true_new_customer=float(horizon_probability(BASE_SCALE, horizon=horizon)),
        test_rates=ChurnRates(
            true=float(np.mean(event)),
            observed_all=float(np.mean(churned_by_extract & event)),
            observed_full_followup=float(np.mean(event[full])),
        ),
    )
