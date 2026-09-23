"""Check the censored-label constructions and reproduce the article's numbers.

The closed-form twelve-month probability is checked against SciPy's Weibull
distribution and against sampling. The vectorised customer-month construction
is checked on hand-worked customers and row for row against the article's
Python loop. A covariate-free life table built from those rows removes most of
the snapshot label's bias but not all of it: the article's construction keeps
the month in which a customer is censored and counts it as survived, which
leaves the estimate about 1.5 points low; dropping those partial months
recovers the true rate. The same small downward bias plausibly explains why the
hazard model sits 0.006 to 0.007 below the truth in every cohort of the
article's table.

The article's code uses the figure's seed and design, so its tables are pinned.
Counts, true probabilities and churn rates are pure NumPy and match exactly at
the printed precision; values from fitted models are allowed one unit in the
last printed digit (the current scikit-learn gives, for example, 0.088 where the
article prints 0.089 for the naive model in the newest cohort).

One prose claim is not checked because the article does not give its code: that
a naive model without tenure "predicts around" a 42 percent snapshot positive
rate. In this simulation the snapshot label (churned by the extract) is
positive for 54 percent of customers, and a tenure-free boosting model on it
predicts 0.54 on average; 42 percent is the twelve-month churn rate counted from
the snapshot, which the evaluation section reports and which is pinned here.
"""

import numpy as np
import pytest
from scipy.stats import weibull_min

from blog_reproducibility.machine_learning.censored_labels import (
    BASE_SCALE,
    COHORTS,
    HORIZON,
    WEIBULL_SHAPE,
    PersonMonths,
    example_payload,
    hazard_design,
    horizon_probability,
    life_table_probability,
    person_months,
    simulate_customers,
    weibull_scale,
)

SUMMARY = example_payload()
SCORES = {score.model: score for score in SUMMARY.scores}


def _article_person_months(
    x1: np.ndarray, x2: np.ndarray, churn_time: np.ndarray, followup: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the rows with the article's loop, kept verbatim as a reference."""
    rows, labels = [], []
    for i in range(x1.size):
        last = int(np.ceil(min(churn_time[i], followup[i], HORIZON)))
        for m in range(1, max(last, 1) + 1):
            rows.append((x1[i], x2[i], m))
            labels.append(
                int(churn_time[i] <= m and churn_time[i] > m - 1 and m <= followup[i] + 1e-9)
            )
    return np.array(rows), np.array(labels)


def test_horizon_probability_matches_scipy_weibull() -> None:
    """The closed form is the Weibull CDF at the horizon."""
    scales = np.array([3.0, 18.0, 60.0])
    expected = weibull_min.cdf(HORIZON, WEIBULL_SHAPE, scale=scales)

    assert horizon_probability(scales) == pytest.approx(expected, rel=1e-12)


def test_horizon_probability_matches_sampling() -> None:
    """Scaled draws from NumPy's Weibull churn within twelve months at the stated rate."""
    rng = np.random.default_rng(3)
    draws = BASE_SCALE * rng.weibull(WEIBULL_SHAPE, 400_000)

    assert np.mean(draws <= HORIZON) == pytest.approx(
        float(horizon_probability(BASE_SCALE)), abs=0.003
    )


def test_weibull_scale_at_average_features() -> None:
    """Average features give the base scale; x1 shortens and x2 lengthens it."""
    zero = np.zeros(1)
    one = np.ones(1)

    assert weibull_scale(zero, zero) == pytest.approx([BASE_SCALE])
    assert weibull_scale(one, zero) == pytest.approx([BASE_SCALE * np.exp(-0.8)])
    assert weibull_scale(zero, one) == pytest.approx([BASE_SCALE * np.exp(0.6)])


def test_person_months_on_hand_worked_customers() -> None:
    """Rows stop at churn, censoring or the horizon; only the churn month is positive."""
    churn_time = np.array([2.5, 20.0, 0.3, 0.7, 30.0])
    followup = np.array([10.0, 4.2, 5.0, 0.5, 30.0])
    ids = np.arange(5, dtype=np.float64)
    rows = person_months(ids, -ids, churn_time, followup)

    # Churns in month 3; censored after 4.2 months (five rows, no event); churns
    # in month 1; censored at half a month before churning (one row, no event);
    # survives the whole horizon (twelve rows, no event).
    assert rows.x1.tolist() == [0.0] * 3 + [1.0] * 5 + [2.0] + [3.0] + [4.0] * 12
    assert rows.x2.tolist() == (-rows.x1).tolist()
    assert rows.month.tolist() == [1, 2, 3, 1, 2, 3, 4, 5, 1, 1, *range(1, 13)]
    assert rows.churned.tolist() == [0, 0, 1] + [0] * 5 + [1] + [0] + [0] * 12


def test_person_months_matches_the_article_loop() -> None:
    """The array construction reproduces the article's rows in the same order."""
    base = simulate_customers(3000, seed=5)
    rows = person_months(base.x1, base.x2, base.churn_time, base.followup)
    reference, labels = _article_person_months(base.x1, base.x2, base.churn_time, base.followup)

    np.testing.assert_array_equal(np.column_stack([rows.x1, rows.x2, rows.month]), reference)
    np.testing.assert_array_equal(rows.churned, labels)


def test_life_table_under_censoring() -> None:
    """Customer-month rows remove most of the snapshot's bias, and complete months all of it.

    The article's rows include the month in which a customer is censored and
    count it as survived, which pulls the estimate about 1.5 points low. Dropping
    those partial months gives the true marginal rate.
    """
    base = simulate_customers(20_000, seed=11)
    # The features are not used to build rows, so the first one carries the customer index.
    customer = np.arange(base.x1.size, dtype=np.float64)
    rows = person_months(customer, base.x2, base.churn_time, base.followup)
    owner = rows.x1.astype(np.int64)
    complete = (rows.month <= base.followup[owner]) | (rows.churned == 1)
    complete_rows = PersonMonths(
        rows.x1[complete], rows.x2[complete], rows.month[complete], rows.churned[complete]
    )
    truth = float(np.mean(horizon_probability(base.scale)))
    snapshot = float(np.mean((base.churn_time <= base.followup) & (base.churn_time <= HORIZON)))
    article = life_table_probability(rows)

    assert snapshot < truth - 0.05
    assert snapshot < article < truth - 0.005
    assert life_table_probability(complete_rows) == pytest.approx(truth, abs=0.006)


def test_hazard_design_has_one_dummy_per_month() -> None:
    """Two feature columns, then a one-hot month."""
    design = hazard_design(np.array([0.5, -1.0]), np.array([2.0, 3.0]), np.array([1, 12]))

    assert design.shape == (2, 2 + HORIZON)
    assert design[:, :2].tolist() == [[0.5, 2.0], [-1.0, 3.0]]
    assert design[0, 2] == 1.0 and design[1, 2 + HORIZON - 1] == 1.0
    assert design[:, 2:].sum(axis=1).tolist() == [1.0, 1.0]


def test_training_rows_match_the_article() -> None:
    """115,084 customer-month rows from 14,952 customers, 5,014 with under a year."""
    assert SUMMARY.training_customers == 14_952
    assert SUMMARY.short_followup_training == 5_014
    assert SUMMARY.person_month_rows == 115_084


def test_cohort_table_matches_the_article() -> None:
    """Counts and true probabilities exactly; model means to one unit in the third decimal."""
    published = (
        (410, 0.501, 0.089, 0.505, 0.495),
        (415, 0.456, 0.219, 0.465, 0.450),
        (890, 0.476, 0.396, 0.478, 0.469),
        (1700, 0.484, 0.607, 0.488, 0.477),
        (1633, 0.490, 0.737, 0.494, 0.483),
    )
    assert tuple((row.lower, row.upper) for row in SUMMARY.cohorts) == COHORTS
    for row, (customers, true, naive, fixed, hazard) in zip(
        SUMMARY.cohorts, published, strict=True
    ):
        assert row.customers == customers
        assert round(row.true, 3) == true
        observed = (row.naive, row.fixed_horizon, row.hazard)
        assert observed == pytest.approx((naive, fixed, hazard), abs=0.0015)


def test_scores_match_the_article() -> None:
    """AUC against the true outcome and mean absolute probability error per model."""
    published = {
        "naive": (0.764, 0.193),
        "fixed horizon": (0.839, 0.051),
        "hazard": (0.849, 0.007),
    }
    for model, (auc, error) in published.items():
        assert SCORES[model].auc == pytest.approx(auc, abs=0.0015)
        assert SCORES[model].mean_absolute_error == pytest.approx(error, abs=0.0015)


def test_naive_model_looks_excellent_on_its_own_labels() -> None:
    """AUC 0.896 on the snapshot label, and 0.006 against 0.446 for a new average customer."""
    assert SUMMARY.naive_own_label_auc == pytest.approx(0.896, abs=0.0015)
    assert SUMMARY.naive_new_customer == pytest.approx(0.006, abs=0.0015)
    assert round(SUMMARY.true_new_customer, 3) == 0.446
    assert 50 < SUMMARY.true_new_customer / SUMMARY.naive_new_customer < 100


def test_test_set_churn_rates_match_the_article() -> None:
    """True 48.6 percent; 41.8 percent from the snapshot; 49.0 percent with full follow-up."""
    rates = SUMMARY.test_rates

    assert (round(rates.true, 3), round(rates.observed_all, 3)) == (0.486, 0.418)
    assert round(rates.observed_full_followup, 3) == 0.490


def test_figure_claims_hold() -> None:
    """The naive model rises with tenure from far below to far above the truth.

    The fixed-horizon and hazard models track the true probability in every cohort.
    """
    naive = [row.naive for row in SUMMARY.cohorts]
    first, last = SUMMARY.cohorts[0], SUMMARY.cohorts[-1]

    assert naive == sorted(naive)
    assert first.naive < first.true / 5
    assert last.naive > last.true + 0.2
    for row in SUMMARY.cohorts:
        assert abs(row.fixed_horizon - row.true) < 0.015
        assert abs(row.hazard - row.true) < 0.01
    errors = [SCORES[model].mean_absolute_error for model in ("hazard", "fixed horizon", "naive")]
    assert errors == sorted(errors)


def test_simulation_is_deterministic() -> None:
    """The same seed gives the same extract."""
    first, second = simulate_customers(50, seed=4), simulate_customers(50, seed=4)

    np.testing.assert_array_equal(first.churn_time, second.churn_time)
    np.testing.assert_array_equal(first.test, second.test)


def test_invalid_inputs_are_rejected() -> None:
    """Bad sizes, seeds, scales, horizons, months and mismatched vectors are refused."""
    with pytest.raises(ValueError):
        simulate_customers(1)
    with pytest.raises(TypeError):
        simulate_customers(100, seed=True)
    with pytest.raises(ValueError):
        horizon_probability(np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        horizon_probability(1.0, horizon=-1.0)
    with pytest.raises(ValueError):
        hazard_design(np.zeros(1), np.zeros(1), np.array([13]))
    with pytest.raises(ValueError):
        person_months(np.zeros(2), np.zeros(2), np.ones(3), np.ones(2))
    with pytest.raises(ValueError):
        person_months(np.zeros(1), np.zeros(1), np.array([-1.0]), np.ones(1))
    with pytest.raises(ValueError):
        person_months(np.zeros(1), np.zeros(1), np.ones(1), np.ones(1), horizon=0)
