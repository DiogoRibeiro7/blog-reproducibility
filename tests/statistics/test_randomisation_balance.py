"""Check the allocation distribution against explicit enumeration.

The website test compared the closed form with a count over every possible
assignment in a smaller population. That check is kept, and the exact
hypergeometric identities are added to it.
"""

from collections import Counter
from itertools import combinations
from math import isclose
from statistics import mean, pvariance

import pytest

from blog_reproducibility.statistics.randomisation_balance import (
    adjusted_sd,
    adjustment_curve,
    balance_probabilities,
    example_payload,
)

SUMMARY = example_payload()


def test_allocation_probabilities_against_explicit_enumeration() -> None:
    """Counting every assignment of a small population gives the same distribution."""
    size, high, treated = 8, 4, 4
    units = list(range(size))
    carriers = set(range(high))

    counts: Counter[int] = Counter()
    for allocation in combinations(units, treated):
        counts[len(carriers.intersection(allocation))] += 1
    total = sum(counts.values())

    rows = balance_probabilities(size, high, treated)
    assert {row.high_treated for row in rows} == set(counts)
    for row in rows:
        assert row.probability == pytest.approx(counts[row.high_treated] / total)


def test_the_distribution_is_a_distribution() -> None:
    """The probabilities sum to one and the differences are symmetric about zero."""
    rows = SUMMARY.allocations

    assert sum(row.probability for row in rows) == pytest.approx(1.0)
    assert all(row.probability > 0 for row in rows)

    differences = [row.difference for row in rows]
    assert differences == pytest.approx([-value for value in reversed(differences)])
    assert min(differences) == -1.0
    assert max(differences) == 1.0


def test_published_probabilities() -> None:
    """The draft quotes perfect balance in 34.4% of allocations."""
    assert SUMMARY.perfect_balance_probability == pytest.approx(0.34372, abs=5e-6)
    assert round(100 * SUMMARY.perfect_balance_probability, 1) == 34.4
    assert SUMMARY.imbalance_at_least_40_points == pytest.approx(0.17890, abs=5e-6)

    # A large imbalance is not rare: about one allocation in six.
    assert 0.16 < SUMMARY.imbalance_at_least_40_points < 0.19


def test_adjustment_helps_near_the_truth_and_not_beyond_it() -> None:
    """The population was built with a coefficient of four, and four is the best."""
    sds = SUMMARY.adjustment_sd

    assert sds["4"] < sds["2"] < sds["0"]
    # Twice the true coefficient is exactly as bad as not adjusting at all.
    assert sds["8"] == pytest.approx(sds["0"])
    assert sds["0"] == pytest.approx(1.12390, abs=5e-6)
    assert sds["4"] == pytest.approx(0.64889, abs=5e-6)


def test_adjusted_sd_against_the_residual_variance_written_out() -> None:
    """Recomputing the residuals by hand gives the same standard deviation."""
    covariates = tuple([0.0] * 10 + [1.0] * 10)
    noise = [-2.0, -1.0, 0.0, 1.0, 2.0] * 4
    outcomes = tuple(
        10 + 4 * covariate + error for covariate, error in zip(covariates, noise, strict=True)
    )

    for slope in (0.0, 1.5, 4.0, 8.0):
        residuals = [
            outcome - slope * covariate
            for outcome, covariate in zip(outcomes, covariates, strict=True)
        ]
        # variance() here is the sample variance, matching the model.
        spread = (sum((r - mean(residuals)) ** 2 for r in residuals) / (len(residuals) - 1)) ** 0.5
        expected = spread * (1 / 10 + 1 / 10) ** 0.5
        assert adjusted_sd(outcomes, covariates, 10, slope) == pytest.approx(expected)

    # pvariance is not what the model uses; this states the difference deliberately.
    assert pvariance(outcomes) < sum((value - mean(outcomes)) ** 2 for value in outcomes) / (
        len(outcomes) - 1
    )


def test_the_adjustment_curve_is_smooth_and_has_one_minimum() -> None:
    """The curve the figure draws falls to a single minimum and rises again."""
    curve = adjustment_curve()

    assert curve[0][0] == 0.0
    assert curve[-1][0] == pytest.approx(8.0)
    best = min(curve, key=lambda point: point[1])
    assert isclose(best[0], 4.0, abs_tol=0.11)


def test_invalid_populations_and_allocations() -> None:
    """Impossible sizes, arms, and mismatched lengths are rejected."""
    with pytest.raises(ValueError):
        balance_probabilities(size=10, high=11, treated=5)
    with pytest.raises(ValueError):
        balance_probabilities(size=10, high=5, treated=10)
    with pytest.raises(ValueError):
        balance_probabilities(size=10, high=5, treated=0)

    with pytest.raises(ValueError):
        adjusted_sd((1.0, 2.0), (1.0,), 1, 0.0)
    with pytest.raises(ValueError):
        adjusted_sd((1.0, 2.0), (1.0, 2.0), 2, 0.0)
    with pytest.raises(ValueError):
        adjusted_sd((1.0, 2.0), (1.0, 2.0), 1, float("nan"))
