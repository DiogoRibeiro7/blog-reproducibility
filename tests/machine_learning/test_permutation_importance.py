"""Check the importance methods on known models and reproduce the article's tables.

The batched permutation importance is checked against scikit-learn's own
``permutation_importance`` on a small forest, and every method is checked on a
linear model whose importances are known: a column the model ignores scores
exactly zero, and shuffling a column with weight ``w`` and unit variance raises
the error by ``2 w**2`` in expectation. Conditional permutation with a single bin
is ordinary permutation, and conditioning a column on an exact copy of itself
leaves it almost unchanged.

The seeded simulation then reproduces the article's permutation, training-set,
duplicate-removed, group, conditional and drop-column numbers. They come from
fitted forests, so they are compared at the printed precision with a tolerance
of one unit in the last place; the extrapolation numbers are pure NumPy and
match exactly. The bootstrap intervals, the five-seed range of the driver's
share and the twelve-sensor table are not pinned: the bootstrap needs fifty
further importance runs, the seed study's code is not shown, and the sensor
example is a different simulation.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.inspection import permutation_importance

from blog_reproducibility.machine_learning.permutation_importance import (
    FEATURES,
    conditional_importance,
    example_payload,
    extrapolation,
    fit_forest,
    group_importance,
    permutation_importances,
    sensor_split,
)

SUMMARY = example_payload()


@dataclass(frozen=True)
class Linear:
    """A fixed linear predictor, so the true importances are known."""

    weights: NDArray[np.float64]

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the weighted sum of each row."""
        return features @ self.weights


def _linear_data(
    rows: int, weights: tuple[float, ...]
) -> tuple[Linear, NDArray[np.float64], NDArray[np.float64]]:
    model = Linear(np.array(weights))
    features = np.random.default_rng(5).normal(size=(rows, len(weights)))
    return model, features, model.predict(features)


def test_batched_permutation_matches_scikit_learn() -> None:
    """Same shuffles, same random numbers, same importances as scikit-learn."""
    split = sensor_split(400, seed=3)
    forest = fit_forest(split.train_features, split.train_target, trees=10)
    reference = permutation_importance(
        forest,
        split.test_features,
        split.test_target,
        n_repeats=5,
        random_state=7,
        scoring="neg_mean_squared_error",
    ).importances_mean

    ours = permutation_importances(
        forest, split.test_features, split.test_target, repeats=5, seed=7
    )

    assert ours == pytest.approx(reference, rel=1e-12, abs=1e-12)


def test_ignored_column_scores_exactly_zero() -> None:
    """Shuffling a column the model never reads cannot change a prediction."""
    model, features, target = _linear_data(300, (1.5, 0.0))

    assert permutation_importances(model, features, target)[1] == 0.0
    assert group_importance(model, features, target, (1,)) == 0.0
    assert conditional_importance(model, features, target, 1, 0) == 0.0


def test_linear_importance_is_twice_the_squared_weight() -> None:
    """E[(w (x - x'))**2] = 2 w**2 for independent unit-variance x and x'."""
    model, features, target = _linear_data(20_000, (3.0, 1.0))

    importances = permutation_importances(model, features, target, repeats=5)
    together = group_importance(model, features, target, (0, 1), repeats=5)

    assert importances == pytest.approx((18.0, 2.0), rel=0.05)
    assert together == pytest.approx(20.0, rel=0.05)


def test_one_bin_conditional_is_group_permutation() -> None:
    """With a single bin, conditional permutation shuffles over all rows."""
    model, features, target = _linear_data(500, (1.0, -2.0, 0.5))

    conditional = conditional_importance(model, features, target, 1, 0, bins=1, seed=4)

    assert conditional == pytest.approx(group_importance(model, features, target, (1,), seed=4))


def test_conditioning_on_an_exact_copy_removes_the_importance() -> None:
    """Within narrow bins of its own copy, a column barely moves when shuffled."""
    model, features, target = _linear_data(4000, (2.0, 0.0))
    features[:, 1] = features[:, 0]

    marginal = group_importance(model, features, target, (0,))
    conditional = conditional_importance(model, features, target, 0, 1, bins=200)

    assert conditional < 0.01 * marginal


def test_permutation_table_matches_the_article() -> None:
    """The duplicate, which does nothing, outranks the weak feature that does."""
    assert SUMMARY.permutation == pytest.approx((3.794, 0.987, 0.451, 0.001), abs=0.0015)
    assert SUMMARY.permutation[1] > 2 * SUMMARY.permutation[2]


def test_removing_the_duplicate_restores_the_driver() -> None:
    """The driver's score rises to 8.142; the pair had summed to 4.8, losing over a third."""
    assert SUMMARY.without_duplicate[0] == pytest.approx(8.142, abs=0.0015)
    pair = SUMMARY.permutation[0] + SUMMARY.permutation[1]
    assert round(pair, 1) == 4.8
    assert pair < (2 / 3) * SUMMARY.without_duplicate[0]
    assert SUMMARY.permutation[0] < (2 / 3) * SUMMARY.without_duplicate[0]


def test_training_set_importance_matches_the_article() -> None:
    """Training-set importance credits memorised noise, a hundred times its test value."""
    assert SUMMARY.training_permutation == pytest.approx((3.904, 1.190, 0.743, 0.147), abs=0.0015)
    assert SUMMARY.train_mse == pytest.approx(0.57, abs=0.015)
    assert SUMMARY.test_mse == pytest.approx(1.07, abs=0.015)
    assert SUMMARY.training_permutation[3] > 100 * SUMMARY.permutation[3]
    assert SUMMARY.training_permutation[2] / SUMMARY.permutation[2] == pytest.approx(
        5 / 3, rel=0.02
    )


def test_shuffled_rows_leave_the_data() -> None:
    """Correlation 0.98 falls to 0.03, and 48 percent of rows now differ by over one unit."""
    shift = SUMMARY.extrapolation

    assert round(shift.real_correlation, 2) == 0.98
    assert round(shift.permuted_correlation, 2) == 0.03
    assert shift.real_far_share == 0.0
    assert round(shift.permuted_far_share, 2) == 0.48


def test_group_and_conditional_match_the_article() -> None:
    """The pair together scores 8.168; conditionally the driver adds 0.291, the copy 0.015."""
    assert SUMMARY.group_pair == pytest.approx(8.168, abs=0.0015)
    assert SUMMARY.group_pair == pytest.approx(SUMMARY.without_duplicate[0], rel=0.01)
    assert SUMMARY.conditional[:2] == pytest.approx((0.291, 0.015), abs=0.0015)


def test_drop_column_table_matches_the_article() -> None:
    """Refitting without the driver costs less than refitting without the weak feature."""
    assert SUMMARY.drop_column == pytest.approx((0.183, -0.001, 0.265, 0.004), abs=0.0015)
    assert SUMMARY.drop_column[0] < SUMMARY.drop_column[2]


def test_each_method_ranks_the_features_differently() -> None:
    """The figure's claim: permutation, conditional and drop-column give three orders."""
    orders = {
        method: tuple(np.argsort(scores)[::-1].tolist())
        for method, scores in (
            ("permutation", SUMMARY.permutation),
            ("conditional", SUMMARY.conditional),
            ("drop", SUMMARY.drop_column),
        )
    }

    assert orders["permutation"] == (0, 1, 2, 3)
    assert orders["conditional"] == (2, 0, 1, 3)
    assert orders["drop"] == (2, 0, 3, 1)
    assert SUMMARY.features == FEATURES


def test_simulation_is_deterministic() -> None:
    """The same seed gives the same data."""
    first, second = sensor_split(100, seed=2), sensor_split(100, seed=2)

    assert np.array_equal(first.test_features, second.test_features)
    assert np.array_equal(first.train_target, second.train_target)


def test_duplicate_correlation_follows_its_noise() -> None:
    """corr(x1, x1 + 0.2 e) = 1 / sqrt(1.04), and an exact copy never lies far away."""
    split = sensor_split(20_000)
    exact = extrapolation(sensor_split(200, duplicate_noise=0.0).test_features)

    assert extrapolation(split.test_features).real_correlation == pytest.approx(
        1 / np.sqrt(1.04), abs=0.005
    )
    assert exact.real_correlation == pytest.approx(1.0)
    assert exact.real_far_share == 0.0


def test_invalid_inputs_are_rejected() -> None:
    """Bad sizes, columns, bins and shapes are refused."""
    model, features, target = _linear_data(50, (1.0, 1.0))
    with pytest.raises(ValueError):
        sensor_split(2)
    with pytest.raises(ValueError):
        sensor_split(duplicate_noise=-0.1)
    with pytest.raises(TypeError):
        sensor_split(seed=True)
    with pytest.raises(ValueError):
        permutation_importances(model, features, target[:-1])
    with pytest.raises(ValueError):
        permutation_importances(model, features, target, repeats=0)
    with pytest.raises(ValueError):
        group_importance(model, features, target, ())
    with pytest.raises(ValueError):
        group_importance(model, features, target, (2,))
    with pytest.raises(ValueError):
        conditional_importance(model, features, target, 0, 1, bins=0)
    with pytest.raises(ValueError):
        extrapolation(features, far_gap=0.0)
