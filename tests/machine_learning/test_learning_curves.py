"""Check the learning-curve model and reproduce what the article's simulation still gives.

The building blocks are checked independently: the true probabilities on
hand-worked inputs, the Bayes error on a small case and against the error of
the Bayes rule on simulated labels, the noisy-test floor against simulation, the
power-law fit on noiseless data, and the seeded training draws.

The seeded simulation is then compared with the article. The Bayes error, the
positive rate and everything about the logistic regression reproduce at the
printed precision (one unit of tolerance in the last digit for fitted values).
The gradient boosting curve does not fully reproduce: the article was produced
with scikit-learn 1.6.1, where this exact code gives the published table (checked
by running it there: boosting test error 0.266 at 500 examples, fitted floor
0.206 and exponent 0.60, predictions 0.218 and 0.211). Under the locked
scikit-learn, ``HistGradientBoostingClassifier`` with early stopping gives
0.261 at 500 examples, and the fit to the five smallest sizes has floor 0.216
and exponent 0.78, predicting 0.221 at 8,000 (actual 0.218) and 0.218 at 32,000
(actual 0.211). Only the boosting rows that still match (250 and 32,000
examples) are pinned; the figure's claim is tested as it holds here: the
extrapolations miss the held-out errors by less than a point, and the boosting
curve heads for a floor near the Bayes rate while the logistic one stalls far
above it.

Two sentences of the article's prose do not hold under either version and are
not tested as written: the boosting model is *better* than the logistic one at
250 examples (0.296 against 0.326), and the spread over the three draws at 250
examples, 0.025, is smaller than the gap between the models there, 0.030. The
"floor three times higher" of the figure's alt text matches no ratio of the
fitted floors or of their excess over the Bayes rate, so it is not tested.

The label-noise run is computed for the five smallest sizes only, which is all
its fit uses; its full eight-size run takes as long as the main simulation.
"""

import numpy as np
import pytest

from blog_reproducibility.machine_learning.learning_curves import (
    FITTED_SIZES,
    POOL_SIZE,
    TRAINING_SIZES,
    bayes_error,
    example_payload,
    fit_power_law,
    flip_labels,
    learning_curve,
    make_model,
    noisy_test_floor,
    power_law,
    simulate_task,
    training_indices,
    true_probability,
)

SUMMARY = example_payload(label_noise_sizes=TRAINING_SIZES[:FITTED_SIZES])
LOGISTIC = SUMMARY.curve("logistic")
BOOSTING = SUMMARY.curve("boosting")
# One unit in the third decimal, for values that come from a fitted estimator.
FITTED = 0.0015


def test_true_probability_on_hand_worked_inputs() -> None:
    """At the origin the log-odds is -1; at x0 = x1 = 1 it is 1.2 - 1.0 + 1.5 - 1 = 0.7."""
    features = np.zeros((2, 12))
    features[1, :2] = 1.0

    expected = 1 / (1 + np.exp([1.0, -0.7]))
    assert true_probability(features) == pytest.approx(expected)


def test_bayes_error_on_a_small_case() -> None:
    """The Bayes rule errs with probability min(p, 1 - p), symmetric in the two classes."""
    probabilities = np.array([0.1, 0.5, 0.9, 0.3])

    assert bayes_error(probabilities) == pytest.approx((0.1 + 0.5 + 0.1 + 0.3) / 4)
    assert bayes_error(1 - probabilities) == pytest.approx(bayes_error(probabilities))


def test_bayes_rule_attains_the_bayes_error_on_simulated_labels() -> None:
    """Predicting the more probable class errs at the Bayes rate, and flipped test
    labels raise that rate to e + (1 - 2 e) times it."""
    task = simulate_task()
    bayes_rule = (task.test_probabilities > 0.5).astype(np.int64)
    bayes = bayes_error(task.test_probabilities)
    noisy = flip_labels(task.test_labels, 0.1, seed=11)

    assert np.mean(bayes_rule != task.test_labels) == pytest.approx(bayes, abs=0.006)
    assert np.mean(bayes_rule != noisy) == pytest.approx(noisy_test_floor(bayes, 0.1), abs=0.006)


def test_flip_labels_flips_the_requested_share_and_is_an_involution() -> None:
    """About ten percent of labels change, and the same flips applied twice undo."""
    labels = np.random.default_rng(1).integers(0, 2, 50_000)
    flipped = flip_labels(labels, 0.1, seed=3)

    assert np.mean(flipped != labels) == pytest.approx(0.1, abs=0.005)
    assert np.array_equal(flip_labels(flipped, 0.1, seed=3), labels)


def test_power_law_fit_recovers_noiseless_parameters() -> None:
    """Exact power-law errors at five sizes give back their three parameters."""
    sizes = TRAINING_SIZES[:FITTED_SIZES]
    errors = tuple(power_law(n, 0.2, 3.0, 0.6) for n in sizes)
    fit = fit_power_law(sizes, errors)

    assert (fit.floor, fit.scale, fit.exponent) == pytest.approx((0.2, 3.0, 0.6), rel=1e-5)
    assert fit.predict(32_000) == pytest.approx(0.2 + 3.0 * 32_000**-0.6, rel=1e-6)


def test_training_indices_are_distinct_seeded_and_in_range() -> None:
    """Each training set is a seeded sample without replacement from the pool."""
    rows = training_indices(1000, 2)

    assert np.unique(rows).size == 1000
    assert rows.min() >= 0 and rows.max() < POOL_SIZE
    assert np.array_equal(rows, training_indices(1000, 2))
    assert not np.array_equal(rows, training_indices(1000, 1))


def test_bayes_error_and_positive_rate_match_the_article() -> None:
    """Bayes error rate = 0.205; positive rate 0.506; noisy-test floor 0.264."""
    assert round(SUMMARY.bayes_error, 3) == 0.205
    assert round(SUMMARY.positive_rate, 3) == 0.506
    assert round(SUMMARY.noisy_test_floor, 3) == 0.264


def test_logistic_curve_matches_the_article() -> None:
    """Logistic test and training errors at all eight sizes."""
    test = (0.326, 0.317, 0.304, 0.301, 0.300, 0.298, 0.298, 0.298)
    training = (0.297, 0.289, 0.306, 0.283, 0.297, 0.292, 0.296, 0.296)

    assert LOGISTIC.sizes == TRAINING_SIZES
    assert LOGISTIC.test_errors == pytest.approx(test, abs=FITTED)
    assert [point.training_error for point in LOGISTIC.points] == pytest.approx(
        training, abs=FITTED
    )


def test_logistic_fit_matches_the_article() -> None:
    """Floor 0.294, exponent 0.69, predictions 0.297 at 8,000 and 0.295 at 32,000."""
    fit = SUMMARY.fit("logistic")

    assert fit.floor == pytest.approx(0.294, abs=FITTED)
    assert fit.exponent == pytest.approx(0.69, abs=0.015)
    assert fit.predict(8000) == pytest.approx(0.297, abs=FITTED)
    assert fit.predict(32_000) == pytest.approx(0.295, abs=FITTED)


def test_boosting_rows_that_still_reproduce() -> None:
    """At 250 examples 0.296 test, 0.127 training, spread 0.025; at 32,000 0.211 and 0.182."""
    first, last = BOOSTING.points[0], BOOSTING.points[-1]

    assert (first.test_error, first.training_error) == pytest.approx((0.296, 0.127), abs=FITTED)
    assert first.test_error_sd == pytest.approx(0.025, abs=FITTED)
    assert (last.test_error, last.training_error) == pytest.approx((0.211, 0.182), abs=FITTED)


def test_boosting_keeps_learning_and_logistic_stalls() -> None:
    """Boosting falls at every size and ends within a point of the Bayes rate; the
    logistic regression is flat from 1,000 examples, almost ten points above it."""
    errors = BOOSTING.test_errors

    assert all(later < earlier for earlier, later in zip(errors, errors[1:], strict=False))
    assert 0 < errors[-1] - SUMMARY.bayes_error < 0.01
    assert all(b < a for a, b in zip(LOGISTIC.test_errors, errors, strict=True))
    assert max(LOGISTIC.test_errors[2:]) - min(LOGISTIC.test_errors[2:]) < 0.007
    assert LOGISTIC.test_errors[-1] - SUMMARY.bayes_error > 0.09


def test_train_test_gap_separates_variance_from_bias() -> None:
    """Boosting's gap at 250 examples is seventeen points; logistic's is three, then none."""
    boosting, logistic = BOOSTING.points[0], LOGISTIC.points[0]

    assert boosting.test_error - boosting.training_error == pytest.approx(0.17, abs=0.005)
    assert logistic.test_error - logistic.training_error == pytest.approx(0.03, abs=0.005)
    assert abs(LOGISTIC.points[2].test_error - LOGISTIC.points[2].training_error) < 0.005
    assert BOOSTING.points[2].test_error_sd < 0.005


def test_fit_to_small_sizes_predicts_the_held_out_sizes() -> None:
    """The figure's claim: fits to the five smallest sizes land within a point of the
    three held-out errors, the boosting floor near the Bayes rate and the logistic
    floor far above it."""
    for curve in (LOGISTIC, BOOSTING):
        fit = SUMMARY.fit(curve.model)
        for point in curve.points[FITTED_SIZES:]:
            assert abs(fit.predict(point.size) - point.test_error) < 0.008
    boosting, logistic = SUMMARY.fit("boosting"), SUMMARY.fit("logistic")

    assert 0 < boosting.floor - SUMMARY.bayes_error < 0.012
    assert logistic.floor - boosting.floor > 0.07


def test_label_noise_table_and_floor() -> None:
    """Flipped labels cost data, not floor: 0.336, 0.267 and 0.231 at 250, 1,000 and
    4,000 examples, about the clean error at half the size, with no higher a floor."""
    noisy = dict(zip(SUMMARY.noisy_curve.sizes, SUMMARY.noisy_curve.test_errors, strict=True))
    clean = dict(zip(BOOSTING.sizes, BOOSTING.test_errors, strict=True))

    assert (noisy[250], noisy[1000], noisy[4000]) == pytest.approx(
        (0.336, 0.267, 0.231), abs=FITTED
    )
    assert all(noisy[n] > clean[n] for n in noisy)
    assert all(abs(noisy[2 * n] - clean[n]) < 0.01 for n in (250, 500, 1000, 2000))
    assert SUMMARY.noisy_fit.floor < SUMMARY.fit("boosting").floor


def test_learning_curve_is_deterministic() -> None:
    """Seeded draws and a seeded model give the same curve twice."""
    task = simulate_task()

    assert learning_curve("boosting", task, sizes=(250,)) == learning_curve(
        "boosting", task, sizes=(250,)
    )


def test_invalid_inputs_are_rejected() -> None:
    """Unknown models, too few sizes, bad probabilities and mismatched labels are refused."""
    task = simulate_task()
    with pytest.raises(ValueError):
        make_model("forest")
    with pytest.raises(ValueError):
        fit_power_law((250, 500, 1000), (0.3, 0.29, 0.28))
    with pytest.raises(ValueError):
        bayes_error(np.array([0.2, 1.2]))
    with pytest.raises(ValueError):
        noisy_test_floor(0.2, 1.5)
    with pytest.raises(ValueError):
        training_indices(10, 0, pool_size=5)
    with pytest.raises(ValueError):
        learning_curve("logistic", task, sizes=(250,), pool_labels=task.test_labels)
    with pytest.raises(TypeError):
        training_indices(True, 0)
