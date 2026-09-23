"""Check the selection step directly and test the figure's claim about leakage.

The selection is compared with a ranking built from ``np.corrcoef`` and with
planted features the label fully determines. The full figure sweep takes about
16 s, so it is not run here. Its first point is the first thing the seeded
generator computes, so a sweep over the smallest pool alone reproduces it draw
for draw; the counts of correct predictions come from fitted logistic
regressions, so they are compared with a tolerance of two predictions in
4,000. The claim in the title and alt text (selection outside the folds climbs
above 80 percent as the pool grows, selection inside stays at chance) is
tested on a shorter sweep with 15 datasets per pool size. The article's table
uses 100 replications and other designs, so its numbers are not pinned.
"""

import numpy as np
import pytest

from blog_reproducibility.machine_learning.selection_leakage import (
    CANDIDATE_FEATURES,
    cross_validated_accuracy,
    leakage_sweep,
    noise_dataset,
    top_correlated_features,
)

SHORT_SWEEP = leakage_sweep(candidate_features=(20, 300, 10000), replications=15)


def test_first_figure_point_is_reproduced() -> None:
    """With 20 noise features the figure starts at 56.7 percent outside and 50.9 inside."""
    summary = leakage_sweep(candidate_features=CANDIDATE_FEATURES[:1])
    (row,) = summary.rows

    assert row.candidate_features == 20
    assert row.selected_outside == pytest.approx(2267 / 4000, abs=0.0005)
    assert row.selected_inside == pytest.approx(2036 / 4000, abs=0.0005)


def test_selection_outside_the_folds_manufactures_accuracy() -> None:
    """Outside-fold accuracy on noise rises with the pool and passes 80 percent."""
    outside = [row.selected_outside for row in SHORT_SWEEP.rows]

    assert outside == sorted(outside)
    assert outside[-1] > 0.8


def test_selection_inside_the_folds_stays_at_chance() -> None:
    """Inside-fold accuracy stays within a few points of a coin toss at every pool size."""
    for row in SHORT_SWEEP.rows:
        assert abs(row.selected_inside - 0.5) < 0.05
        assert row.selected_outside > row.selected_inside


def test_top_features_match_a_corrcoef_ranking() -> None:
    """The selected columns are the ones with the largest absolute Pearson correlation."""
    rng = np.random.default_rng(3)
    features = rng.normal(size=(60, 40))
    labels = rng.integers(0, 2, 60)
    correlation = np.abs(np.corrcoef(features.T, labels)[-1, :-1])

    chosen = top_correlated_features(features, labels, 5)

    assert set(chosen) == set(np.argsort(correlation)[-5:])
    assert list(correlation[chosen]) == sorted(correlation[chosen])


def test_planted_features_are_selected() -> None:
    """Columns that equal the label, or its negation, have correlation one and are kept."""
    rng = np.random.default_rng(4)
    labels = np.repeat([0, 1], 20)
    features = rng.normal(size=(40, 30))
    features[:, 7] = labels
    features[:, 19] = -3.0 * labels

    assert set(top_correlated_features(features, labels, 2)) == {7, 19}


def test_a_real_signal_is_found_inside_the_folds() -> None:
    """A column that separates the classes gives perfect accuracy with honest selection."""
    rng = np.random.default_rng(5)
    features, labels = noise_dataset(100, 50, rng)
    features[:, 0] += 10.0 * (2 * labels - 1)

    accuracy = cross_validated_accuracy(features, labels, 1, rng, select_inside=True)

    assert accuracy == 1.0


def test_noise_dataset_is_balanced() -> None:
    """The label has equal halves and the features have the requested shape."""
    features, labels = noise_dataset(100, 7, np.random.default_rng(0))

    assert features.shape == (100, 7)
    assert labels.sum() == 50


def test_cross_validation_draws_one_fold_seed() -> None:
    """Each run consumes exactly one integer, so the sweep's draw order is fixed."""
    features, labels = noise_dataset(40, 12, np.random.default_rng(1))
    rng = np.random.default_rng(2)
    mirror = np.random.default_rng(2)

    first = cross_validated_accuracy(features, labels, 3, rng, select_inside=False)
    mirror.integers(1_000_000_000)

    assert rng.integers(1_000_000_000) == mirror.integers(1_000_000_000)
    again = cross_validated_accuracy(
        features, labels, 3, np.random.default_rng(2), select_inside=False
    )
    assert first == again


def test_invalid_inputs_are_rejected() -> None:
    """Mismatched shapes, impossible selections, and odd sample sizes are refused."""
    features = np.zeros((10, 4))
    labels = np.repeat([0, 1], 5)
    with pytest.raises(ValueError):
        top_correlated_features(features, labels[:9], 2)
    with pytest.raises(ValueError):
        top_correlated_features(features, labels, 5)
    with pytest.raises(ValueError):
        top_correlated_features(features, labels, 0)
    with pytest.raises(ValueError):
        noise_dataset(11, 3, np.random.default_rng(0))
    with pytest.raises(TypeError):
        leakage_sweep(replications=True)
