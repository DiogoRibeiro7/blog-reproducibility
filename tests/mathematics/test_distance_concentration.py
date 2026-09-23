"""Check relative contrast directly and reproduce the article's two tables.

Contrast is checked on configurations whose distances are known, then the seeded
sweeps reproduce the article's contrast and classifier tables. Classifier
accuracies depend on scikit-learn's solvers, so they are compared to the
published three decimals with a tolerance of one unit in the last place.
"""

import numpy as np
import pytest

from blog_reproducibility.mathematics.distance_concentration import (
    contrast_sweep,
    example_payload,
    labelled_dataset,
    relative_contrast,
)

SUMMARY = example_payload()


def test_relative_contrast_on_known_distances() -> None:
    """Points at distance 1 and 3 from the query give contrast (3 - 1) / 1 = 2."""
    points = np.array([[1.0, 0.0], [0.0, 3.0], [0.0, -2.0]])
    queries = np.array([[0.0, 0.0]])

    assert relative_contrast(points, queries) == pytest.approx(2.0)


def test_relative_contrast_averages_over_queries() -> None:
    """The contrast is the mean of each query's own contrast."""
    points = np.array([[0.0], [1.0], [4.0]])
    queries = np.array([[0.5], [2.0]])
    first = (3.5 - 0.5) / 0.5
    second = (2.0 - 1.0) / 1.0

    assert relative_contrast(points, queries) == pytest.approx((first + second) / 2)


def test_contrast_table_matches_the_article() -> None:
    """Relative contrast falls from ten thousand on a line to 0.08 in 2,000 dimensions."""
    published = {
        1: (10_867, 10_578),
        2: (93, 102),
        5: (7.4, 7.7),
        10: (2.9, 3.2),
        20: (1.4, 1.5),
        50: (0.69, 0.76),
        100: (0.45, 0.50),
        500: (0.17, 0.20),
        2000: (0.08, 0.09),
    }
    for row in SUMMARY.contrast:
        uniform, gaussian = published[row.dimension]
        assert row.uniform == pytest.approx(uniform, rel=0.03, abs=0.005)
        assert row.gaussian == pytest.approx(gaussian, rel=0.03, abs=0.005)


def test_contrast_falls_with_dimension() -> None:
    """Every added dimension compresses the ranking further."""
    uniform = [row.uniform for row in SUMMARY.contrast]
    gaussian = [row.gaussian for row in SUMMARY.contrast]

    assert uniform == sorted(uniform, reverse=True)
    assert gaussian == sorted(gaussian, reverse=True)


def test_classifier_table_matches_the_article() -> None:
    """k-NN decays toward chance while projection and logistic regression hold up."""
    published = {
        0: (0.834, 0.834, 0.856),
        5: (0.820, 0.831, 0.851),
        20: (0.781, 0.823, 0.844),
        50: (0.744, 0.821, 0.843),
        100: (0.694, 0.811, 0.822),
        500: (0.610, 0.690, 0.749),
    }
    for row in SUMMARY.accuracy:
        expected = published[row.noise_dimensions]
        observed = (row.knn, row.knn_after_pca, row.logistic)
        assert observed == pytest.approx(expected, abs=0.0015)
        if row.noise_dimensions:
            assert row.knn < row.knn_after_pca < row.logistic


def test_labelled_dataset_shape_and_signal() -> None:
    """Two informative columns separate the classes; the rest are pure noise."""
    features, labels = labelled_dataset(4000, 3, np.random.default_rng(0))

    assert features.shape == (4000, 5)
    assert set(np.unique(labels)) == {0, 1}
    gap = features[labels == 1].mean(axis=0) - features[labels == 0].mean(axis=0)
    assert gap[:2] == pytest.approx([1.5, 1.5], abs=0.1)
    assert np.all(np.abs(gap[2:]) < 0.1)


def test_contrast_sweep_is_deterministic() -> None:
    """The same seed gives the same contrasts."""
    assert contrast_sweep((3,), seed=2) == contrast_sweep((3,), seed=2)


def test_invalid_inputs_are_rejected() -> None:
    """Mismatched shapes, too few points, and coincident queries are refused."""
    with pytest.raises(ValueError):
        relative_contrast(np.zeros((3, 2)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        relative_contrast(np.zeros((1, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError):
        relative_contrast(np.array([[0.0], [1.0]]), np.array([[1.0]]))
    with pytest.raises(ValueError):
        contrast_sweep((0,))
