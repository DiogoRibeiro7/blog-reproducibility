"""Distance concentration, for the article on nearest neighbours in high dimensions.

Relative contrast is the gap between a query's farthest and nearest neighbour
as a share of the nearest distance. Beyer and colleagues showed it tends to
zero as dimension grows for a wide class of distributions, so distance
rankings lose their meaning. The article measures it for uniform and Gaussian
clouds of 1,000 points with 50 queries, then shows what irrelevant dimensions
do to a 5-nearest-neighbour classifier, against the same classifier after a
two-component PCA and against logistic regression.

Both simulations follow the article's generators draw for draw: the contrast
sweep uses one generator seeded at 0, and each classifier replication uses a
generator seeded at ``10 * noise_dims + replication``. Distances are computed
one query at a time rather than as a full query-by-point-by-dimension array,
which would need 800 MB at 2,000 dimensions.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from blog_reproducibility.common.validation import count

__all__ = [
    "DIMENSIONS",
    "NOISE_DIMENSIONS",
    "AccuracyRow",
    "ContrastRow",
    "DistanceSummary",
    "classifier_accuracy",
    "contrast_sweep",
    "example_payload",
    "labelled_dataset",
    "relative_contrast",
]

DIMENSIONS: Final[tuple[int, ...]] = (1, 2, 5, 10, 20, 50, 100, 500, 2000)
NOISE_DIMENSIONS: Final[tuple[int, ...]] = (0, 5, 20, 50, 100, 500)
POINTS: Final[int] = 1000
QUERIES: Final[int] = 50
TRAIN_SIZE: Final[int] = 1000
TEST_SIZE: Final[int] = 2000
REPLICATIONS: Final[int] = 5
NEIGHBOURS: Final[int] = 5
CLASS_SHIFT: Final[float] = 1.5


@dataclass(frozen=True, slots=True)
class ContrastRow:
    """Mean relative contrast at one dimension."""

    dimension: int
    uniform: float
    gaussian: float


@dataclass(frozen=True, slots=True)
class AccuracyRow:
    """Mean test accuracy over replications at one number of noise dimensions."""

    noise_dimensions: int
    knn: float
    knn_after_pca: float
    logistic: float


@dataclass(frozen=True, slots=True)
class DistanceSummary:
    """The article's contrast table and classifier table."""

    contrast: tuple[ContrastRow, ...]
    accuracy: tuple[AccuracyRow, ...]


def relative_contrast(points: NDArray[np.float64], queries: NDArray[np.float64]) -> float:
    """Mean over queries of (farthest - nearest) / nearest Euclidean distance."""
    if points.ndim != 2 or queries.ndim != 2 or points.shape[1] != queries.shape[1]:
        raise ValueError("points and queries must be matrices with the same number of columns")
    if points.shape[0] < 2 or queries.shape[0] == 0:
        raise ValueError("relative contrast needs at least two points and one query")
    contrasts = []
    for query in queries:
        distances = np.sqrt(((points - query) ** 2).sum(axis=1))
        nearest = distances.min()
        if nearest == 0:
            raise ValueError("a query coincides with a point, so the contrast is undefined")
        contrasts.append((distances.max() - nearest) / nearest)
    return float(np.mean(contrasts))


def contrast_sweep(
    dimensions: tuple[int, ...] = DIMENSIONS, *, seed: int = 0
) -> tuple[ContrastRow, ...]:
    """Relative contrast of uniform and Gaussian clouds at each dimension."""
    rng = np.random.default_rng(count(seed, name="seed"))
    rows = []
    for dimension in dimensions:
        width = count(dimension, name="dimension", minimum=1)
        # Points before queries, uniform before Gaussian: the article's draw order.
        samplers: tuple[Callable[..., NDArray[np.float64]], ...] = (rng.uniform, rng.normal)
        uniform, gaussian = (
            relative_contrast(sample(size=(POINTS, width)), sample(size=(QUERIES, width)))
            for sample in samplers
        )
        rows.append(ContrastRow(dimension, uniform, gaussian))
    return tuple(rows)


def labelled_dataset(
    size: int, noise_dimensions: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Two classes shifted apart in two dimensions, padded with pure-noise columns."""
    labels = rng.integers(0, 2, count(size, name="size", minimum=1))
    signal = rng.normal(size=(labels.size, 2)) + CLASS_SHIFT * labels[:, None]
    noise = rng.normal(size=(labels.size, count(noise_dimensions, name="noise_dimensions")))
    return np.column_stack([signal, noise]), labels


def classifier_accuracy(noise_dimensions: int, *, replications: int = REPLICATIONS) -> AccuracyRow:
    """Mean test accuracy of the three classifiers over seeded replications."""
    scores = []
    for replication in range(count(replications, name="replications", minimum=1)):
        rng = np.random.default_rng(10 * noise_dimensions + replication)
        train, train_labels = labelled_dataset(TRAIN_SIZE, noise_dimensions, rng)
        test, test_labels = labelled_dataset(TEST_SIZE, noise_dimensions, rng)

        knn = KNeighborsClassifier(NEIGHBOURS).fit(train, train_labels)
        pca = PCA(2, svd_solver="full").fit(train)
        knn_pca = KNeighborsClassifier(NEIGHBOURS).fit(pca.transform(train), train_labels)
        logistic = LogisticRegression(max_iter=2000).fit(train, train_labels)
        scores.append(
            (
                knn.score(test, test_labels),
                knn_pca.score(pca.transform(test), test_labels),
                logistic.score(test, test_labels),
            )
        )
    knn_mean, pca_mean, logistic_mean = np.mean(scores, axis=0)
    return AccuracyRow(noise_dimensions, float(knn_mean), float(pca_mean), float(logistic_mean))


def example_payload() -> DistanceSummary:
    """Return the article's two tables."""
    return DistanceSummary(
        contrast=contrast_sweep(),
        accuracy=tuple(classifier_accuracy(noise) for noise in NOISE_DIMENSIONS),
    )
