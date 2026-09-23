"""Feature selection before cross-validation, for the article on leakage.

Cross-validation estimates how a procedure performs on data it has not seen,
and the estimate is honest only if every step that shapes the predictions for
a held-out fold was fitted without that fold's labels. Selecting the features
most correlated with the label is such a step. Done once on all the data
before the folds are formed, it lets the held-out labels choose the features
used to predict them, and among many noise features there are always a few
that correlate with the labels by chance.

The figure makes the point on pure noise: 100 samples with a balanced binary
label assigned at random, and a pool of independent standard normal features
whose size runs from 20 to 10,000. The ten features with the largest absolute
Pearson correlation with the label are kept, and a logistic regression is
scored by five-fold stratified cross-validation, once with the selection done
on all the data and once with it redone inside each training fold. Each point
is the mean over 40 datasets.

The whole sweep uses one generator seeded at 0, in the article's draw order:
for each dataset the features, then the label shuffle, then a fold seed for
the outside-selection run, then a fold seed for the inside-selection run. The
article's own table uses 100 replications and different designs, so its
numbers are not reproduced here; the figure's claim is.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from blog_reproducibility.common.validation import count

__all__ = [
    "CANDIDATE_FEATURES",
    "FOLDS",
    "KEPT_FEATURES",
    "REPLICATIONS",
    "SAMPLES",
    "SEED",
    "LeakageRow",
    "SelectionLeakageSummary",
    "cross_validated_accuracy",
    "example_payload",
    "leakage_sweep",
    "noise_dataset",
    "top_correlated_features",
]

SEED: Final[int] = 0
CANDIDATE_FEATURES: Final[tuple[int, ...]] = (20, 50, 100, 300, 1000, 3000, 10000)
SAMPLES: Final[int] = 100
KEPT_FEATURES: Final[int] = 10
FOLDS: Final[int] = 5
REPLICATIONS: Final[int] = 40
FOLD_SEED_BOUND: Final[int] = 1_000_000_000
LOGISTIC_MAX_ITER: Final[int] = 1000


@dataclass(frozen=True, slots=True)
class LeakageRow:
    """Mean cross-validated accuracy on noise at one size of the feature pool."""

    candidate_features: int
    selected_outside: float
    selected_inside: float


@dataclass(frozen=True, slots=True)
class SelectionLeakageSummary:
    """The figure's two curves, one row per size of the feature pool."""

    samples: int
    kept_features: int
    replications: int
    rows: tuple[LeakageRow, ...]


def top_correlated_features(
    features: NDArray[np.float64], labels: NDArray[np.int64], kept: int
) -> NDArray[np.intp]:
    """Indices of the ``kept`` columns with the largest absolute correlation with the label.

    The indices come in increasing order of correlation, as ``np.argsort``
    returns them; a tiny constant in the denominator guards constant columns.
    """
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.size:
        raise ValueError("features must be a matrix with one row per label")
    k = count(kept, name="kept", minimum=1)
    if k > features.shape[1]:
        raise ValueError("cannot keep more features than there are columns")
    centred = labels - labels.mean()
    correlation = np.abs((features - features.mean(0)).T @ centred) / (
        features.std(0) * centred.std() * labels.size + 1e-12
    )
    return np.argsort(correlation)[-k:]


def noise_dataset(
    samples: int, candidate_features: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Standard normal features and a balanced label shuffled independently of them."""
    n = count(samples, name="samples", minimum=2)
    if n % 2:
        raise ValueError("samples must be even so the label can be balanced")
    features = rng.normal(
        0, 1, (n, count(candidate_features, name="candidate_features", minimum=1))
    )
    labels = np.repeat([0, 1], n // 2)
    rng.shuffle(labels)
    return features, labels


def cross_validated_accuracy(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    kept: int,
    rng: np.random.Generator,
    *,
    select_inside: bool,
    folds: int = FOLDS,
) -> float:
    """Stratified k-fold accuracy of a logistic regression on the selected features.

    With ``select_inside`` false the features are chosen once on every label,
    held-out ones included; with it true they are rechosen on each training
    fold. One integer is drawn from ``rng`` to seed the fold assignment.
    """
    splitter = StratifiedKFold(
        count(folds, name="folds", minimum=2),
        shuffle=True,
        random_state=int(rng.integers(FOLD_SEED_BOUND)),
    )
    columns = None if select_inside else top_correlated_features(features, labels, kept)
    correct = 0
    for train, test in splitter.split(features, labels):
        chosen = (
            top_correlated_features(features[train], labels[train], kept)
            if columns is None
            else columns
        )
        model = LogisticRegression(max_iter=LOGISTIC_MAX_ITER).fit(
            features[train][:, chosen], labels[train]
        )
        correct += int((model.predict(features[test][:, chosen]) == labels[test]).sum())
    return correct / labels.size


def leakage_sweep(
    seed: int = SEED,
    *,
    candidate_features: tuple[int, ...] = CANDIDATE_FEATURES,
    samples: int = SAMPLES,
    kept: int = KEPT_FEATURES,
    replications: int = REPLICATIONS,
) -> SelectionLeakageSummary:
    """Mean accuracy with selection outside and inside the folds at each pool size."""
    rng = np.random.default_rng(count(seed, name="seed"))
    reps = count(replications, name="replications", minimum=1)
    rows = []
    for pool in candidate_features:
        outside, inside = [], []
        for _ in range(reps):
            features, labels = noise_dataset(samples, pool, rng)
            outside.append(
                cross_validated_accuracy(features, labels, kept, rng, select_inside=False)
            )
            inside.append(cross_validated_accuracy(features, labels, kept, rng, select_inside=True))
        rows.append(LeakageRow(pool, float(np.mean(outside)), float(np.mean(inside))))
    return SelectionLeakageSummary(
        samples=samples, kept_features=kept, replications=reps, rows=tuple(rows)
    )


def example_payload() -> SelectionLeakageSummary:
    """Return the figure's two curves."""
    return leakage_sweep()
