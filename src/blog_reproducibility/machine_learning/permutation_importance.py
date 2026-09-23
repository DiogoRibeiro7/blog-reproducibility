"""Four feature-importance methods on correlated inputs, for the permutation-importance article.

Permutation importance shuffles one column of the evaluation set and reports how
much the fitted model's error rises. With correlated inputs it measures what
this model relies on, not what drives the outcome: two near-duplicate columns
split the credit, part of it disappears because the unshuffled partner covers
for the shuffled one, and the shuffled rows combine values that never occur
together, so the model is scored off the data it was trained on.

The article's simulation has 4,000 rows and four features: a driver ``x1``, a
near-copy ``x2 = x1 + 0.2 e`` with no effect of its own, a weak independent
feature ``x3`` and pure noise ``x4``, with ``y = 2 x1 + 0.5 x3 + e``. A random
forest (300 trees, leaves of at least five, two candidate features per split) is
trained on half and evaluated on the other half, and four methods rank the
features:

- permutation importance, each column shuffled on its own;
- group permutation, the correlated pair shuffled together with one permutation;
- conditional permutation, a column shuffled only within twenty quantile bins of
  its partner, which keeps the shuffled rows on the joint distribution;
- drop-column importance, the forest refitted without the column.

The data, split and forests use the article's seeds (a generator seeded at 0,
``random_state=0`` for the split and every forest). Group permutation draws from
its own generator seeded at 0, and each conditional importance starts a fresh
generator seeded at 0, as the article's code does, so every number is reproduced
draw for draw.

scikit-learn's ``permutation_importance`` predicts once per shuffle, and a
300-tree forest spends most of each prediction in per-tree overhead, so the
eighty shuffles cost about ten seconds. :func:`permutation_importances` makes the
same shuffles with the same random numbers (one ``RandomState`` seeded from the
caller's seed, each column's permutations compounding) and scores them in one
batched prediction. Its equivalence with scikit-learn is checked in the tests.
"""

from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from blog_reproducibility.common.validation import count, non_negative, positive, real

__all__ = [
    "BINS",
    "FEATURES",
    "REPEATS",
    "SAMPLES",
    "SEED",
    "Extrapolation",
    "ImportanceSummary",
    "Regressor",
    "SensorSplit",
    "conditional_importance",
    "drop_column_importances",
    "example_payload",
    "extrapolation",
    "fit_forest",
    "group_importance",
    "permutation_importances",
    "sensor_split",
]

SEED: Final[int] = 0
SAMPLES: Final[int] = 4000
DUPLICATE_NOISE: Final[float] = 0.2
DRIVER_EFFECT: Final[float] = 2.0
WEAK_EFFECT: Final[float] = 0.5
TEST_SHARE: Final[float] = 0.5
TREES: Final[int] = 300
MIN_LEAF: Final[int] = 5
CANDIDATES_PER_SPLIT: Final[int] = 2
REPEATS: Final[int] = 20
BINS: Final[int] = 20
FAR_GAP: Final[float] = 1.0
FEATURES: Final[tuple[str, ...]] = ("x1 driver", "x2 duplicate", "x3 weak", "x4 noise")
DRIVER: Final[int] = 0
DUPLICATE: Final[int] = 1


class Regressor(Protocol):
    """Anything with a ``predict`` method, such as a fitted scikit-learn regressor."""

    def predict(self, features: NDArray[np.float64], /) -> Any:
        """Return one prediction per row."""


@dataclass(frozen=True, slots=True)
class SensorSplit:
    """The article's simulated sensors, split in half for training and evaluation."""

    train_features: NDArray[np.float64]
    train_target: NDArray[np.float64]
    test_features: NDArray[np.float64]
    test_target: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Extrapolation:
    """How far the rows produced by shuffling the duplicate lie from the real ones."""

    real_correlation: float
    permuted_correlation: float
    real_far_share: float
    permuted_far_share: float


@dataclass(frozen=True, slots=True)
class ImportanceSummary:
    """The article's importance tables and the figure's four panels.

    Every importance is an increase in mean squared error. ``without_duplicate``
    covers ``x1``, ``x3`` and ``x4`` after refitting without ``x2``, and
    ``conditional`` conditions ``x1`` on ``x2`` and every other feature on ``x1``.
    """

    features: tuple[str, ...]
    train_mse: float
    test_mse: float
    permutation: tuple[float, ...]
    training_permutation: tuple[float, ...]
    without_duplicate: tuple[float, ...]
    group_pair: float
    conditional: tuple[float, ...]
    drop_column: tuple[float, ...]
    extrapolation: Extrapolation


def sensor_split(
    samples: int = SAMPLES, *, duplicate_noise: float = DUPLICATE_NOISE, seed: int = SEED
) -> SensorSplit:
    """Simulate the driver, its duplicate, the weak feature and noise, and split them."""
    size = count(samples, name="samples", minimum=4)
    noise = non_negative(duplicate_noise, name="duplicate_noise")
    rng = np.random.default_rng(count(seed, name="seed"))
    driver = rng.normal(size=size)
    duplicate = driver + rng.normal(scale=noise, size=size)
    weak = rng.normal(size=size)
    irrelevant = rng.normal(size=size)
    target = DRIVER_EFFECT * driver + WEAK_EFFECT * weak + rng.normal(size=size)
    features = np.column_stack([driver, duplicate, weak, irrelevant])
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=TEST_SHARE, random_state=0
    )
    return SensorSplit(train_x, train_y, test_x, test_y)


def fit_forest(
    features: NDArray[np.float64], target: NDArray[np.float64], *, trees: int = TREES
) -> Any:
    """Fit the article's random forest, seeded at 0."""
    return RandomForestRegressor(
        n_estimators=count(trees, name="trees", minimum=1),
        min_samples_leaf=MIN_LEAF,
        max_features=CANDIDATES_PER_SPLIT,
        random_state=0,
        n_jobs=-1,
    ).fit(features, target)


def _mse(target: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
    return float(mean_squared_error(target, predictions))


def _loss_increases(
    model: Regressor,
    target: NDArray[np.float64],
    baseline: float,
    shuffled: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Increase in MSE for each shuffled copy, predicted in one batch."""
    predictions = np.asarray(model.predict(np.concatenate(shuffled)), dtype=np.float64)
    blocks = predictions.reshape(len(shuffled), target.size)
    return np.array([_mse(target, block) for block in blocks]) - baseline


def _check_data(features: NDArray[np.float64], target: NDArray[np.float64]) -> None:
    if features.ndim != 2 or target.ndim != 1 or features.shape[0] != target.size:
        raise ValueError("features must be a matrix with one row per target value")
    if target.size < 2:
        raise ValueError("importance needs at least two rows")


def _column(value: int, features: NDArray[np.float64], *, name: str) -> int:
    index = count(value, name=name)
    if index >= features.shape[1]:
        raise ValueError(f"{name} must index a column of the features")
    return index


def permutation_importances(
    model: Regressor,
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    repeats: int = REPEATS,
    seed: int = SEED,
) -> tuple[float, ...]:
    """Mean increase in MSE when each column is shuffled on its own.

    Reproduces ``sklearn.inspection.permutation_importance`` with
    ``scoring="neg_mean_squared_error"`` shuffle for shuffle: one seed is drawn
    from ``RandomState(seed)``, every column restarts a ``RandomState`` from it,
    and each repeat shuffles the index array again and applies it to the
    already-shuffled column.
    """
    _check_data(features, target)
    rounds = count(repeats, name="repeats", minimum=1)
    column_seed = np.random.RandomState(count(seed, name="seed")).randint(
        np.iinfo(np.int32).max + 1
    )
    baseline = _mse(target, np.asarray(model.predict(features), dtype=np.float64))
    importances = []
    for column in range(features.shape[1]):
        shuffler = np.random.RandomState(column_seed)
        order = np.arange(target.size)
        shuffled = features.copy()
        copies = []
        for _ in range(rounds):
            shuffler.shuffle(order)
            shuffled[:, column] = shuffled[order, column]
            copies.append(shuffled.copy())
        importances.append(float(np.mean(_loss_increases(model, target, baseline, copies))))
    return tuple(importances)


def group_importance(
    model: Regressor,
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    columns: tuple[int, ...],
    *,
    repeats: int = REPEATS,
    seed: int = SEED,
) -> float:
    """Mean increase in MSE when ``columns`` are shuffled together by one permutation."""
    _check_data(features, target)
    if not columns:
        raise ValueError("a group needs at least one column")
    group = [_column(column, features, name="column") for column in columns]
    rounds = count(repeats, name="repeats", minimum=1)
    rng = np.random.default_rng(count(seed, name="seed"))
    baseline = _mse(target, np.asarray(model.predict(features), dtype=np.float64))
    copies = []
    for _ in range(rounds):
        shuffled = features.copy()
        shuffled[:, group] = features[rng.permutation(target.size)][:, group]
        copies.append(shuffled)
    return float(np.mean(_loss_increases(model, target, baseline, copies)))


def conditional_importance(
    model: Regressor,
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    column: int,
    condition: int,
    *,
    bins: int = BINS,
    repeats: int = REPEATS,
    seed: int = SEED,
) -> float:
    """Mean increase in MSE when ``column`` is shuffled within quantile bins of ``condition``."""
    _check_data(features, target)
    shuffled_column = _column(column, features, name="column")
    partner = _column(condition, features, name="condition")
    groups = count(bins, name="bins", minimum=1)
    rounds = count(repeats, name="repeats", minimum=1)
    rng = np.random.default_rng(count(seed, name="seed"))
    baseline = _mse(target, np.asarray(model.predict(features), dtype=np.float64))
    edges = np.quantile(features[:, partner], np.linspace(0, 1, groups + 1))
    labels = np.clip(np.searchsorted(edges, features[:, partner], side="right") - 1, 0, groups - 1)
    members = [np.where(labels == label)[0] for label in range(groups)]
    copies = []
    for _ in range(rounds):
        shuffled = features.copy()
        for rows in members:
            shuffled[rows, shuffled_column] = features[rng.permutation(rows), shuffled_column]
        copies.append(shuffled)
    return float(np.mean(_loss_increases(model, target, baseline, copies)))


def drop_column_importances(split: SensorSplit, baseline: float) -> tuple[float, ...]:
    """Increase in test MSE over ``baseline`` after refitting without each column."""
    reference = real(baseline, name="baseline")
    width = split.train_features.shape[1]
    increases = []
    for dropped in range(width):
        kept = [column for column in range(width) if column != dropped]
        model = fit_forest(split.train_features[:, kept], split.train_target)
        predictions = np.asarray(model.predict(split.test_features[:, kept]), dtype=np.float64)
        increases.append(_mse(split.test_target, predictions) - reference)
    return tuple(increases)


def extrapolation(
    features: NDArray[np.float64],
    *,
    column: int = DUPLICATE,
    partner: int = DRIVER,
    far_gap: float = FAR_GAP,
    seed: int = SEED,
) -> Extrapolation:
    """Correlation and share of far-apart rows, before and after shuffling ``column``."""
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must be a matrix with at least two rows")
    shuffled_column = _column(column, features, name="column")
    other = _column(partner, features, name="partner")
    gap = positive(far_gap, name="far_gap")
    rng = np.random.default_rng(count(seed, name="seed"))
    shuffled = features[rng.permutation(features.shape[0]), shuffled_column]
    real_values = features[:, shuffled_column]
    anchor = features[:, other]
    return Extrapolation(
        real_correlation=float(np.corrcoef(anchor, real_values)[0, 1]),
        permuted_correlation=float(np.corrcoef(anchor, shuffled)[0, 1]),
        real_far_share=float(np.mean(np.abs(real_values - anchor) > gap)),
        permuted_far_share=float(np.mean(np.abs(shuffled - anchor) > gap)),
    )


def example_payload() -> ImportanceSummary:
    """Return the article's importance tables and the figure's four panels."""
    split = sensor_split()
    forest = fit_forest(split.train_features, split.train_target)
    test_x, test_y = split.test_features, split.test_target
    test_mse = _mse(test_y, np.asarray(forest.predict(test_x), dtype=np.float64))
    train_mse = _mse(
        split.train_target, np.asarray(forest.predict(split.train_features), dtype=np.float64)
    )

    kept = [DRIVER, 2, 3]
    without = fit_forest(split.train_features[:, kept], split.train_target)
    return ImportanceSummary(
        features=FEATURES,
        train_mse=train_mse,
        test_mse=test_mse,
        permutation=permutation_importances(forest, test_x, test_y),
        training_permutation=permutation_importances(
            forest, split.train_features, split.train_target
        ),
        without_duplicate=permutation_importances(without, test_x[:, kept], test_y),
        group_pair=group_importance(forest, test_x, test_y, (DRIVER, DUPLICATE)),
        conditional=(
            conditional_importance(forest, test_x, test_y, DRIVER, DUPLICATE),
            conditional_importance(forest, test_x, test_y, DUPLICATE, DRIVER),
            conditional_importance(forest, test_x, test_y, 2, DRIVER),
            conditional_importance(forest, test_x, test_y, 3, DRIVER),
        ),
        drop_column=drop_column_importances(split, test_mse),
        extrapolation=extrapolation(test_x),
    )
