"""Multiple comparisons in drift monitoring, for the article on alerts that never stop.

A drift monitor compares each of 200 features with its reference window every
day, using a two-sample Kolmogorov-Smirnov test on 2,000 reference records and
500 new ones. At level ``alpha`` with ``m`` stable features, the chance of at
least one false alert a day is ``1 - (1 - alpha)^m`` and the expected count is
``alpha * m``.

Three decision rules are compared. Uncorrected testing flags ``p < alpha``.
Bonferroni flags ``p < alpha / m`` and controls the family-wise error rate.
Benjamini-Hochberg sorts the p-values, finds the largest ``k`` with
``p_(k) <= k q / m`` and flags those ``k`` features, controlling the false
discovery rate under independence or positive dependence. Benjamini-Yekutieli
is the same procedure with ``q`` divided by the harmonic number ``H_m``, and
holds under any dependence.

The power sweep follows the article's generator seeded at 7 draw for draw: for
each of six shift sizes, 30 days, and within each day, feature by feature, a
reference sample and then a batch, the first five features shifted. The burst
comparison follows the generator seeded at 21: ten-factor loadings first, then
for each of 60 days 200 independent feature pairs and a correlated reference
and batch in which the factors carry 80 percent of each feature's variance.

Drawing one day as a single ``(features, reference + batch)`` block consumes
the generator exactly as the article's per-feature calls do. The KS statistic
is computed for all features at once as an integer on the lattice
``1 / lcm(n1, n2)``, the same rounding SciPy applies before its exact p-value,
and each distinct lattice value is sent to ``scipy.stats.ks_2samp`` once. The
p-values are therefore SciPy's own, at a fraction of the cost of 60,000
separate calls.
"""

from dataclasses import dataclass
from math import lcm
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "ALPHA",
    "BATCH_SIZE",
    "BURST_DAYS",
    "BURST_SEED",
    "DAYS",
    "DRIFTING",
    "FACTORS",
    "FDR_LEVEL",
    "FEATURES",
    "POWER_SEED",
    "REFERENCE_SIZE",
    "NOISE_VARIANCE",
    "SHARED_VARIANCE",
    "SHIFTS",
    "AlertCounts",
    "BurstComparison",
    "MonitoringSummary",
    "PowerRow",
    "alert_bursts",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "example_payload",
    "false_alert_probability",
    "ks_lattice_statistics",
    "ks_pvalues",
    "persistence_false_alerts",
    "power_curves",
]

FEATURES: Final[int] = 200
REFERENCE_SIZE: Final[int] = 2000
BATCH_SIZE: Final[int] = 500
ALPHA: Final[float] = 0.05
FDR_LEVEL: Final[float] = 0.05
DRIFTING: Final[int] = 5
DAYS: Final[int] = 30
SHIFTS: Final[tuple[float, ...]] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
POWER_SEED: Final[int] = 7
BURST_SEED: Final[int] = 21
BURST_DAYS: Final[int] = 60
FACTORS: Final[int] = 10
SHARED_VARIANCE: Final[float] = 0.8
# Written out rather than as 1 - 0.8, which is 0.19999999999999996 and would move the draws.
NOISE_VARIANCE: Final[float] = 0.2
# The article's burst statistics: days above this count, and days at or below the quiet one.
BURST_THRESHOLD: Final[int] = 20
QUIET_THRESHOLD: Final[int] = 2
# The first table of the article, and the persistence rule's run length.
TABLE_FEATURE_COUNTS: Final[tuple[int, ...]] = (10, 40, 200)
PERSISTENCE_DAYS: Final[int] = 3

# Exact KS p-values depend only on the sample sizes and the lattice statistic.
_PVALUE_CACHE: dict[tuple[int, int, int], float] = {}


@dataclass(frozen=True, slots=True)
class PowerRow:
    """Share of drifting features flagged per day at one shift, by procedure."""

    shift: float
    uncorrected: float
    bonferroni: float
    benjamini_hochberg: float
    benjamini_yekutieli: float
    uncorrected_true_per_day: float
    uncorrected_false_per_day: float
    uncorrected_fdr: float


@dataclass(frozen=True, slots=True)
class AlertCounts:
    """Daily false-alert counts with no drift, and the article's summaries of them."""

    counts: tuple[int, ...]
    mean: float
    sd: float
    maximum: int
    days_over_threshold: int
    quiet_days: int


@dataclass(frozen=True, slots=True)
class BurstComparison:
    """False alerts per day for independent and for factor-correlated features."""

    independent: AlertCounts
    correlated: AlertCounts


@dataclass(frozen=True, slots=True)
class MonitoringSummary:
    """Every number the article reports from the two seeded figure simulations."""

    false_alert_probability: tuple[tuple[int, float], ...]
    expected_false_alerts_per_day: float
    yekutieli_divisor: float
    persistence_false_alerts_per_month: float
    power: tuple[PowerRow, ...]
    bursts: BurstComparison


def false_alert_probability(features: int, alpha: float = ALPHA) -> float:
    """Chance of at least one false alert among independent tests of stable features."""
    m = count(features, name="features")
    level = probability(alpha, name="alpha")
    return 1.0 - (1.0 - level) ** m


def persistence_false_alerts(
    days: int = DAYS,
    features: int = FEATURES,
    alpha: float = ALPHA,
    run: int = PERSISTENCE_DAYS,
) -> float:
    """Expected false alerts over ``days`` when a feature must fail ``run`` days in a row.

    Each day from the ``run``-th onwards, a stable feature fires when the last
    ``run`` independent tests all rejected, which has probability ``alpha^run``.
    """
    horizon = count(days, name="days")
    length = count(run, name="run", minimum=1)
    windows = max(horizon - length + 1, 0)
    level = probability(alpha, name="alpha")
    return windows * count(features, name="features") * level**length


def _as_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty (features, records) matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def ks_lattice_statistics(reference: ArrayLike, batch: ArrayLike) -> NDArray[np.int64]:
    """Two-sided KS statistics row by row, as integers in units of ``1 / lcm(n1, n2)``.

    Rows are features. With ``a = lcm / n1`` and ``b = lcm / n2``, the scaled
    difference ``a F1(x) - b F2(x)`` rises only at reference points and falls
    only at batch points. Its maximum is reached just before some sorted batch
    point ``y_i``, where it equals ``a #(reference < y_i) - b i``, and its
    minimum at some ``y_i``, where it equals ``a #(reference <= y_i) - b (i + 1)``.
    Two binary searches per feature therefore give the statistic exactly,
    ties included, as SciPy defines it with ``searchsorted(..., side="right")``.
    """
    first = _as_matrix(reference, name="reference")
    second = _as_matrix(batch, name="batch")
    if first.shape[0] != second.shape[0]:
        raise ValueError("reference and batch must have the same number of features")
    n1, n2 = first.shape[1], second.shape[1]
    step = lcm(n1, n2)
    a, b = step // n1, step // n2

    ordered_first = np.sort(first, axis=1)
    ordered_second = np.sort(second, axis=1)
    before = np.arange(n2) * b
    statistics = np.empty(first.shape[0], dtype=np.int64)
    for row, (sample, points) in enumerate(zip(ordered_first, ordered_second, strict=True)):
        below = np.searchsorted(sample, points, side="left") * a - before
        at_or_below = np.searchsorted(sample, points, side="right") * a - before - b
        statistics[row] = max(int(below.max()), -int(at_or_below.min()), 0)
    return statistics


def ks_pvalues(reference: ArrayLike, batch: ArrayLike) -> NDArray[np.float64]:
    """SciPy's two-sided ``ks_2samp`` p-value for each feature (row)."""
    first = _as_matrix(reference, name="reference")
    second = _as_matrix(batch, name="batch")
    lattice = ks_lattice_statistics(first, second)
    n1, n2 = first.shape[1], second.shape[1]

    values, rows = np.unique(lattice, return_index=True)
    for value, row in zip(values.tolist(), rows.tolist(), strict=True):
        key = (n1, n2, int(value))
        if key not in _PVALUE_CACHE:
            _PVALUE_CACHE[key] = float(stats.ks_2samp(first[row], second[row]).pvalue)
    return np.array([_PVALUE_CACHE[(n1, n2, int(value))] for value in lattice])


def _pvalue_vector(pvalues: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pvalues must be a non-empty vector")
    if np.any((p < 0) | (p > 1)) or not np.all(np.isfinite(p)):
        raise ValueError("pvalues must lie in [0, 1]")
    return p


def benjamini_hochberg(pvalues: ArrayLike, q: float = FDR_LEVEL) -> NDArray[np.bool_]:
    """Flag the ``k`` smallest p-values, ``k`` the largest with ``p_(k) <= k q / m``."""
    p = _pvalue_vector(pvalues)
    level = probability(q, name="q", inclusive=False)
    m = p.size
    order = np.argsort(p)
    below = np.flatnonzero(p[order] <= level * np.arange(1, m + 1) / m)
    reject = np.zeros(m, dtype=bool)
    if below.size:
        reject[order[: below.max() + 1]] = True
    return reject


def benjamini_yekutieli(pvalues: ArrayLike, q: float = FDR_LEVEL) -> NDArray[np.bool_]:
    """Benjamini-Hochberg at ``q / H_m``, valid under arbitrary dependence."""
    p = _pvalue_vector(pvalues)
    level = probability(q, name="q", inclusive=False)
    harmonic = float(np.sum(1.0 / np.arange(1, p.size + 1)))
    return benjamini_hochberg(p, level / harmonic)


def _independent_day(
    rng: np.random.Generator, shifts: NDArray[np.float64], reference_size: int, batch_size: int
) -> NDArray[np.float64]:
    """One day of per-feature KS p-values, drawn as the article's feature-by-feature loop."""
    block = rng.normal(size=(shifts.size, reference_size + batch_size))
    batch = block[:, reference_size:] + shifts[:, None]
    return ks_pvalues(block[:, :reference_size], batch)


def power_curves(
    seed: int = POWER_SEED,
    *,
    shifts: tuple[float, ...] = SHIFTS,
    days: int = DAYS,
    features: int = FEATURES,
    drifting: int = DRIFTING,
    reference_size: int = REFERENCE_SIZE,
    batch_size: int = BATCH_SIZE,
    alpha: float = ALPHA,
    q: float = FDR_LEVEL,
) -> tuple[PowerRow, ...]:
    """Average daily power of each procedure at each shift, from one generator."""
    rng = np.random.default_rng(count(seed, name="seed"))
    m = count(features, name="features", minimum=1)
    moved = count(drifting, name="drifting", minimum=1)
    if moved > m:
        raise ValueError("drifting cannot exceed features")
    horizon = count(days, name="days", minimum=1)
    n_ref = count(reference_size, name="reference_size", minimum=1)
    n_batch = count(batch_size, name="batch_size", minimum=1)
    level = probability(alpha, name="alpha", inclusive=False)
    fdr_level = probability(q, name="q", inclusive=False)

    rows = []
    for raw_shift in shifts:
        shift = real(raw_shift, name="shift")
        offsets = np.zeros(m)
        offsets[:moved] = shift
        power: dict[str, list[float]] = {key: [] for key in ("raw", "bonf", "bh", "by")}
        true_alerts, false_alerts, fdr = [], [], []
        for _ in range(horizon):
            p = _independent_day(rng, offsets, n_ref, n_batch)
            flags = {
                "raw": p < level,
                "bonf": p < level / m,
                "bh": benjamini_hochberg(p, fdr_level),
                "by": benjamini_yekutieli(p, fdr_level),
            }
            for key, flagged in flags.items():
                power[key].append(float(flagged[:moved].mean()))
            hits = int(flags["raw"][:moved].sum())
            misses = int(flags["raw"].sum()) - hits
            true_alerts.append(hits)
            false_alerts.append(misses)
            fdr.append(misses / (hits + misses) if hits + misses else 0.0)
        rows.append(
            PowerRow(
                shift=shift,
                uncorrected=float(np.mean(power["raw"])),
                bonferroni=float(np.mean(power["bonf"])),
                benjamini_hochberg=float(np.mean(power["bh"])),
                benjamini_yekutieli=float(np.mean(power["by"])),
                uncorrected_true_per_day=float(np.mean(true_alerts)),
                uncorrected_false_per_day=float(np.mean(false_alerts)),
                uncorrected_fdr=float(np.mean(fdr)),
            )
        )
    return tuple(rows)


def _summarise(counts: list[int]) -> AlertCounts:
    values = np.asarray(counts)
    return AlertCounts(
        counts=tuple(int(value) for value in values),
        mean=float(values.mean()),
        sd=float(values.std()),
        maximum=int(values.max()),
        days_over_threshold=int(np.sum(values > BURST_THRESHOLD)),
        quiet_days=int(np.sum(values <= QUIET_THRESHOLD)),
    )


def alert_bursts(
    seed: int = BURST_SEED,
    *,
    days: int = BURST_DAYS,
    features: int = FEATURES,
    factors: int = FACTORS,
    shared_variance: float = SHARED_VARIANCE,
    noise_variance: float = NOISE_VARIANCE,
    reference_size: int = REFERENCE_SIZE,
    batch_size: int = BATCH_SIZE,
    alpha: float = ALPHA,
) -> BurstComparison:
    """Daily false-alert counts with no drift, for independent and correlated features."""
    rng = np.random.default_rng(count(seed, name="seed"))
    m = count(features, name="features", minimum=1)
    k = count(factors, name="factors", minimum=1)
    shared = probability(shared_variance, name="shared_variance")
    horizon = count(days, name="days", minimum=1)
    n_ref = count(reference_size, name="reference_size", minimum=1)
    n_batch = count(batch_size, name="batch_size", minimum=1)
    level = probability(alpha, name="alpha", inclusive=False)
    noise_scale = np.sqrt(positive(noise_variance, name="noise_variance"))

    loadings = rng.normal(size=(m, k)) * np.sqrt(shared / k)

    def correlated_batch(records: int) -> NDArray[np.float64]:
        latent = rng.normal(size=(records, k))
        return np.asarray(latent @ loadings.T + rng.normal(scale=noise_scale, size=(records, m)))

    independent, correlated = [], []
    for _ in range(horizon):
        p_independent = _independent_day(rng, np.zeros(m), n_ref, n_batch)
        reference, batch = correlated_batch(n_ref), correlated_batch(n_batch)
        p_correlated = ks_pvalues(reference.T, batch.T)
        independent.append(int(np.sum(p_independent < level)))
        correlated.append(int(np.sum(p_correlated < level)))
    return BurstComparison(_summarise(independent), _summarise(correlated))


def example_payload() -> MonitoringSummary:
    """Return the closed-form numbers and both seeded simulations the article reports."""
    return MonitoringSummary(
        false_alert_probability=tuple(
            (m, false_alert_probability(m)) for m in TABLE_FEATURE_COUNTS
        ),
        expected_false_alerts_per_day=ALPHA * FEATURES,
        yekutieli_divisor=float(np.sum(1.0 / np.arange(1, FEATURES + 1))),
        persistence_false_alerts_per_month=persistence_false_alerts(),
        power=power_curves(),
        bursts=alert_bursts(),
    )
