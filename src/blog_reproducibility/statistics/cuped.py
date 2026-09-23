"""CUPED and stratification against the plain difference in means, for the CUPED article.

If a pre-experiment covariate ``X`` has correlation ``rho`` with the outcome
``Y``, the residual ``Y - theta (X - mean X)`` with the least-squares ``theta``
has variance ``(1 - rho^2) Var(Y)``. Randomisation balances ``X`` across arms,
so comparing residuals estimates the same effect with a standard error smaller
by ``sqrt(1 - rho^2)``. That is CUPED. Stratifying on covariate quartiles and
averaging the within-stratum differences recovers most, but not all, of the
gain, because four levels keep only part of the linear information.

The figure simulates two arms of 2,000 users whose covariate and outcome have
mean 50 and standard deviation 10, at ten correlations from 0 to 0.9, and
takes the spread of each estimator over 1,200 A/A experiments per
correlation, all from one generator seeded at 0. It is reproduced draw for
draw: the experiments are generated in blocks, which consumes the generator
exactly as the figure's one-at-a-time loop does, and the estimators are
vectorised across a block.

The article's tables come from a different design (correlations 0, 0.3, 0.5,
0.7 and 0.85, 4,000 replications, a generator shared across its code blocks)
and are not reproduced. Its sample-size table is closed form, ``n = 2 sigma^2
(1 - rho^2) (z_0.975 + z_0.8)^2 / delta^2`` per arm for an effect of one unit,
and is reproduced exactly.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "CORRELATIONS",
    "EFFECT",
    "OUTCOME_MEAN",
    "OUTCOME_SD",
    "REPLICATIONS",
    "SAMPLE_SIZE_CORRELATIONS",
    "SEED",
    "STRATA",
    "USERS_PER_ARM",
    "CupedSummary",
    "SampleSizeRow",
    "StandardErrorRow",
    "cuped_effect",
    "difference_in_means",
    "draw_experiment",
    "example_payload",
    "sample_size_rows",
    "simulate_estimates",
    "standard_error_rows",
    "stratified_effect",
    "theoretical_standard_error",
    "users_per_arm_for_power",
]

SEED: Final[int] = 0
USERS_PER_ARM: Final[int] = 2000
OUTCOME_MEAN: Final[float] = 50.0
OUTCOME_SD: Final[float] = 10.0
REPLICATIONS: Final[int] = 1200
STRATA: Final[int] = 4
# The figure's grid, exactly as np.linspace spaces it (0.30000000000000004 and so on).
CORRELATIONS: Final[tuple[float, ...]] = tuple(np.linspace(0.0, 0.9, 10).tolist())
EFFECT: Final[float] = 1.0
SAMPLE_SIZE_CORRELATIONS: Final[tuple[float, ...]] = (0.0, 0.3, 0.5, 0.7, 0.85)
_BLOCK: Final[int] = 200

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StandardErrorRow:
    """Spread of each estimator over the simulated A/A experiments at one correlation."""

    correlation: float
    difference_in_means: float
    cuped: float
    stratified: float
    theory: float


@dataclass(frozen=True, slots=True)
class SampleSizeRow:
    """Users per arm for 80 percent power at one correlation, and the saving."""

    correlation: float
    users_per_arm: float
    saving: float


@dataclass(frozen=True, slots=True)
class CupedSummary:
    """The figure's simulated standard errors and the article's sample-size table."""

    rows: tuple[StandardErrorRow, ...]
    sample_sizes: tuple[SampleSizeRow, ...]


def _correlation(value: float) -> float:
    rho = real(value, name="correlation")
    if not -1.0 < rho < 1.0:
        raise ValueError("correlation must lie in (-1, 1)")
    return rho


def draw_experiment(
    correlation: float,
    rng: np.random.Generator,
    *,
    users_per_arm: int = USERS_PER_ARM,
    sd: float = OUTCOME_SD,
) -> tuple[NDArray[np.int64], FloatArray, FloatArray]:
    """Draw one A/A experiment: treatment flags, covariates and outcomes.

    The draws are the figure's: all ``2 n`` covariates, then all ``2 n``
    outcome noise terms. The first ``n`` users are control.
    """
    rho = _correlation(correlation)
    n = count(users_per_arm, name="users_per_arm", minimum=2)
    spread = positive(sd, name="sd")
    treatment = np.repeat(np.array([0, 1], dtype=np.int64), n)
    covariate = rng.normal(OUTCOME_MEAN, spread, 2 * n)
    noise = rng.normal(0, spread * np.sqrt(1 - rho**2), 2 * n)
    outcome = OUTCOME_MEAN + rho * (covariate - OUTCOME_MEAN) + noise
    return treatment, covariate, outcome


def difference_in_means(treatment: NDArray[np.int64], outcome: FloatArray) -> float:
    """Mean outcome of treated users minus mean outcome of control users."""
    return float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())


def cuped_effect(treatment: NDArray[np.int64], covariate: FloatArray, outcome: FloatArray) -> float:
    """Difference in means of ``Y - theta (X - mean X)``, with ``theta`` pooled across arms."""
    theta = np.cov(covariate, outcome, ddof=1)[0, 1] / covariate.var(ddof=1)
    return difference_in_means(treatment, outcome - theta * (covariate - covariate.mean()))


def stratified_effect(
    treatment: NDArray[np.int64],
    covariate: FloatArray,
    outcome: FloatArray,
    strata: int = STRATA,
) -> float:
    """Within-stratum differences in means, weighted by stratum size, on covariate quantiles."""
    k = count(strata, name="strata", minimum=1)
    edges = np.quantile(covariate, np.linspace(0, 1, k + 1)[1:-1])
    stratum = np.digitize(covariate, edges)
    return float(
        sum(
            np.mean(stratum == j)
            * difference_in_means(treatment[stratum == j], outcome[stratum == j])
            for j in range(k)
        )
    )


def _block_estimates(
    covariate: FloatArray, outcome: FloatArray, n: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute the three estimators for each row of a block of experiments (control first).

    CUPED's adjusted means are the raw means less ``theta`` times the arm means
    of the centred covariate, which avoids forming the adjusted outcomes.
    """
    rows, users = covariate.shape
    difference = outcome[:, n:].mean(axis=1) - outcome[:, :n].mean(axis=1)

    centred = covariate - covariate.mean(axis=1, keepdims=True)
    covariance = np.einsum("ij,ij->i", centred, outcome - outcome.mean(axis=1, keepdims=True))
    theta = covariance / np.einsum("ij,ij->i", centred, centred)
    cuped = difference - theta * (centred[:, n:].mean(axis=1) - centred[:, :n].mean(axis=1))

    # Cells index (experiment, quartile, arm). A user's quartile is the number of
    # quartile edges at or below its covariate, which is how np.digitize assigns it.
    edges = np.quantile(covariate, [0.25, 0.5, 0.75], axis=1)
    cell = np.arange(0, 2 * STRATA * rows, 2 * STRATA)[:, None] + np.repeat([0, 1], n)
    for edge in edges:
        cell += 2 * (covariate >= edge[:, None])
    totals = np.bincount(cell.ravel(), weights=outcome.ravel(), minlength=2 * STRATA * rows)
    sizes = np.bincount(cell.ravel(), minlength=2 * STRATA * rows).reshape(rows, STRATA, 2)
    means = totals.reshape(rows, STRATA, 2) / sizes
    weights = sizes.sum(axis=2) / users
    stratified = np.zeros(rows)
    for j in range(STRATA):
        stratified = stratified + weights[:, j] * (means[:, j, 1] - means[:, j, 0])
    return difference, cuped, stratified


def simulate_estimates(
    correlation: float,
    rng: np.random.Generator,
    *,
    replications: int = REPLICATIONS,
    users_per_arm: int = USERS_PER_ARM,
    sd: float = OUTCOME_SD,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Simulate difference-in-means, CUPED and stratified estimates over A/A experiments.

    Each experiment consumes ``2 n`` covariate normals and then ``2 n`` noise
    normals, as :func:`draw_experiment` does, so drawing a block of them at
    once leaves the generator in the same state and gives the same data.
    ``normal(loc, scale)`` is ``loc + scale * z``, and the in-place arithmetic
    below performs the same operations, so the data are equal bit for bit.
    """
    rho = _correlation(correlation)
    reps = count(replications, name="replications", minimum=1)
    n = count(users_per_arm, name="users_per_arm", minimum=2)
    spread = positive(sd, name="sd")
    noise_scale = spread * np.sqrt(1 - rho**2)
    parts: list[tuple[FloatArray, FloatArray, FloatArray]] = []
    done = 0
    while done < reps:
        block = min(_BLOCK, reps - done)
        normals = rng.standard_normal((block, 2, 2 * n))
        covariate, noise = normals[:, 0, :], normals[:, 1, :]
        covariate *= spread
        covariate += OUTCOME_MEAN
        noise *= noise_scale
        outcome = covariate - OUTCOME_MEAN
        outcome *= rho
        outcome += OUTCOME_MEAN
        outcome += noise
        parts.append(_block_estimates(covariate, outcome, n))
        done += block
    difference, cuped, stratified = (np.concatenate(series) for series in zip(*parts, strict=True))
    return difference, cuped, stratified


def theoretical_standard_error(
    correlation: float, *, users_per_arm: int = USERS_PER_ARM, sd: float = OUTCOME_SD
) -> float:
    """Standard error of the adjusted estimate: ``sigma sqrt(2 / n) sqrt(1 - rho^2)``."""
    rho = _correlation(correlation)
    n = count(users_per_arm, name="users_per_arm", minimum=1)
    return positive(sd, name="sd") * sqrt(2 / n) * sqrt(1 - rho**2)


def standard_error_rows(
    seed: int = SEED,
    *,
    correlations: tuple[float, ...] = CORRELATIONS,
    replications: int = REPLICATIONS,
    users_per_arm: int = USERS_PER_ARM,
) -> tuple[StandardErrorRow, ...]:
    """Simulate the figure: the spread of each estimator at every correlation, in draw order."""
    rng = np.random.default_rng(count(seed, name="seed"))
    rows = []
    for rho in correlations:
        difference, cuped, stratified = simulate_estimates(
            rho, rng, replications=replications, users_per_arm=users_per_arm
        )
        rows.append(
            StandardErrorRow(
                correlation=rho,
                difference_in_means=float(np.std(difference)),
                cuped=float(np.std(cuped)),
                stratified=float(np.std(stratified)),
                theory=theoretical_standard_error(rho, users_per_arm=users_per_arm),
            )
        )
    return tuple(rows)


def users_per_arm_for_power(
    correlation: float,
    effect: float = EFFECT,
    sd: float = OUTCOME_SD,
    *,
    significance: float = 0.05,
    power: float = 0.80,
) -> float:
    """Users per arm for the given power with the variance scaled by ``1 - rho^2``."""
    rho = _correlation(correlation)
    alpha = probability(significance, name="significance", inclusive=False)
    beta = probability(power, name="power", inclusive=False)
    z = float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(beta))
    spread = positive(sd, name="sd")
    return 2 * spread**2 * (1 - rho**2) * z**2 / positive(effect, name="effect") ** 2


def sample_size_rows(
    correlations: tuple[float, ...] = SAMPLE_SIZE_CORRELATIONS,
) -> tuple[SampleSizeRow, ...]:
    """The article's sample-size table: users per arm and the share saved."""
    unadjusted = users_per_arm_for_power(0.0)
    return tuple(
        SampleSizeRow(
            correlation=rho,
            users_per_arm=users_per_arm_for_power(rho),
            saving=1 - users_per_arm_for_power(rho) / unadjusted,
        )
        for rho in correlations
    )


def example_payload() -> CupedSummary:
    """Return the figure's simulated standard errors and the closed-form sample sizes."""
    return CupedSummary(rows=standard_error_rows(), sample_sizes=sample_size_rows())
