"""Regression, matching, weighting and doubly robust estimates, for the propensity score article.

Each simulated study has 4,000 units with three standard normal confounders
``x``. Treatment is assigned with probability ``sigma(z)``, the logistic function
of ``z = s (0.9 x_1 + 0.6 x_2 - 0.5 x_3)`` at confounding strength ``s``, and the
outcome is ``10 + 2 x_1 + x_2 + 0.5 x_3 + 2 d + e`` with ``e ~ N(0, 2^2)``; in
the misspecified-outcome condition it also has ``x_1^3 - x_1 x_2``. The
propensity score is fitted by 40 Newton steps of a logistic regression and
clipped to ``[0.01, 0.99]``. Five estimators are compared: the naive difference
in means, regression adjustment (the coefficient on ``d`` in a linear model),
one-to-one nearest-neighbour matching on the score with replacement,
normalised inverse probability weighting, and the augmented (doubly robust)
estimator, which corrects arm-wise linear predictions with weighted residuals.

Under four conditions (both models correct; the outcome model missing the
cubic and interaction terms; the propensity model missing ``x_3``; strength 2.5,
which leaves little overlap) regression fails only when the outcome model is
wrong, matching and weighting only when the propensity model is wrong, and the
doubly robust estimator stays unbiased throughout.

Some limits have closed forms. Write ``x = a z + u`` with ``a = beta / |beta|^2``
and ``u`` independent of ``z``, whose covariance is ``I - beta beta^T / |beta|^2``.
Every moment ``E[sigma(z) m(x)]`` of a polynomial ``m`` then reduces to a
one-dimensional Gaussian integral over ``z``, computed here by Gauss-Hermite
quadrature. That gives the naive difference (3.67 with a linear outcome, 5.63
with the nonlinear one, 4.48 at strength 2.5), the limit of regression
adjustment under the misspecified outcome (1.81), the standardised
differences in the covariates before adjustment, and the share of true scores
outside ``[0.1, 0.9]``, ``2 Phi(-logit(0.9) / |beta|)``.

The figure runs 400 studies in each condition, one after another from a
generator seeded at 0, and is reproduced draw for draw. The article's four
tables are the same computation (its fits take 60 Newton steps rather than 40,
which converge to the same scores), so they are reproduced too. Newton's map
is deterministic, so once an iterate repeats the rest of the 40 steps cycle,
and the fit reads the 40th iterate off the cycle rather than computing it; the
scores are identical. Under poor overlap many scores are clipped to the same
0.01 or 0.99, so a treated unit often has several equally near controls; the
match is whichever SciPy's k-d tree returns first, and that choice alone moves
a study's matching estimate by up to about half a point. The article's overlap,
trimming and balance tables continue along the generator after the figure's
studies, with other draws, and are not reproduced; the tests check them
against the closed forms.
"""

from dataclasses import dataclass
from math import log, sqrt
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.spatial import cKDTree

from blog_reproducibility.common.validation import count, non_negative, probability, real

__all__ = [
    "CLIP",
    "CONDITIONS",
    "ESTIMATORS",
    "INTERCEPT",
    "NEWTON_STEPS",
    "NOISE_SD",
    "OUTCOME_COEFFICIENTS",
    "OVERLAP_STRENGTHS",
    "REPLICATIONS",
    "SEED",
    "TREATMENT_COEFFICIENTS",
    "TRUE_EFFECT",
    "UNITS",
    "BalanceRow",
    "Condition",
    "ConditionRow",
    "Estimates",
    "OverlapRow",
    "PropensitySummary",
    "condition_estimates",
    "doubly_robust",
    "draw_study",
    "estimate_all",
    "example_payload",
    "fit_propensity",
    "inverse_probability_weighting",
    "naive_difference",
    "naive_difference_limit",
    "propensity_matching",
    "regression_adjustment",
    "regression_limit",
    "share_outside",
    "skip_studies",
    "standardised_difference",
    "summarise",
]

SEED: Final[int] = 0
TRUE_EFFECT: Final[float] = 2.0
UNITS: Final[int] = 4000
REPLICATIONS: Final[int] = 400
TREATMENT_COEFFICIENTS: Final[tuple[float, float, float]] = (0.9, 0.6, -0.5)
OUTCOME_COEFFICIENTS: Final[tuple[float, float, float]] = (2.0, 1.0, 0.5)
INTERCEPT: Final[float] = 10.0
NOISE_SD: Final[float] = 2.0
NEWTON_STEPS: Final[int] = 40
CLIP: Final[tuple[float, float]] = (0.01, 0.99)
ESTIMATORS: Final[tuple[str, ...]] = (
    "Naive difference",
    "Regression adjustment",
    "Propensity matching",
    "Inverse probability weighting",
    "Doubly robust (AIPW)",
)
# The article's overlap table: confounding strengths, and the score band it counts outside of.
OVERLAP_STRENGTHS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 3.0)
_OVERLAP_BAND: Final[float] = 0.1
_QUADRATURE_NODES: Final[int] = 160

_Study = tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class Condition:
    """One of the figure's four conditions."""

    name: str
    strength: float = 1.0
    nonlinear: bool = False
    wrong_propensity: bool = False


CONDITIONS: Final[tuple[Condition, ...]] = (
    Condition("both models correct"),
    Condition("outcome model misspecified", nonlinear=True),
    Condition("propensity model misspecified", wrong_propensity=True),
    Condition("strong confounding, poor overlap", strength=2.5),
)


@dataclass(frozen=True, slots=True)
class Estimates:
    """One value for each of the five estimators, in the figure's order."""

    naive: float
    regression: float
    matching: float
    weighting: float
    doubly_robust: float


@dataclass(frozen=True, slots=True)
class ConditionRow:
    """Each estimator's mean, spread, error and bias over one condition's studies."""

    condition: Condition
    replications: int
    mean: Estimates
    standard_deviation: Estimates
    root_mean_squared_error: Estimates
    absolute_bias: Estimates
    naive_limit: float
    regression_limit: float


@dataclass(frozen=True, slots=True)
class OverlapRow:
    """Share of true propensity scores outside ``[0.1, 0.9]`` at one confounding strength."""

    strength: float
    share_outside: float


@dataclass(frozen=True, slots=True)
class BalanceRow:
    """Standardised difference in one covariate between the arms, before adjustment."""

    covariate: int
    standardised_difference: float


@dataclass(frozen=True, slots=True)
class PropensitySummary:
    """The figure's four conditions and the closed forms behind the article's other tables."""

    effect: float
    units: int
    rows: tuple[ConditionRow, ...]
    overlap: tuple[OverlapRow, ...]
    balance: tuple[BalanceRow, ...]


def _strength(strength: float) -> float:
    return non_negative(strength, name="strength")


def draw_study(
    rng: np.random.Generator,
    units: int = UNITS,
    *,
    strength: float = 1.0,
    nonlinear: bool = False,
) -> _Study:
    """Draw confounders, treatment and outcome, in the article's order."""
    n = count(units, name="units", minimum=2)
    s = _strength(strength)
    x = rng.normal(0, 1, (n, 3))
    b1, b2, b3 = TREATMENT_COEFFICIENTS
    logit = s * (b1 * x[:, 0] + b2 * x[:, 1] + b3 * x[:, 2])
    d = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    c1, c2, c3 = OUTCOME_COEFFICIENTS
    base = c1 * x[:, 0] + c2 * x[:, 1] + c3 * x[:, 2]
    if nonlinear:
        base = base + 1.0 * x[:, 0] ** 3 - 1.0 * x[:, 0] * x[:, 1]
    y: NDArray[np.float64] = INTERCEPT + base + TRUE_EFFECT * d + rng.normal(0, NOISE_SD, n)
    return x, d, y


def skip_studies(rng: np.random.Generator, studies: int, units: int = UNITS) -> None:
    """Advance ``rng`` past ``studies`` draws of :func:`draw_study`, whatever their condition."""
    n = count(units, name="units", minimum=2)
    for _ in range(count(studies, name="studies")):
        rng.normal(0, 1, (n, 3))
        rng.random(n)
        rng.normal(0, NOISE_SD, n)


def fit_propensity(
    covariates: ArrayLike, treatment: ArrayLike, steps: int = NEWTON_STEPS
) -> NDArray[np.float64]:
    """Fitted probabilities after ``steps`` damped Newton steps of a logistic regression.

    Each step solves ``(A^T W A + 1e-8 I) delta = A^T (d - p)`` with
    ``W = p (1 - p) + 1e-9``, from zero coefficients. When an iterate repeats, the
    remaining steps go round the same cycle, so the final iterate is read off it.
    """
    x = np.asarray(covariates, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    d = np.asarray(treatment)
    budget = count(steps, name="steps")
    if d.shape != (design.shape[0],):
        raise ValueError("treatment must have one entry per row of the covariates")
    ridge = 1e-8 * np.eye(design.shape[1])
    b = np.zeros(design.shape[1])
    iterates = [b]
    seen = {b.tobytes(): 0}
    for step in range(1, budget + 1):
        p = 1 / (1 + np.exp(-design @ b))
        w = p * (1 - p) + 1e-9
        b = b + np.linalg.solve((design * w[:, None]).T @ design + ridge, design.T @ (d - p))
        key = b.tobytes()
        if key in seen:
            start = seen[key]
            b = iterates[start + (budget - start) % (step - start)]
            break
        seen[key] = step
        iterates.append(b)
    scores: NDArray[np.float64] = 1 / (1 + np.exp(-design @ b))
    return scores


def _estimates(
    x: NDArray[np.float64], d: NDArray[np.int64], y: NDArray[np.float64], ps: NDArray[np.float64]
) -> tuple[float, float, float, float, float]:
    """The website's five estimates, in its order of operations."""
    design = np.column_stack([np.ones(len(x)), d, x])
    regression = np.linalg.lstsq(design, y, rcond=None)[0][1]
    treated, controls = np.where(d == 1)[0], np.where(d == 0)[0]
    _, match = cKDTree(ps[controls][:, None]).query(ps[treated][:, None])
    matching = np.mean(y[treated] - y[controls][match])
    w1, w0 = d / ps, (1 - d) / (1 - ps)
    weighting = np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0)
    arm = np.column_stack([np.ones(len(x)), x])
    m1 = arm @ np.linalg.lstsq(arm[d == 1], y[d == 1], rcond=None)[0]
    m0 = arm @ np.linalg.lstsq(arm[d == 0], y[d == 0], rcond=None)[0]
    robust = np.mean(m1 - m0 + d * (y - m1) / ps - (1 - d) * (y - m0) / (1 - ps))
    naive = y[d == 1].mean() - y[d == 0].mean()
    return (
        float(naive),
        float(regression),
        float(matching),
        float(weighting),
        float(robust),
    )


def _study(
    covariates: ArrayLike, treatment: ArrayLike, outcome: ArrayLike, scores: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(covariates, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    d = np.asarray(treatment)
    y = np.asarray(outcome, dtype=np.float64)
    n = x.shape[0]
    if x.ndim != 2 or d.shape != (n,) or y.shape != (n,):
        raise ValueError("covariates, treatment and outcome must describe the same units")
    if not np.all((d == 0) | (d == 1)):
        raise ValueError("treatment must hold only zeros and ones")
    d = d.astype(np.int64)
    if np.count_nonzero(d) < 2 or np.count_nonzero(d == 0) < 2:
        raise ValueError("each arm needs at least two units")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("covariates and outcome must be finite")
    ps = np.full(n, 0.5) if scores is None else np.asarray(scores, dtype=np.float64)
    if ps.shape != (n,) or not np.all((ps > 0) & (ps < 1)):
        raise ValueError("scores must be one probability in (0, 1) per unit")
    return x, d, y, ps


def naive_difference(covariates: ArrayLike, treatment: ArrayLike, outcome: ArrayLike) -> float:
    """Mean outcome of the treated minus that of the untreated."""
    _, d, y, _ = _study(covariates, treatment, outcome, None)
    return float(y[d == 1].mean() - y[d == 0].mean())


def regression_adjustment(covariates: ArrayLike, treatment: ArrayLike, outcome: ArrayLike) -> float:
    """Coefficient on treatment in a linear regression of the outcome on it and the covariates."""
    x, d, y, _ = _study(covariates, treatment, outcome, None)
    design = np.column_stack([np.ones(len(x)), d, x])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def propensity_matching(
    covariates: ArrayLike, treatment: ArrayLike, outcome: ArrayLike, scores: ArrayLike
) -> float:
    """Each treated unit against the control with the nearest score, with replacement."""
    _, d, y, ps = _study(covariates, treatment, outcome, scores)
    treated, controls = np.where(d == 1)[0], np.where(d == 0)[0]
    _, match = cKDTree(ps[controls][:, None]).query(ps[treated][:, None])
    return float(np.mean(y[treated] - y[controls][match]))


def inverse_probability_weighting(
    covariates: ArrayLike,
    treatment: ArrayLike,
    outcome: ArrayLike,
    scores: ArrayLike,
    *,
    trim: float = 0.0,
) -> float:
    """Normalised inverse probability weights, dropping scores within ``trim`` of 0 or 1."""
    _, d, y, ps = _study(covariates, treatment, outcome, scores)
    cut = probability(trim, name="trim")
    if cut >= 0.5:
        raise ValueError("trim must be below one half")
    keep = (ps > cut) & (ps < 1 - cut)
    d, y, ps = d[keep], y[keep], ps[keep]
    if np.count_nonzero(d) == 0 or np.count_nonzero(d == 0) == 0:
        raise ValueError("trimming left an arm empty")
    w1, w0 = d / ps, (1 - d) / (1 - ps)
    return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))


def doubly_robust(
    covariates: ArrayLike, treatment: ArrayLike, outcome: ArrayLike, scores: ArrayLike
) -> float:
    """Augmented inverse probability weighting on arm-wise linear predictions."""
    x, d, y, ps = _study(covariates, treatment, outcome, scores)
    arm = np.column_stack([np.ones(len(x)), x])
    m1 = arm @ np.linalg.lstsq(arm[d == 1], y[d == 1], rcond=None)[0]
    m0 = arm @ np.linalg.lstsq(arm[d == 0], y[d == 0], rcond=None)[0]
    return float(np.mean(m1 - m0 + d * (y - m1) / ps - (1 - d) * (y - m0) / (1 - ps)))


def estimate_all(
    covariates: ArrayLike, treatment: ArrayLike, outcome: ArrayLike, scores: ArrayLike
) -> Estimates:
    """All five estimators on one study, given its propensity scores."""
    return Estimates(*_estimates(*_study(covariates, treatment, outcome, scores)))


def _simulate(
    rng: np.random.Generator, condition: Condition, estimated: int, skipped: int, units: int
) -> NDArray[np.float64]:
    values = np.empty((estimated, len(ESTIMATORS)))
    for rep in range(estimated):
        x, d, y = draw_study(rng, units, strength=condition.strength, nonlinear=condition.nonlinear)
        features = x[:, :2] if condition.wrong_propensity else x
        ps = np.clip(fit_propensity(features, d), *CLIP)
        values[rep] = _estimates(x, d, y, ps)
    skip_studies(rng, skipped, units)
    return values


def condition_estimates(
    seed: int = SEED,
    *,
    replications: int = REPLICATIONS,
    estimated: int | None = None,
    units: int = UNITS,
) -> tuple[NDArray[np.float64], ...]:
    """The figure's loop: ``replications`` studies per condition, one condition after another.

    Only the first ``estimated`` studies of each condition are estimated (all of
    them by default); the rest are drawn and discarded, which keeps the
    generator where the figure's loop has it. Returns one ``(estimated, 5)``
    array per condition.
    """
    rng = np.random.default_rng(count(seed, name="seed"))
    reps = count(replications, name="replications", minimum=1)
    first = reps if estimated is None else count(estimated, name="estimated", minimum=1)
    if first > reps:
        raise ValueError("estimated cannot exceed replications")
    n = count(units, name="units", minimum=2)
    return tuple(_simulate(rng, condition, first, reps - first, n) for condition in CONDITIONS)


def _moments(strength: float) -> tuple[NDArray[np.float64], NDArray[np.float64], Any]:
    """Loadings ``a``, residual covariance ``C`` and ``E[z^k sigma(z)]`` for k = 0 to 3."""
    beta = _strength(strength) * np.array(TREATMENT_COEFFICIENTS)
    scale = float(beta @ beta)
    nodes, weights = np.polynomial.hermite_e.hermegauss(_QUADRATURE_NODES)
    z = sqrt(scale) * nodes
    sigmoid = 1 / (1 + np.exp(-z))
    expect = [float(weights @ (z**k * sigmoid)) / sqrt(2 * np.pi) for k in range(4)]
    loadings = beta / scale
    residual = np.eye(3) - np.outer(beta, beta) / scale
    return loadings, residual, expect


def _population(
    strength: float, nonlinear: bool
) -> tuple[float, float, float, NDArray[np.float64]]:
    """``P(d = 1)``, ``E[y]``, ``E[d y]`` and ``E[d x]`` for the study design."""
    if _strength(strength) == 0.0:
        raise ValueError("strength must be positive for the treatment to depend on the confounders")
    a, cov, e = _moments(strength)
    share = e[0]
    dx = a * e[1]
    c = np.array(OUTCOME_COEFFICIENTS)
    mean_y = INTERCEPT + TRUE_EFFECT * share
    dy = INTERCEPT * share + float(c @ dx) + TRUE_EFFECT * share
    if nonlinear:
        cube = a[0] ** 3 * e[3] + 3 * a[0] * cov[0, 0] * e[1]
        cross = a[0] * a[1] * e[2] + cov[0, 1] * e[0]
        dy += cube - cross
    return share, mean_y, dy, dx


def naive_difference_limit(strength: float = 1.0, *, nonlinear: bool = False) -> float:
    """Large-sample naive difference: ``E[y | d = 1] - E[y | d = 0]``."""
    share, mean_y, dy, _ = _population(strength, nonlinear)
    return dy / share - (mean_y - dy) / (1 - share)


def regression_limit(strength: float = 1.0, *, nonlinear: bool = False) -> float:
    """Large-sample coefficient on treatment in the linear regression on ``1, d, x``.

    With the linear outcome it is the true effect. With the nonlinear one the
    regression omits ``x_1^3 - x_1 x_2``, whose projection on ``d`` given ``x``
    is the bias.
    """
    share, mean_y, dy, dx = _population(strength, nonlinear)
    gram = np.zeros((5, 5))
    gram[0, 0], gram[0, 1], gram[1, 0], gram[1, 1] = 1.0, share, share, share
    gram[1, 2:], gram[2:, 1] = dx, dx
    gram[2:, 2:] = np.eye(3)
    cross = np.zeros(5)
    cross[0], cross[1] = mean_y, dy
    cross[2:] = np.array(OUTCOME_COEFFICIENTS) + TRUE_EFFECT * dx
    if nonlinear:
        cross[2] += 3.0  # E[x_1 x_1^3]; E[x_k x_1 x_2] and the other odd moments vanish.
    return float(np.linalg.solve(gram, cross)[1])


def standardised_difference(strength: float = 1.0) -> tuple[float, float, float]:
    """Difference in each covariate's mean between the arms, in its standard deviations."""
    share, _, _, dx = _population(strength, False)
    values = dx / share + dx / (1 - share)
    return float(values[0]), float(values[1]), float(values[2])


def share_outside(strength: float, band: float = _OVERLAP_BAND) -> float:
    """Share of true scores outside ``[band, 1 - band]``: ``2 Phi(-logit(1 - band) / |beta|)``."""
    edge = probability(band, name="band", inclusive=False)
    if edge >= 0.5:
        raise ValueError("band must be below one half")
    spread = _strength(strength) * sqrt(sum(b**2 for b in TREATMENT_COEFFICIENTS))
    if spread == 0.0:
        return 0.0
    return float(2 * stats.norm.cdf(-log((1 - edge) / edge) / spread))


def _stats(values: NDArray[np.float64]) -> tuple[Estimates, Estimates, Estimates, Estimates]:
    mean = values.mean(axis=0)
    return (
        Estimates(*(float(v) for v in mean)),
        Estimates(*(float(v) for v in values.std(axis=0))),
        Estimates(*(float(v) for v in np.sqrt(np.mean((values - TRUE_EFFECT) ** 2, axis=0)))),
        Estimates(*(float(v) for v in np.abs(mean - TRUE_EFFECT))),
    )


def summarise(condition: Condition, values: ArrayLike) -> ConditionRow:
    """Mean, spread, error and bias of each estimator over a condition's studies."""
    table = np.asarray(values, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != len(ESTIMATORS) or table.shape[0] < 1:
        raise ValueError("values must have one row per study and one column per estimator")
    mean, sd, rmse, bias = _stats(table)
    strength = real(condition.strength, name="strength")
    return ConditionRow(
        condition=condition,
        replications=table.shape[0],
        mean=mean,
        standard_deviation=sd,
        root_mean_squared_error=rmse,
        absolute_bias=bias,
        naive_limit=naive_difference_limit(strength, nonlinear=condition.nonlinear),
        regression_limit=regression_limit(strength, nonlinear=condition.nonlinear),
    )


def example_payload() -> PropensitySummary:
    """Return the figure's four conditions (400 studies each) and the article's closed forms."""
    values = condition_estimates()
    return PropensitySummary(
        effect=TRUE_EFFECT,
        units=UNITS,
        rows=tuple(
            summarise(condition, table) for condition, table in zip(CONDITIONS, values, strict=True)
        ),
        overlap=tuple(OverlapRow(s, share_outside(s)) for s in OVERLAP_STRENGTHS),
        balance=tuple(
            BalanceRow(j + 1, value) for j, value in enumerate(standardised_difference(1.0))
        ),
    )
