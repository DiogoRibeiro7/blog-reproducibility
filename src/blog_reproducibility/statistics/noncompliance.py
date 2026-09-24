"""Intention to treat, as treated, per protocol and Wald estimates, for the non-compliance article.

An experiment assigns a feature to half its users, ``z = 1``, and some users
use it, ``d = 1``, whatever their assignment. Behind any adoption rate there are
three kinds of user: compliers, a share ``c``, who use the feature exactly when
assigned it; always-takers, a share ``a``, who use it regardless; and
never-takers, the remaining ``1 - c - a``, who never do. The kinds differ in
their baselines (10 for compliers, 13 for always-takers, 8 for never-takers),
using the feature adds ``tau = 2``, and the outcome has normal noise with
standard deviation 5.

Four estimators are computed on each experiment. Intention to treat compares
the arms as assigned and converges to ``tau c``, the effect of the offer. The
Wald estimator divides it by the difference in take-up between the arms, ``c``,
and converges to ``tau``, the effect of use among compliers. As treated
compares users who used the feature with users who did not, and per protocol
compares assigned users who used it with unassigned users who did not; both mix
kinds with different baselines, so their limits are

    as treated   = tau + (c/2 mu_c + a mu_a) / (c/2 + a) - (c/2 mu_c + n mu_n) / (c/2 + n),
    per protocol = tau + (c mu_c + a mu_a) / (c + a) - (c mu_c + n mu_n) / (c + n),

with ``n = 1 - c - a``, which exceed ``tau`` whenever always-takers or
never-takers exist. Each difference of means has the large-sample standard
deviation ``sqrt(v_1 / (N p_1) + v_0 / (N p_0))`` over the two groups' shares
``p`` and outcome variances ``v``, and the Wald estimate, by the delta method,
``sqrt(4 w / N) / c`` with ``w`` the variance of ``y - tau d``: 27.01 at the
article's 60 percent compliance, the "27" its sample-size formula uses.

The figure runs 500 experiments of 4,000 users at each of eight compliance
rates from 20 to 90 percent, with 10 percent always-takers, from one generator
seeded at 0, and is reproduced draw for draw. The article's tables run 3,000
experiments from a generator seeded at 0 in a different order (60 percent
compliance first), so they are not reproduced; the tests check them against the
closed forms above instead. Its sample sizes are closed forms and are pinned.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "ALWAYS_TAKERS",
    "ALWAYS_TAKER_SHIFT",
    "ARTICLE_OUTCOME_VARIANCE",
    "BASELINE",
    "COMPLIANCE_RATES",
    "ESTIMATORS",
    "NEVER_TAKER_SHIFT",
    "NOISE_SD",
    "POWER",
    "REPLICATIONS",
    "SAMPLE_SIZE_COMPLIANCE",
    "SEED",
    "SIGNIFICANCE",
    "TRUE_EFFECT",
    "USERS",
    "ComplianceRow",
    "Estimates",
    "NoncomplianceSummary",
    "SampleSizeRow",
    "ScenarioRow",
    "as_treated",
    "compliance_row",
    "draw_experiment",
    "estimate_all",
    "estimator_spread",
    "example_payload",
    "expected_estimates",
    "intention_to_treat",
    "per_protocol",
    "residual_variance",
    "scenario_row",
    "simulate_estimates",
    "users_per_arm",
    "wald",
]

SEED: Final[int] = 0
TRUE_EFFECT: Final[float] = 2.0
USERS: Final[int] = 4000
REPLICATIONS: Final[int] = 500
ALWAYS_TAKERS: Final[float] = 0.1
COMPLIANCE_RATES: Final[tuple[float, ...]] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BASELINE: Final[float] = 10.0
ALWAYS_TAKER_SHIFT: Final[float] = 3.0
NEVER_TAKER_SHIFT: Final[float] = -2.0
NOISE_SD: Final[float] = 5.0
ESTIMATORS: Final[tuple[str, ...]] = (
    "Intention to treat",
    "As treated",
    "Per protocol",
    "Wald (complier effect)",
)
# The article's sample-size section: "27 is about the outcome variance".
ARTICLE_OUTCOME_VARIANCE: Final[float] = 27.0
SAMPLE_SIZE_COMPLIANCE: Final[tuple[float, ...]] = (0.9, 0.6, 0.4, 0.2)
POWER: Final[float] = 0.8
SIGNIFICANCE: Final[float] = 0.05

_Arrays = tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class Estimates:
    """One value for each of the four estimators, in the figure's order."""

    intention_to_treat: float
    as_treated: float
    per_protocol: float
    wald: float


@dataclass(frozen=True, slots=True)
class ComplianceRow:
    """Mean of each estimator over the figure's experiments at one compliance rate."""

    compliance: float
    always_takers: float
    replications: int
    simulated: Estimates
    expected: Estimates
    spread: Estimates


@dataclass(frozen=True, slots=True)
class ScenarioRow:
    """Limits and single-experiment spreads of the estimators for one of the article's designs."""

    compliance: float
    always_takers: float
    effect: float
    expected: Estimates
    spread: Estimates


@dataclass(frozen=True, slots=True)
class SampleSizeRow:
    """Users per arm for 80 percent power on the effect of use, by the article's formula."""

    compliance: float
    users_per_arm: float
    residual_variance: float


@dataclass(frozen=True, slots=True)
class NoncomplianceSummary:
    """The figure's simulated means beside their limits, and the article's closed forms."""

    effect: float
    users: int
    rows: tuple[ComplianceRow, ...]
    article: ScenarioRow
    inert_feature: ScenarioRow
    one_sided: ScenarioRow
    sample_sizes: tuple[SampleSizeRow, ...]


def _shares(compliers: float, always_takers: float) -> tuple[float, float, float]:
    """Shares of compliers, always-takers and never-takers."""
    c = probability(compliers, name="compliers")
    a = probability(always_takers, name="always_takers")
    if c + a > 1.0 + 1e-12:
        raise ValueError("compliers and always_takers cannot exceed one together")
    return c, a, max(0.0, 1.0 - c - a)


def _kind_means() -> tuple[float, float, float]:
    """Untreated mean outcome of compliers, always-takers and never-takers."""
    return BASELINE, BASELINE + ALWAYS_TAKER_SHIFT, BASELINE + NEVER_TAKER_SHIFT


def draw_experiment(
    rng: np.random.Generator,
    users: int = USERS,
    compliers: float = 0.6,
    *,
    always_takers: float = ALWAYS_TAKERS,
    effect: float = TRUE_EFFECT,
) -> _Arrays:
    """Assign ``users`` 50/50 and return assignment, use and outcome, in the article's draw order.

    Assignment, then a uniform that decides each user's kind, then the noise.
    """
    n = count(users, name="users", minimum=1)
    c, a, _ = _shares(compliers, always_takers)
    tau = real(effect, name="effect")
    z = rng.integers(0, 2, n)
    u = rng.random(n)
    complier = u < c
    always = ~complier & (u < c + a)
    never = ~complier & ~always
    d = np.where(complier, z, np.where(always, 1, 0))
    base = BASELINE + np.where(always, ALWAYS_TAKER_SHIFT, np.where(never, NEVER_TAKER_SHIFT, 0.0))
    y: NDArray[np.float64] = base + tau * d + rng.normal(0, NOISE_SD, n)
    return z, d, y


def _itt(z: NDArray[np.int64], d: NDArray[np.int64], y: NDArray[np.float64]) -> float:
    return float(y[z == 1].mean() - y[z == 0].mean())


def _as_treated(z: NDArray[np.int64], d: NDArray[np.int64], y: NDArray[np.float64]) -> float:
    return float(y[d == 1].mean() - y[d == 0].mean())


def _per_protocol(z: NDArray[np.int64], d: NDArray[np.int64], y: NDArray[np.float64]) -> float:
    return float(y[(z == 1) & (d == 1)].mean() - y[(z == 0) & (d == 0)].mean())


def _wald(z: NDArray[np.int64], d: NDArray[np.int64], y: NDArray[np.float64]) -> float:
    return _itt(z, d, y) / float(d[z == 1].mean() - d[z == 0].mean())


def _experiment(assignment: ArrayLike, treatment: ArrayLike, outcome: ArrayLike) -> _Arrays:
    z = np.asarray(assignment)
    d = np.asarray(treatment)
    y = np.asarray(outcome, dtype=np.float64)
    if z.ndim != 1 or z.shape != d.shape or z.shape != y.shape:
        raise ValueError("assignment, treatment and outcome must be 1-D arrays of one length")
    for name, values in (("assignment", z), ("treatment", d)):
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(f"{name} must hold only zeros and ones")
    if not np.all(np.isfinite(y)):
        raise ValueError("outcome must be finite")
    z, d = z.astype(np.int64), d.astype(np.int64)
    groups = (z == 1, z == 0, d == 1, d == 0, (z == 1) & (d == 1), (z == 0) & (d == 0))
    if not all(np.any(group) for group in groups):
        raise ValueError("every arm and take-up group must contain at least one user")
    return z, d, y


def intention_to_treat(assignment: ArrayLike, treatment: ArrayLike, outcome: ArrayLike) -> float:
    """Mean outcome of the assigned arm minus that of the unassigned arm."""
    return _itt(*_experiment(assignment, treatment, outcome))


def as_treated(assignment: ArrayLike, treatment: ArrayLike, outcome: ArrayLike) -> float:
    """Mean outcome of users who used the feature minus that of users who did not."""
    return _as_treated(*_experiment(assignment, treatment, outcome))


def per_protocol(assignment: ArrayLike, treatment: ArrayLike, outcome: ArrayLike) -> float:
    """Assigned users who used the feature against unassigned users who did not."""
    return _per_protocol(*_experiment(assignment, treatment, outcome))


def wald(assignment: ArrayLike, treatment: ArrayLike, outcome: ArrayLike) -> float:
    """Intention to treat on the outcome over intention to treat on take-up."""
    z, d, y = _experiment(assignment, treatment, outcome)
    if d[z == 1].mean() == d[z == 0].mean():
        raise ValueError("take-up must differ between the arms")
    return _wald(z, d, y)


def estimate_all(assignment: ArrayLike, treatment: ArrayLike, outcome: ArrayLike) -> Estimates:
    """All four estimators on one experiment."""
    z, d, y = _experiment(assignment, treatment, outcome)
    return Estimates(
        intention_to_treat=_itt(z, d, y),
        as_treated=_as_treated(z, d, y),
        per_protocol=_per_protocol(z, d, y),
        wald=wald(z, d, y),
    )


def simulate_estimates(
    rng: np.random.Generator,
    compliers: float,
    *,
    always_takers: float = ALWAYS_TAKERS,
    effect: float = TRUE_EFFECT,
    users: int = USERS,
    replications: int = REPLICATIONS,
) -> NDArray[np.float64]:
    """Each estimator on ``replications`` experiments drawn one after another from ``rng``.

    Returns an array of shape ``(replications, 4)`` in the order of :data:`ESTIMATORS`.
    """
    reps = count(replications, name="replications", minimum=1)
    values = np.empty((reps, len(ESTIMATORS)))
    for rep in range(reps):
        z, d, y = draw_experiment(rng, users, compliers, always_takers=always_takers, effect=effect)
        values[rep] = (_itt(z, d, y), _as_treated(z, d, y), _per_protocol(z, d, y), _wald(z, d, y))
    return values


def _group(weights: tuple[float, ...], means: tuple[float, ...]) -> tuple[float, float, float]:
    """Share, mean outcome and outcome variance of a group mixing kinds of user."""
    share = sum(weights)
    if share <= 0.0:
        raise ValueError("every group must have a positive share of users")
    mean = sum(w * m for w, m in zip(weights, means, strict=True)) / share
    spread = sum(w * (m - mean) ** 2 for w, m in zip(weights, means, strict=True)) / share
    return share, mean, NOISE_SD**2 + spread


def _contrast(
    treated: tuple[float, float, float], control: tuple[float, float, float], users: int
) -> tuple[float, float]:
    """Limit and large-sample standard deviation of a difference of two group means."""
    p1, m1, v1 = treated
    p0, m0, v0 = control
    return m1 - m0, sqrt(v1 / (users * p1) + v0 / (users * p0))


def residual_variance(compliers: float, always_takers: float = ALWAYS_TAKERS) -> float:
    """Variance of ``y - tau d``: the noise plus the spread of the kinds' baselines."""
    c, a, n = _shares(compliers, always_takers)
    return _group((c, a, n), _kind_means())[2]


def _estimates_and_spread(
    compliers: float, always_takers: float, effect: float, users: int
) -> tuple[Estimates, Estimates]:
    c, a, n = _shares(compliers, always_takers)
    if c <= 0.0:
        raise ValueError("compliers must be positive for the Wald estimator to exist")
    tau = real(effect, name="effect")
    n_users = count(users, name="users", minimum=1)
    mu_c, mu_a, mu_n = _kind_means()
    itt = _contrast(
        _group((c / 2, a / 2, n / 2), (mu_c + tau, mu_a + tau, mu_n)),
        _group((c / 2, a / 2, n / 2), (mu_c, mu_a + tau, mu_n)),
        n_users,
    )
    treated = _contrast(
        _group((c / 2, a), (mu_c + tau, mu_a + tau)),
        _group((c / 2, n), (mu_c, mu_n)),
        n_users,
    )
    protocol = _contrast(
        _group((c / 2, a / 2), (mu_c + tau, mu_a + tau)),
        _group((c / 2, n / 2), (mu_c, mu_n)),
        n_users,
    )
    wald_sd = sqrt(4 * residual_variance(c, a) / n_users) / c
    expected = Estimates(itt[0], treated[0], protocol[0], tau)
    spread = Estimates(itt[1], treated[1], protocol[1], wald_sd)
    return expected, spread


def expected_estimates(
    compliers: float, always_takers: float = ALWAYS_TAKERS, effect: float = TRUE_EFFECT
) -> Estimates:
    """Large-sample limit of each estimator: ``tau c``, the two confounded contrasts, ``tau``."""
    return _estimates_and_spread(compliers, always_takers, effect, USERS)[0]


def estimator_spread(
    compliers: float,
    always_takers: float = ALWAYS_TAKERS,
    effect: float = TRUE_EFFECT,
    users: int = USERS,
) -> Estimates:
    """Large-sample standard deviation of each estimator across experiments of ``users``."""
    return _estimates_and_spread(compliers, always_takers, effect, users)[1]


def users_per_arm(
    compliance: float,
    *,
    variance: float = ARTICLE_OUTCOME_VARIANCE,
    effect: float = TRUE_EFFECT,
    power: float = POWER,
    significance: float = SIGNIFICANCE,
) -> float:
    """The article's ``2 sigma^2 (z_{1 - alpha/2} + z_power)^2 / (tau c)^2`` users per arm."""
    c = probability(compliance, name="compliance", inclusive=False)
    alpha = probability(significance, name="significance", inclusive=False)
    beta = probability(power, name="power", inclusive=False)
    z = float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(beta))
    return (
        2 * positive(variance, name="variance") * z**2 / (positive(effect, name="effect") * c) ** 2
    )


def compliance_row(
    rng: np.random.Generator,
    compliers: float,
    *,
    replications: int = REPLICATIONS,
    users: int = USERS,
) -> ComplianceRow:
    """Mean of each estimator over ``replications`` experiments, beside its limit and spread."""
    values = simulate_estimates(rng, compliers, users=users, replications=replications)
    means = [float(np.mean(column)) for column in values.T]
    expected, spread = _estimates_and_spread(compliers, ALWAYS_TAKERS, TRUE_EFFECT, users)
    return ComplianceRow(
        compliance=float(compliers),
        always_takers=ALWAYS_TAKERS,
        replications=values.shape[0],
        simulated=Estimates(*means),
        expected=expected,
        spread=spread,
    )


def scenario_row(compliers: float, always_takers: float, effect: float) -> ScenarioRow:
    """Limits and spreads for one of the article's designs of 4,000 users."""
    expected, spread = _estimates_and_spread(compliers, always_takers, effect, USERS)
    return ScenarioRow(
        compliance=float(compliers),
        always_takers=float(always_takers),
        effect=float(effect),
        expected=expected,
        spread=spread,
    )


def example_payload() -> NoncomplianceSummary:
    """Return the figure's means (500 experiments per rate) and the article's closed forms."""
    rng = np.random.default_rng(SEED)
    return NoncomplianceSummary(
        effect=TRUE_EFFECT,
        users=USERS,
        rows=tuple(compliance_row(rng, c) for c in COMPLIANCE_RATES),
        article=scenario_row(0.6, ALWAYS_TAKERS, TRUE_EFFECT),
        inert_feature=scenario_row(0.6, ALWAYS_TAKERS, 0.0),
        one_sided=scenario_row(0.7, 0.0, TRUE_EFFECT),
        sample_sizes=tuple(
            SampleSizeRow(
                compliance=c,
                users_per_arm=users_per_arm(c),
                residual_variance=residual_variance(c),
            )
            for c in SAMPLE_SIZE_COMPLIANCE
        ),
    )
