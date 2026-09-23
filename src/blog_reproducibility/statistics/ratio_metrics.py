"""Ratio metrics and the delta method, for the article on per-session metrics in A/B tests.

A per-session conversion rate in a user-randomised experiment is a ratio of two
user-level sums, ``R = sum(Y_i) / sum(X_i)``, where ``X_i`` counts user ``i``'s
sessions and ``Y_i`` her conversions. The session-level z-test treats every
session as an independent trial and gives each arm's rate the variance
``p (1 - p) / sum(X_i)``. The delta method keeps users as the unit:

    Var(R) ~ [s_Y^2 - 2 (m_Y / m_X) s_XY + (m_Y / m_X)^2 s_X^2] / (n m_X^2),

with the means, variances and covariance taken over the ``n`` users of an arm.
It is the variance of the mean of the linearised per-user values
``Y_i - R X_i``, scaled by ``1 / m_X^2``.

In the article's model each user's propensity to convert is Beta with mean 0.10
and concentration ``kappa``, sessions are ``1 + Poisson(m - 1)``, and
conversions are binomial in the sessions at the user's propensity. Sessions of
one user then have intraclass correlation ``rho = 1 / (kappa + 1)``. With
exactly ``m`` sessions per user, the session-level variance is too small by the
design effect ``1 + (m - 1) rho``; with Poisson session counts the delta-method
variance exceeds it by ``1 + (m - 1 / m) rho``. Under the null the session-level
z statistic is then roughly normal with that ratio as its variance, so its false
positive rate is about ``2 Phi(-1.96 / sqrt(ratio))``.

The figure runs 1,200 A/A tests with 2,000 users per arm at each of five mean
session counts, for kappa 20 and then kappa 4, from one generator seeded at 0,
and is reproduced draw for draw. The article's tables come from its own code,
which runs a different sequence of simulations (2,000 replications, other
concentrations, a user bootstrap) and are not reproduced; the closed-form
numbers it derives are.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "BASE_RATE",
    "CONCENTRATIONS",
    "CRITICAL_Z",
    "MEAN_SESSIONS",
    "MILD_HETEROGENEITY",
    "REPLICATIONS",
    "SEED",
    "STRONG_HETEROGENEITY",
    "USERS_PER_ARM",
    "Arm",
    "ArticleNumbers",
    "FalsePositiveRow",
    "RatioMetricSummary",
    "article_numbers",
    "delta_method_variance",
    "delta_method_z",
    "design_effect",
    "draw_arm",
    "example_payload",
    "false_positive_rates",
    "intraclass_correlation",
    "propensity_sd",
    "session_level_false_positive_rate",
    "session_level_z",
    "variance_ratio",
]

SEED: Final[int] = 0
USERS_PER_ARM: Final[int] = 2000
BASE_RATE: Final[float] = 0.10
MILD_HETEROGENEITY: Final[float] = 20.0
STRONG_HETEROGENEITY: Final[float] = 4.0
CONCENTRATIONS: Final[tuple[float, ...]] = (MILD_HETEROGENEITY, STRONG_HETEROGENEITY)
MEAN_SESSIONS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10)
REPLICATIONS: Final[int] = 1200
CRITICAL_Z: Final[float] = 1.96
# The article's worked example: three sessions per user at kappa 4.
ARTICLE_SESSIONS: Final[int] = 3

Arm = tuple[NDArray[np.int64], NDArray[np.int64]]
"""One arm of an experiment: per-user sessions and per-user conversions."""


@dataclass(frozen=True, slots=True)
class FalsePositiveRow:
    """Share of A/A tests declared significant at one design, with the closed-form prediction."""

    concentration: float
    mean_sessions: int
    variance_ratio: float
    predicted_session_level: float
    session_level: float
    delta_method: float


@dataclass(frozen=True, slots=True)
class ArticleNumbers:
    """Closed-form numbers the article derives for kappa 4 and three sessions per user."""

    intraclass_correlation: float
    propensity_sd: float
    design_effect: float
    variance_ratio: float
    standard_error_shortfall: float


@dataclass(frozen=True, slots=True)
class RatioMetricSummary:
    """The figure's false positive rates and the article's closed-form numbers."""

    rows: tuple[FalsePositiveRow, ...]
    article: ArticleNumbers


def intraclass_correlation(concentration: float) -> float:
    """Correlation between two sessions of one user when propensities are Beta(kappa)."""
    return 1 / (positive(concentration, name="concentration") + 1)


def propensity_sd(concentration: float, base_rate: float = BASE_RATE) -> float:
    """Standard deviation of users' propensities: ``sqrt(mu (1 - mu) / (kappa + 1))``."""
    mu = probability(base_rate, name="base_rate", inclusive=False)
    return sqrt(mu * (1 - mu) * intraclass_correlation(concentration))


def design_effect(sessions_per_user: float, concentration: float) -> float:
    """Variance inflation ``1 + (m - 1) rho`` when every user has exactly ``m`` sessions."""
    m = real(sessions_per_user, name="sessions_per_user")
    if m < 1:
        raise ValueError("sessions_per_user must be 1 or greater")
    return 1 + (m - 1) * intraclass_correlation(concentration)


def variance_ratio(mean_sessions: float, concentration: float) -> float:
    """Delta-method over session-level variance for ``1 + Poisson(m - 1)`` sessions per user.

    The per-user residual ``Y - mu X`` has variance
    ``mu (1 - mu) [m kappa + E(X^2)] / (kappa + 1)`` with ``E(X^2) = m^2 + m - 1``;
    dividing by ``m`` times the per-session variance ``mu (1 - mu)`` gives
    ``1 + (m - 1 / m) rho``. The base rate cancels.
    """
    m = real(mean_sessions, name="mean_sessions")
    if m < 1:
        raise ValueError("mean_sessions must be 1 or greater")
    return 1 + (m - 1 / m) * intraclass_correlation(concentration)


def session_level_false_positive_rate(ratio: float, *, critical: float = CRITICAL_Z) -> float:
    """Large-sample A/A rejection rate of a test whose variance is too small by ``ratio``."""
    inflation = positive(ratio, name="ratio")
    return float(2 * stats.norm.sf(positive(critical, name="critical") / sqrt(inflation)))


def draw_arm(
    rng: np.random.Generator,
    users: int,
    concentration: float,
    mean_sessions: int,
    *,
    base_rate: float = BASE_RATE,
) -> Arm:
    """Draw one arm: propensities, then session counts, then conversions, as the article does."""
    n = count(users, name="users", minimum=2)
    kappa = positive(concentration, name="concentration")
    sessions_mean = count(mean_sessions, name="mean_sessions", minimum=1)
    mu = probability(base_rate, name="base_rate", inclusive=False)
    propensity = rng.beta(mu * kappa, (1 - mu) * kappa, n)
    sessions = 1 + rng.poisson(sessions_mean - 1, n)
    return sessions, rng.binomial(sessions, propensity)


_UserArrays = tuple[NDArray[np.float64], NDArray[np.float64]]


def _user_arrays(sessions: ArrayLike, conversions: ArrayLike) -> _UserArrays:
    s = np.asarray(sessions, dtype=np.float64)
    c = np.asarray(conversions, dtype=np.float64)
    if s.ndim != 1 or s.shape != c.shape or s.size < 2:
        raise ValueError("sessions and conversions must be equal-length sequences of 2+ users")
    if not np.all(np.isfinite(s + c)) or np.any(s < 0) or np.any(c < 0):
        raise ValueError("sessions and conversions must be finite and non-negative")
    if s.sum() <= 0:
        raise ValueError("the arm must have at least one session")
    return s, c


def _delta_variance(s: NDArray[np.float64], c: NDArray[np.float64]) -> float:
    n, mx, my = len(s), s.mean(), c.mean()
    vx, vy, cxy = s.var(ddof=1), c.var(ddof=1), np.cov(s, c, ddof=1)[0, 1]
    return float((vy - 2 * (my / mx) * cxy + (my / mx) ** 2 * vx) / (mx**2 * n))


def _session_z(control: _UserArrays, treatment: _UserArrays) -> float:
    (s_a, c_a), (s_b, c_b) = control, treatment
    sessions_a, sessions_b = s_a.sum(), s_b.sum()
    conversions_a, conversions_b = c_a.sum(), c_b.sum()
    pooled = (conversions_a + conversions_b) / (sessions_a + sessions_b)
    if not 0 < pooled < 1:
        raise ValueError("the pooled conversion rate must lie strictly between 0 and 1")
    rate_a, rate_b = conversions_a / sessions_a, conversions_b / sessions_b
    return float(
        (rate_b - rate_a) / np.sqrt(pooled * (1 - pooled) * (1 / sessions_a + 1 / sessions_b))
    )


def _delta_z(control: _UserArrays, treatment: _UserArrays) -> float:
    (s_a, c_a), (s_b, c_b) = control, treatment
    variance = _delta_variance(s_a, c_a) + _delta_variance(s_b, c_b)
    if variance <= 0:
        raise ValueError("the delta-method variance must be positive")
    return float((c_b.sum() / s_b.sum() - c_a.sum() / s_a.sum()) / np.sqrt(variance))


def delta_method_variance(sessions: ArrayLike, conversions: ArrayLike) -> float:
    """Delta-method variance of ``sum(conversions) / sum(sessions)``, users as the unit."""
    return _delta_variance(*_user_arrays(sessions, conversions))


def session_level_z(
    control: tuple[ArrayLike, ArrayLike], treatment: tuple[ArrayLike, ArrayLike]
) -> float:
    """Pooled two-proportion z statistic that counts every session as an independent trial."""
    return _session_z(_user_arrays(*control), _user_arrays(*treatment))


def delta_method_z(
    control: tuple[ArrayLike, ArrayLike], treatment: tuple[ArrayLike, ArrayLike]
) -> float:
    """z statistic for the difference in ratios with the delta-method variance of each arm."""
    return _delta_z(_user_arrays(*control), _user_arrays(*treatment))


def false_positive_rates(
    seed: int = SEED,
    *,
    concentrations: tuple[float, ...] = CONCENTRATIONS,
    mean_sessions: tuple[int, ...] = MEAN_SESSIONS,
    users: int = USERS_PER_ARM,
    replications: int = REPLICATIONS,
) -> tuple[FalsePositiveRow, ...]:
    """Run A/A tests for every concentration, then every session count, in the figure's order."""
    rng = np.random.default_rng(count(seed, name="seed"))
    reps = count(replications, name="replications", minimum=1)
    n = count(users, name="users", minimum=2)
    kappas = tuple(positive(kappa, name="concentration") for kappa in concentrations)
    sessions = tuple(count(m, name="mean_sessions", minimum=1) for m in mean_sessions)

    rows = []
    for kappa in kappas:
        for m in sessions:
            naive = delta = 0
            for _ in range(reps):
                control = _user_arrays(*draw_arm(rng, n, kappa, m))
                treatment = _user_arrays(*draw_arm(rng, n, kappa, m))
                naive += abs(_session_z(control, treatment)) > CRITICAL_Z
                delta += abs(_delta_z(control, treatment)) > CRITICAL_Z
            ratio = variance_ratio(m, kappa)
            rows.append(
                FalsePositiveRow(
                    concentration=kappa,
                    mean_sessions=m,
                    variance_ratio=ratio,
                    predicted_session_level=session_level_false_positive_rate(ratio),
                    session_level=naive / reps,
                    delta_method=delta / reps,
                )
            )
    return tuple(rows)


def article_numbers() -> ArticleNumbers:
    """The intraclass correlation, spread, design effect and variance ratio the article quotes."""
    ratio = variance_ratio(ARTICLE_SESSIONS, STRONG_HETEROGENEITY)
    return ArticleNumbers(
        intraclass_correlation=intraclass_correlation(STRONG_HETEROGENEITY),
        propensity_sd=propensity_sd(STRONG_HETEROGENEITY),
        design_effect=design_effect(ARTICLE_SESSIONS, STRONG_HETEROGENEITY),
        variance_ratio=ratio,
        standard_error_shortfall=1 - 1 / sqrt(ratio),
    )


def example_payload() -> RatioMetricSummary:
    """Return the figure's A/A false positive rates and the article's closed forms."""
    return RatioMetricSummary(rows=false_positive_rates(), article=article_numbers())
