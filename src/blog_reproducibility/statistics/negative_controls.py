"""Negative control outcomes, for the article on finding bias you cannot see.

Users have an unobserved engagement ``u``, standard normal, and adopt a feature
with probability ``expit(a u - 0.5)``. Engagement drives two outcomes: the real
one, ``y = 5 + c_y u + 0.3 A + e``, which the feature raises by 0.3, and a
negative control, ``n = 5 + c_n u + e'``, which the feature cannot reach, with
unit-variance noise in both. Comparing adopters with everyone else estimates
the effect plus ``c_y g`` and finds ``c_n g`` on the control, where

    g = E[u | adopt] - E[u | not] = E[u expit(a u - 0.5)] / (p (1 - p))

is the engagement gap that adoption selects and ``p = E[expit(a u - 0.5)]`` the
adoption rate; both are one-dimensional normal integrals, evaluated here by
quadrature. With ``a = 1`` the adopters' density ``phi(u) expit(u - 1/2)`` is
symmetric about one half, so their mean engagement is exactly 0.5 and
``g = 1 / (2 (1 - p))``: ``p = 0.398`` and ``g = 0.831``. So the control's
signal detects the bias whatever its size, but equals it only when
``c_n = c_y``: their ratio is the ratio of the two exposures, and subtracting
the control leaves ``0.3 + (c_y - c_n) g``. The large-sample standard error of a
difference is ``sqrt(v_1 / (p N) + v_0 / ((1 - p) N))``, with ``v`` the outcome's
variance within each group, ``1 + c^2 Var(u | group)``, which gives the chance a
control is flagged at 1.96 standard errors. An A/A test on an outcome with unit
spread and a hidden bias ``b`` has standard error ``sqrt(2 / n)`` and flags with
probability ``Phi(b / se - 1.96) + Phi(-b / se - 1.96)``.

The figure raises the outcome's exposure from 0 to 1.6 in 17 steps with the
control's held at 0.6, averaging 12 populations of 30,000 users at each step,
all drawn in turn from one generator seeded at 67. It is reproduced draw for
draw. The article's tables come from other designs and generators (40,000 users
from seed 67 at an exposure of 0.8, then seeds 3, 5, 7 and 11) and are not
pinned; its quantities are computed here in closed form and the tests compare
its printed simulations with them.
"""

from dataclasses import dataclass
from math import exp, pi, sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, special, stats

from blog_reproducibility.common.validation import count, non_negative, positive, real

__all__ = [
    "AA_HIDDEN_BIASES",
    "AA_USERS_PER_ARM",
    "ADOPTION_EXPOSURE",
    "ADOPTION_OFFSET",
    "ARTICLE_OUTCOME_EXPOSURE",
    "ARTICLE_USERS",
    "BASELINE",
    "CONFOUNDING_PAIRS",
    "CONTROL_EXPOSURE",
    "CORRECTION_EXPOSURES",
    "CRITICAL_Z",
    "DETECTION_EXPOSURE",
    "DETECTION_USERS",
    "RUNS_PER_STRENGTH",
    "SEED",
    "STRENGTHS",
    "TRUE_EFFECT",
    "USERS",
    "AATestRow",
    "ArticleNumbers",
    "ConfoundingRow",
    "CorrectionRow",
    "DetectionRow",
    "EngagementSelection",
    "NegativeControlSummary",
    "Population",
    "TrackingCurve",
    "aa_flag_rate",
    "article_numbers",
    "bias_and_signal",
    "detection_rate",
    "difference_standard_error",
    "draw_population",
    "engagement_selection",
    "example_payload",
    "naive_difference",
    "tracking_curve",
    "tracking_point",
]

SEED: Final[int] = 67
USERS: Final[int] = 30_000
TRUE_EFFECT: Final[float] = 0.30
BASELINE: Final[float] = 5.0
CONTROL_EXPOSURE: Final[float] = 0.60
ADOPTION_EXPOSURE: Final[float] = 1.0
ADOPTION_OFFSET: Final[float] = 0.5
# The figure: the outcome's exposure from 0 to 1.6, 12 populations at each step.
STRENGTHS: Final[tuple[float, ...]] = tuple(float(s) for s in np.linspace(0.0, 1.6, 17))
RUNS_PER_STRENGTH: Final[int] = 12
CRITICAL_Z: Final[float] = 1.96
# The article's designs.
ARTICLE_USERS: Final[int] = 40_000
ARTICLE_OUTCOME_EXPOSURE: Final[float] = 0.80
# Exposures of the outcome and of the control, in the order of the article's table.
CONFOUNDING_PAIRS: Final[tuple[tuple[float, float], ...]] = (
    (0.8, 0.6),
    (0.8, 0.8),
    (0.4, 0.6),
    (1.2, 0.6),
    (0.0, 0.6),
)
DETECTION_EXPOSURE: Final[float] = 0.25
DETECTION_USERS: Final[tuple[int, ...]] = (1000, 5000, 20_000, 100_000)
CORRECTION_EXPOSURES: Final[tuple[float, ...]] = (0.6, 0.3, 0.9)
AA_USERS_PER_ARM: Final[tuple[int, ...]] = (2000, 10_000, 50_000)
AA_HIDDEN_BIASES: Final[tuple[float, ...]] = (0.0, 0.02, 0.05)


@dataclass(frozen=True, slots=True)
class Population:
    """One simulated population: who adopted, both outcomes and the hidden engagement."""

    adopted: NDArray[np.bool_]
    outcome: NDArray[np.float64]
    control: NDArray[np.float64]
    engagement: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrackingCurve:
    """The figure: bias and control signal against the outcome's exposure, and in expectation."""

    users: int
    runs: int
    control_exposure: float
    strengths: tuple[float, ...]
    biases: tuple[float, ...]
    signals: tuple[float, ...]
    expected_biases: tuple[float, ...]
    expected_signal: float


@dataclass(frozen=True, slots=True)
class EngagementSelection:
    """How adoption sorts users by engagement, in closed form."""

    adoption_rate: float
    adopter_mean: float
    other_mean: float
    adopter_variance: float
    other_variance: float
    gap: float


@dataclass(frozen=True, slots=True)
class ConfoundingRow:
    """Expected bias in the outcome and signal on the control at one pair of exposures."""

    outcome_exposure: float
    control_exposure: float
    bias: float
    signal: float
    ratio: float


@dataclass(frozen=True, slots=True)
class DetectionRow:
    """Expected control signal and the chance it is flagged, at one number of users."""

    users: int
    signal: float
    standard_error: float
    flag_rate: float


@dataclass(frozen=True, slots=True)
class CorrectionRow:
    """The raw estimate and the estimate after subtracting the control, in expectation."""

    control_exposure: float
    share_of_confounding: float
    raw: float
    corrected: float


@dataclass(frozen=True, slots=True)
class AATestRow:
    """Chance an A/A test flags a hidden bias, for an outcome with unit spread."""

    users_per_arm: int
    hidden_bias: float
    flag_rate: float


@dataclass(frozen=True, slots=True)
class ArticleNumbers:
    """The article's quantities in closed form, for comparison with its simulations."""

    adoption_rate: float
    naive_effect: float
    naive_standard_error: float
    control_signal: float
    control_standard_error: float
    confounding: tuple[ConfoundingRow, ...]
    detection: tuple[DetectionRow, ...]
    correction: tuple[CorrectionRow, ...]
    aa_tests: tuple[AATestRow, ...]


@dataclass(frozen=True, slots=True)
class NegativeControlSummary:
    """The figure's curve, the selection behind it and the article's closed forms."""

    curve: TrackingCurve
    selection: EngagementSelection
    article: ArticleNumbers


def draw_population(
    rng: np.random.Generator,
    users: int = USERS,
    *,
    outcome_exposure: float = ARTICLE_OUTCOME_EXPOSURE,
    control_exposure: float = CONTROL_EXPOSURE,
    effect: float = TRUE_EFFECT,
    adoption_exposure: float = ADOPTION_EXPOSURE,
) -> Population:
    """Draw engagement, adoption, the outcome's noise and the control's noise, in that order."""
    n = count(users, name="users", minimum=4)
    c_y = real(outcome_exposure, name="outcome_exposure")
    c_n = real(control_exposure, name="control_exposure")
    tau = real(effect, name="effect")
    a = real(adoption_exposure, name="adoption_exposure")
    u = rng.normal(0, 1, n)
    adopted = rng.random(n) < 1 / (1 + np.exp(-(a * u - ADOPTION_OFFSET)))
    outcome = BASELINE + c_y * u + tau * adopted + rng.normal(0, 1, n)
    control = BASELINE + c_n * u + rng.normal(0, 1, n)
    return Population(adopted=adopted, outcome=outcome, control=control, engagement=u)


def _groups(
    adopted: ArrayLike, values: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mask = np.asarray(adopted)
    data = np.asarray(values, dtype=np.float64)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise ValueError("adopted must be a one-dimensional boolean array")
    if data.shape != mask.shape:
        raise ValueError("values must have one entry per user")
    if not np.all(np.isfinite(data)):
        raise ValueError("values must be finite")
    if min(int(mask.sum()), int((~mask).sum())) < 2:
        raise ValueError("each group needs at least two users")
    return data[~mask], data[mask]


def naive_difference(adopted: ArrayLike, values: ArrayLike) -> tuple[float, float]:
    """Adopters' mean minus everyone else's, with its unpooled standard error, as the article."""
    others, adopters = _groups(adopted, values)
    difference = adopters.mean() - others.mean()
    se = np.sqrt(others.var(ddof=1) / others.size + adopters.var(ddof=1) / adopters.size)
    return float(difference), float(se)


def bias_and_signal(population: Population, effect: float = TRUE_EFFECT) -> tuple[float, float]:
    """The naive estimate's error on the outcome and its estimate on the control, as the figure."""
    others, adopters = _groups(population.adopted, population.outcome)
    bias = (adopters.mean() - others.mean()) - real(effect, name="effect")
    control_others, control_adopters = _groups(population.adopted, population.control)
    signal = control_adopters.mean() - control_others.mean()
    return float(bias), float(signal)


def tracking_point(
    rng: np.random.Generator,
    outcome_exposure: float,
    *,
    users: int = USERS,
    runs: int = RUNS_PER_STRENGTH,
    control_exposure: float = CONTROL_EXPOSURE,
) -> tuple[float, float]:
    """Mean bias and control signal over ``runs`` populations drawn in turn from ``rng``."""
    draws = count(runs, name="runs", minimum=1)
    pairs = [
        bias_and_signal(
            draw_population(
                rng, users, outcome_exposure=outcome_exposure, control_exposure=control_exposure
            )
        )
        for _ in range(draws)
    ]
    bias, signal = np.mean(pairs, axis=0)
    return float(bias), float(signal)


def tracking_curve(
    seed: int = SEED,
    *,
    strengths: tuple[float, ...] = STRENGTHS,
    users: int = USERS,
    runs: int = RUNS_PER_STRENGTH,
    control_exposure: float = CONTROL_EXPOSURE,
) -> TrackingCurve:
    """The figure: every exposure in turn from one generator, with the closed forms beside it."""
    if not strengths:
        raise ValueError("strengths must not be empty")
    exposures = tuple(real(s, name="strength") for s in strengths)
    c_n = real(control_exposure, name="control_exposure")
    rng = np.random.default_rng(count(seed, name="seed"))
    points = [
        tracking_point(rng, s, users=users, runs=runs, control_exposure=c_n) for s in exposures
    ]
    gap = engagement_selection().gap
    return TrackingCurve(
        users=users,
        runs=runs,
        control_exposure=c_n,
        strengths=exposures,
        biases=tuple(bias for bias, _ in points),
        signals=tuple(signal for _, signal in points),
        expected_biases=tuple(s * gap for s in exposures),
        expected_signal=c_n * gap,
    )


def _normal_moment(power: int, adoption_exposure: float, offset: float) -> float:
    """``E[u^power expit(a u - offset)]`` for standard normal ``u``."""

    def integrand(u: float) -> float:
        return u**power * float(special.expit(adoption_exposure * u - offset)) * exp(-u * u / 2)

    value, _ = integrate.quad(integrand, -np.inf, np.inf, epsabs=1e-13, epsrel=1e-12)
    return float(value) / sqrt(2 * pi)


def engagement_selection(
    adoption_exposure: float = ADOPTION_EXPOSURE, offset: float = ADOPTION_OFFSET
) -> EngagementSelection:
    """Adoption rate and the mean and variance of engagement among adopters and the rest."""
    a = real(adoption_exposure, name="adoption_exposure")
    c = real(offset, name="offset")
    p = _normal_moment(0, a, c)
    first = _normal_moment(1, a, c)
    second = _normal_moment(2, a, c)
    adopter_mean, other_mean = first / p, -first / (1 - p)
    return EngagementSelection(
        adoption_rate=p,
        adopter_mean=adopter_mean,
        other_mean=other_mean,
        adopter_variance=second / p - adopter_mean**2,
        other_variance=(1 - second) / (1 - p) - other_mean**2,
        gap=adopter_mean - other_mean,
    )


def difference_standard_error(
    users: int, exposure: float, selection: EngagementSelection | None = None
) -> float:
    """Large-sample standard error of the naive difference on an outcome with this exposure."""
    n = count(users, name="users", minimum=1)
    c = real(exposure, name="exposure")
    s = selection if selection is not None else engagement_selection()
    adopters = (1 + c * c * s.adopter_variance) / (s.adoption_rate * n)
    others = (1 + c * c * s.other_variance) / ((1 - s.adoption_rate) * n)
    return sqrt(adopters + others)


def _two_sided_rate(shift: float, critical: float) -> float:
    return float(stats.norm.cdf(shift - critical) + stats.norm.cdf(-shift - critical))


def detection_rate(
    users: int,
    exposure: float,
    *,
    selection: EngagementSelection | None = None,
    critical: float = CRITICAL_Z,
) -> float:
    """Chance the control's naive difference is flagged as non-zero, in large samples."""
    s = selection if selection is not None else engagement_selection()
    signal = real(exposure, name="exposure") * s.gap
    se = difference_standard_error(users, exposure, s)
    return _two_sided_rate(signal / se, positive(critical, name="critical"))


def aa_flag_rate(
    users_per_arm: int,
    hidden_bias: float,
    *,
    sd: float = 1.0,
    critical: float = CRITICAL_Z,
) -> float:
    """Chance a two-sided z-test flags an A/A difference: ``se = sd sqrt(2 / n)``."""
    n = count(users_per_arm, name="users_per_arm", minimum=1)
    se = positive(sd, name="sd") * sqrt(2 / n)
    shift = non_negative(hidden_bias, name="hidden_bias") / se
    return _two_sided_rate(shift, positive(critical, name="critical"))


def article_numbers(selection: EngagementSelection | None = None) -> ArticleNumbers:
    """Every table in the article, in expectation."""
    s = selection if selection is not None else engagement_selection()
    g = s.gap
    c_y = ARTICLE_OUTCOME_EXPOSURE
    return ArticleNumbers(
        adoption_rate=s.adoption_rate,
        naive_effect=TRUE_EFFECT + c_y * g,
        naive_standard_error=difference_standard_error(ARTICLE_USERS, c_y, s),
        control_signal=CONTROL_EXPOSURE * g,
        control_standard_error=difference_standard_error(ARTICLE_USERS, CONTROL_EXPOSURE, s),
        confounding=tuple(
            ConfoundingRow(
                outcome_exposure=outcome,
                control_exposure=control,
                bias=outcome * g,
                signal=control * g,
                ratio=outcome / control,
            )
            for outcome, control in CONFOUNDING_PAIRS
        ),
        detection=tuple(
            DetectionRow(
                users=n,
                signal=DETECTION_EXPOSURE * g,
                standard_error=difference_standard_error(n, DETECTION_EXPOSURE, s),
                flag_rate=detection_rate(n, DETECTION_EXPOSURE, selection=s),
            )
            for n in DETECTION_USERS
        ),
        correction=tuple(
            CorrectionRow(
                control_exposure=control,
                share_of_confounding=control / c_y,
                raw=TRUE_EFFECT + c_y * g,
                corrected=TRUE_EFFECT + (c_y - control) * g,
            )
            for control in CORRECTION_EXPOSURES
        ),
        aa_tests=tuple(
            AATestRow(users_per_arm=n, hidden_bias=b, flag_rate=aa_flag_rate(n, b))
            for n in AA_USERS_PER_ARM
            for b in AA_HIDDEN_BIASES
        ),
    )


def example_payload() -> NegativeControlSummary:
    """Return the figure's curve (204 populations of 30,000) and the article's closed forms."""
    selection = engagement_selection()
    return NegativeControlSummary(
        curve=tracking_curve(),
        selection=selection,
        article=article_numbers(selection),
    )
