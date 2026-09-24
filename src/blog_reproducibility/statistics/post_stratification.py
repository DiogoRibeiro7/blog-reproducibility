"""Post-stratification and its cost in precision, for the article on weighting a survey.

A population falls into four age bands with shares ``s_a`` of 28, 27, 25 and 20
percent and three regions, independently, and a satisfaction score has a mean
that rises by 0.6 per age band and 0.4 per region. People answer with a
probability that is logistic in their age band, ``p_a = 1 / (1 + exp(1.6 -
b a))``, so the older answer more and the unweighted mean is biased upward.
Post-stratification on age gives every respondent in band ``a`` the weight
``w_a = s_a / r_a``, the band's population share over its share of respondents.

In the limit of a large population the respondents' band shares are
``r_a = s_a p_a / P`` with ``P = sum_a s_a p_a``, so ``w_a = P / p_a``. Kish's
effective sample ``(sum w)^2 / sum w^2``, as a share of the respondents, is then

    n_eff / n = 1 / (E[p] E[1 / p]),

both expectations over the population's age bands, which is one when the
response rate is the same everywhere and falls as it spreads out. The largest
weight over the smallest is ``max p / min p``. The unweighted bias is ``sum_a
r_a mu_a - sum_a s_a mu_a`` with ``mu_a`` the band's mean score, and weighting
on age removes all of it, because response depends on nothing else.

The article's last table adds an attitude ``k``, standard normal, that raises the
score by 0.7 per unit and, in its second column, the log-odds of answering by
0.5 per unit. Within a band the respondents' mean attitude is then
``kappa_a = E[k sigma(eta_a + 0.5 k)] / E[sigma(eta_a + 0.5 k)]``, one integral
over a normal each, and weighting on age leaves a bias of ``0.7 sum_a s_a
kappa_a``, 0.227 of an unweighted 0.457. Weighting on age and region jointly, or
raking to both, leaves the same in the limit, since the region carries no
information about the attitude.

The figure is one computation from a generator seeded at 73: the age band,
region and score noise of 120,000 people, then one draw per person of whether
they answer at each of 16 response slopes from 0.05 to 1.6. It plots the
effective sample and the bias surviving weighting, as shares of the respondents
and of the unweighted bias, against the largest weight over the smallest. It is
reproduced draw for draw. The article's tables use a population of 200,000 with
the attitude and other seeds, so they are compared with the closed forms rather
than reproduced.
"""

from dataclasses import dataclass
from math import exp, pi, sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate

from blog_reproducibility.common.validation import count, non_negative, positive, real

__all__ = [
    "AGE_SHARES",
    "ARTICLE_SLOPES",
    "ATTITUDE_EFFECT",
    "ATTITUDE_RESPONSE",
    "CELL_MEANS",
    "INTERCEPT",
    "NOISE_SD",
    "POPULATION",
    "REGION_SHARES",
    "SEED",
    "SLOPE_POINTS",
    "SLOPE_START",
    "SLOPE_STOP",
    "SURVEY_SLOPE",
    "AttitudeRow",
    "ImbalanceRow",
    "Population",
    "PostStratificationSummary",
    "WeightingCurve",
    "attitude_row",
    "band_means",
    "draw_population",
    "example_payload",
    "expected_effective_share",
    "expected_unweighted_bias",
    "imbalance_row",
    "kish_effective_size",
    "post_stratification_weights",
    "respond",
    "respondent_shares",
    "response_probabilities",
    "response_rate",
    "weighting_curve",
]

SEED: Final[int] = 73
POPULATION: Final[int] = 120_000
AGE_SHARES: Final[tuple[float, ...]] = (0.28, 0.27, 0.25, 0.20)
REGION_SHARES: Final[tuple[float, ...]] = (0.50, 0.30, 0.20)
# Mean score by age band (rows) and region (columns).
CELL_MEANS: Final[tuple[tuple[float, ...], ...]] = (
    (6.0, 6.4, 6.8),
    (6.6, 7.0, 7.4),
    (7.2, 7.6, 8.0),
    (7.8, 8.2, 8.6),
)
NOISE_SD: Final[float] = 1.2
# Log-odds of answering: INTERCEPT + slope x age band.
INTERCEPT: Final[float] = -1.6
SLOPE_START: Final[float] = 0.05
SLOPE_STOP: Final[float] = 1.6
SLOPE_POINTS: Final[int] = 16
# The article: its survey at slope 0.55, its imbalance table, and the unmeasured attitude.
SURVEY_SLOPE: Final[float] = 0.55
ARTICLE_SLOPES: Final[tuple[float, ...]] = (0.2, 0.55, 1.0, 1.5)
ATTITUDE_EFFECT: Final[float] = 0.7
ATTITUDE_RESPONSE: Final[float] = 0.5


@dataclass(frozen=True, slots=True)
class Population:
    """Each person's age band, region and score."""

    age: NDArray[np.int64]
    region: NDArray[np.int64]
    score: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WeightingCurve:
    """The figure: effective sample, surviving bias and weight ratio at each response slope."""

    slopes: tuple[float, ...]
    population_mean: float
    respondents: tuple[int, ...]
    effective_share: tuple[float, ...]
    surviving_bias: tuple[float, ...]
    weight_ratio: tuple[float, ...]
    unweighted_bias: tuple[float, ...]
    weighted_bias: tuple[float, ...]
    expected_effective_share: tuple[float, ...]
    expected_weight_ratio: tuple[float, ...]
    expected_unweighted_bias: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ImbalanceRow:
    """One response slope in closed form: who answers, the weights and what they cost."""

    slope: float
    response_rate: float
    respondent_shares: tuple[float, ...]
    youngest_response: float
    oldest_response: float
    weight_ratio: float
    effective_share: float
    unweighted_bias: float


@dataclass(frozen=True, slots=True)
class AttitudeRow:
    """Biases when an unmeasured attitude also drives answering, in closed form."""

    attitude_response: float
    unweighted_bias: float
    weighted_bias: float


@dataclass(frozen=True, slots=True)
class PostStratificationSummary:
    """The figure's curves and the closed forms behind the article's tables."""

    curve: WeightingCurve
    population_mean: float
    survey: ImbalanceRow
    imbalance: tuple[ImbalanceRow, ...]
    attitude: tuple[AttitudeRow, ...]


def _logistic(x: float) -> float:
    return 1 / (1 + exp(-x))


def band_means() -> tuple[float, ...]:
    """Mean score of each age band, over its regions."""
    return tuple(sum(m * r for m, r in zip(row, REGION_SHARES, strict=True)) for row in CELL_MEANS)


def response_probabilities(slope: float, *, intercept: float = INTERCEPT) -> tuple[float, ...]:
    """Chance of answering in each age band: ``1 / (1 + exp(-(intercept + slope a)))``."""
    b, c = real(slope, name="slope"), real(intercept, name="intercept")
    return tuple(_logistic(c + b * a) for a in range(len(AGE_SHARES)))


def response_rate(slope: float) -> float:
    """Share of the population that answers: ``sum_a s_a p_a``."""
    return sum(s * p for s, p in zip(AGE_SHARES, response_probabilities(slope), strict=True))


def respondent_shares(slope: float) -> tuple[float, ...]:
    """Age bands' shares of the respondents: ``s_a p_a / P``."""
    rate = response_rate(slope)
    return tuple(
        s * p / rate for s, p in zip(AGE_SHARES, response_probabilities(slope), strict=True)
    )


def expected_effective_share(slope: float) -> float:
    """Kish's effective sample over the respondents, weighting on age: ``1 / (E[p] E[1/p])``."""
    probabilities = response_probabilities(slope)
    inverse = sum(s / p for s, p in zip(AGE_SHARES, probabilities, strict=True))
    return 1 / (response_rate(slope) * inverse)


def expected_unweighted_bias(slope: float) -> float:
    """Respondents' mean score less the population's."""
    means = band_means()
    shares = respondent_shares(slope)
    return sum(r * m for r, m in zip(shares, means, strict=True)) - sum(
        s * m for s, m in zip(AGE_SHARES, means, strict=True)
    )


def imbalance_row(slope: float) -> ImbalanceRow:
    """Who answers at one response slope, and what weighting them costs, in closed form."""
    probabilities = response_probabilities(slope)
    return ImbalanceRow(
        slope=slope,
        response_rate=response_rate(slope),
        respondent_shares=respondent_shares(slope),
        youngest_response=probabilities[0],
        oldest_response=probabilities[-1],
        weight_ratio=max(probabilities) / min(probabilities),
        effective_share=expected_effective_share(slope),
        unweighted_bias=expected_unweighted_bias(slope),
    )


def attitude_row(
    attitude_response: float,
    *,
    slope: float = SURVEY_SLOPE,
    attitude_effect: float = ATTITUDE_EFFECT,
) -> AttitudeRow:
    """Unweighted bias and the bias left by weighting on age, with an unmeasured attitude.

    The attitude is standard normal, adds ``attitude_effect`` per unit to the
    score and ``attitude_response`` per unit to the log-odds of answering.
    """
    b = non_negative(attitude_response, name="attitude_response")
    effect = real(attitude_effect, name="attitude_effect")

    def density(k: float) -> float:
        return exp(-k * k / 2) / sqrt(2 * pi)

    answer, attitude = [], []
    for eta in (real(slope, name="slope") * a + INTERCEPT for a in range(len(AGE_SHARES))):
        rate = integrate.quad(
            lambda k, e=eta: _logistic(e + b * k) * density(k), -12, 12, epsabs=1e-14
        )[0]
        lean = integrate.quad(
            lambda k, e=eta: k * _logistic(e + b * k) * density(k), -12, 12, epsabs=1e-14
        )[0]
        answer.append(float(rate))
        attitude.append(float(lean) / float(rate))
    total = sum(s * p for s, p in zip(AGE_SHARES, answer, strict=True))
    means = band_means()
    respondents = sum(
        s * p / total * (m + effect * k)
        for s, p, m, k in zip(AGE_SHARES, answer, means, attitude, strict=True)
    )
    population = sum(s * m for s, m in zip(AGE_SHARES, means, strict=True))
    return AttitudeRow(
        attitude_response=b,
        unweighted_bias=respondents - population,
        weighted_bias=effect * sum(s * k for s, k in zip(AGE_SHARES, attitude, strict=True)),
    )


def draw_population(rng: np.random.Generator, size: int = POPULATION) -> Population:
    """Draw every age band, then every region, then the score noise, as the website."""
    n = count(size, name="size", minimum=1)
    age = rng.choice(len(AGE_SHARES), n, p=list(AGE_SHARES)).astype(np.int64)
    region = rng.choice(len(REGION_SHARES), n, p=list(REGION_SHARES)).astype(np.int64)
    means = np.asarray(CELL_MEANS, dtype=np.float64)
    score = means[age, region] + rng.normal(0, NOISE_SD, n)
    return Population(age=age, region=region, score=score)


def respond(rng: np.random.Generator, age: ArrayLike, slope: float) -> NDArray[np.bool_]:
    """One uniform draw per person against the logistic chance of answering."""
    bands = np.asarray(age)
    b = real(slope, name="slope")
    answered: NDArray[np.bool_] = rng.random(bands.size) < 1 / (
        1 + np.exp(-(INTERCEPT + b * bands))
    )
    return answered


def post_stratification_weights(
    groups: ArrayLike, shares: tuple[float, ...] = AGE_SHARES
) -> NDArray[np.float64]:
    """Each respondent's group share in the population over its share of the respondents."""
    labels = np.asarray(groups)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional array")
    weights = np.zeros(labels.size)
    for group, share in enumerate(shares):
        members = labels == group
        if members.sum():
            weights[members] = positive(share, name="share") / (members.sum() / members.size)
    return weights


def kish_effective_size(weights: ArrayLike) -> float:
    """Kish's effective sample size: ``(sum w)^2 / sum w^2``."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0 or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be non-negative, one-dimensional and not all zero")
    return float(w.sum() ** 2 / np.sum(w**2))


def weighting_curve(
    seed: int = SEED, *, size: int = POPULATION, slopes: tuple[float, ...] | None = None
) -> WeightingCurve:
    """The figure: one population, then one response draw per slope from the same generator."""
    rng = np.random.default_rng(count(seed, name="seed"))
    people = draw_population(rng, size)
    truth = float(people.score.mean())
    grid = (
        tuple(float(b) for b in np.linspace(SLOPE_START, SLOPE_STOP, SLOPE_POINTS))
        if slopes is None
        else tuple(real(b, name="slope") for b in slopes)
    )
    if not grid:
        raise ValueError("slopes must not be empty")
    respondents, effective, surviving, ratio, raw, weighted = [], [], [], [], [], []
    for slope in grid:
        answered = respond(rng, people.age, slope)
        weights = post_stratification_weights(people.age[answered])
        scores = people.score[answered]
        raw_bias = float(scores.mean()) - truth
        weighted_bias = float(np.average(scores, weights=weights)) - truth
        respondents.append(int(answered.sum()))
        effective.append(kish_effective_size(weights) / int(answered.sum()))
        surviving.append(weighted_bias / raw_bias)
        ratio.append(float(weights.max() / weights.min()))
        raw.append(raw_bias)
        weighted.append(weighted_bias)
    return WeightingCurve(
        slopes=grid,
        population_mean=truth,
        respondents=tuple(respondents),
        effective_share=tuple(effective),
        surviving_bias=tuple(surviving),
        weight_ratio=tuple(ratio),
        unweighted_bias=tuple(raw),
        weighted_bias=tuple(weighted),
        expected_effective_share=tuple(expected_effective_share(b) for b in grid),
        expected_weight_ratio=tuple(imbalance_row(b).weight_ratio for b in grid),
        expected_unweighted_bias=tuple(expected_unweighted_bias(b) for b in grid),
    )


def example_payload() -> PostStratificationSummary:
    """Return the figure's curves and the closed forms behind the article's tables."""
    return PostStratificationSummary(
        curve=weighting_curve(),
        population_mean=sum(s * m for s, m in zip(AGE_SHARES, band_means(), strict=True)),
        survey=imbalance_row(SURVEY_SLOPE),
        imbalance=tuple(imbalance_row(b) for b in ARTICLE_SLOPES),
        attitude=tuple(attitude_row(b) for b in (0.0, ATTITUDE_RESPONSE)),
    )
