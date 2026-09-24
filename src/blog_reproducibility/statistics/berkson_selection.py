"""Collider bias from selecting on a sum, for the article on Berkson's paradox in operational data.

Support tickets have a severity and a customer value, independent standard
normals, and are escalated when a score ``S = severity + value + d + e`` clears
the quantile that keeps a share ``s`` of them, with judgement noise ``e`` of
standard deviation 0.5 and, in the article's second version, an unrecorded
difficulty ``d``, standard normal. Everything is jointly normal and the
selection depends on ``S`` alone, so each input splits into its regression on
``S`` and a residual independent of ``S``, and truncation changes only the
first part. With ``sigma^2 = Var(S)``, ``a = Phi^{-1}(1 - s)`` and the inverse
Mills ratio ``lambda = phi(a) / s``, the score's variance among the selected
is ``1 - k`` times its variance in all tickets, with ``k = lambda (lambda - a)``,
and for inputs of unit weight

    Cov(X_i, X_j | selected) = delta_ij Var(X_i) - Var(X_i) Var(X_j) k / sigma^2.

So two independent inputs of unit variance have correlation
``-k / (sigma^2 - k)`` among the selected: 0 before any selection, -0.39 at half,
and approaching ``-1 / (1 + 0.25) = -0.8`` as the rule becomes strict; without
the judgement noise, as for hiring on talent plus polish, the limit is -1.

Least squares on the selected tickets converges to the covariances above. With
resolution time ``4 + 3 severity + h d`` plus noise of standard deviation 2 and
``q = k / sigma^2``, the regression on both attributes converges to
``3 - h q / (1 - 2 q)`` for severity and ``-h q / (1 - 2 q)`` for value, which is
``(3, 0)`` when the selection used only recorded inputs (``h = 0``) and
``(2.02, -0.98)`` when the agent's judgement of difficulty drove both
(``h = 2``). Value alone picks up ``3`` times the induced correlation. The
selected means are ``lambda / sigma`` for each unit-weight input, which gives
the fitted intercept and so the error of the selected model on every ticket in
closed form. Selecting when either input exceeds 1 has closed-form moments
too, from the quadrant it leaves out.

The figure and the article are one computation: 200,000 tickets from a
generator seeded at 0 and the correlation among the top 50, 30, 15, 5, 2 and 1
percent by score. The article's later blocks continue the same generator
(resolution hours, then difficulty, its score and hours, then talent and polish
for hiring), so every number it prints is reproduced here exactly.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from blog_reproducibility.common.validation import count, non_negative, probability, real

__all__ = [
    "BASE_HOURS",
    "ESCALATED_SHARE",
    "EITHER_THRESHOLD",
    "HIGH_VALUE",
    "HIRING_SHARES",
    "HOURS_NOISE_SD",
    "HOURS_PER_DIFFICULTY",
    "HOURS_PER_SEVERITY",
    "JUDGEMENT_SD",
    "LOW_SEVERITY",
    "SEED",
    "SELECTION_SHARES",
    "TICKETS",
    "BerksonSummary",
    "EitherRule",
    "HiringRow",
    "ModelTransfer",
    "RegressionRow",
    "SelectedFit",
    "SelectionCurve",
    "Tickets",
    "correlation",
    "draw_tickets",
    "either_rule_correlation",
    "either_rule_share",
    "example_payload",
    "least_squares_slopes",
    "selected_correlation",
    "selected_least_squares",
    "selection_curve",
    "top_share",
    "transfer_errors",
    "truncation_factor",
]

SEED: Final[int] = 0
TICKETS: Final[int] = 200_000
JUDGEMENT_SD: Final[float] = 0.5
# Shares of tickets escalated, in the figure's order: more selective to the right.
SELECTION_SHARES: Final[tuple[float, ...]] = (0.5, 0.3, 0.15, 0.05, 0.02, 0.01)
ESCALATED_SHARE: Final[float] = 0.15
BASE_HOURS: Final[float] = 4.0
HOURS_PER_SEVERITY: Final[float] = 3.0
HOURS_PER_DIFFICULTY: Final[float] = 2.0
HOURS_NOISE_SD: Final[float] = 2.0
# The subgroups on which the article compares the two models.
HIGH_VALUE: Final[float] = 1.5
LOW_SEVERITY: Final[float] = -1.0
EITHER_THRESHOLD: Final[float] = 1.0
HIRING_SHARES: Final[tuple[float, ...]] = (0.1, 0.01)


@dataclass(frozen=True, slots=True)
class Tickets:
    """Severity, customer value and the escalation score of every ticket."""

    severity: NDArray[np.float64]
    value: NDArray[np.float64]
    score: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SelectionCurve:
    """The figure: correlation among the escalated tickets at each share, and its closed form."""

    tickets: int
    population_correlation: float
    shares: tuple[float, ...]
    selected: tuple[int, ...]
    correlations: tuple[float, ...]
    expected_correlations: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RegressionRow:
    """Least-squares slopes of resolution hours, in predictor order, and their limits."""

    label: str
    coefficients: tuple[float, ...]
    expected: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SelectedFit:
    """The limit of least squares of hours on severity and value among the selected tickets."""

    intercept: float
    severity: float
    value: float


@dataclass(frozen=True, slots=True)
class ModelTransfer:
    """Errors on all tickets of a model trained on the escalated ones and of one trained on all."""

    rmse_selected: float
    rmse_population: float
    high_value_error_selected: float
    high_value_error_population: float
    low_severity_error_selected: float
    low_severity_error_population: float
    expected_rmse_selected: float
    expected_rmse_population: float
    expected_high_value_error: float
    expected_low_severity_error: float
    recorded_only_rmse_selected: float
    recorded_only_rmse_population: float


@dataclass(frozen=True, slots=True)
class EitherRule:
    """Escalating when either attribute exceeds a threshold: share kept and correlation."""

    threshold: float
    share: float
    correlation: float
    expected_share: float
    expected_correlation: float


@dataclass(frozen=True, slots=True)
class HiringRow:
    """Correlation of talent and polish among those hired on their sum."""

    share: float
    correlation: float
    expected_correlation: float


@dataclass(frozen=True, slots=True)
class BerksonSummary:
    """The figure's curve and every table in the article, simulated and in closed form."""

    curve: SelectionCurve
    regressions: tuple[RegressionRow, ...]
    hidden_regressions: tuple[RegressionRow, ...]
    transfer: ModelTransfer
    either: EitherRule
    hiring: tuple[HiringRow, ...]


def _share(value: float) -> float:
    return probability(value, name="share", inclusive=False)


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array of two or more values")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} must be finite")
    return data


def draw_tickets(
    rng: np.random.Generator, tickets: int = TICKETS, *, judgement_sd: float = JUDGEMENT_SD
) -> Tickets:
    """Draw severity, then value, then the judgement noise in the score, as the article."""
    n = count(tickets, name="tickets", minimum=2)
    noise = non_negative(judgement_sd, name="judgement_sd")
    severity = rng.normal(0, 1, n)
    value = rng.normal(0, 1, n)
    score = severity + value + rng.normal(0, noise, n)
    return Tickets(severity=severity, value=value, score=score)


def top_share(score: ArrayLike, share: float) -> NDArray[np.bool_]:
    """Tickets whose score is above the quantile that keeps ``share`` of them."""
    values = _vector(score, "score")
    kept: NDArray[np.bool_] = values > np.quantile(values, 1 - _share(share))
    return kept


def correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Pearson correlation, as ``np.corrcoef`` gives it."""
    first, second = _vector(x, "x"), _vector(y, "y")
    if first.shape != second.shape:
        raise ValueError("x and y must have the same length")
    return float(np.corrcoef(first, second)[0, 1])


def truncation_factor(share: float) -> float:
    """``k = lambda (lambda - a)``: the share of its variance a normal loses when cut to its top."""
    s = _share(share)
    a = float(stats.norm.isf(s))
    mills = float(stats.norm.pdf(a)) / s
    return mills * (mills - a)


def selected_correlation(share: float, *, noise_sd: float = JUDGEMENT_SD) -> float:
    """Correlation of two independent standard normals among the top ``s`` of their noisy sum."""
    variance = 2 + non_negative(noise_sd, name="noise_sd") ** 2
    k = truncation_factor(share)
    return -k / (variance - k)


def selection_curve(
    seed: int = SEED, *, tickets: int = TICKETS, shares: tuple[float, ...] = SELECTION_SHARES
) -> SelectionCurve:
    """The figure: one draw of tickets, then the correlation among the top of each share."""
    if not shares:
        raise ValueError("shares must not be empty")
    data = draw_tickets(np.random.default_rng(count(seed, name="seed")), tickets)
    return _curve(data, tuple(_share(s) for s in shares))


def _curve(data: Tickets, shares: tuple[float, ...]) -> SelectionCurve:
    masks = [top_share(data.score, s) for s in shares]
    return SelectionCurve(
        tickets=int(data.score.size),
        population_correlation=correlation(data.severity, data.value),
        shares=shares,
        selected=tuple(int(mask.sum()) for mask in masks),
        correlations=tuple(correlation(data.severity[m], data.value[m]) for m in masks),
        expected_correlations=tuple(selected_correlation(s) for s in shares),
    )


def least_squares_slopes(predictors: ArrayLike, outcome: ArrayLike) -> tuple[float, ...]:
    """Slopes of an ordinary least-squares fit with an intercept, as the article's ``ols``."""
    x = np.asarray(predictors, dtype=np.float64)
    y = _vector(outcome, "outcome")
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("predictors must be a two-dimensional array with one row per outcome")
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0][1:]
    return tuple(float(c) for c in coefficients)


def selected_least_squares(
    share: float,
    *,
    hidden_sd: float = 0.0,
    hidden_hours: float = 0.0,
    noise_sd: float = JUDGEMENT_SD,
) -> SelectedFit:
    """Limit of least squares of hours on severity and value among the top ``s`` by score.

    The score adds an unrecorded input of standard deviation ``hidden_sd`` that
    also adds ``hidden_hours`` per unit to the resolution time.
    """
    v_d = non_negative(hidden_sd, name="hidden_sd") ** 2
    h = real(hidden_hours, name="hidden_hours")
    sigma2 = 2 + v_d + non_negative(noise_sd, name="noise_sd") ** 2
    s = _share(share)
    k = truncation_factor(s)
    q = k / sigma2
    leak = h * v_d * q / (1 - 2 * q)
    severity, value = HOURS_PER_SEVERITY - leak, 0.0 - leak
    # Each unit-weight input has mean lambda / sigma among the selected, the hidden one v_d times.
    shift = float(stats.norm.pdf(stats.norm.isf(s))) / s / sqrt(sigma2)
    mean_hours = BASE_HOURS + (HOURS_PER_SEVERITY + h * v_d) * shift
    return SelectedFit(
        intercept=mean_hours - (severity + value) * shift,
        severity=severity,
        value=value,
    )


def transfer_errors(
    fit: SelectedFit, *, hidden_sd: float = 1.0, hidden_hours: float = HOURS_PER_DIFFICULTY
) -> tuple[float, float, float]:
    """Root mean squared error on all tickets, and mean errors on high-value and low-severity ones.

    The tickets are the population, where severity, value and the hidden input
    are independent, so the error of ``a + b_s severity + b_v value`` is normal
    around ``a - 4`` with every coefficient's shortfall adding its own variance.
    """
    v_d = non_negative(hidden_sd, name="hidden_sd") ** 2
    h = real(hidden_hours, name="hidden_hours")
    offset = fit.intercept - BASE_HOURS
    rmse = sqrt(
        HOURS_NOISE_SD**2
        + (HOURS_PER_SEVERITY - fit.severity) ** 2
        + fit.value**2
        + h * h * v_d
        + offset**2
    )
    high = HIGH_VALUE
    high_value_mean = float(stats.norm.pdf(high) / stats.norm.sf(high))
    low = LOW_SEVERITY
    low_severity_mean = float(-stats.norm.pdf(low) / stats.norm.cdf(low))
    return (
        rmse,
        offset + fit.value * high_value_mean,
        offset + (fit.severity - HOURS_PER_SEVERITY) * low_severity_mean,
    )


def either_rule_share(threshold: float = EITHER_THRESHOLD) -> float:
    """Share of tickets with either attribute above the threshold: ``1 - Phi(t)^2``."""
    t = real(threshold, name="threshold")
    return 1 - float(stats.norm.cdf(t)) ** 2


def either_rule_correlation(threshold: float = EITHER_THRESHOLD) -> float:
    """Correlation of the two attributes among tickets with either above the threshold.

    The rule leaves out the quadrant below ``t`` in both, whose moments factor:
    ``E[X; kept] = phi(t) Phi(t)``, ``E[X^2; kept] = 1 - (Phi(t) - t phi(t)) Phi(t)``
    and ``E[XY; kept] = -phi(t)^2``.
    """
    t = real(threshold, name="threshold")
    density, cdf = float(stats.norm.pdf(t)), float(stats.norm.cdf(t))
    kept = either_rule_share(t)
    mean = density * cdf / kept
    variance = (1 - (cdf - t * density) * cdf) / kept - mean**2
    covariance = -(density**2) / kept - mean**2
    return covariance / variance


def _hours(severity: NDArray[np.float64], noise: NDArray[np.float64]) -> NDArray[np.float64]:
    hours: NDArray[np.float64] = BASE_HOURS + HOURS_PER_SEVERITY * severity + noise
    return hours


def _fit_predict(
    mask: NDArray[np.bool_],
    outcome: NDArray[np.float64],
    severity: NDArray[np.float64],
    value: NDArray[np.float64],
) -> NDArray[np.float64]:
    design = np.column_stack([np.ones(severity.size), severity, value])
    coefficients = np.linalg.lstsq(design[mask], outcome[mask], rcond=None)[0]
    predictions: NDArray[np.float64] = design @ coefficients
    return predictions


def _rmse(outcome: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean((outcome - predictions) ** 2)))


def _mean_error(
    predictions: NDArray[np.float64], outcome: NDArray[np.float64], mask: NDArray[np.bool_]
) -> float:
    return float(np.mean(predictions[mask] - outcome[mask]))


def example_payload(seed: int = SEED) -> BerksonSummary:
    """Run the article's code blocks in turn from one generator and put the closed forms beside."""
    rng = np.random.default_rng(count(seed, name="seed"))
    data = draw_tickets(rng, TICKETS)
    curve = _curve(data, SELECTION_SHARES)
    severity, value, score = data.severity, data.value, data.score
    n = severity.size
    escalated = top_share(score, ESCALATED_SHARE)

    hours = _hours(severity, rng.normal(0, HOURS_NOISE_SD, n))
    both = np.column_stack([severity, value])
    recorded = selected_least_squares(ESCALATED_SHARE)
    marginal = HOURS_PER_SEVERITY * selected_correlation(ESCALATED_SHARE)
    regressions = (
        RegressionRow(
            "all tickets, both attributes",
            least_squares_slopes(both, hours),
            (HOURS_PER_SEVERITY, 0.0),
        ),
        RegressionRow(
            "escalated, both attributes",
            least_squares_slopes(both[escalated], hours[escalated]),
            (recorded.severity, recorded.value),
        ),
        RegressionRow(
            "escalated, severity alone",
            least_squares_slopes(severity[escalated, None], hours[escalated]),
            (HOURS_PER_SEVERITY,),
        ),
        RegressionRow(
            "escalated, value alone",
            least_squares_slopes(value[escalated, None], hours[escalated]),
            (marginal,),
        ),
        RegressionRow(
            "escalated, both attributes and the score",
            least_squares_slopes(np.column_stack([both, score])[escalated], hours[escalated]),
            (HOURS_PER_SEVERITY, 0.0, 0.0),
        ),
    )

    difficulty = rng.normal(0, 1, n)
    hidden_score = severity + value + difficulty + rng.normal(0, JUDGEMENT_SD, n)
    hidden_escalated = top_share(hidden_score, ESCALATED_SHARE)
    hidden_hours = (
        BASE_HOURS
        + HOURS_PER_SEVERITY * severity
        + HOURS_PER_DIFFICULTY * difficulty
        + rng.normal(0, HOURS_NOISE_SD, n)
    )
    hidden = selected_least_squares(
        ESCALATED_SHARE, hidden_sd=1.0, hidden_hours=HOURS_PER_DIFFICULTY
    )
    hidden_regressions = (
        RegressionRow(
            "all tickets, both attributes",
            least_squares_slopes(both, hidden_hours),
            (HOURS_PER_SEVERITY, 0.0),
        ),
        RegressionRow(
            "escalated, both attributes",
            least_squares_slopes(both[hidden_escalated], hidden_hours[hidden_escalated]),
            (hidden.severity, hidden.value),
        ),
    )

    everyone = np.ones(n, dtype=bool)
    predicted_selected = _fit_predict(hidden_escalated, hidden_hours, severity, value)
    predicted_population = _fit_predict(everyone, hidden_hours, severity, value)
    high, low = value > HIGH_VALUE, severity < LOW_SEVERITY
    expected_rmse, expected_high, expected_low = transfer_errors(hidden)
    population_fit = SelectedFit(BASE_HOURS, HOURS_PER_SEVERITY, 0.0)
    transfer = ModelTransfer(
        rmse_selected=_rmse(hidden_hours, predicted_selected),
        rmse_population=_rmse(hidden_hours, predicted_population),
        high_value_error_selected=_mean_error(predicted_selected, hidden_hours, high),
        high_value_error_population=_mean_error(predicted_population, hidden_hours, high),
        low_severity_error_selected=_mean_error(predicted_selected, hidden_hours, low),
        low_severity_error_population=_mean_error(predicted_population, hidden_hours, low),
        expected_rmse_selected=expected_rmse,
        expected_rmse_population=transfer_errors(population_fit)[0],
        expected_high_value_error=expected_high,
        expected_low_severity_error=expected_low,
        recorded_only_rmse_selected=_rmse(hours, _fit_predict(escalated, hours, severity, value)),
        recorded_only_rmse_population=_rmse(hours, _fit_predict(everyone, hours, severity, value)),
    )

    either_kept = (severity > EITHER_THRESHOLD) | (value > EITHER_THRESHOLD)
    either = EitherRule(
        threshold=EITHER_THRESHOLD,
        share=float(either_kept.mean()),
        correlation=correlation(severity[either_kept], value[either_kept]),
        expected_share=either_rule_share(),
        expected_correlation=either_rule_correlation(),
    )

    talent, polish = rng.normal(0, 1, n), rng.normal(0, 1, n)
    merit = talent + polish
    hiring = []
    for share in HIRING_SHARES:
        hired = top_share(merit, share)
        hiring.append(
            HiringRow(
                share=share,
                correlation=correlation(talent[hired], polish[hired]),
                expected_correlation=selected_correlation(share, noise_sd=0.0),
            )
        )

    return BerksonSummary(
        curve=curve,
        regressions=regressions,
        hidden_regressions=hidden_regressions,
        transfer=transfer,
        either=either,
        hiring=tuple(hiring),
    )
