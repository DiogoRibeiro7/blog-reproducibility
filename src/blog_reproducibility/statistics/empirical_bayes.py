"""Empirical Bayes shrinkage across a programme of experiments, for the article on past results.

Each experiment in a programme measures its own true effect ``theta_i`` with a
known standard error ``s_i``. If the true effects are spread around zero with
standard deviation ``tau``, the estimates ``y_i = theta_i + e_i`` have variance
``tau^2 + s_i^2``, so the variance of the estimates minus the mean squared
standard error estimates ``tau^2`` from the programme itself (floored just above
zero). Given ``tau``, the posterior mean of each effect is ``B_i y_i`` with

    B_i = tau^2 / (tau^2 + s_i^2),

the share of the estimate's variance that is signal: precise experiments keep
most of their measured effect and imprecise ones little of it.

With the standard errors uniform on ``[a, b]``, as in the article, the averages
are closed form. The raw estimates have root mean squared error
``sqrt(E s^2) = sqrt((a^2 + a b + b^2) / 3)``; shrinkage with the true ``tau``
leaves the posterior variance, ``tau^2 (1 - E B)``, with
``E B = tau / (b - a) (atan(b / tau) - atan(a / tau))``. For the winners, the
results with ``y_i > 1.96 s_i``, write ``v = tau^2 + s^2`` and
``c = 1.96 s / sqrt(v)``: an experiment with standard error ``s`` wins with
probability ``Phi(-c)``, and ``E[y; win] = sqrt(v) phi(c)`` while
``E[theta; win] = tau^2 phi(c) / sqrt(v)``. Because ``E[theta | y] = B y``,
selecting on the estimate does not bias the shrunk estimate: in expectation the
shrunk winners are worth exactly what the true winners are, which is why
shrinkage undoes the winner's curse. Integrating over a band of standard errors
gives the expected number of winners, their raw and true means, and the factor
by which the raw estimates overstate the truth.

The figure draws one programme of 4,000 experiments from a generator seeded at
71, with ``tau = 0.01`` and standard errors uniform on ``[0.004, 0.020]``, and is
reproduced draw for draw. Its description speaks of two hundred experiments,
the size of the article's programme, but the figure's generator draws 4,000.
The article's tables come from other generators (seed 71 with 200 experiments,
seeds 200 to 599, 7, 400 to 599 and 13) and are not reproduced; the tests check
them against the closed forms here.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import atan, sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, stats

from blog_reproducibility.common.validation import count, positive

__all__ = [
    "CRITICAL_Z",
    "EFFECT_SPREAD",
    "EXPERIMENTS",
    "FORECAST_SPREADS",
    "PRECISION_BANDS",
    "PROGRAMME_SIZE",
    "SEED",
    "SE_HIGH",
    "SE_LOW",
    "VARIANCE_FLOOR",
    "ClosedForms",
    "EmpiricalBayesSummary",
    "FigureSummary",
    "Programme",
    "ShrunkProgramme",
    "WinnerBand",
    "WinnerExpectation",
    "closed_forms",
    "draw_programme",
    "estimate_spread",
    "example_payload",
    "figure_programme",
    "figure_summary",
    "mean_shrinkage_factor",
    "oracle_rmse",
    "raw_rmse",
    "shrink",
    "shrink_programme",
    "shrinkage_factors",
    "significant_winners",
    "winner_expectation",
]

SEED: Final[int] = 71
# The figure pools 4,000 experiments; the article's programme has 200.
EXPERIMENTS: Final[int] = 4000
PROGRAMME_SIZE: Final[int] = 200
EFFECT_SPREAD: Final[float] = 0.010
SE_LOW: Final[float] = 0.004
SE_HIGH: Final[float] = 0.020
CRITICAL_Z: Final[float] = 1.96
# The spread estimate floors the variance of the true effects here, not at zero.
VARIANCE_FLOOR: Final[float] = 1e-12
PRECISION_BANDS: Final[tuple[tuple[float, float], ...]] = (
    (0.004, 0.008),
    (0.008, 0.014),
    (0.014, 0.020),
)
FORECAST_SPREADS: Final[tuple[float, ...]] = (0.004, 0.010, 0.020)


@dataclass(frozen=True, slots=True)
class Programme:
    """True effects, their standard errors and the estimates of one programme of experiments."""

    effects: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    estimates: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ShrunkProgramme:
    """A programme with its estimated spread, shrunk estimates and positive significant results."""

    programme: Programme
    estimated_spread: float
    shrunk: NDArray[np.float64]
    winners: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class WinnerBand:
    """The figure's winners measured with standard errors in one band."""

    lower: float
    upper: float
    winners: int
    raw_mean: float
    true_mean: float
    shrunk_mean: float
    overstatement: float


@dataclass(frozen=True, slots=True)
class FigureSummary:
    """The figure's programme: errors of the raw and shrunk estimates, overall and for winners."""

    experiments: int
    estimated_spread: float
    raw_rmse: float
    shrunk_rmse: float
    winners: int
    winner_raw_mean: float
    winner_true_mean: float
    winner_shrunk_mean: float
    winner_raw_rmse: float
    winner_shrunk_rmse: float
    bands: tuple[WinnerBand, ...]


@dataclass(frozen=True, slots=True)
class WinnerExpectation:
    """Expected winners per programme with standard errors in ``[lower, upper)``, in closed form.

    The means are ratios of expectations: the expected sum over the winners
    divided by their expected number.
    """

    spread: float
    lower: float
    upper: float
    winners: float
    raw_total: float
    true_total: float
    raw_mean: float
    true_mean: float
    overstatement: float


@dataclass(frozen=True, slots=True)
class ClosedForms:
    """The article's model in closed form, for a spread of 0.01 and a programme of 200."""

    mean_shrinkage_factor: float
    raw_rmse: float
    oracle_rmse: float
    rmse_reduction: float
    bands: tuple[WinnerExpectation, ...]
    forecasts: tuple[WinnerExpectation, ...]


@dataclass(frozen=True, slots=True)
class EmpiricalBayesSummary:
    """The figure's simulated programme and the article's closed forms."""

    figure: FigureSummary
    closed_form: ClosedForms


def _standard_error_range(se_low: float, se_high: float) -> tuple[float, float]:
    low = positive(se_low, name="se_low")
    high = positive(se_high, name="se_high")
    if high <= low:
        raise ValueError("se_high must be greater than se_low")
    return low, high


def _checked(
    estimates: ArrayLike, standard_errors: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(estimates, dtype=np.float64)
    errors = np.asarray(standard_errors, dtype=np.float64)
    if values.ndim != 1 or values.shape != errors.shape or values.size < 2:
        raise ValueError("estimates and standard errors must be matching 1-D arrays of two or more")
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(errors))):
        raise ValueError("estimates and standard errors must be finite")
    if np.any(errors <= 0):
        raise ValueError("standard errors must be positive")
    return values, errors


def draw_programme(
    rng: np.random.Generator,
    experiments: int = PROGRAMME_SIZE,
    *,
    spread: float = EFFECT_SPREAD,
    se_low: float = SE_LOW,
    se_high: float = SE_HIGH,
) -> Programme:
    """Draw true effects, then standard errors, then one estimate per experiment.

    The draws are the article's: normal true effects with standard deviation
    ``spread``, standard errors uniform on ``[se_low, se_high)`` because tests
    ran for different lengths, and normal measurement error of that size.
    """
    k = count(experiments, name="experiments", minimum=2)
    tau = positive(spread, name="spread")
    low, high = _standard_error_range(se_low, se_high)
    effects = rng.normal(0.0, tau, k)
    standard_errors = rng.uniform(low, high, k)
    estimates = effects + rng.normal(0, standard_errors)
    return Programme(effects=effects, standard_errors=standard_errors, estimates=estimates)


def estimate_spread(estimates: ArrayLike, standard_errors: ArrayLike) -> float:
    """Spread of the true effects: the variance of the estimates less the mean sampling variance."""
    values, errors = _checked(estimates, standard_errors)
    variance = float(values.var(ddof=1)) - float(np.mean(errors**2))
    return sqrt(max(variance, VARIANCE_FLOOR))


def shrinkage_factors(standard_errors: ArrayLike, spread: float) -> NDArray[np.float64]:
    """Share of each estimate kept: ``tau^2 / (tau^2 + s^2)``."""
    errors = np.asarray(standard_errors, dtype=np.float64)
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError("standard errors must be finite and positive")
    tau = positive(spread, name="spread")
    factors: NDArray[np.float64] = tau**2 / (tau**2 + errors**2)
    return factors


def shrink(estimates: ArrayLike, standard_errors: ArrayLike, spread: float) -> NDArray[np.float64]:
    """Posterior mean of each effect under a normal prior of scale ``spread`` centred on zero."""
    values = np.asarray(estimates, dtype=np.float64)
    factors = shrinkage_factors(standard_errors, spread)
    if values.shape != factors.shape or not np.all(np.isfinite(values)):
        raise ValueError("estimates must be finite and match the standard errors")
    shrunk: NDArray[np.float64] = factors * values
    return shrunk


def significant_winners(
    estimates: ArrayLike, standard_errors: ArrayLike, *, critical: float = CRITICAL_Z
) -> NDArray[np.bool_]:
    """Positive results significant at the two-sided 5 percent level: the ones a programme ships."""
    values, errors = _checked(estimates, standard_errors)
    winners: NDArray[np.bool_] = values > positive(critical, name="critical") * errors
    return winners


def shrink_programme(programme: Programme, *, critical: float = CRITICAL_Z) -> ShrunkProgramme:
    """Estimate the spread from the programme, shrink every estimate, and mark the winners."""
    spread = estimate_spread(programme.estimates, programme.standard_errors)
    return ShrunkProgramme(
        programme=programme,
        estimated_spread=spread,
        shrunk=shrink(programme.estimates, programme.standard_errors, spread),
        winners=significant_winners(
            programme.estimates, programme.standard_errors, critical=critical
        ),
    )


def figure_programme(seed: int = SEED, experiments: int = EXPERIMENTS) -> ShrunkProgramme:
    """The figure's programme: 4,000 experiments from one generator, shrunk and filtered."""
    rng = np.random.default_rng(count(seed, name="seed"))
    return shrink_programme(draw_programme(rng, experiments))


def _rmse(errors: NDArray[np.float64]) -> float:
    return sqrt(float(np.mean(errors**2)))


def figure_summary(result: ShrunkProgramme | None = None) -> FigureSummary:
    """Errors of the raw and shrunk estimates for the figure's programme and its winners."""
    shrunk = result if result is not None else figure_programme()
    truth = shrunk.programme.effects
    raw = shrunk.programme.estimates
    errors = shrunk.programme.standard_errors
    win = shrunk.winners
    bands = []
    for lower, upper in PRECISION_BANDS:
        band = win & (errors >= lower) & (errors < upper)
        raw_mean, true_mean = float(np.mean(raw[band])), float(np.mean(truth[band]))
        bands.append(
            WinnerBand(
                lower=lower,
                upper=upper,
                winners=int(np.count_nonzero(band)),
                raw_mean=raw_mean,
                true_mean=true_mean,
                shrunk_mean=float(np.mean(shrunk.shrunk[band])),
                overstatement=raw_mean / true_mean,
            )
        )
    return FigureSummary(
        experiments=truth.size,
        estimated_spread=shrunk.estimated_spread,
        raw_rmse=_rmse(raw - truth),
        shrunk_rmse=_rmse(shrunk.shrunk - truth),
        winners=int(np.count_nonzero(win)),
        winner_raw_mean=float(np.mean(raw[win])),
        winner_true_mean=float(np.mean(truth[win])),
        winner_shrunk_mean=float(np.mean(shrunk.shrunk[win])),
        winner_raw_rmse=_rmse(raw[win] - truth[win]),
        winner_shrunk_rmse=_rmse(shrunk.shrunk[win] - truth[win]),
        bands=tuple(bands),
    )


def mean_shrinkage_factor(
    spread: float = EFFECT_SPREAD, *, se_low: float = SE_LOW, se_high: float = SE_HIGH
) -> float:
    """Average of ``tau^2 / (tau^2 + s^2)`` over standard errors uniform on ``[low, high]``."""
    tau = positive(spread, name="spread")
    low, high = _standard_error_range(se_low, se_high)
    return tau / (high - low) * (atan(high / tau) - atan(low / tau))


def raw_rmse(*, se_low: float = SE_LOW, se_high: float = SE_HIGH) -> float:
    """Root mean squared error of the raw estimates: the root of the mean squared standard error."""
    low, high = _standard_error_range(se_low, se_high)
    return sqrt((low**2 + low * high + high**2) / 3)


def oracle_rmse(
    spread: float = EFFECT_SPREAD, *, se_low: float = SE_LOW, se_high: float = SE_HIGH
) -> float:
    """Root mean squared error after shrinking with the true spread: the mean posterior variance."""
    tau = positive(spread, name="spread")
    return tau * sqrt(1 - mean_shrinkage_factor(tau, se_low=se_low, se_high=se_high))


def winner_expectation(
    lower: float = SE_LOW,
    upper: float = SE_HIGH,
    *,
    spread: float = EFFECT_SPREAD,
    experiments: int = PROGRAMME_SIZE,
    se_low: float = SE_LOW,
    se_high: float = SE_HIGH,
    critical: float = CRITICAL_Z,
) -> WinnerExpectation:
    """Expected winners with standard errors in ``[lower, upper)`` and what they are worth.

    Integrates the probability of winning, ``Phi(-c)``, and the expected raw and
    true effects of a winner, ``sqrt(v) phi(c)`` and ``tau^2 phi(c) / sqrt(v)``,
    over the uniform density of the standard errors.
    """
    tau = positive(spread, name="spread")
    k = count(experiments, name="experiments", minimum=1)
    low, high = _standard_error_range(se_low, se_high)
    z = positive(critical, name="critical")
    if not low <= lower < upper <= high:
        raise ValueError("the band must lie within the range of the standard errors")

    def threshold(s: float) -> float:
        return z * s / sqrt(tau**2 + s**2)

    def integral(integrand: Callable[[float], float]) -> float:
        return k * float(integrate.quad(integrand, lower, upper)[0]) / (high - low)

    winners = integral(lambda s: stats.norm.sf(threshold(s)))
    raw_total = integral(lambda s: sqrt(tau**2 + s**2) * stats.norm.pdf(threshold(s)))
    true_total = integral(lambda s: tau**2 / sqrt(tau**2 + s**2) * stats.norm.pdf(threshold(s)))
    return WinnerExpectation(
        spread=tau,
        lower=lower,
        upper=upper,
        winners=winners,
        raw_total=raw_total,
        true_total=true_total,
        raw_mean=raw_total / winners,
        true_mean=true_total / winners,
        overstatement=raw_total / true_total,
    )


def closed_forms() -> ClosedForms:
    """The article's quantities in closed form: errors, winners by precision, and forecasts."""
    raw, oracle = raw_rmse(), oracle_rmse()
    return ClosedForms(
        mean_shrinkage_factor=mean_shrinkage_factor(),
        raw_rmse=raw,
        oracle_rmse=oracle,
        rmse_reduction=1 - oracle / raw,
        bands=tuple(winner_expectation(lower, upper) for lower, upper in PRECISION_BANDS),
        forecasts=tuple(winner_expectation(spread=spread) for spread in FORECAST_SPREADS),
    )


def example_payload() -> EmpiricalBayesSummary:
    """Return the figure's programme summary and the article's closed forms."""
    return EmpiricalBayesSummary(figure=figure_summary(), closed_form=closed_forms())
