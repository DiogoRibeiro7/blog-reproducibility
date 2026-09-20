"""Gaussian probability models used by the p-value evidence article."""

from dataclasses import asdict, dataclass
from math import erfc, exp, hypot, isfinite, log, sqrt
from statistics import NormalDist
from typing import Final

STANDARD_NORMAL: Final[NormalDist] = NormalDist()


@dataclass(frozen=True, slots=True)
class SelectedStudies:
    """Expected study counts after selection by a significance threshold."""

    alpha: float
    power: float
    signal_rejections: float
    null_rejections: float
    signal_nonrejections: float
    null_nonrejections: float
    signal_given_rejection: float
    repeat_rejection_given_first_rejection: float


@dataclass(frozen=True, slots=True)
class StudySummary:
    """Normal-model summary for an estimate with known standard error."""

    estimate: float
    se: float
    p_value: float
    ci95: tuple[float, float]


def _real(value: object, *, name: str) -> float:
    """Return a finite real value while rejecting booleans and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")

    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _probability(value: object, *, name: str, inclusive: bool = True) -> float:
    """Validate a finite probability."""
    probability = _real(value, name=name)
    valid = 0.0 <= probability <= 1.0 if inclusive else 0.0 < probability < 1.0

    if not valid:
        interval = "[0, 1]" if inclusive else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}")
    return probability


def two_sided_p(z: float) -> float:
    """Return the two-sided standard-normal tail probability."""
    statistic = _real(z, name="z")
    return erfc(abs(statistic) / sqrt(2.0))


def rejection_probability(effect: float = 2.0, alpha: float = 0.05) -> float:
    """Return P(|Z| >= c) for Z ~ N(effect, 1)."""
    standardized_effect = _real(effect, name="effect")
    significance = _probability(alpha, name="alpha", inclusive=False)
    cutoff = STANDARD_NORMAL.inv_cdf(1.0 - significance / 2.0)

    return (
        erfc((cutoff - standardized_effect) / sqrt(2.0))
        + erfc((cutoff + standardized_effect) / sqrt(2.0))
    ) / 2.0


def selected_studies(
    prior_signal: float = 0.1,
    alpha: float = 0.05,
    effect: float = 2.0,
    studies: float = 10_000.0,
) -> SelectedStudies:
    """Return expected counts when only rejected tests are retained."""
    signal_prior = _probability(prior_signal, name="prior_signal")
    significance = _probability(alpha, name="alpha", inclusive=False)
    standardized_effect = _real(effect, name="effect")
    study_count = _real(studies, name="studies")
    if study_count <= 0.0:
        raise ValueError("studies must be positive")

    power = rejection_probability(standardized_effect, significance)
    signal_rejections = study_count * signal_prior * power
    null_rejections = study_count * (1.0 - signal_prior) * significance
    retained = signal_rejections + null_rejections
    signal_given_rejection = signal_rejections / retained

    return SelectedStudies(
        alpha=significance,
        power=power,
        signal_rejections=signal_rejections,
        null_rejections=null_rejections,
        signal_nonrejections=study_count * signal_prior * (1.0 - power),
        null_nonrejections=study_count * (1.0 - signal_prior) * (1.0 - significance),
        signal_given_rejection=signal_given_rejection,
        repeat_rejection_given_first_rejection=(
            signal_given_rejection * power + (1.0 - signal_given_rejection) * significance
        ),
    )


def posterior_signal(z: float, prior_signal: float = 0.1, effect: float = 2.0) -> float:
    """Return P(signal | signed z) for N(effect,1) versus N(0,1)."""
    statistic = _real(z, name="z")
    signal_prior = _probability(prior_signal, name="prior_signal")
    standardized_effect = _real(effect, name="effect")

    if signal_prior in (0.0, 1.0):
        return signal_prior

    log_odds = (
        log(signal_prior)
        - log(1.0 - signal_prior)
        + standardized_effect * statistic
        - standardized_effect**2 / 2.0
    )
    if log_odds >= 0.0:
        return 1.0 / (1.0 + exp(-log_odds))

    odds = exp(log_odds)
    return odds / (1.0 + odds)


def study_summary(estimate: float, se: float) -> StudySummary:
    """Return a normal-model p-value and 95% confidence interval."""
    point_estimate = _real(estimate, name="estimate")
    standard_error = _real(se, name="se")
    if standard_error <= 0.0:
        raise ValueError("se must be positive")

    half_width = STANDARD_NORMAL.inv_cdf(0.975) * standard_error
    return StudySummary(
        estimate=point_estimate,
        se=standard_error,
        p_value=two_sided_p(point_estimate / standard_error),
        ci95=(point_estimate - half_width, point_estimate + half_width),
    )


def example_payload() -> dict[str, object]:
    """Return the deterministic numerical examples reported in the article."""
    return {
        "z_2_p": two_sided_p(2.0),
        "z_2_likelihood_ratio": exp(2.0),
        "negative_z_2_likelihood_ratio": exp(-6.0),
        "thresholds": [asdict(selected_studies(alpha=alpha)) for alpha in (0.05, 0.01, 0.001)],
        "prior_sensitivity": [
            {
                "prior": prior,
                "given_z_2": posterior_signal(2.0, prior),
                "given_rejection": selected_studies(prior).signal_given_rejection,
            }
            for prior in (0.01, 0.1, 0.5)
        ],
        "study_a": asdict(study_summary(0.20, 0.10)),
        "study_b": asdict(study_summary(0.19, 0.11)),
        "study_difference": asdict(study_summary(0.01, hypot(0.10, 0.11))),
        "same_p_small_effect": asdict(study_summary(0.002, 0.001)),
        "same_p_large_effect": asdict(study_summary(2.0, 1.0)),
        "any_rejection_20_independent_null_tests": 1.0 - 0.95**20,
        "bonferroni_20_independent_null_tests": 1.0 - (1.0 - 0.05 / 20.0) ** 20,
    }
