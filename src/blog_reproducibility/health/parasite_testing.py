"""Diagnostic arithmetic for the parasite-testing article.

Everything here is Bayes' rule and repeated sampling applied to published test
accuracies: how much a second and a third specimen add, what probability of
infection survives a run of negative results, and how far a symptom checklist
can move a prior compared with a laboratory test.

Repeated specimens are treated as independent, which is optimistic: shedding is
intermittent and correlated within a person, so the independent calculation is
an upper bound on what a second specimen adds. The article uses it that way, and
the observed two-specimen yield is reported next to it for comparison.

Sources are named beside each constant.
"""

from dataclasses import dataclass
from typing import Final

from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "BRANDA_PUBLISHED_NPV",
    "BRANDA_SENSITIVITY",
    "CARTWRIGHT",
    "GARCIA",
    "PRIORS",
    "TAPE",
    "ParasiteSummary",
    "after_positive",
    "detected_by",
    "example_payload",
    "left_after_negatives",
    "likelihood_ratio",
    "specificity_lower_bound",
]

# Cartwright CP. J Clin Microbiol 1999;37(8):2408-2411. Patients with three stool
# specimens examined. A case is a parasite found in any of the three, so these are
# upper bounds on true sensitivity.
CARTWRIGHT: Final[dict[str, int]] = {
    "cases": 373,
    "found by the first specimen": 283,
    "found by the first two": 343,
}
# Branda JA, et al. Clin Infect Dis 2006;42(7):972-978. Sensitivity of the first of
# at least three specimens, and the negative predictive values the authors give.
BRANDA_SENSITIVITY: Final[float] = 0.72
BRANDA_PUBLISHED_NPV: Final[dict[float, int]] = {0.05: 98, 0.10: 97, 0.15: 95, 0.20: 93}
# Wendt S, et al. Dtsch Arztebl Int 2019;116(13):213-219. Adhesive tape test for pinworm.
TAPE: Final[dict[str, float]] = {"one morning": 0.50, "three mornings": 0.90}
# Garcia LS, Shimizu RY. J Clin Microbiol 1997;35(6):1526-1529. Giardia antigen
# immunoassays; specificity was 100% on 50 negative specimens.
GARCIA: Final[dict[str, float]] = {
    "lowest sensitivity": 0.94,
    "negative specimens": 50,
    "false positives": 0,
}
PRIORS: Final[tuple[float, ...]] = (0.20, 0.05, 0.01, 0.001)


@dataclass(frozen=True, slots=True)
class ParasiteSummary:
    """Every number the parasite article reports."""

    first_specimen_percent: float
    two_specimens_observed_percent: float
    two_specimens_if_independent_percent: float
    tape_three_mornings_if_independent_percent: float
    negative_predictive_value_percent: dict[str, float]
    left_after_negatives_percent: dict[str, tuple[float, float, float]]
    left_after_three_at_half_sensitivity_percent: dict[str, float]
    checklist_likelihood_ratio: dict[str, float]
    after_a_positive_checklist_percent: dict[str, float]
    antigen_specificity_lower_bound_percent: float
    antigen_likelihood_ratio: float
    after_a_positive_antigen_test_percent: float


def detected_by(samples: int, sensitivity: float) -> float:
    """Share of infections found by at least one of independent samples."""
    number = count(samples, name="samples", minimum=0)
    rate = probability(sensitivity, name="sensitivity")
    return 1.0 - (1.0 - rate) ** number


def left_after_negatives(
    prior: float,
    negatives: int,
    sensitivity: float,
    specificity: float = 1.0,
) -> float:
    """Probability of infection after independent negative results, by Bayes' rule."""
    before = probability(prior, name="prior")
    number = count(negatives, name="negatives", minimum=0)
    rate = probability(sensitivity, name="sensitivity")
    correct = probability(specificity, name="specificity")

    infected = before * (1.0 - rate) ** number
    healthy = (1.0 - before) * correct**number
    if infected + healthy == 0.0:
        raise ValueError("No probability mass is left under either hypothesis")
    return infected / (infected + healthy)


def likelihood_ratio(sensitivity: float, false_positive_share: float) -> float:
    """Return the positive likelihood ratio of a test or a checklist."""
    rate = probability(sensitivity, name="sensitivity")
    false_positives = probability(false_positive_share, name="false_positive_share")
    if false_positives == 0.0:
        raise ValueError("false_positive_share must be positive")
    return rate / false_positives


def after_positive(prior: float, ratio: float) -> float:
    """Update a prior by a likelihood ratio and return the posterior probability."""
    before = probability(prior, name="prior", inclusive=False)
    factor = positive(ratio, name="ratio")

    odds = before / (1.0 - before) * factor
    return odds / (1.0 + odds)


def specificity_lower_bound(negatives: int, confidence: float = 0.95) -> float:
    """One-sided lower limit for a specificity observed as all negatives correct.

    With no false positives in ``n`` negative specimens, the exact one-sided
    limit is ``(1 - confidence) ** (1 / n)``. The familiar rule of three,
    ``1 - 3 / n``, is its first-order approximation.
    """
    number = count(negatives, name="negatives", minimum=1)
    level = probability(confidence, name="confidence", inclusive=False)
    return float((1.0 - level) ** (1.0 / number))


def example_payload() -> ParasiteSummary:
    """Return the numbers the article reports."""
    first = CARTWRIGHT["found by the first specimen"] / CARTWRIGHT["cases"]
    two = CARTWRIGHT["found by the first two"] / CARTWRIGHT["cases"]
    bound = specificity_lower_bound(int(GARCIA["negative specimens"]))
    antigen = likelihood_ratio(GARCIA["lowest sensitivity"], 1.0 - bound)

    return ParasiteSummary(
        first_specimen_percent=round(100 * first, 1),
        two_specimens_observed_percent=round(100 * two, 1),
        two_specimens_if_independent_percent=round(100 * detected_by(2, first), 1),
        tape_three_mornings_if_independent_percent=round(
            100 * detected_by(3, TAPE["one morning"]), 1
        ),
        negative_predictive_value_percent={
            str(prevalence): round(
                100 * (1 - left_after_negatives(prevalence, 1, BRANDA_SENSITIVITY)), 1
            )
            for prevalence in BRANDA_PUBLISHED_NPV
        },
        left_after_negatives_percent={
            str(prior): (
                round(100 * left_after_negatives(prior, 1, BRANDA_SENSITIVITY), 3),
                round(100 * left_after_negatives(prior, 2, BRANDA_SENSITIVITY), 3),
                round(100 * left_after_negatives(prior, 3, BRANDA_SENSITIVITY), 3),
            )
            for prior in PRIORS
        },
        left_after_three_at_half_sensitivity_percent={
            str(prior): round(100 * left_after_negatives(prior, 3, 0.5), 2) for prior in PRIORS
        },
        checklist_likelihood_ratio={
            str(share): round(likelihood_ratio(0.95, share), 2) for share in (0.8, 0.5, 0.2)
        },
        after_a_positive_checklist_percent={
            str(share): round(100 * after_positive(0.01, likelihood_ratio(0.95, share)), 1)
            for share in (0.8, 0.5, 0.2)
        },
        antigen_specificity_lower_bound_percent=round(100 * bound, 1),
        antigen_likelihood_ratio=round(antigen, 1),
        after_a_positive_antigen_test_percent=round(100 * after_positive(0.01, antigen), 1),
    )
