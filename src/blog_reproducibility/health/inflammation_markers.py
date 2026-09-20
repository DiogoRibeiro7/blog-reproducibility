"""C-reactive protein as a measurement, for the inflammation article.

Two questions. How far must two CRP results from one person differ before the
difference means anything, and how wide is the range of single results around a
person's usual level? Both follow from the within-subject variation.

CRP is skewed, so the article reports the log-normal version of the reference
change value alongside the symmetric one: the rise and the fall that two results
must exceed are not mirror images.

A bivariate log-normal model of repeat testing was tried when the article was
written and dropped, because it predicted 51% reclassification where a national
survey observed 32%, and 41% where a cohort observed 69%. The article quotes the
observations instead, and the reclassification model is deliberately absent here
too.

Sources are named beside each constant.
"""

from dataclasses import dataclass
from math import exp, log, sqrt
from statistics import NormalDist
from typing import Final

from blog_reproducibility.common.validation import non_negative, positive

__all__ = [
    "BOWER",
    "CANTOS",
    "DEGOMA_ICC",
    "EFLM_WITHIN",
    "ESTIMATES",
    "MACY",
    "Z",
    "Estimate",
    "InflammationSummary",
    "example_payload",
    "lognormal_change_value",
    "single_result_interval",
    "symmetric_change_value",
]

STANDARD: Final[NormalDist] = NormalDist()
Z: Final[float] = STANDARD.inv_cdf(0.975)

# EFLM Biological Variation Database (https://biologicalvariation.eu), C-reactive
# protein, six studies: within-subject CV, %.
EFLM_WITHIN: Final[float] = 34.7
# Macy EM, et al. Clin Chem 1997;43:52-58: analytical and within-subject CV, and
# the critical difference the authors report, %.
MACY: Final[dict[str, float]] = {
    "analytical": 5.2,
    "within subject": 42.2,
    "published critical difference": 118.0,
}
# Bower JK, et al. Arch Intern Med 2012;172:1519-1521. NHANES, 541 people tested
# twice 18.9 days apart. Visser M, et al. JAMA 1999;282:2131-2135: 6.7% of US
# adults had CRP above 1 mg/dL.
BOWER: Final[dict[str, float]] = {
    "intraclass correlation": 0.77,
    "above 1 mg/dL who were below it on repeat, %": 32.0,
    "share above 1 mg/dL": 0.067,
}
# DeGoma EM, et al. Atherosclerosis 2012;224:274-279 (MESA, serial values in 255 people).
DEGOMA_ICC: Final[float] = 0.62
# CANTOS event rates per 100 person-years, and fatal infection rates.
CANTOS: Final[dict[str, float]] = {
    "placebo": 4.50,
    "150 mg": 3.86,
    "fatal infection, canakinumab": 0.31,
    "fatal infection, placebo": 0.18,
}


@dataclass(frozen=True, slots=True)
class Estimate:
    """A hazard or risk ratio with its 95% interval and what produced it."""

    label: str
    ratio: float
    low: float
    high: float
    note: str


# The first row is a Mendelian randomisation estimate per 1 SD of genetically
# raised ln CRP; the others are randomised trials against placebo.
ESTIMATES: Final[tuple[Estimate, ...]] = (
    Estimate(
        "CRP raised by genes (Mendelian randomisation)",
        1.00,
        0.90,
        1.13,
        "no drug: tests whether CRP is a cause",
    ),
    Estimate("Methotrexate, CIRT, n = 4,786", 0.96, 0.79, 1.16, "did not lower CRP, IL-6 or IL-1β"),
    Estimate(
        "Canakinumab 150 mg, CANTOS, n = 10,061",
        0.85,
        0.74,
        0.98,
        "lowered CRP 37 points more than placebo",
    ),
    Estimate(
        "Colchicine, COLCOT, n = 4,745",
        0.77,
        0.61,
        0.96,
        "after a recent myocardial infarction",
    ),
    Estimate("Colchicine, LoDoCo2, n = 5,522", 0.69, 0.57, 0.83, "chronic coronary disease"),
)


@dataclass(frozen=True, slots=True)
class InflammationSummary:
    """Every number the inflammation article reports."""

    critical_difference_from_macy_percent: int
    change_needed_lognormal_percent: tuple[int, int]
    single_results_from_3_mg_per_litre: tuple[float, float]
    single_results_from_1_mg_per_litre: tuple[float, float]
    cantos_events_prevented_per_1000: float
    cantos_extra_fatal_infections_per_1000: float
    cantos_events_prevented_per_extra_infection: float


def symmetric_change_value(within_cv: float, analytical_cv: float = 0.0) -> float:
    """Reference change value in %, treating results as symmetric around the mean."""
    biological = non_negative(within_cv, name="within_cv")
    analytical = non_negative(analytical_cv, name="analytical_cv")
    return Z * sqrt(2.0) * sqrt(biological**2 + analytical**2)


def lognormal_change_value(within_cv: float, analytical_cv: float = 0.0) -> tuple[float, float]:
    """Rise and fall, in %, that two results must exceed when CRP is log-normal.

    The two are not mirror images: a rise of 155% and a fall of 61% are the same
    multiplicative step in opposite directions.
    """
    biological = non_negative(within_cv, name="within_cv")
    analytical = non_negative(analytical_cv, name="analytical_cv")

    sigma = sqrt(log(1 + (biological / 100) ** 2) + log(1 + (analytical / 100) ** 2))
    factor = exp(Z * sqrt(2.0) * sigma)
    return (100 * (factor - 1), 100 * (1 / factor - 1))


def single_result_interval(usual: float, within_cv: float) -> tuple[float, float]:
    """95% range of one result from a person whose usual (median) level is ``usual``."""
    median = positive(usual, name="usual")
    biological = non_negative(within_cv, name="within_cv")

    sigma = sqrt(log(1 + (biological / 100) ** 2))
    return (median * exp(-Z * sigma), median * exp(Z * sigma))


def example_payload() -> InflammationSummary:
    """Return the numbers the article reports."""
    rise, fall = lognormal_change_value(EFLM_WITHIN)
    low_3, high_3 = single_result_interval(3.0, EFLM_WITHIN)
    low_1, high_1 = single_result_interval(1.0, EFLM_WITHIN)
    prevented = CANTOS["placebo"] - CANTOS["150 mg"]
    infections = CANTOS["fatal infection, canakinumab"] - CANTOS["fatal infection, placebo"]

    return InflammationSummary(
        critical_difference_from_macy_percent=round(
            symmetric_change_value(MACY["within subject"], MACY["analytical"])
        ),
        change_needed_lognormal_percent=(round(rise), round(fall)),
        single_results_from_3_mg_per_litre=(round(low_3, 1), round(high_3, 1)),
        single_results_from_1_mg_per_litre=(round(low_1, 2), round(high_1, 2)),
        cantos_events_prevented_per_1000=round(10 * prevented, 1),
        cantos_extra_fatal_infections_per_1000=round(10 * infections, 1),
        cantos_events_prevented_per_extra_infection=round(prevented / infections, 1),
    )
