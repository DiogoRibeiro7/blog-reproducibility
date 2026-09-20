"""Surrogate-test and absolute-risk arithmetic for the leaky-gut article.

Two calculations. The first asks what a positive result on a surrogate test says
about the quantity the test stands for, given the correlation between them: if a
commercial kit correlates with measured permeability at 0.06, calling the top
fifth of its results "positive" identifies the top fifth of true permeability
barely more often than picking people at random.

The second turns a published hazard ratio into absolute risks, because a ratio
of three on a base risk of 3.5% is a different statement from a ratio of three
on a base risk of 30%.

Sources are named beside each constant.
"""

from dataclasses import dataclass
from math import atanh, sqrt, tanh
from typing import Final

from blog_reproducibility.common.bivariate import conditional_top_share
from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "POWER",
    "TOP",
    "TURPIN",
    "ZHOU",
    "CorrelationEstimate",
    "LeakyGutSummary",
    "correlation_interval",
    "example_payload",
    "risks_by_test",
    "share_truly_high",
]

# Power N, et al. Front Physiol 2021;12:645303. Serum marker from a commercial kit
# against the lactulose-mannitol ratio in 39 healthy first-degree relatives of
# patients with Crohn's disease.
POWER: Final[dict[str, float]] = {"r squared": 0.004, "participants": 39}
# Turpin W, et al. Gastroenterology 2020;159:2092-2100. First-degree relatives
# followed for a median of 7.8 years.
TURPIN: Final[dict[str, float]] = {
    "relatives": 1420,
    "developed Crohn's disease": 50,
    "hazard ratio": 3.03,
    "interval low": 1.64,
    "interval high": 5.63,
}
# Zhou Q, et al. Gut 2019;68:996-1002. Responders among those who completed eight weeks.
ZHOU: Final[dict[str, tuple[int, int]]] = {"glutamine": (43, 54), "placebo": (3, 52)}
# Call the highest fifth of test results "positive" and the highest fifth of true
# values "leaky". The choice of a fifth is a convention, not a clinical cut-off.
TOP: Final[float] = 0.20


@dataclass(frozen=True, slots=True)
class CorrelationEstimate:
    """A correlation with its 95% interval by Fisher's transformation."""

    correlation: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class LeakyGutSummary:
    """Every number the leaky-gut article reports."""

    correlation: CorrelationEstimate
    variance_explained_percent: float
    truly_high_among_positives_percent: dict[str, float]
    relatives_who_developed_crohns_percent: float
    risk_by_test_percent: dict[str, tuple[float, float]]
    glutamine_trial_responders_percent: tuple[float, float]


def share_truly_high(correlation: float, *, top: float = TOP, step: float = 0.001) -> float:
    """P(true value in its top share | test result in its top share).

    Test and truth are treated as a standard bivariate normal pair, so this is
    the shared tail-agreement integral.
    """
    return conditional_top_share(correlation, top=top, step=step)


def correlation_interval(
    r_squared: float,
    participants: int,
    *,
    z: float = 1.959964,
) -> CorrelationEstimate:
    """Correlation and its 95% interval by Fisher's transformation.

    Only ``r squared`` was published, so the sign is taken as positive. The
    interval still reaches below zero, which is the point: this study cannot
    rule out that the kit measures nothing.
    """
    explained = probability(r_squared, name="r_squared")
    people = count(participants, name="participants", minimum=4)
    critical = positive(z, name="z")

    correlation = sqrt(explained)
    centre = atanh(correlation)
    half_width = critical / sqrt(people - 3)
    return CorrelationEstimate(
        correlation=correlation,
        low=tanh(centre - half_width),
        high=tanh(centre + half_width),
    )


def risks_by_test(share_abnormal: float, overall: float, ratio: float) -> tuple[float, float]:
    """Absolute risk with a normal and with an abnormal test.

    The overall risk is a weighted average of the two, so fixing the ratio
    between them and the share with an abnormal test determines both.
    """
    abnormal_share = probability(share_abnormal, name="share_abnormal")
    base = probability(overall, name="overall")
    relative = positive(ratio, name="ratio")

    denominator = 1.0 + (relative - 1.0) * abnormal_share
    if denominator <= 0.0:
        raise ValueError("The stated ratio and share leave no valid normal-test risk")

    normal = base / denominator
    return (normal, relative * normal)


def example_payload() -> LeakyGutSummary:
    """Return the numbers the article reports."""
    estimate = correlation_interval(POWER["r squared"], int(POWER["participants"]))
    overall = TURPIN["developed Crohn's disease"] / TURPIN["relatives"]
    ratio = TURPIN["hazard ratio"]
    (glutamine_yes, glutamine_all), (placebo_yes, placebo_all) = ZHOU["glutamine"], ZHOU["placebo"]

    return LeakyGutSummary(
        correlation=CorrelationEstimate(
            correlation=round(estimate.correlation, 3),
            low=round(estimate.low, 2),
            high=round(estimate.high, 2),
        ),
        variance_explained_percent=round(100 * POWER["r squared"], 1),
        truly_high_among_positives_percent={
            "at the observed correlation": round(100 * share_truly_high(estimate.correlation), 1),
            "at the top of its interval": round(100 * share_truly_high(estimate.high), 1),
            "at 0.5": round(100 * share_truly_high(0.5), 1),
            "at 0.9": round(100 * share_truly_high(0.9), 1),
            "by chance": round(100 * TOP, 1),
        },
        relatives_who_developed_crohns_percent=round(100 * overall, 1),
        risk_by_test_percent={
            str(share): (
                round(100 * risks_by_test(share, overall, ratio)[0], 1),
                round(100 * risks_by_test(share, overall, ratio)[1], 1),
            )
            for share in (0.1, 0.2, 0.3)
        },
        glutamine_trial_responders_percent=(
            round(100 * glutamine_yes / glutamine_all, 1),
            round(100 * placebo_yes / placebo_all, 1),
        ),
    )
