"""Synthetic cohorts and length-biased sampling for the screening article.

The article follows the same 1,000 synthetic people through five scenarios to
separate three things that all raise five-year survival: earlier diagnosis
(lead time), extra diagnoses of disease that would never have caused harm
(overdiagnosis), and deaths actually postponed. Only the last changes mortality,
and the cohorts make that visible because every person's history is carried
through unchanged.

A second model covers length-biased sampling: a snapshot of prevalent cases
over-represents slow disease, because slow disease is present for longer, and a
first screening round sees a snapshot while later rounds see incident cases.

No patient records, fitted parameters, or random draws are used. Every number
here is either a count of stipulated histories or an exact stationary
expectation.
"""

from dataclasses import dataclass
from typing import Final

from blog_reproducibility.common.validation import count, positive, real

__all__ = [
    "SCENARIOS",
    "Scenario",
    "CohortSummary",
    "DurationSelection",
    "Person",
    "ScreeningSummary",
    "cohort",
    "example_payload",
    "sampling",
    "summarise",
]

POPULATION: Final[int] = 1_000
PROGRESSIVE: Final[int] = 120
INDOLENT_END: Final[int] = 300
CANCER_DEATHS: Final[int] = 75


@dataclass(frozen=True, slots=True)
class Person:
    """One synthetic history: what they had, when it was found, and how they died."""

    person: int
    kind: str
    diagnosis: int | None
    death: int
    cause: str


@dataclass(frozen=True, slots=True)
class CohortSummary:
    """Death risks over the whole cohort and survival among the diagnosed."""

    people: int
    diagnoses: int
    five_year_survivors: int
    five_year_survival: float | None
    cancer_deaths: int
    all_deaths: int
    cancer_death_risk: float
    all_death_risk: float


@dataclass(frozen=True, slots=True)
class DurationSelection:
    """Stationary stocks and flows for two disease speeds under screening."""

    incident_rates: tuple[float, ...]
    snapshot_stock: tuple[float, ...]
    incident_weights: tuple[float, ...]
    snapshot_weights: tuple[float, ...]
    detection_probabilities: tuple[float, ...]
    detected_annual_rates: tuple[float, ...]
    repeated_weights: tuple[float, ...]
    incident_mean_duration: float
    snapshot_mean_duration: float


def cohort(
    *,
    earlier: bool = False,
    additional: bool = False,
    postponed_deaths: int = 0,
) -> tuple[Person, ...]:
    """Return paired histories for the same 1,000 people under one scenario.

    Progressive disease: people 0 to 119, of whom 75 die of cancer at year 9 and
    45 of other causes at year 20. Indolent lesions: people 120 to 299, who die
    of other causes at year 20. Of the rest, 50 die at year 10 and 650 at year
    20. ``earlier`` moves diagnosis from year 6 to year 2. ``postponed_deaths``
    moves that many cancer deaths to year 20, when those people die of another
    cause instead. Every history is complete: nobody is censored.
    """
    postponed = count(postponed_deaths, name="postponed_deaths", minimum=0)
    if postponed > CANCER_DEATHS:
        raise ValueError(f"postponed_deaths must be at most {CANCER_DEATHS}")
    if postponed and not earlier:
        raise ValueError("The benefit scenario requires earlier diagnosis")

    rows: list[Person] = []
    for person in range(POPULATION):
        progressive = person < PROGRESSIVE
        indolent = PROGRESSIVE <= person < INDOLENT_END
        diagnosed = progressive or (additional and indolent)
        cancer_death = postponed <= person < CANCER_DEATHS

        if cancer_death:
            death = 9
        elif INDOLENT_END <= person < 350:
            death = 10
        else:
            death = 20

        rows.append(
            Person(
                person=person,
                kind="progressive" if progressive else ("indolent" if indolent else "none"),
                diagnosis=(2 if earlier else 6) if diagnosed else None,
                death=death,
                cause="cancer" if cancer_death else "other",
            )
        )
    return tuple(rows)


def summarise(
    rows: tuple[Person, ...],
    *,
    horizon: int = 12,
    survival_years: int = 5,
) -> CohortSummary:
    """Use every person for death risks and only the diagnosed for survival.

    A death at the horizon counts as a death. Surviving means living strictly
    beyond ``diagnosis + survival_years``. Complete follow-up is assumed, and
    that assumption is what makes the comparison across scenarios exact.
    """
    limit = count(horizon, name="horizon", minimum=0)
    window = count(survival_years, name="survival_years", minimum=0)
    if not rows:
        raise ValueError("The cohort must not be empty")

    diagnosed = [row for row in rows if row.diagnosis is not None]
    cancer = sum(row.death <= limit and row.cause == "cancer" for row in rows)
    deaths = sum(row.death <= limit for row in rows)
    survivors = sum(
        row.death > row.diagnosis + window for row in diagnosed if row.diagnosis is not None
    )

    return CohortSummary(
        people=len(rows),
        diagnoses=len(diagnosed),
        five_year_survivors=survivors,
        five_year_survival=survivors / len(diagnosed) if diagnosed else None,
        cancer_deaths=cancer,
        all_deaths=deaths,
        cancer_death_risk=cancer / len(rows),
        all_death_risk=deaths / len(rows),
    )


def sampling(
    rates: tuple[float, ...] = (40.0, 40.0),
    durations: tuple[float, ...] = (1.0, 4.0),
    interval: float = 2.0,
) -> DurationSelection:
    """Stationary prevalence and repeated-screen selection, with perfect sensitivity.

    Entry phases are uniform relative to regular screens, incident cases are
    counted once at first detection, and nobody leaves the detectable window by
    another route. The results are expected stocks and annual flows, not
    patients.

    The stock of each type is its incidence times its duration, which is why a
    snapshot over-represents slow disease even when both types arise equally
    often.
    """
    if len(rates) != len(durations) or not rates:
        raise ValueError("rates and durations must be non-empty and the same length")

    screen_interval = positive(interval, name="interval")
    incidence = tuple(real(rate, name="rate") for rate in rates)
    if any(rate < 0 for rate in incidence):
        raise ValueError("rates must be zero or greater")
    if sum(incidence) <= 0:
        raise ValueError("the total incidence must be positive")
    window = tuple(positive(value, name="duration") for value in durations)

    stock = tuple(rate * length for rate, length in zip(incidence, window, strict=True))
    detection = tuple(min(1.0, length / screen_interval) for length in window)
    detected = tuple(rate * chance for rate, chance in zip(incidence, detection, strict=True))
    incident_weights = tuple(rate / sum(incidence) for rate in incidence)
    snapshot_weights = tuple(value / sum(stock) for value in stock)

    return DurationSelection(
        incident_rates=incidence,
        snapshot_stock=stock,
        incident_weights=incident_weights,
        snapshot_weights=snapshot_weights,
        detection_probabilities=detection,
        detected_annual_rates=detected,
        repeated_weights=tuple(value / sum(detected) for value in detected),
        incident_mean_duration=sum(
            weight * length for weight, length in zip(incident_weights, window, strict=True)
        ),
        snapshot_mean_duration=sum(
            weight * length for weight, length in zip(snapshot_weights, window, strict=True)
        ),
    )


@dataclass(frozen=True, slots=True)
class Scenario:
    """One row of the article's table: what screening did to this cohort."""

    name: str
    earlier: bool = False
    additional: bool = False
    postponed_deaths: int = 0


SCENARIOS: Final[tuple[Scenario, ...]] = (
    Scenario("Clinical diagnosis"),
    Scenario("Earlier diagnosis only", earlier=True),
    Scenario("Additional diagnoses only", additional=True),
    Scenario("Both, no effect on death", earlier=True, additional=True),
    Scenario("Both, with 18 deaths postponed", earlier=True, additional=True, postponed_deaths=18),
)


@dataclass(frozen=True, slots=True)
class ScreeningSummary:
    """Every number the screening article reports."""

    cohort_scenarios: dict[str, CohortSummary]
    duration_selection: DurationSelection


def example_payload() -> ScreeningSummary:
    """Return the numbers the article reports."""
    return ScreeningSummary(
        cohort_scenarios={
            scenario.name: summarise(
                cohort(
                    earlier=scenario.earlier,
                    additional=scenario.additional,
                    postponed_deaths=scenario.postponed_deaths,
                )
            )
            for scenario in SCENARIOS
        },
        duration_selection=sampling(),
    )
