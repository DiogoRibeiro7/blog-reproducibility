"""Finite-population bookkeeping for the poll-selection article.

The article works a synthetic population of twenty million people in which
supporters are recorded at four times the rate of everyone else. Every quantity
here is exact arithmetic on stipulated counts: nothing is sampled, and no
opinion is simulated. The point of the example is that a larger number of
recorded responses shrinks a naive binomial interval without moving the error,
so both are computed side by side.

The identity behind the article is Meng's decomposition of the error of a
recorded mean,

    q - p = rho * sd(Y) * sqrt((1 - f) / f),

with ``rho`` the correlation between answering and the answer, ``sd(Y)`` the
population standard deviation, and ``f`` the recorded fraction.
"""

from dataclasses import dataclass
from math import sqrt

from blog_reproducibility.common.validation import count, probability, real

__all__ = [
    "FiniteSummary",
    "NaiveInterval",
    "PollSelectionExamples",
    "SampleSizeRow",
    "WeightedGroup",
    "WeightingExample",
    "example_payload",
    "finite_summary",
    "naive_interval",
    "population_from_ratio",
    "srs_standard_error",
    "weighting_example",
]


@dataclass(frozen=True, slots=True)
class NaiveInterval:
    """A binomial normal interval around a recorded share.

    It is reported to show what it claims, not because it is valid: it
    describes sampling variation among respondents and says nothing about the
    population when recording depends on the answer.
    """

    half_width: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class FiniteSummary:
    """Exact bookkeeping for one stipulated population and recorded subset."""

    population: int
    sample: int
    population_share: float
    respondent_share: float
    recorded_fraction: float
    yes_recording_rate: float
    no_recording_rate: float
    covariance: float
    data_defect_correlation: float | None
    error: float
    population_sd: float
    no_assumption_bounds: tuple[float, float]


@dataclass(frozen=True, slots=True)
class WeightedGroup:
    """One weighting cell: its population, its respondents, and its weight."""

    group: str
    total: int
    yes: int
    seen_yes: int
    seen_no: int
    recorded: int
    weight: float
    population_share: float
    respondent_share: float


@dataclass(frozen=True, slots=True)
class WeightingExample:
    """A two-group population with weights that reproduce the group totals."""

    groups: tuple[WeightedGroup, ...]
    population: int
    sample: int
    population_share: float
    unweighted_share: float
    weighted_share: float


@dataclass(frozen=True, slots=True)
class SampleSizeRow:
    """One row of the reach table: the exact summary and the interval it invites."""

    summary: FiniteSummary
    naive_interval: NaiveInterval


@dataclass(frozen=True, slots=True)
class PollSelectionExamples:
    """Every number the poll-selection article reports."""

    main: FiniteSummary
    main_naive_interval: NaiveInterval
    sample_size_examples: tuple[SampleSizeRow, ...]
    srs_1000_standard_error: float
    weighting_by_group_only: WeightingExample
    weighting_within_group_selection: WeightingExample
    compatible_worlds: tuple[FiniteSummary, ...]
    sensitivity: dict[str, float]


def finite_summary(
    yes_total: int,
    no_total: int,
    yes_seen: int,
    no_seen: int,
) -> FiniteSummary:
    """Summarise a population and the subset of it that was recorded.

    ``data_defect_correlation`` is ``None`` for a census, where the recorded
    fraction is one and the correlation is undefined rather than zero.
    """
    yes_population = count(yes_total, name="yes_total")
    no_population = count(no_total, name="no_total")
    yes_recorded = count(yes_seen, name="yes_seen")
    no_recorded = count(no_seen, name="no_seen")

    if yes_population == 0 or no_population == 0:
        raise ValueError("The population must contain both outcomes")
    if yes_recorded > yes_population or no_recorded > no_population:
        raise ValueError("Cannot record more people than the population contains")
    if yes_recorded + no_recorded == 0:
        raise ValueError("The recorded sample must not be empty")

    population = yes_population + no_population
    sample = yes_recorded + no_recorded
    share = yes_population / population
    respondent_share = yes_recorded / sample
    recorded_fraction = sample / population
    covariance = yes_recorded / population - share * recorded_fraction

    correlation: float | None = None
    if recorded_fraction < 1.0:
        spread = sqrt(share * (1.0 - share) * recorded_fraction * (1.0 - recorded_fraction))
        correlation = covariance / spread

    return FiniteSummary(
        population=population,
        sample=sample,
        population_share=share,
        respondent_share=respondent_share,
        recorded_fraction=recorded_fraction,
        yes_recording_rate=yes_recorded / yes_population,
        no_recording_rate=no_recorded / no_population,
        covariance=covariance,
        data_defect_correlation=correlation,
        error=respondent_share - share,
        population_sd=sqrt(share * (1.0 - share)),
        no_assumption_bounds=(
            recorded_fraction * respondent_share,
            recorded_fraction * respondent_share + 1.0 - recorded_fraction,
        ),
    )


def naive_interval(respondent_share: float, sample: int) -> NaiveInterval:
    """Return the 95% binomial normal interval around a recorded share."""
    share = probability(respondent_share, name="respondent_share")
    size = count(sample, name="sample", minimum=1)

    half_width = 1.96 * sqrt(share * (1.0 - share) / size)
    return NaiveInterval(
        half_width=half_width,
        lower=share - half_width,
        upper=share + half_width,
    )


def srs_standard_error(share: float, population: int, sample: int) -> float:
    """Design standard error of a binary mean under simple random sampling.

    Sampling is without replacement, so the finite-population correction takes
    the standard error to exactly zero at a census.
    """
    proportion = probability(share, name="share", inclusive=False)
    population_size = count(population, name="population", minimum=2)
    sample_size = count(sample, name="sample", minimum=1)

    if sample_size > population_size:
        raise ValueError("sample must not exceed population")

    variance = proportion * (1.0 - proportion)
    correction = (population_size - sample_size) / (sample_size * (population_size - 1))
    return sqrt(variance * correction)


def population_from_ratio(respondent_share: float, ratio: float) -> float:
    """Invert outcome-dependent recording, with ``ratio`` the odds of being recorded.

    ``ratio`` is the recording rate among supporters divided by the rate among
    everyone else. A ratio of one returns the respondent share unchanged.
    """
    share = probability(respondent_share, name="respondent_share", inclusive=False)
    odds = real(ratio, name="ratio")
    if odds <= 0.0:
        raise ValueError("ratio must be positive")

    return share / (odds * (1.0 - share) + share)


def weighting_example(*, outcome_dependent: bool = False) -> WeightingExample:
    """Two population groups whose weights reproduce the population exactly.

    Group A is 40% of the population and supports at 90%; group B is 60% and
    supports at 40%. When ``outcome_dependent`` is false, recording differs
    between groups but not between answers inside a group, and weighting to the
    group totals recovers the population share. When it is true, recording also
    differs by answer, the weights still match the group totals exactly, and
    the weighted estimate is wrong anyway.
    """
    specification = (
        ("A", 40_000, 36_000, 3_600, 100 if outcome_dependent else 400),
        ("B", 60_000, 24_000, 480, 180 if outcome_dependent else 720),
    )

    groups: list[WeightedGroup] = []
    for name, total, yes, seen_yes, seen_no in specification:
        recorded = seen_yes + seen_no
        groups.append(
            WeightedGroup(
                group=name,
                total=total,
                yes=yes,
                seen_yes=seen_yes,
                seen_no=seen_no,
                recorded=recorded,
                weight=total / recorded,
                population_share=yes / total,
                respondent_share=seen_yes / recorded,
            )
        )

    population = sum(group.total for group in groups)
    sample = sum(group.recorded for group in groups)

    return WeightingExample(
        groups=tuple(groups),
        population=population,
        sample=sample,
        population_share=sum(group.yes for group in groups) / population,
        unweighted_share=sum(group.seen_yes for group in groups) / sample,
        weighted_share=sum(group.total * group.respondent_share for group in groups) / population,
    )


def sample_size_examples() -> tuple[SampleSizeRow, ...]:
    """Hold the population and the four-to-one recording ratio while reach grows.

    Every row records six supporters for each non-supporter, so the respondent
    share stays at 6/7 and the error stays at 9/35 while the naive interval
    narrows by a factor of ``sqrt(n)``.
    """
    rows: list[SampleSizeRow] = []
    for size in (70, 700, 7_000, 70_000, 700_000, 1_120_000):
        summary = finite_summary(12_000_000, 8_000_000, 6 * size // 7, size // 7)
        rows.append(
            SampleSizeRow(
                summary=summary,
                naive_interval=naive_interval(summary.respondent_share, size),
            )
        )
    return tuple(rows)


def example_payload() -> PollSelectionExamples:
    """Return every number the article reports."""
    return PollSelectionExamples(
        main=finite_summary(12_000_000, 8_000_000, 960_000, 160_000),
        main_naive_interval=naive_interval(6.0 / 7.0, 1_120_000),
        sample_size_examples=sample_size_examples(),
        srs_1000_standard_error=srs_standard_error(0.6, 20_000_000, 1_000),
        weighting_by_group_only=weighting_example(),
        weighting_within_group_selection=weighting_example(outcome_dependent=True),
        compatible_worlds=tuple(
            finite_summary(yes, 20_000_000 - yes, 960_000, 160_000)
            for yes in (10_000_000, 12_000_000, 15_000_000)
        ),
        sensitivity={
            str(ratio): population_from_ratio(6.0 / 7.0, ratio) for ratio in (1, 2, 3, 4, 6, 8)
        },
    )
