"""Check the polling model against individual records and exhaustive sampling.

Each check reconstructs the quantity a different way — from person-level
records, from every possible sample, or from every completion of the unobserved
people — rather than calling the model twice.
"""

from itertools import combinations, pairwise
from math import sqrt
from statistics import mean, pvariance

import pytest

from blog_reproducibility.statistics.poll_selection import (
    example_payload,
    finite_summary,
    naive_interval,
    population_from_ratio,
    sample_size_examples,
    srs_standard_error,
    weighting_example,
)


def test_finite_identity_against_individual_records() -> None:
    """The defect correlation should match the one computed from person-level data."""
    answers = [1] * 600 + [0] * 400
    recorded = [1] * 48 + [0] * 552 + [1] * 8 + [0] * 392

    covariance = mean(
        [
            (answer - mean(answers)) * (seen - mean(recorded))
            for answer, seen in zip(answers, recorded, strict=True)
        ]
    )
    correlation = covariance / sqrt(pvariance(answers) * pvariance(recorded))

    summary = finite_summary(600, 400, 48, 8)
    observed = mean([answer for answer, seen in zip(answers, recorded, strict=True) if seen])

    assert summary.respondent_share == observed
    assert summary.data_defect_correlation == pytest.approx(correlation)
    assert correlation * sqrt(pvariance(answers)) * sqrt(
        (1 - mean(recorded)) / mean(recorded)
    ) == pytest.approx(observed - mean(answers))


def test_srs_variance_against_every_possible_sample() -> None:
    """The design standard error should equal the spread over all samples of that size."""
    population = [1] * 6 + [0] * 4
    estimates = [mean(sample) for sample in combinations(population, 4)]

    assert mean(estimates) == pytest.approx(0.6)
    assert srs_standard_error(0.6, 10, 4) ** 2 == pytest.approx(pvariance(estimates))
    assert srs_standard_error(0.6, 10, 10) == 0.0


def test_expanding_reach_preserves_error_while_naive_intervals_shrink() -> None:
    """More responses shrink the interval by sqrt(n) and leave the error at 9/35."""
    rows = sample_size_examples()
    widths = [row.naive_interval.half_width for row in rows]

    assert all(row.summary.error == pytest.approx(9 / 35) for row in rows)
    assert all(wide > narrow for wide, narrow in pairwise(widths))
    assert widths[0] / widths[-1] == pytest.approx(
        sqrt(rows[-1].summary.sample / rows[0].summary.sample)
    )
    assert all(row.naive_interval.lower > 0.6 for row in rows)

    for row in rows:
        summary = row.summary
        fraction = summary.recorded_fraction
        assert summary.data_defect_correlation is not None
        reconstructed = (
            summary.data_defect_correlation
            * summary.population_sd
            * sqrt((1 - fraction) / fraction)
        )
        assert reconstructed == pytest.approx(summary.error)


def test_weighting_by_expanding_each_observed_record() -> None:
    """Weighting should equal a direct weighted mean over the expanded records."""
    for dependent in (False, True):
        result = weighting_example(outcome_dependent=dependent)
        pairs = [
            (answer, group.weight)
            for group in result.groups
            for answer, repeats in ((1, group.seen_yes), (0, group.seen_no))
            for _ in range(repeats)
        ]
        direct = sum(answer * weight for answer, weight in pairs) / sum(
            weight for _, weight in pairs
        )

        assert result.weighted_share == pytest.approx(direct)
        for group in result.groups:
            assert group.weight * group.recorded == pytest.approx(group.total)

    assert weighting_example().weighted_share == pytest.approx(0.6)
    biased = weighting_example(outcome_dependent=True)
    assert biased.weighted_share == pytest.approx(0.4 * (36 / 37) + 0.6 * (8 / 11))
    assert biased.weighted_share > 0.82


def test_distinct_populations_have_identical_observed_data() -> None:
    """Three populations produce the same recorded counts and respondent share."""
    worlds = example_payload().compatible_worlds
    assert [world.population_share for world in worlds] == [0.5, 0.6, 0.75]

    for world in worlds:
        assert world.sample == 1_120_000
        assert world.respondent_share == pytest.approx(6 / 7)
        ratio = world.yes_recording_rate / world.no_recording_rate
        assert population_from_ratio(world.respondent_share, ratio) == pytest.approx(
            world.population_share
        )

    assert population_from_ratio(6 / 7, 2) == pytest.approx(0.75)
    assert population_from_ratio(6 / 7, 6) == pytest.approx(0.5)
    assert population_from_ratio(6 / 7, 1) == pytest.approx(6 / 7)


def test_bounds_against_every_completion_of_unseen_people() -> None:
    """The assumption-free bounds should span every possible unobserved completion."""
    rows = [finite_summary(3 + unseen_yes, 7 - unseen_yes, 3, 1) for unseen_yes in range(7)]
    shares = [row.population_share for row in rows]

    assert rows[0].no_assumption_bounds == pytest.approx((min(shares), max(shares)))
    assert finite_summary(600, 400, 60, 40).error == 0.0

    census = finite_summary(600, 400, 600, 400)
    assert census.error == 0.0
    assert census.data_defect_correlation is None
    assert census.no_assumption_bounds == pytest.approx((0.6, 0.6))


def test_published_headline_values() -> None:
    """The article's headline numbers should come back unchanged."""
    payload = example_payload()
    main = payload.main

    assert main.population == 20_000_000
    assert main.sample == 1_120_000
    assert main.population_share == pytest.approx(0.6)
    assert main.respondent_share == pytest.approx(6 / 7)
    assert main.yes_recording_rate / main.no_recording_rate == pytest.approx(4.0)
    assert payload.srs_1000_standard_error == pytest.approx(0.0154915, abs=5e-7)
    assert payload.sensitivity["4"] == pytest.approx(0.6)


def test_invalid_counts_and_sensitivity_inputs() -> None:
    """Impossible populations, samples, and recording ratios are rejected."""
    for counts in ((600, 400, 601, 1), (600, 400, 0, 0), (0, 4, 0, 1)):
        with pytest.raises(ValueError):
            finite_summary(*counts)

    with pytest.raises(TypeError):
        finite_summary(600, 400, True, 1)

    for ratio in (0.0, -1.0):
        with pytest.raises(ValueError):
            population_from_ratio(0.8, ratio)
    for ratio in (float("inf"), float("nan")):
        with pytest.raises(ValueError):
            population_from_ratio(0.8, ratio)

    with pytest.raises(ValueError):
        naive_interval(0.5, 0)
    with pytest.raises(ValueError):
        srs_standard_error(0.6, 10, 11)
