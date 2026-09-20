"""Check the screening cohorts and length-biased sampling.

The scenarios share one set of people, so the strongest checks compare histories
person by person rather than comparing summaries. Only the final scenario is
allowed to change a death time, and exactly 18 of them.
"""

import pytest

from blog_reproducibility.health.screening_survival import (
    SCENARIOS,
    cohort,
    example_payload,
    sampling,
    summarise,
)

SUMMARY = example_payload()


def test_only_the_benefit_scenario_changes_a_death_history() -> None:
    """Lead time and overdiagnosis move diagnoses, never deaths."""
    baseline = cohort()

    for scenario in SCENARIOS[:4]:
        rows = cohort(earlier=scenario.earlier, additional=scenario.additional)
        for before, after in zip(baseline, rows, strict=True):
            assert (before.death, before.cause) == (after.death, after.cause)

    benefit = cohort(earlier=True, additional=True, postponed_deaths=18)
    changed = [
        (before, after)
        for before, after in zip(baseline, benefit, strict=True)
        if (before.death, before.cause) != (after.death, after.cause)
    ]
    assert len(changed) == 18
    for before, after in changed:
        assert before.cause == "cancer"
        assert after.cause == "other"
        assert after.death > before.death


def test_every_scenario_keeps_the_same_thousand_people() -> None:
    """The cohort is paired: the same identities appear in every scenario."""
    baseline = cohort()
    assert len(baseline) == 1_000
    assert [row.person for row in baseline] == list(range(1_000))

    for scenario in SCENARIOS:
        rows = cohort(
            earlier=scenario.earlier,
            additional=scenario.additional,
            postponed_deaths=scenario.postponed_deaths,
        )
        assert [row.person for row in rows] == [row.person for row in baseline]
        assert [row.kind for row in rows] == [row.kind for row in baseline]


def test_survival_rises_without_mortality_moving() -> None:
    """Four scenarios raise five-year survival and leave cancer deaths at 75."""
    scenarios = SUMMARY.cohort_scenarios
    baseline = scenarios["Clinical diagnosis"]

    assert baseline.five_year_survival == 0.375
    assert scenarios["Earlier diagnosis only"].five_year_survival == 1.0
    assert scenarios["Additional diagnoses only"].five_year_survival == 0.75
    assert scenarios["Both, no effect on death"].five_year_survival == 1.0

    for name in (
        "Clinical diagnosis",
        "Earlier diagnosis only",
        "Additional diagnoses only",
        "Both, no effect on death",
    ):
        assert scenarios[name].cancer_deaths == 75
        assert scenarios[name].all_deaths == 125


def test_only_the_benefit_scenario_moves_mortality() -> None:
    """Postponing 18 deaths is the only change that mortality registers."""
    benefit = SUMMARY.cohort_scenarios["Both, with 18 deaths postponed"]

    assert benefit.cancer_deaths == 57
    assert benefit.all_deaths == 107
    assert benefit.cancer_death_risk == pytest.approx(0.057)
    assert benefit.all_death_risk == pytest.approx(0.107)
    assert benefit.five_year_survival == 1.0  # survival was already at its ceiling


def test_diagnosis_counts_follow_from_the_cohort_definition() -> None:
    """Extra diagnoses are exactly the 180 indolent lesions."""
    scenarios = SUMMARY.cohort_scenarios

    assert scenarios["Clinical diagnosis"].diagnoses == 120
    assert scenarios["Earlier diagnosis only"].diagnoses == 120
    assert scenarios["Additional diagnoses only"].diagnoses == 300
    assert scenarios["Both, no effect on death"].diagnoses == 300


def test_survival_counted_a_second_way() -> None:
    """Recomputing survivors from the histories gives the same count."""
    for scenario in SCENARIOS:
        rows = cohort(
            earlier=scenario.earlier,
            additional=scenario.additional,
            postponed_deaths=scenario.postponed_deaths,
        )
        summary = summarise(rows)
        survivors = sum(
            1 for row in rows if row.diagnosis is not None and row.death > row.diagnosis + 5
        )
        assert summary.five_year_survivors == survivors


def test_the_horizon_and_window_are_honoured() -> None:
    """A longer window and a shorter horizon both change what is counted."""
    rows = cohort()

    assert summarise(rows, survival_years=0).five_year_survivors == 120
    assert summarise(rows, survival_years=15).five_year_survivors == 0
    assert summarise(rows, horizon=8).cancer_deaths == 0
    assert summarise(rows, horizon=20).all_deaths == 1_000


def test_length_biased_sampling_over_represents_slow_disease() -> None:
    """Equal incidence, unequal duration, and an unequal snapshot."""
    selection = sampling()

    assert selection.incident_weights == (0.5, 0.5)
    assert selection.snapshot_stock == (40.0, 160.0)
    assert selection.snapshot_weights == pytest.approx((0.2, 0.8))
    assert selection.repeated_weights == pytest.approx((1 / 3, 2 / 3))
    assert selection.incident_mean_duration == pytest.approx(2.5)
    assert selection.snapshot_mean_duration == pytest.approx(3.4)

    # The snapshot exaggerates duration; repeated screening pulls it back part way.
    assert selection.snapshot_weights[1] > selection.repeated_weights[1]
    assert selection.repeated_weights[1] > selection.incident_weights[1]


def test_equal_durations_remove_the_selection() -> None:
    """With the same detectable window there is nothing for duration to select on."""
    selection = sampling(rates=(40.0, 40.0), durations=(3.0, 3.0))

    assert selection.snapshot_weights == selection.incident_weights
    assert selection.repeated_weights == selection.incident_weights
    assert selection.snapshot_mean_duration == pytest.approx(selection.incident_mean_duration)


def test_rescaling_time_leaves_the_composition_alone() -> None:
    """Doubling every duration and the screening interval changes no weight."""
    original = sampling(rates=(40.0, 40.0), durations=(1.0, 4.0), interval=2.0)
    rescaled = sampling(rates=(40.0, 40.0), durations=(2.0, 8.0), interval=4.0)

    assert rescaled.snapshot_weights == pytest.approx(original.snapshot_weights)
    assert rescaled.repeated_weights == pytest.approx(original.repeated_weights)
    assert rescaled.snapshot_mean_duration == pytest.approx(2 * original.snapshot_mean_duration)


def test_a_window_longer_than_the_interval_is_always_detected() -> None:
    """Detection probability is capped at one, not allowed to exceed it."""
    selection = sampling(rates=(10.0, 10.0), durations=(5.0, 9.0), interval=2.0)

    assert selection.detection_probabilities == (1.0, 1.0)
    assert selection.repeated_weights == selection.incident_weights


def test_invalid_inputs() -> None:
    """Impossible scenarios, cohorts, and sampling parameters are rejected."""
    with pytest.raises(ValueError):
        cohort(postponed_deaths=76)
    with pytest.raises(ValueError):
        cohort(postponed_deaths=18)  # the benefit scenario needs earlier diagnosis
    with pytest.raises(TypeError):
        cohort(postponed_deaths=True)

    with pytest.raises(ValueError):
        summarise(())
    with pytest.raises(ValueError):
        sampling(rates=(40.0,), durations=(1.0, 4.0))
    with pytest.raises(ValueError):
        sampling(rates=(40.0, 40.0), durations=(0.0, 4.0))
    with pytest.raises(ValueError):
        sampling(rates=(0.0, 0.0), durations=(1.0, 4.0))
    with pytest.raises(ValueError):
        sampling(interval=0.0)
