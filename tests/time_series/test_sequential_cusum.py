"""Tests for the sequential CUSUM example supporting the published article."""

from math import isclose

import pytest

from blog_reproducibility.time_series.sequential_cusum import (
    seeded_mean_shift_example,
    upper_cusum,
)


def test_scores_equal_exhaustive_suffix_maxima() -> None:
    """The recurrence must equal the direct maximum over all suffix sums."""
    data = [-3.0, 1.0, 4.0, -2.0, 7.0, 2.0, 0.0]

    for target_shift in (0.5, 2.0, 4.0):
        result = upper_cusum(
            data,
            baseline=1.0,
            target_shift=target_shift,
            threshold=3.0,
        )
        expected = [
            max(
                [0.0]
                + [sum(value - 1.0 - target_shift / 2.0 for value in data[k:n]) for k in range(n)]
            )
            for n in range(1, len(data) + 1)
        ]

        assert all(
            isclose(actual, wanted, abs_tol=1e-12)
            for actual, wanted in zip(result.scores, expected, strict=True)
        )
        assert result.alarm_index == next(
            (index for index, value in enumerate(expected) if value > 3.0),
            None,
        )


def test_strict_threshold_and_complete_trajectory() -> None:
    """Equality with the threshold is not an alarm and scores continue afterwards."""
    empty = upper_cusum([], baseline=0.0, target_shift=2.0, threshold=5.0)
    no_alarm = upper_cusum([0.0, -4.0, 1.0], baseline=0.0, target_shift=2.0, threshold=5.0)
    equality = upper_cusum([6.0], baseline=0.0, target_shift=2.0, threshold=5.0)
    alarm = upper_cusum([8.0, 8.0, 8.0], baseline=0.0, target_shift=2.0, threshold=5.0)

    assert empty.scores == ()
    assert empty.alarm_index is None
    assert no_alarm.scores == (0.0, 0.0, 0.0)
    assert no_alarm.alarm_index is None
    assert equality.scores == (5.0,)
    assert equality.alarm_index is None
    assert alarm.scores == (7.0, 14.0, 21.0)
    assert alarm.alarm_index == 0


def test_scores_and_alarm_respect_measurement_units() -> None:
    """Location shifts leave scores unchanged and positive rescaling preserves alarms."""
    data = [0.0, 3.0, 2.0, -1.0, 6.0, 4.0]
    original = upper_cusum(data, baseline=0.0, target_shift=2.0, threshold=5.0)
    shifted = upper_cusum(
        [value + 10.0 for value in data],
        baseline=10.0,
        target_shift=2.0,
        threshold=5.0,
    )
    scaled = upper_cusum(
        [value * 3.0 for value in data],
        baseline=0.0,
        target_shift=6.0,
        threshold=15.0,
    )

    assert original.scores == shifted.scores
    assert all(
        isclose(3.0 * score, scaled_score, abs_tol=1e-12)
        for score, scaled_score in zip(original.scores, scaled.scores, strict=True)
    )
    assert original.alarm_index == shifted.alarm_index == scaled.alarm_index


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_observations_are_rejected(bad_value: float) -> None:
    """CUSUM inputs must contain only finite observations."""
    with pytest.raises(ValueError):
        upper_cusum(
            [0.0, bad_value],
            baseline=0.0,
            target_shift=2.0,
            threshold=5.0,
        )


@pytest.mark.parametrize(
    ("target_shift", "threshold"),
    [(0.0, 5.0), (-2.0, 5.0), (2.0, 0.0), (2.0, -1.0)],
)
def test_non_positive_detection_parameters_are_rejected(
    target_shift: float,
    threshold: float,
) -> None:
    """The target shift and threshold must both be positive."""
    with pytest.raises(ValueError):
        upper_cusum(
            [0.0],
            baseline=0.0,
            target_shift=target_shift,
            threshold=threshold,
        )


def test_non_numeric_inputs_are_rejected() -> None:
    """Runtime validation rejects values outside the numerical API."""
    with pytest.raises(TypeError):
        upper_cusum(
            [0.0, "bad"],  # type: ignore[list-item]
            baseline=0.0,
            target_shift=2.0,
            threshold=5.0,
        )

    with pytest.raises(TypeError):
        upper_cusum(
            [0.0],
            baseline=True,
            target_shift=2.0,
            threshold=5.0,
        )


def test_seeded_example_matches_published_claims() -> None:
    """The migrated implementation reproduces the numerical values in the article."""
    example = seeded_mean_shift_example()

    assert example.first_changed_index == 60
    assert example.alarm_index == 64
    assert len(example.data) == 100
    assert len(example.scores) == 100
    assert example.scores[63] == pytest.approx(3.3772820199681686)
    assert example.scores[64] == pytest.approx(5.330177777284745)
    assert example.scores[63] < example.threshold < example.scores[64]


def test_seeded_example_is_deterministic() -> None:
    """Identical seeds reproduce the complete observations and score trajectory."""
    first = seeded_mean_shift_example(seed=42)
    second = seeded_mean_shift_example(seed=42)

    assert first == second
