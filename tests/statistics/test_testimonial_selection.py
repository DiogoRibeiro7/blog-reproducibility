"""Check the testimonial model against numerical integration and simulated people.

The exact moments use a truncated-normal formula. The checks below integrate
the joint model directly and simulate person-level data, so agreement is not a
matter of calling the same expression twice.
"""

from itertools import pairwise
from math import sqrt
from statistics import NormalDist, mean, variance

import pytest

from blog_reproducibility.statistics.testimonial_selection import (
    conditional_prediction,
    example_payload,
    selected_moments,
    simulate_pairs,
)


def test_selected_moments_against_direct_numerical_integration() -> None:
    """Integrating the conditional mean over the baseline tail reproduces the moments."""
    normal = NormalDist(50, 10)
    step = 0.002
    grid = [65 + (index + 0.5) * step for index in range(35_000)]
    weights = [normal.pdf(x) * step for x in grid]
    mass = sum(weights)

    row = selected_moments()
    assert row.selected_fraction == pytest.approx(mass, abs=1e-9)
    assert row.baseline_mean == pytest.approx(
        sum(x * w for x, w in zip(grid, weights, strict=True)) / mass, abs=1e-6
    )

    followup = sum((50 + 0.64 * (x - 50)) * w for x, w in zip(grid, weights, strict=True)) / mass
    assert row.followup_mean == pytest.approx(followup, abs=1e-6)


def test_latent_variable_simulation_agrees_with_exact_selected_change() -> None:
    """Simulated people reproduce the exact mean change and its variance."""
    pairs = simulate_pairs(n=300_000)
    changes = [after - before for before, after in pairs if before >= 65]
    row = selected_moments()

    standard_error = sqrt(row.change_variance / len(changes))
    assert abs(mean(changes) - row.mean_change) < 5 * standard_error
    assert variance(changes) == pytest.approx(row.change_variance, rel=0.04)


def test_simulation_is_deterministic_and_independent_of_global_state() -> None:
    """The same seed gives the same people; a different seed gives different ones."""
    import random

    random.seed(1)
    first = simulate_pairs(n=50)
    random.seed(999)
    second = simulate_pairs(n=50)

    assert first == second
    assert simulate_pairs(n=50, seed=7) != first


def test_counterfactual_effect_is_preserved_despite_all_three_improving() -> None:
    """Every scenario improves, and the contrast between them is still the effect."""
    control = selected_moments()

    for effect in (-3.0, 3.0):
        intervention = selected_moments(effect=effect)
        assert intervention.mean_change < 0
        assert intervention.followup_mean - control.followup_mean == pytest.approx(effect)
        assert intervention.mean_change - control.mean_change == pytest.approx(effect)


def test_perfect_repeatability_and_independent_measurement_limits() -> None:
    """With no measurement noise nothing changes; with no stable part nothing persists."""
    perfect = selected_moments(within_sd=0.0)
    independent = selected_moments(between_sd=0.0)

    assert perfect.mean_change == 0.0
    assert perfect.change_variance == 0.0
    assert independent.followup_mean == pytest.approx(50.0)
    assert independent.slope == 0.0


def test_score_units_and_baseline_averaging() -> None:
    """Rescaling the score scales the moments; averaging baselines shrinks the effect."""
    original = selected_moments()
    rescaled = selected_moments(cutoff=200, mu=155, between_sd=24, within_sd=18)

    assert rescaled.selected_fraction == pytest.approx(original.selected_fraction)
    assert rescaled.mean_change == pytest.approx(3 * original.mean_change)
    assert rescaled.change_variance == pytest.approx(9 * original.change_variance)

    rows = [selected_moments(baseline_count=k) for k in (1, 4, 16)]
    assert all(left.mean_change < right.mean_change < 0 for left, right in pairwise(rows))
    assert all(left.selected_fraction > right.selected_fraction for left, right in pairwise(rows))


def test_published_headline_values() -> None:
    """Every number the article's tables quote should come back unchanged."""
    payload = example_payload()
    cohort = payload.selected_cohort

    assert cohort.slope == pytest.approx(0.64)
    assert cohort.selected_fraction == pytest.approx(0.0668, abs=5e-5)
    assert cohort.baseline_mean == pytest.approx(69.39, abs=5e-3)
    assert cohort.followup_mean == pytest.approx(62.41, abs=5e-3)
    assert cohort.mean_change == pytest.approx(-6.98, abs=5e-3)

    # The article's intervention table: every arm falls from the same baseline.
    interventions = payload.interventions
    assert interventions["-3.0"].followup_mean == pytest.approx(59.41, abs=5e-3)
    assert interventions["0.0"].followup_mean == pytest.approx(62.41, abs=5e-3)
    assert interventions["3.0"].followup_mean == pytest.approx(65.41, abs=5e-3)

    prediction = conditional_prediction(70.0)
    assert prediction.mean == pytest.approx(62.8)
    assert prediction.prediction_95[0] == pytest.approx(47.74, abs=5e-3)
    assert prediction.prediction_95[1] == pytest.approx(77.86, abs=5e-3)

    assert payload.unselected_drop_at_least_10 == pytest.approx(0.1193, abs=5e-5)
    assert payload.at_least_one_large_drop_among_20_independent_people == pytest.approx(
        0.9212, abs=5e-5
    )


def test_invalid_model_parameters() -> None:
    """Impossible standard deviations, counts, and effects are rejected."""
    with pytest.raises(ValueError):
        selected_moments(within_sd=-1.0)
    with pytest.raises(ValueError):
        selected_moments(between_sd=0.0, within_sd=0.0)
    with pytest.raises(ValueError):
        selected_moments(baseline_count=0)
    with pytest.raises(ValueError):
        selected_moments(effect=float("nan"))

    for bad_count in (1.5, True):
        with pytest.raises(TypeError):
            selected_moments(baseline_count=bad_count)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        conditional_prediction(70.0, correlation=1.0)
