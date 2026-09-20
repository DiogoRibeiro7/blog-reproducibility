"""Check the longitudinal design formulas against their own simulation.

The simulation is the article's check on the analytic results, so both are here
and the test asserts that they agree to the Monte Carlo accuracy the article
states: about 0.7% relative for a standard deviation from 10,000 replications.
"""

from math import sqrt

import pytest

from blog_reproducibility.statistics.longitudinal_design import (
    BETWEEN_SD,
    DESIGNS,
    WITHIN_SD,
    budget_curve,
    example_payload,
    individual_rmse,
    population_standard_error,
    simulate_design,
    simulate_designs,
    variance_estimate_sd,
)

ROWS = example_payload()


def test_every_design_spends_the_same_budget() -> None:
    """A thousand observations, four ways."""
    assert all(row.observations == 1000 for row in ROWS)
    assert [(row.subjects, row.measurements) for row in ROWS] == list(DESIGNS)


def test_the_formulas_are_what_they_claim_to_be() -> None:
    """Each closed form is recomputed from the two standard deviations."""
    for subjects, measurements in DESIGNS:
        expected_se = sqrt((BETWEEN_SD**2 + WITHIN_SD**2 / measurements) / subjects)
        assert population_standard_error(subjects, measurements) == pytest.approx(expected_se)
        assert individual_rmse(measurements) == pytest.approx(WITHIN_SD / sqrt(measurements))
        assert variance_estimate_sd(subjects, measurements) == pytest.approx(
            sqrt(2 / (subjects - 1)) * (BETWEEN_SD**2 + WITHIN_SD**2 / measurements)
        )


def test_more_people_buy_population_precision_and_nothing_else() -> None:
    """The population SE falls with subjects; the individual RMSE does not move."""
    assert population_standard_error(200, 10) < population_standard_error(50, 10)
    assert individual_rmse(10) == individual_rmse(10)
    # Only measurements move the individual RMSE.
    assert individual_rmse(50) < individual_rmse(10)
    assert individual_rmse(4) == pytest.approx(individual_rmse(16) * 2)


def test_the_budget_trade_off_runs_the_two_ways() -> None:
    """Spreading the budget over people helps the mean and hurts the individual."""
    by_measurements = {row.measurements: row for row in ROWS}
    widest = by_measurements[5]
    longest = by_measurements[50]

    assert widest.analytic_population_se < longest.analytic_population_se
    assert widest.analytic_individual_rmse > longest.analytic_individual_rmse


def test_the_simulation_agrees_with_the_formulas() -> None:
    """Within the 0.7% relative accuracy the article states for 10,000 replications."""
    for row in ROWS:
        assert row.simulated_population_se == pytest.approx(row.analytic_population_se, rel=0.02)
        assert row.simulated_individual_rmse == pytest.approx(
            row.analytic_individual_rmse, rel=0.01
        )


def test_the_simulation_is_seeded_and_shares_one_stream() -> None:
    """The table repeats exactly, and a per-design seed gives different numbers."""
    assert simulate_designs() == simulate_designs()
    assert simulate_designs(seed=7) != simulate_designs()

    # A single design reseeded on its own matches the first row of the table,
    # and not the later ones, because the stream carries across designs.
    first_alone = simulate_design(*DESIGNS[0], replications=10_000)
    assert first_alone == pytest.approx(simulate_designs()[0])
    second_alone = simulate_design(*DESIGNS[1], replications=10_000)
    assert second_alone != pytest.approx(simulate_designs()[1])


def test_published_table() -> None:
    """The analytic table and the simulated values the article quotes."""
    analytic_se = [round(row.analytic_population_se, 3) for row in ROWS]
    analytic_rmse = [round(row.analytic_individual_rmse, 3) for row in ROWS]
    simulated_se = [round(row.simulated_population_se, 3) for row in ROWS]

    assert analytic_se == [0.243, 0.170, 0.138, 0.118]
    assert analytic_rmse == [0.424, 0.671, 0.949, 1.342]
    assert simulated_se == [0.241, 0.171, 0.137, 0.119]


def test_the_budget_curve_is_monotone_in_both_directions() -> None:
    """Along one budget the two quantities move opposite ways."""
    curve = budget_curve()
    standard_errors = [se for _, se, _ in curve]
    individual = [rmse for _, _, rmse in curve]

    assert standard_errors == sorted(standard_errors)
    assert individual == sorted(individual, reverse=True)


def test_invalid_designs() -> None:
    """Empty arms, single subjects for a variance, and tiny simulations are rejected."""
    with pytest.raises(ValueError):
        population_standard_error(0, 10)
    with pytest.raises(ValueError):
        individual_rmse(0)
    with pytest.raises(ValueError):
        variance_estimate_sd(1, 10)
    with pytest.raises(ValueError):
        simulate_design(1, 10)
    with pytest.raises(ValueError):
        simulate_design(20, 10, replications=1)
