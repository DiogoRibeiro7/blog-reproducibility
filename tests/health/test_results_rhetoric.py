"""Check the testimonial-wall calculation and the retention arithmetic.

The exact wall mean is checked against order statistics with closed forms, against
a seeded simulation, and against the truncated-normal approximation the article
quotes as a sanity check.
"""

from math import pi, sqrt
from statistics import NormalDist

import pytest

from blog_reproducibility.health.results_rhetoric import (
    KROGSBOLL,
    SCALE,
    WALL,
    clients_needed,
    example_payload,
    wall_mean,
    wall_mean_by_simulation,
)

SUMMARY = example_payload()


def test_exact_wall_mean_against_known_order_statistics() -> None:
    """The expected maximum of two and of three standard normals has a closed form."""
    assert wall_mean(2, wall=1) == pytest.approx(1 / sqrt(pi), abs=1e-4)
    assert wall_mean(3, wall=1) == pytest.approx(3 / (2 * sqrt(pi)), abs=1e-4)

    # A wall that holds everybody shows the mean effect and nothing else.
    assert wall_mean(20, wall=20, effect=0.5) == 0.5
    assert wall_mean(50, wall=50) == 0.0


def test_exact_wall_mean_against_simulation() -> None:
    """Drawing walls reproduces the integral to its own standard error."""
    assert wall_mean(1_000) == pytest.approx(wall_mean_by_simulation(1_000), abs=0.03)
    assert wall_mean(200, wall=5) == pytest.approx(wall_mean_by_simulation(200, wall=5), abs=0.05)


def test_simulation_is_seeded_and_independent_of_global_state() -> None:
    """The same seed repeats; a different seed gives a different estimate."""
    import random

    random.seed(11)
    first = wall_mean_by_simulation(200, wall=5, repeats=40)
    random.seed(12)

    assert wall_mean_by_simulation(200, wall=5, repeats=40) == first
    assert wall_mean_by_simulation(200, wall=5, repeats=40, seed=7) != first


def test_truncated_normal_approximation_is_close_when_the_wall_is_small() -> None:
    """The article's shortcut agrees with the exact value for a small wall."""
    standard = NormalDist()
    for clients in (1_000, 100_000):
        share = WALL / clients
        approximation = standard.pdf(standard.inv_cdf(1 - share)) / share
        assert approximation == pytest.approx(wall_mean(clients), rel=0.01)


def test_the_effect_enters_as_a_pure_shift() -> None:
    """A programme that works moves the wall by its effect and no more."""
    assert wall_mean(1_000, effect=1.0) == pytest.approx(1.0 + wall_mean(1_000), abs=1e-9)
    assert wall_mean(1_000, effect=0.5) == pytest.approx(0.5 + wall_mean(1_000), abs=1e-9)

    walls = list(SUMMARY.wall_by_client_count.values())
    assert walls == sorted(walls)
    assert walls[0] > 1.5


def test_a_narrower_wall_selects_harder() -> None:
    """Showing fewer of the best results raises their mean."""
    assert wall_mean(1_000, wall=5) > wall_mean(1_000, wall=20) > wall_mean(1_000, wall=100)


def test_a_programme_with_no_effect_matches_an_effective_one_at_the_quoted_size() -> None:
    """24,000 clients and no effect produce the wall of 1,000 clients and a large effect."""
    matched = SUMMARY.clients_at_which_no_effect_matches
    target = SUMMARY.wall_with_effect_1sd_1000_clients

    assert matched == 24_000
    assert wall_mean(matched) == pytest.approx(target, abs=0.02)


def test_arithmetic_on_the_published_retention_figures() -> None:
    """The published percentages turn into the client counts the article quotes."""
    finley = SUMMARY.finley

    assert finley.left_before_a_year_percent == 93.4
    assert finley.still_attending_at_a_year == 3971
    assert finley.left_within_four_weeks == 16244
    assert 15 <= finley.enrolled_per_client_still_attending < 15.5
    assert finley.left_within_four_weeks_percent == 27.0


def test_published_numbers() -> None:
    """Every value the article prints should come back unchanged."""
    assert SUMMARY.wall_by_client_count == {
        200: 1.74,
        1_000: 2.41,
        10_000: 3.16,
        100_000: 3.78,
        1_000_000: 4.32,
    }
    assert SUMMARY.wall_with_effect_1sd_1000_clients == 3.41
    assert SUMMARY.wall_with_effect_half_sd_1000_clients == 2.91
    assert SUMMARY.share_of_the_wall_that_is_selection == 0.71
    assert SUMMARY.wall_as_percent_of_body_weight == {1_000: 12.3, 100_000: 19.3}
    assert SUMMARY.share_of_change_the_treatment_explains == 0.56
    assert SUMMARY.before_after_change_over_treatment_effect == 1.8
    assert SCALE == 5.1


def test_krogsboll_decomposition_is_internally_consistent() -> None:
    """The share the treatment explains and the factor are two views of one ratio."""
    active = KROGSBOLL["active treatment"]
    own = active - KROGSBOLL["placebo"]

    assert SUMMARY.share_of_change_the_treatment_explains == pytest.approx(own / active, abs=0.005)
    assert SUMMARY.before_after_change_over_treatment_effect == pytest.approx(
        active / own, abs=0.05
    )
    assert SUMMARY.share_of_change_the_treatment_explains * (
        SUMMARY.before_after_change_over_treatment_effect
    ) == pytest.approx(1.0, abs=0.02)


def test_invalid_inputs() -> None:
    """A wall larger than the client list, and bad counts, are rejected."""
    with pytest.raises(ValueError):
        wall_mean(10, wall=20)
    with pytest.raises(ValueError):
        wall_mean(100, wall=0)
    with pytest.raises(ValueError):
        wall_mean(100, effect=float("nan"))
    with pytest.raises(ValueError):
        wall_mean_by_simulation(10, wall=20)
    with pytest.raises(TypeError):
        wall_mean(True)


def test_clients_needed_is_the_inverse_of_the_wall() -> None:
    """The count it returns reaches the target, and a smaller one does not."""
    target = 3.0
    matched = clients_needed(target)

    assert wall_mean(matched) >= target - 0.02
    assert wall_mean(max(WALL, matched // 2)) < target
