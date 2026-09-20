"""Check test inversion against the values the confidence-sets article prints.

The accepted set has a closed form, so most checks compare it with a direct
scan of the acceptance condition rather than with itself. The coverage figures
come from a seeded simulation and are checked to Monte Carlo accuracy.
"""

from math import isclose, sqrt

import pytest

from blog_reproducibility.statistics.confidence_sets import (
    SIGMA,
    Z,
    accepted_set,
    bisection_floor,
    contains,
    convex_hull,
    coverage_summary,
    example_payload,
    fieller_set,
    grid_miss_probability,
    local_minimum,
    mean_function,
    measure,
    p_value,
    profile_projection_bound,
    profile_search,
)


def test_published_table_of_accepted_sets() -> None:
    """The five rows the article prints should come back exactly."""
    expected = {
        -0.20: (),
        0.04: ((-1.202, -0.746), (0.746, 1.202)),
        0.50: ((-1.346, -1.259), (-0.644, -0.435), (0.435, 0.644), (1.259, 1.346)),
        1.00: ((-1.441, -1.385), (-0.286, 0.286), (1.385, 1.441)),
        1.20: ((-1.471, -1.422), (1.422, 1.471)),
    }
    for y, rounded in expected.items():
        pieces = accepted_set(y)
        assert len(pieces) == len(rounded)
        for (low, high), (want_low, want_high) in zip(pieces, rounded, strict=True):
            assert low == pytest.approx(want_low, abs=5e-4)
            assert high == pytest.approx(want_high, abs=5e-4)


def test_accepted_set_agrees_with_a_direct_scan_of_the_condition() -> None:
    """Membership should match evaluating the acceptance condition point by point."""
    for y in (-0.20, 0.04, 0.30, 0.50, 1.00, 1.20):
        pieces = accepted_set(y)
        for index in range(3001):
            theta = -1.8 + index * (3.6 / 3000)
            accepted = abs(y - mean_function(theta)) < Z * SIGMA
            # Points within a float of a boundary can fall either way.
            if min((abs(theta - edge) for piece in pieces for edge in piece), default=1.0) < 1e-6:
                continue
            assert contains(pieces, theta) == accepted, (y, theta)


def test_empty_set_occurs_at_exactly_the_nominal_error_rate() -> None:
    """Below the lower critical value nothing is accepted, and that is the 2.5% tail."""
    assert accepted_set(-Z * SIGMA - 1e-12) == ()
    assert accepted_set(-0.20) == ()
    assert len(accepted_set(-Z * SIGMA + 1e-6)) > 0

    # At theta = ±1 the mean is zero, so an empty set needs y < -z*sigma.
    summary = coverage_summary(1.0, trials=20_000)
    assert summary.component_counts.get(0, 0.0) == pytest.approx(0.025, abs=0.005)


def test_hull_is_valid_and_larger() -> None:
    """The hull covers at least as often as the set, at a multiple of the size."""
    pieces = accepted_set(0.04)
    hull = convex_hull(pieces)
    assert hull is not None
    assert hull[0] == pytest.approx(-1.2015, abs=5e-4)
    assert hull[1] == pytest.approx(1.2015, abs=5e-4)
    assert (hull[1] - hull[0]) / measure(pieces) == pytest.approx(2.6, abs=0.05)

    # The hull asserts compatibility with the most thoroughly rejected value.
    assert hull[0] <= 0.0 <= hull[1]
    assert not contains(pieces, 0.0)
    assert convex_hull(()) is None


def test_p_values_quoted_in_the_figure_discussion() -> None:
    """The dip at ±1 and the floor at zero should match the article."""
    assert p_value(1.0, 0.04) == pytest.approx(0.62, abs=5e-3)
    assert p_value(-1.0, 0.04) == pytest.approx(p_value(1.0, 0.04))
    assert p_value(0.0, 0.04) == pytest.approx(4e-33, rel=0.2)

    # The article corrects an earlier version: those endpoints sit inside the set.
    assert p_value(0.87, 0.04) == pytest.approx(0.81, abs=0.01)
    assert p_value(1.10, 0.04) == pytest.approx(0.96, abs=0.01)


def test_component_widths_and_grid_miss_probabilities() -> None:
    """The four-component case has the widths the grid table is built on."""
    widths = [high - low for low, high in accepted_set(0.50)]
    assert widths[0] == pytest.approx(0.086, abs=5e-4)
    assert widths[1] == pytest.approx(0.208, abs=5e-4)
    assert widths == sorted(widths[:2]) + sorted(widths[2:], reverse=True)

    assert grid_miss_probability(0.086, 0.05) == 0.0
    assert grid_miss_probability(0.086, 0.10) == pytest.approx(0.14, abs=0.005)
    assert grid_miss_probability(0.086, 0.20) == pytest.approx(0.57, abs=0.005)
    assert grid_miss_probability(0.086, 0.50) == pytest.approx(0.83, abs=0.005)
    assert grid_miss_probability(0.208, 0.50) == pytest.approx(0.58, abs=0.005)
    assert grid_miss_probability(0.456, 0.50) == pytest.approx(0.09, abs=0.005)


def test_bisection_stalls_at_the_floating_point_floor() -> None:
    """Bisection ends because doubles run out, not because a tolerance was met."""
    floor = bisection_floor()

    assert floor.halvings == 52
    assert floor.final_width == pytest.approx(1.1e-16, rel=0.05)
    assert floor.boundary == pytest.approx(0.745910039589340, abs=5e-15)
    # The stall is at the spacing of adjacent doubles: no midpoint is representable.
    upper = floor.boundary + floor.final_width
    assert (floor.boundary + upper) / 2.0 in (floor.boundary, upper)


def test_fieller_set_changes_shape_with_the_denominator() -> None:
    """A significant denominator gives an interval; a weak one does not."""
    interval = fieller_set(1.0, 2.0)
    assert interval.kind == "interval"
    assert interval.bounds is not None
    assert interval.bounds[0] == pytest.approx(0.20, abs=5e-3)
    assert interval.bounds[1] == pytest.approx(0.89, abs=5e-3)

    exclusive = fieller_set(1.0, 0.5)
    assert exclusive.kind == "exclusive"
    assert exclusive.bounds is not None
    assert exclusive.bounds[0] == pytest.approx(-11.06, abs=5e-3)
    assert exclusive.bounds[1] == pytest.approx(0.62, abs=5e-3)

    assert fieller_set(0.2, 0.3).kind == "real line"


def test_fieller_endpoints_solve_the_inverted_test() -> None:
    """Each finite endpoint should make the test statistic exactly critical."""
    for numerator, denominator in ((1.0, 2.0), (1.0, 0.5)):
        result = fieller_set(numerator, denominator)
        assert result.bounds is not None
        for rho in result.bounds:
            statistic = abs(numerator - rho * denominator) / sqrt(0.3**2 + rho**2 * 0.3**2)
            assert statistic == pytest.approx(Z)


def test_local_search_finds_the_basin_it_started_in() -> None:
    """The left start rejects and the right start accepts the same target value."""
    left = profile_search(-1.2)
    right = profile_search(1.2)
    middle = profile_search(0.0)

    assert left.minimizer == pytest.approx(-0.96, abs=5e-3)
    assert left.statistic == pytest.approx(133.8, abs=0.05)
    assert left.p_value == pytest.approx(6e-31, rel=0.2)
    assert left.rejects

    assert right.minimizer == pytest.approx(1.036, abs=5e-3)
    assert right.statistic == pytest.approx(2.48, abs=5e-3)
    assert right.p_value == pytest.approx(0.115, abs=5e-4)
    assert not right.rejects
    assert middle.statistic == pytest.approx(right.statistic)


def test_local_minimum_against_a_dense_scan() -> None:
    """The minimiser should agree with a brute-force scan of its own basin."""

    def objective(x: float) -> float:
        return (x**2 - 1.0) ** 2 - 0.3 * x

    for start, low, high in ((-1.2, -1.7, -0.2), (1.2, 0.2, 1.7)):
        found = local_minimum(objective, start)
        scanned = min(
            (low + index * (high - low) / 200_000 for index in range(200_001)),
            key=objective,
        )
        assert found == pytest.approx(scanned, abs=1e-4)
        assert low < found < high


def test_projection_bound_from_each_basin() -> None:
    """The right basin gives the true projection; the left one is too narrow."""
    assert profile_projection_bound(start=1.2) == pytest.approx(0.423, abs=5e-4)
    assert profile_projection_bound(start=-1.2) == pytest.approx(-0.177, abs=5e-4)
    # Anti-conservative: the shortcut discards target values that are accepted.
    assert profile_projection_bound(start=-1.2) < profile_projection_bound(start=1.2)


def test_simulated_coverage_matches_the_published_table() -> None:
    """The set delivers its nominal level and the hull over-covers at more size."""
    expected = {
        1.0: (0.951, 0.975, 0.77, 2.30),
        0.0: (0.952, 1.000, 0.66, 2.88),
        1.3: (0.952, 0.976, 0.60, 2.68),
    }
    for theta, (set_coverage, hull_coverage, set_size, hull_size) in expected.items():
        summary = coverage_summary(theta)
        assert summary.trials == 20_000
        assert summary.set_coverage == pytest.approx(set_coverage, abs=0.006)
        assert summary.hull_coverage == pytest.approx(hull_coverage, abs=0.006)
        assert summary.mean_set_measure == pytest.approx(set_size, abs=0.02)
        assert summary.mean_hull_length == pytest.approx(hull_size, abs=0.02)
        assert summary.hull_coverage >= summary.set_coverage
        assert summary.mean_hull_length > summary.mean_set_measure

    assert coverage_summary(1.3).component_counts == {4: 1.0}


def test_simulation_is_seeded_and_independent_of_global_state() -> None:
    """The same seed repeats; a different seed does not."""
    import random

    random.seed(3)
    first = coverage_summary(1.0, trials=500)
    random.seed(4)
    assert coverage_summary(1.0, trials=500) == first
    assert coverage_summary(1.0, trials=500, seed=11) != first


def test_example_payload_is_internally_consistent() -> None:
    """The payload should report the same sets the functions return."""
    payload = example_payload()
    assert payload.accepted_sets["+0.04"] == accepted_set(0.04)
    assert payload.profile_projection_bound_right == pytest.approx(0.423, abs=5e-4)
    assert payload.profile_projection_bound_left == pytest.approx(-0.177, abs=5e-4)
    assert isclose(payload.p_value_at_one_for_004, p_value(1.0, 0.04))
    assert len(payload.coverage) == 3
    assert [row.step for row in payload.grid_miss] == [0.05, 0.10, 0.20, 0.50]


def test_invalid_inputs() -> None:
    """Degenerate spreads, levels, and brackets are rejected."""
    with pytest.raises(ValueError):
        accepted_set(0.04, sigma=0.0)
    with pytest.raises(ValueError):
        p_value(1.0, 0.04, sigma=-1.0)
    with pytest.raises(ValueError):
        grid_miss_probability(0.0, 0.5)
    with pytest.raises(ValueError):
        coverage_summary(1.0, trials=0)
    with pytest.raises(TypeError):
        local_minimum(3.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        # Both endpoints are rejected, so no boundary lies between them.
        bisection_floor(low=0.0, high=0.1)
