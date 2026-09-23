"""Check the sparse-cell rule directly and what it catches in the simulated sample.

The rule is checked on a hand-built grid. On the simulated sample it should
catch most uniform strays, and the core points it flags should sit in the
tails of the bivariate normal, not near its centre.
"""

import numpy as np
import pytest

from blog_reproducibility.data_science.histogram_outliers import (
    example_payload,
    flag_sparse_cells,
    flagged_sample,
    simulate_points,
)

SUMMARY = example_payload()


def test_rule_flags_lonely_points_only() -> None:
    """Two points share a cell, one is alone: only the lone point is flagged."""
    points = np.array([[0.01, 0.01], [0.02, 0.02], [3.0, -3.0]])

    assert flag_sparse_cells(points).tolist() == [False, False, True]


def test_points_outside_the_grid_join_the_edge_cells() -> None:
    """Points beyond the square are clipped into the outer cells, as the figure does."""
    points = np.array([[-7.0, -7.0], [4.99, 4.99]])

    assert flag_sparse_cells(points).tolist() == [True, True]


def test_rule_catches_most_strays() -> None:
    """Uniform strays rarely share a cell, so most of them are flagged."""
    assert SUMMARY.strays == 45
    assert SUMMARY.strays_flagged >= 0.75 * SUMMARY.strays
    assert SUMMARY.flagged == SUMMARY.strays_flagged + SUMMARY.core_flagged


def test_flagged_core_points_are_in_the_tails() -> None:
    """Core points are flagged only far from the centre of their distribution."""
    sample = flagged_sample()
    covariance = np.array([[1.0, 0.65], [0.65, 1.0]])
    precision = np.linalg.inv(covariance)
    core = sample.points[~sample.is_stray]
    distance = np.sqrt(np.einsum("ij,jk,ik->i", core, precision, core))
    flagged = sample.flagged[~sample.is_stray]

    assert np.median(distance[flagged]) > 2.5
    assert np.median(distance[~flagged]) < 1.5
    # Flagged core points are a small share of the core.
    assert flagged.mean() < 0.03


def test_simulation_is_deterministic() -> None:
    """The same seed gives the same points."""
    first, _ = simulate_points(seed=2)
    second, _ = simulate_points(seed=2)

    np.testing.assert_array_equal(first, second)


def test_invalid_inputs_are_rejected() -> None:
    """Points must be two-dimensional."""
    with pytest.raises(ValueError):
        flag_sparse_cells(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        flag_sparse_cells(np.zeros((3, 2)), bins=0)
