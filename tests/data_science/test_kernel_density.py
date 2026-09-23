"""Check the kernel density estimate directly and the figure's claim about modes.

The estimator is compared with a sum of normal densities written out for a
tiny sample, and with its limits. The figure claims the small bandwidth shows
spurious modes, the middle one the true two, and the large one a single mode.
"""

import numpy as np
import pytest

from blog_reproducibility.data_science.kernel_density import (
    BANDWIDTHS,
    GRID,
    count_modes,
    example_payload,
    gaussian_kde,
    mixture_density,
    mixture_sample,
)

SUMMARY = {row.bandwidth: row for row in example_payload()}


def _normal(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return np.asarray(np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)))


def test_estimate_is_the_average_of_kernels() -> None:
    """Three points give the mean of three normal curves."""
    grid = np.linspace(-3, 3, 7)
    expected = (_normal(grid, -1, 0.5) + _normal(grid, 0, 0.5) + _normal(grid, 2, 0.5)) / 3

    assert gaussian_kde([-1.0, 0.0, 2.0], 0.5, grid) == pytest.approx(expected)


def test_estimate_integrates_to_one() -> None:
    """At every bandwidth the estimate is a density."""
    sample = mixture_sample()
    grid = np.linspace(-12, 12, 24_001)
    for bandwidth in BANDWIDTHS:
        area = float(np.trapezoid(gaussian_kde(sample, bandwidth, grid), grid))
        assert area == pytest.approx(1.0, abs=1e-6)


def test_bandwidth_decides_the_number_of_modes() -> None:
    """The figure's claim: spurious modes, then the true two, then one."""
    assert count_modes(mixture_density(GRID)) == 2
    assert SUMMARY[0.12].modes > 2
    assert SUMMARY[0.45].modes == 2
    assert SUMMARY[1.40].modes == 1


def test_oversmoothing_costs_the_most_accuracy() -> None:
    """The wide bandwidth is far from the truth. The narrow one is closer than the
    middle one in squared error even though its extra modes are spurious: the
    labels are about shape, not integrated error."""
    errors = {bandwidth: row.integrated_squared_error for bandwidth, row in SUMMARY.items()}

    assert errors[1.40] > 5 * errors[0.45]
    assert errors[0.12] < errors[0.45]


def test_count_modes() -> None:
    """Strict local maxima, ignoring the ends."""
    assert count_modes([0, 1, 0, 2, 0]) == 2
    assert count_modes([0, 1, 1, 0]) == 0
    assert count_modes([3, 2, 1]) == 0


def test_invalid_inputs_are_rejected() -> None:
    """Empty samples and non-positive bandwidths are refused."""
    with pytest.raises(ValueError):
        gaussian_kde([], 0.5)
    with pytest.raises(ValueError):
        gaussian_kde([1.0], 0.0)
