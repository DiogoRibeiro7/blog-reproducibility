"""Check the simulated sample means against the exact Gamma distribution of the mean.

The mean of n unit exponentials is Gamma(n, 1/n), whose variance and skewness
are known exactly, so the simulation is checked against theory rather than
against its own output.
"""

import numpy as np
import pytest

from blog_reproducibility.mathematics.central_limit import (
    REPLICATIONS,
    SAMPLE_SIZES,
    example_payload,
    normal_reference,
    sample_means,
)

SUMMARY = example_payload()


def test_moments_match_the_gamma_distribution() -> None:
    """Mean one, variance 1/n, and skewness 2/sqrt(n), within sampling error."""
    for row in SUMMARY:
        # Standard error of the mean is sqrt(1 / (n * replications)).
        assert row.mean == pytest.approx(1.0, abs=4 / np.sqrt(row.sample_size * REPLICATIONS))
        assert row.variance == pytest.approx(row.exact_variance, rel=0.03)
        assert row.skewness == pytest.approx(row.exact_skewness, abs=0.1)


def test_skewness_shrinks_like_one_over_root_n() -> None:
    """The figure's claim: the shape becomes symmetric as n grows."""
    skews = [row.skewness for row in SUMMARY]

    assert skews == sorted(skews, reverse=True)
    assert SUMMARY[-1].exact_skewness == pytest.approx(2 / np.sqrt(30))


def test_normal_reference_is_a_density() -> None:
    """The reference integrates to one and peaks at the population mean."""
    grid = np.linspace(-5, 7, 120_001)
    for size in SAMPLE_SIZES:
        density = normal_reference(grid, size)
        assert float(np.trapezoid(density, grid)) == pytest.approx(1.0, abs=1e-6)
        assert grid[int(np.argmax(density))] == pytest.approx(1.0, abs=1e-3)


def test_simulation_is_deterministic() -> None:
    """The same seed reproduces the same means."""
    first = sample_means((3,), replications=100, seed=1)
    second = sample_means((3,), replications=100, seed=1)

    np.testing.assert_array_equal(first[0], second[0])


def test_invalid_settings_are_rejected() -> None:
    """Sample sizes and replication counts are validated."""
    with pytest.raises(ValueError):
        sample_means((0,))
    with pytest.raises(ValueError):
        sample_means(replications=1)
    with pytest.raises(ValueError):
        normal_reference([1.0], 0)
