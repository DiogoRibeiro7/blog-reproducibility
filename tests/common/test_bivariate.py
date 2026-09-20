"""Tests for the shared bivariate tail-agreement integral."""

from math import sqrt
from random import Random

import pytest

from blog_reproducibility.common.bivariate import STANDARD, conditional_top_share


def test_no_correlation_gives_the_top_share_itself() -> None:
    """Knowing the first value says nothing about the second."""
    for share in (0.05, 0.1, 0.2, 0.5):
        assert conditional_top_share(0.0, top=share) == pytest.approx(share, abs=1e-5)


def test_perfect_correlation_is_certain() -> None:
    """The second value repeats the first exactly."""
    assert conditional_top_share(1.0) == 1.0
    assert conditional_top_share(1.0, top=0.5) == 1.0


def test_agreement_is_increasing_in_the_correlation() -> None:
    """More correlated measurements agree more often."""
    values = [conditional_top_share(rho) for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)]
    assert values == sorted(values)


def test_against_a_seeded_simulation() -> None:
    """Drawing correlated pairs reproduces the integral."""
    rng = Random(20260920)
    correlation, share = 0.6, 0.1
    cut = STANDARD.inv_cdf(1 - share)
    first_high = both_high = 0

    for _ in range(300_000):
        first = rng.gauss(0.0, 1.0)
        if first > cut:
            first_high += 1
            second = correlation * first + sqrt(1 - correlation**2) * rng.gauss(0.0, 1.0)
            both_high += second > cut

    assert conditional_top_share(correlation, top=share) == pytest.approx(
        both_high / first_high, abs=0.012
    )


def test_a_finer_grid_changes_little() -> None:
    """The default step is fine enough that halving it barely moves the answer."""
    coarse = conditional_top_share(0.5, step=0.001)
    fine = conditional_top_share(0.5, step=0.0002)

    assert coarse == pytest.approx(fine, abs=5e-4)


def test_a_wider_definition_of_the_top_agrees_more_often() -> None:
    """Calling half of everything 'high' makes agreement easy."""
    assert conditional_top_share(0.5, top=0.5) > conditional_top_share(0.5, top=0.05)


def test_invalid_inputs() -> None:
    """Correlations outside [0, 1], empty tops, and zero steps are rejected."""
    for correlation in (-0.1, 1.1):
        with pytest.raises(ValueError):
            conditional_top_share(correlation)
    for share in (0.0, 1.0):
        with pytest.raises(ValueError):
            conditional_top_share(0.5, top=share)
    with pytest.raises(ValueError):
        conditional_top_share(0.5, step=0.0)
    with pytest.raises(TypeError):
        conditional_top_share(True)
