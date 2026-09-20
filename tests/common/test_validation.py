"""Tests for the shared input checks."""

import pytest

from blog_reproducibility.common.validation import (
    count,
    non_negative,
    positive,
    probability,
    real,
)


def test_real_accepts_numbers_and_rejects_booleans() -> None:
    """A boolean is an int in Python, and no model means to accept one."""
    assert real(3, name="x") == 3.0
    assert real(-2.5, name="x") == -2.5

    for value in (True, False, "1", None, 1j):
        with pytest.raises(TypeError):
            real(value, name="x")


def test_real_rejects_non_finite_values() -> None:
    """Infinities and nan would propagate silently through every formula."""
    for value in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            real(value, name="x")


def test_positive_and_non_negative_boundaries() -> None:
    """Zero is allowed by one and rejected by the other."""
    assert positive(1e-12, name="x") == 1e-12
    assert non_negative(0.0, name="x") == 0.0

    with pytest.raises(ValueError):
        positive(0.0, name="x")
    with pytest.raises(ValueError):
        non_negative(-1e-12, name="x")


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_probability_inclusive_range(value: float) -> None:
    """The closed range accepts both endpoints."""
    assert probability(value, name="p") == value


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1])
def test_probability_open_range_rejects_endpoints(value: float) -> None:
    """The open range rejects both endpoints and anything outside."""
    with pytest.raises(ValueError):
        probability(value, name="p", inclusive=False)


def test_count_requires_an_integer_at_or_above_the_minimum() -> None:
    """Floats and booleans are rejected even when they look like counts."""
    assert count(0, name="n") == 0
    assert count(5, name="n", minimum=5) == 5

    with pytest.raises(ValueError):
        count(4, name="n", minimum=5)
    for value in (1.0, True, "3"):
        with pytest.raises(TypeError):
            count(value, name="n")
