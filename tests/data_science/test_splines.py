"""Check the three fits against what each method guarantees.

Least-squares polynomials reproduce polynomials of their own degree, and the
smoothing spline keeps its residual sum of squares at the smoothing factor.

The figure's title says the degree-10 polynomial wobbles while the spline
follows the shape. With this design that does not hold: the polynomial is the
closer fit to the true curve for most seeds, at the edges as well as the
interior. The tests therefore check the fits' guarantees and leave the title's
comparison untested; the migration notes record it for the article to revisit.
"""

import numpy as np
import pytest

from blog_reproducibility.data_science.splines import (
    POLYNOMIAL_DEGREE,
    SMOOTHING_PER_POINT,
    example_payload,
    fit_curves,
    true_curve,
)

SUMMARY = {row.fit: row for row in example_payload()}


def test_spline_meets_its_smoothing_bound() -> None:
    """The residual sum of squares equals the smoothing factor, n times 0.09."""
    fits = fit_curves()
    residual = float(np.sum((fits.y - fits.spline) ** 2))

    assert residual == pytest.approx(fits.x.size * SMOOTHING_PER_POINT, rel=1e-3)


def test_polynomial_reproduces_polynomials_of_its_degree() -> None:
    """A degree-10 fit through an exact degree-10 polynomial recovers it."""
    x = np.linspace(-1, 1, 90)
    exact = np.polyval(np.arange(1, POLYNOMIAL_DEGREE + 2, dtype=float), x)
    fitted = np.polyval(np.polyfit(x, exact, POLYNOMIAL_DEGREE), x)

    np.testing.assert_allclose(fitted, exact, atol=1e-8)


def test_curved_fits_beat_the_line() -> None:
    """Both flexible fits follow the curvature a straight line cannot."""
    line = SUMMARY["line"].overall

    assert SUMMARY["spline"].overall < line / 2
    assert SUMMARY["polynomial"].overall < line / 2


def test_fits_are_close_to_the_truth_but_not_to_the_noise() -> None:
    """Both flexible fits sit well inside the noise level of 0.28."""
    for fit in ("spline", "polynomial"):
        assert SUMMARY[fit].overall < 0.28


def test_true_curve() -> None:
    """sin(x) + 0.15 x at a few points."""
    assert true_curve([0.0, np.pi / 2, 10.0]) == pytest.approx(
        [0.0, 1 + 0.15 * np.pi / 2, np.sin(10) + 1.5]
    )


def test_fits_are_deterministic() -> None:
    """The same seed reproduces every fit."""
    first, second = fit_curves(seed=4), fit_curves(seed=4)

    np.testing.assert_array_equal(first.spline, second.spline)
    np.testing.assert_array_equal(first.polynomial, second.polynomial)
