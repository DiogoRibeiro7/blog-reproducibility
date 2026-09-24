"""Check the three fits against what each method guarantees, then the figure's claim.

Least-squares polynomials reproduce polynomials of their own degree, and the
smoothing spline keeps its residual sum of squares at the smoothing factor.

The figure's title says the spline bends locally while the degree-10 polynomial
wobbles globally. On a sharp peak that holds: the spline is closer to the true
curve than the polynomial overall and at the edges, in the figure and for almost
every other seed.
"""

import numpy as np
import pytest

from blog_reproducibility.data_science.splines import (
    NOISE_SD,
    POLYNOMIAL_DEGREE,
    SMOOTHING_PER_POINT,
    example_payload,
    fit_curves,
    fit_errors,
    true_curve,
)

SUMMARY = {row.fit: row for row in example_payload()}


def test_spline_meets_its_smoothing_bound() -> None:
    """The residual sum of squares equals the smoothing factor, n times 0.025."""
    fits = fit_curves()
    residual = float(np.sum((fits.y - fits.spline) ** 2))

    assert residual == pytest.approx(fits.x.size * SMOOTHING_PER_POINT, rel=1e-3)


def test_polynomial_reproduces_polynomials_of_its_degree() -> None:
    """A degree-10 fit through an exact degree-10 polynomial recovers it."""
    x = np.linspace(-1, 1, 90)
    exact = np.polyval(np.arange(1, POLYNOMIAL_DEGREE + 2, dtype=float), x)
    fitted = np.polyval(np.polyfit(x, exact, POLYNOMIAL_DEGREE), x)

    np.testing.assert_allclose(fitted, exact, atol=1e-8)


def test_polynomial_matches_numpy_polynomial_fit() -> None:
    """The degree-10 fit agrees with NumPy's scaled-domain Polynomial.fit."""
    fits = fit_curves()
    reference = np.polynomial.Polynomial.fit(fits.x, fits.y, POLYNOMIAL_DEGREE)

    np.testing.assert_allclose(fits.polynomial, reference(fits.x), rtol=1e-6)


def test_curved_fits_beat_the_line() -> None:
    """Both flexible fits follow the peak a straight line cannot."""
    line = SUMMARY["line"].overall

    assert SUMMARY["spline"].overall < line / 2
    assert SUMMARY["polynomial"].overall < line / 2


def test_spline_is_the_closer_fit_in_the_figure() -> None:
    """The polynomial's error is about twice the spline's overall, more at the edges."""
    spline, polynomial = SUMMARY["spline"], SUMMARY["polynomial"]

    assert spline.overall < 0.6 * polynomial.overall
    assert spline.edges < polynomial.edges / 2


def test_spline_follows_the_signal_not_the_noise() -> None:
    """The spline sits inside the noise level; the polynomial's wobble does not."""
    assert SUMMARY["spline"].overall < NOISE_SD
    assert SUMMARY["spline"].edges < NOISE_SD / 2
    assert SUMMARY["polynomial"].overall > NOISE_SD


def test_spline_is_the_closer_fit_for_almost_every_seed() -> None:
    """Over 200 seeds the spline beats the polynomial overall and at the edges."""
    wins = 0
    for seed in range(200):
        errors = {row.fit: row for row in fit_errors(fit_curves(seed=seed))}
        spline, polynomial = errors["spline"], errors["polynomial"]
        wins += spline.overall < polynomial.overall and spline.edges < polynomial.edges

    assert wins >= 190


def test_true_curve() -> None:
    """A peak of 2.5 at x = 5, half-width 0.5, on a trend of 0.15 x."""
    assert true_curve([0.0, 4.5, 5.0, 10.0]) == pytest.approx(
        [2.5 / 101, 1.25 + 0.675, 2.5 + 0.75, 2.5 / 101 + 1.5]
    )


def test_fits_are_deterministic() -> None:
    """The same seed reproduces every fit."""
    first, second = fit_curves(seed=4), fit_curves(seed=4)

    np.testing.assert_array_equal(first.spline, second.spline)
    np.testing.assert_array_equal(first.polynomial, second.polynomial)
