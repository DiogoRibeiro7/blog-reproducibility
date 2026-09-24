"""Check the measurement error model against the website's code, closed forms and the article.

The figure and the article's first three code blocks draw from one generator
seeded at 0 in the same order, so the computation runs once here and a
transcription of the website generator, with the article's two-predictor
regressions added where the figure only consumes the draws, gives identical
slopes, coefficients and fits. Every number the article prints from them is
pinned at its printed precision, and all of them match: the attenuation table,
the correlation of 0.67 and the leakage table, the SIMEX slopes 0.597 to 0.335,
the extrapolations 0.836 and 0.982, regression calibration 0.996 and the fitted
denominator 2.55 against the theoretical 2.50.

The closed forms explain them: each fitted slope and coefficient lies within
three standard errors of its population value, the rational extrapolant with the
exact parameters passes through every exact SIMEX slope and reaches the true
slope at minus one, and a quadratic fitted to the exact slopes reaches only
0.845, so its under-correction is the extrapolant's, not the noise's. The
article's field deployment section (random forests on a generator seeded at 1)
is a different computation and is not reproduced.
"""

from math import sqrt

import numpy as np
import pytest
from scipy.optimize import curve_fit

from blog_reproducibility.statistics.measurement_error import (
    ADDED_NOISE_SDS,
    OBSERVATIONS,
    RELIABILITIES,
    SIMEX_MULTIPLES,
    attenuated_slope,
    example_payload,
    fit_quadratic,
    fit_rational,
    leakage_coefficients,
    noise_sd_for_reliability,
    observed_correlation,
    ols_coefficients,
    predictor_correlation,
    rational_extrapolant,
    rational_parameters,
    simex_slope,
    simulate,
)

SUMMARY = example_payload()
SIMEX = SUMMARY.simex


def _website() -> dict[str, list[float]]:
    """Transcribe the website generator, with the article's two-predictor regressions."""
    r = np.random.default_rng(0)
    n = 20000
    x_true = r.normal(size=n)
    y = 1.0 * x_true + r.normal(scale=1.0, size=n)
    rels = [1.0, 0.8, 0.6, 0.4, 0.2]
    slopes_rel = []
    for rel in rels:
        su = np.sqrt((1 - rel) / rel)
        slopes_rel.append(float(np.polyfit(x_true + r.normal(scale=su, size=n), y, 1)[0]))
    z = r.normal(size=n)
    x1 = z + r.normal(scale=0.7, size=n)
    x2 = z + r.normal(scale=0.7, size=n)
    y2 = 1.0 * x1 + 0.0 * x2 + r.normal(size=n)
    leakage: list[float] = []
    for su in (0.0, 0.5, 1.0, 1.5):
        design = np.column_stack([np.ones(n), x1 + r.normal(scale=su, size=n), x2])
        leakage.extend(float(b) for b in np.linalg.lstsq(design, y2, rcond=None)[0][1:])
    rel = 0.6
    su = np.sqrt((1 - rel) / rel)
    x_obs = x_true + r.normal(scale=su, size=n)
    lams = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    slopes = np.array(
        [
            np.mean(
                [
                    np.polyfit(x_obs + r.normal(scale=np.sqrt(k) * su, size=n), y, 1)[0]
                    for _ in range(20)
                ]
            )
            for k in lams
        ]
    )
    quad = np.polyfit(lams, slopes, 2)

    def rational(lam: float, a: float, b: float) -> float:
        return a / (b + lam)

    (a, b), _ = curve_fit(rational, lams, slopes, p0=[1.0, 2.0])
    return {
        "attenuation": slopes_rel,
        "leakage": leakage,
        "slopes": [float(value) for value in slopes],
        "quadratic": [float(value) for value in quad],
        "rational": [float(a), float(b)],
        "correlation": [float(np.corrcoef(x1, x2)[0, 1])],
    }


def _slope_se(reliability: float) -> float:
    """Standard error of a slope on a predictor of reliability ``lambda``.

    The outcome has variance two and the noisy predictor ``1 / lambda``, which
    explains ``lambda`` of the outcome's variance; the residual variance
    ``2 - lambda`` over ``n / lambda`` gives ``sqrt(lambda (2 - lambda) / n)``.
    """
    return sqrt(reliability * (2 - reliability) / OBSERVATIONS)


def test_the_computation_is_the_websites() -> None:
    """Same generator, same order: identical slopes, coefficients and fits."""
    website = _website()
    assert [row.fitted_slope for row in SUMMARY.attenuation] == website["attenuation"]
    ours = [c for row in SUMMARY.leakage for c in (row.coefficient_x1, row.coefficient_x2)]
    assert ours == website["leakage"]
    assert list(SIMEX.slopes) == website["slopes"]
    assert list(SIMEX.quadratic) == website["quadratic"]
    assert [SIMEX.rational_numerator, SIMEX.rational_denominator] == website["rational"]
    assert [SUMMARY.predictor_correlation] == website["correlation"]


def test_the_article_attenuation_table() -> None:
    """Fitted slopes 1.003, 0.808, 0.601, 0.397 and 0.196 against the reliability."""
    assert tuple(row.reliability for row in SUMMARY.attenuation) == RELIABILITIES
    fitted = tuple(round(row.fitted_slope, 3) for row in SUMMARY.attenuation)
    assert fitted == (1.003, 0.808, 0.601, 0.397, 0.196)
    predicted = tuple(round(row.predicted_slope, 3) for row in SUMMARY.attenuation)
    assert predicted == (1.000, 0.800, 0.600, 0.400, 0.200)


def test_the_article_leakage_table() -> None:
    """Correlated at 0.67; x1 falls from 0.999 to 0.263 while x2 rises from zero to 0.496."""
    assert round(SUMMARY.predictor_correlation, 2) == 0.67
    assert tuple(row.added_noise_sd for row in SUMMARY.leakage) == ADDED_NOISE_SDS
    table = tuple(
        (round(row.coefficient_x1, 3), round(row.coefficient_x2, 3)) for row in SUMMARY.leakage
    )
    assert table == ((0.999, 0.000), (0.761, 0.159), (0.447, 0.369), (0.263, 0.496))
    # The prose: "from 1.0 to 0.26" and "from zero to 0.50".
    first, last = SUMMARY.leakage[0], SUMMARY.leakage[-1]
    assert (round(first.coefficient_x1, 1), round(last.coefficient_x1, 2)) == (1.0, 0.26)
    assert (round(first.coefficient_x2, 2), round(last.coefficient_x2, 2)) == (0.0, 0.50)


def test_the_article_simex_numbers() -> None:
    """Slopes 0.597 to 0.335; corrected 0.836, 0.982 and 0.996; b is 2.55 against 2.50."""
    assert SIMEX.multiples == SIMEX_MULTIPLES
    assert tuple(round(value, 3) for value in SIMEX.slopes) == (0.597, 0.499, 0.430, 0.376, 0.335)
    assert round(SIMEX.quadratic_estimate, 3) == 0.836
    assert round(SIMEX.rational_estimate, 3) == 0.982
    assert round(SIMEX.regression_calibration, 3) == 0.996
    assert round(SIMEX.rational_denominator, 2) == 2.55
    assert round(SIMEX.exact_denominator, 2) == 2.50


def test_the_simulation_agrees_with_the_closed_forms() -> None:
    """Every fitted slope and coefficient within three standard errors of its population value."""
    for row in SUMMARY.attenuation:
        assert abs(row.fitted_slope - row.predicted_slope) < 3 * _slope_se(row.reliability)
    for slope, exact in zip(SIMEX.slopes, SIMEX.exact_slopes, strict=True):
        # With a unit true slope, the exact slope is also the effective reliability.
        assert abs(slope - exact) < 3 * _slope_se(exact)
    assert abs(SIMEX.regression_calibration - 1.0) < 3 * _slope_se(0.6) / 0.6

    # The normal equations of the population regression, solved numerically.
    p2 = 0.7**2
    for leak in SUMMARY.leakage:
        covariance = np.array([[1 + p2 + leak.added_noise_sd**2, 1.0], [1.0, 1 + p2]])
        population = np.linalg.solve(covariance, [1 + p2, 1.0])
        assert (leak.exact_x1, leak.exact_x2) == pytest.approx(tuple(population), rel=1e-12)
        residual = (1 + p2) + 1 - float(population @ [1 + p2, 1.0])
        errors = np.sqrt(residual * np.diag(np.linalg.inv(covariance)) / OBSERVATIONS)
        assert abs(leak.coefficient_x1 - leak.exact_x1) < 3 * errors[0]
        assert abs(leak.coefficient_x2 - leak.exact_x2) < 3 * errors[1]
    correlation_se = (1 - predictor_correlation() ** 2) / sqrt(OBSERVATIONS)
    gap = SUMMARY.predictor_correlation - SUMMARY.exact_predictor_correlation
    assert abs(gap) < 3 * correlation_se


def test_the_figure_claims() -> None:
    """Noise pulls the slope toward zero; the quadratic under-corrects, the rational recovers it."""
    fitted = [row.fitted_slope for row in SUMMARY.attenuation]
    assert fitted == sorted(fitted, reverse=True)
    for row in SUMMARY.attenuation[1:]:
        assert row.fitted_slope < SUMMARY.true_slope - 5 * _slope_se(row.reliability)
    assert SIMEX.quadratic_estimate < 0.85
    assert abs(SIMEX.rational_estimate - SUMMARY.true_slope) < 0.02
    assert abs(SIMEX.rational_estimate - 1) < abs(SIMEX.quadratic_estimate - 1) / 5
    # Without any noise in the slopes, the quadratic still stops at 0.845.
    assert round(SIMEX.exact_quadratic_estimate, 3) == 0.845
    # The fitted curves pass near the refits they were fitted to.
    fitted_rational = rational_extrapolant(
        SIMEX.multiples, SIMEX.rational_numerator, SIMEX.rational_denominator
    )
    assert fitted_rational == pytest.approx(SIMEX.slopes, abs=0.002)
    assert np.polyval(SIMEX.quadratic, SIMEX.multiples) == pytest.approx(SIMEX.slopes, abs=0.005)


def test_the_closed_forms() -> None:
    """Spearman's 0.3, the rational form with exact parameters, and the leakage limits."""
    assert observed_correlation(0.5, 0.6, 0.6) == pytest.approx(0.3)
    assert noise_sd_for_reliability(0.6) ** 2 == pytest.approx(2 / 3)
    assert attenuated_slope(0.6, 2.0) == pytest.approx(1.2)
    a, b = rational_parameters(0.6)
    assert (a, b) == pytest.approx((1.5, 2.5))
    for multiple in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 7.0):
        assert float(rational_extrapolant(multiple, a, b)) == pytest.approx(
            simex_slope(multiple, 0.6)
        )
    assert simex_slope(-1.0, 0.6) == pytest.approx(1.0)
    assert simex_slope(0.0, 0.6) == pytest.approx(attenuated_slope(0.6))
    assert predictor_correlation() == pytest.approx(1 / 1.49)
    assert leakage_coefficients(0.0) == pytest.approx((1.0, 0.0))
    assert leakage_coefficients(1e6) == pytest.approx((0.0, 1 / 1.49), abs=1e-9)
    # Refitting on exact data returns the exact parameters.
    multiples = np.array(SIMEX_MULTIPLES)
    exact = [simex_slope(float(k), 0.6) for k in multiples]
    assert fit_rational(multiples, exact) == pytest.approx((1.5, 2.5), rel=1e-6)
    # A quadratic fits the exact slopes to half a point on the data, and still misses at -1.
    assert np.polyval(fit_quadratic(multiples, exact), multiples) == pytest.approx(exact, abs=0.005)


def test_ols_coefficients_recover_a_known_plane() -> None:
    """Noise-free data: the intercept is dropped and the slopes are exact."""
    x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [4.0, 1.0]])
    y = 5.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1]
    assert ols_coefficients(x, y) == pytest.approx((2.0, -3.0))


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a run; another seed does not."""
    assert simulate(3, observations=300, refits=2) == simulate(3, observations=300, refits=2)
    assert simulate(3, observations=300, refits=2) != simulate(4, observations=300, refits=2)


def test_invalid_inputs_are_rejected() -> None:
    """Reliabilities outside (0, 1], impossible multiples and mismatched data are refused."""
    with pytest.raises(ValueError):
        noise_sd_for_reliability(0.0)
    with pytest.raises(ValueError):
        noise_sd_for_reliability(1.2)
    with pytest.raises(TypeError):
        attenuated_slope(True)
    with pytest.raises(ValueError):
        observed_correlation(1.5, 0.6, 0.6)
    with pytest.raises(ValueError):
        rational_parameters(1.0)
    with pytest.raises(ValueError):
        simex_slope(-3.0, 0.6)
    with pytest.raises(ValueError):
        leakage_coefficients(-1.0)
    with pytest.raises(ValueError):
        predictor_correlation(-0.1)
    with pytest.raises(ValueError):
        ols_coefficients(np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError):
        ols_coefficients(np.zeros((4, 2)), np.zeros(3))
    with pytest.raises(ValueError):
        simulate(observations=2)
    with pytest.raises(ValueError):
        simulate(refits=0)
