"""Check the quantile regression model against the website, the article and closed forms.

The figure and the article are one computation from a generator seeded at 0, so
the website's draws are transcribed and matched bit for bit, and every number
the article prints for least squares and linear quantile regression is pinned
at its printed precision: the coefficient table, coverage and width by load
segment, the misses above and below, the pinball losses, the mean and median,
and the share of deliveries later than the least-squares prediction. The
article fits with statsmodels' iterative ``QuantReg``; the exact linear program
solved here gives the same printed numbers (statsmodels stops within about
``2e-6`` of the optimum). The boosting columns are not reproduced.

The fits are checked independently of any solver: the dual weights at each
vertex certify optimality, an intercept-only fit is the sample quantile,
noise-free data are fitted exactly, and scikit-learn's ``QuantileRegressor``
finds the same vertex. The model's conditional quantiles are linear in distance
and load, so the fitted coefficients are compared with their population values
in Koenker's asymptotic standard errors (all within 2.3), and the coverage of
each band in each segment with its expected coverage under the model.

Three statements do not hold as written, and the tests pin what is true:

* least squares' 1.17 minutes per unit of load on the mean "is true": load has
  no effect on the mean in the simulation, and 1.17 is 2.3 standard errors of
  sampling noise from zero;
* a saturated network adds "nothing to the median": the fitted median falls by
  1.75 minutes from no load to full load, and the true median by 2.73, because
  a wider right-skewed spread pulls the median below the mean;
* the 1.17 is "the average of a large effect on bad deliveries and none on
  typical ones": averaged over every quantile the load effect is exactly zero,
  the mean effect, and the typical (median) delivery gets faster.

The figure's claims hold: the least-squares band has constant width, the
quantile band is eight times wider at full load than at none, and at full load
the 95th percentile sits more than twice as far above the median as the 5th
sits below it.
"""

from math import exp, sqrt

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate, stats
from sklearn.linear_model import QuantileRegressor

from blog_reproducibility.statistics.quantile_regression import (
    LOAD_SEGMENTS,
    NORMAL_MARGIN,
    QUANTILES,
    TRAINING,
    band_coverage,
    conditional_quantile,
    coverage,
    example_payload,
    late_share_at_mean,
    least_squares,
    pinball_loss,
    population_coefficients,
    promise_quantile,
    quantile_regression,
    shock_quantile,
    simulate_deliveries,
    simulate_fixed_distance,
)

SUMMARY = example_payload()
OLS, Q05, Q50, Q95 = SUMMARY.fits
FIGURE = SUMMARY.figure
DATA = simulate_deliveries(np.random.default_rng(0))
X_TRAIN = np.column_stack([DATA.distance, DATA.load])[:TRAINING]
Y_TRAIN = DATA.minutes[:TRAINING]
FITS = {q: quantile_regression(X_TRAIN, Y_TRAIN, q) for q in QUANTILES}


def _website_draws() -> tuple[NDArray[np.float64], ...]:
    """Transcribe the website generator's draws, in its order."""
    r = np.random.default_rng(0)
    n = 6000
    distance = r.uniform(1, 10, n)
    load = r.uniform(0, 1, n)
    mu = 20 + 3.0 * distance
    sigma = 2 + 12 * load
    y = mu + sigma * (r.gamma(2.0, 1.0, n) - 2.0) / np.sqrt(2.0)
    m = 2500
    lp = r.uniform(0, 1, m)
    yp = 20 + 15.0 + (2 + 12 * lp) * (r.gamma(2.0, 1.0, m) - 2.0) / np.sqrt(2.0)
    return distance, load, y, lp, yp


def _certificate(
    predictors: NDArray[np.float64], outcome: NDArray[np.float64], beta: tuple[float, ...], q: float
) -> NDArray[np.float64]:
    """Dual weights of the observations a vertex interpolates; in [0, 1] if the fit is optimal.

    Optimality needs weights ``a`` with ``X'a = (1 - q) X'1``, ``a = 1`` above the
    fit, ``a = 0`` below it and ``0 <= a <= 1`` on it (the dual of the program).
    """
    design = np.column_stack([np.ones(outcome.size), predictors])
    residual = outcome - design @ np.asarray(beta)
    on_fit = np.abs(residual) < 1e-8
    above = residual >= 1e-8
    target = (1 - q) * design.sum(axis=0) - design[above].sum(axis=0)
    weights: NDArray[np.float64] = np.linalg.solve(design[on_fit].T, target)
    return weights


def _density_at_quantile(q: float, load: NDArray[np.float64]) -> NDArray[np.float64]:
    """Conditional density of the delivery time at its ``q``-quantile."""
    shock = sqrt(2.0) * float(stats.gamma.pdf(2.0 + sqrt(2.0) * shock_quantile(q), 2.0))
    density: NDArray[np.float64] = shock / (2.0 + 12.0 * load)
    return density


def test_the_draws_are_the_websites() -> None:
    """Deliveries, then the figure's fresh deliveries at 5 km, bit for bit."""
    distance, load, y, lp, yp = _website_draws()
    assert np.array_equal(DATA.distance, distance)
    assert np.array_equal(DATA.load, load)
    assert np.array_equal(DATA.minutes, y)
    rng = np.random.default_rng(0)
    simulate_deliveries(rng)
    fresh = simulate_fixed_distance(rng)
    assert np.array_equal(fresh.load, lp)
    assert np.array_equal(fresh.minutes, yp)
    assert np.array_equal(FIGURE.loads, lp)
    assert np.array_equal(FIGURE.minutes, yp)
    assert np.all(fresh.distance == 5.0)


def test_the_article_coefficients() -> None:
    """Least squares 19.46, 3.03, 1.17; quantiles 17.50, 3.02, -13.60 to 22.53, 3.09, 25.32."""
    table = [(row.intercept, row.distance, row.load) for row in SUMMARY.fits]
    printed = [
        (19.46, 3.03, 1.17),
        (17.50, 3.02, -13.60),
        (19.27, 3.01, -1.75),
        (22.53, 3.09, 25.32),
    ]
    for fitted, article in zip(table, printed, strict=True):
        assert tuple(round(value, 2) for value in fitted) == article
    assert [row.quantile for row in SUMMARY.fits] == [None, 0.05, 0.5, 0.95]
    # Distance costs three minutes per kilometre at every quantile.
    assert all(round(row.distance) == 3 for row in SUMMARY.fits)


def test_the_article_coverage_table() -> None:
    """Coverage 99, 96, 93, 83 against 87, 90, 88, 90 percent; widths 29.0 against 10.1 to 39.5."""
    rows = SUMMARY.segments
    assert [(row.lower, row.upper) for row in rows] == list(LOAD_SEGMENTS)
    assert [round(100 * row.ols_coverage) for row in rows] == [99, 96, 93, 83]
    assert [round(100 * row.quantile_coverage) for row in rows] == [87, 90, 88, 90]
    assert [round(row.ols_width, 1) for row in rows] == [29.0] * 4
    assert [round(row.quantile_width, 1) for row in rows] == [10.1, 20.1, 29.8, 39.5]
    assert sum(row.deliveries for row in rows) == 2000
    misses = SUMMARY.misses
    assert (round(100 * misses.ols_coverage), round(100 * misses.quantile_coverage)) == (93, 89)
    assert (round(100 * misses.ols_above, 1), round(100 * misses.ols_below, 1)) == (5.1, 2.0)
    assert (round(100 * misses.quantile_above, 1), round(100 * misses.quantile_below, 1)) == (
        5.2,
        5.9,
    )
    # "Three times wider than needed" at low load; "from 10 minutes to 40".
    assert rows[0].ols_width / rows[0].quantile_width == pytest.approx(3.0, abs=0.15)
    # "Within a few points of 90 percent in every segment."
    assert all(abs(row.quantile_coverage - 0.9) < 0.035 for row in rows)


def test_the_article_pinball_table() -> None:
    """Least squares 0.739, 3.070, 1.257 against 0.500, 2.950, 1.145: a third less in the tail."""
    rows = SUMMARY.pinball
    assert [row.quantile for row in rows] == list(QUANTILES)
    assert [round(row.ols, 3) for row in rows] == [0.739, 3.070, 1.257]
    assert [round(row.quantile_regression, 3) for row in rows] == [0.500, 2.950, 1.145]
    assert all(row.quantile_regression < row.ols for row in rows)
    assert 1 - rows[0].quantile_regression / rows[0].ols == pytest.approx(1 / 3, abs=0.02)


def test_the_article_promise_numbers() -> None:
    """Median 35.9 against a mean of 36.6; 40 percent later than the least-squares prediction."""
    assert round(SUMMARY.median_minutes, 1) == 35.9
    assert round(SUMMARY.mean_minutes, 1) == 36.6
    # 803 of the 2,000 test deliveries, "40 percent".
    assert round(2000 * SUMMARY.late_share) == 803
    assert round(100 * SUMMARY.late_share) == 40
    assert [round(row.quantile, 2) for row in SUMMARY.promises] == [0.5, 0.8, 0.9]
    # The population value: a delivery exceeds its mean with probability 3 exp(-2).
    assert SUMMARY.population_late_share == pytest.approx(3 * exp(-2), rel=1e-12)
    se = sqrt(SUMMARY.population_late_share * (1 - SUMMARY.population_late_share) / 2000)
    assert abs(SUMMARY.late_share - SUMMARY.population_late_share) < 3 * se
    assert SUMMARY.population_mean_minutes == 36.5


def test_the_fits_are_optimal() -> None:
    """Each article fit passes through three deliveries with dual weights inside [0, 1]."""
    for q, beta in FITS.items():
        weights = _certificate(X_TRAIN, Y_TRAIN, beta, q)
        assert weights.size == 3
        assert np.all((weights > -1e-9) & (weights < 1 + 1e-9))
        # Consequently no more than a share q of the deliveries lies below the fit.
        design = np.column_stack([np.ones(TRAINING), X_TRAIN])
        residual = Y_TRAIN - design @ np.asarray(beta)
        below = int(np.count_nonzero(residual < -1e-8))
        assert below <= q * TRAINING <= below + 3


def test_the_summary_uses_the_fits() -> None:
    """The coefficient table, the figure's lines and the refits agree."""
    for row in (Q05, Q50, Q95):
        assert row.quantile is not None
        assert (row.intercept, row.distance, row.load) == FITS[row.quantile]
    coefficients, residual_sd = least_squares(X_TRAIN, Y_TRAIN)
    assert (OLS.intercept, OLS.distance, OLS.load) == coefficients
    assert SUMMARY.residual_sd == residual_sd
    grid = np.array(FIGURE.grid)
    assert np.array_equal(grid, np.linspace(0, 1, 100))
    for line, row in ((FIGURE.lower, Q05), (FIGURE.median, Q50), (FIGURE.upper, Q95)):
        expected = row.intercept + 5.0 * row.distance + row.load * grid
        assert np.allclose(line, expected, rtol=0, atol=1e-12)
    centre = OLS.intercept + 5.0 * OLS.distance + OLS.load * grid
    margin = NORMAL_MARGIN * residual_sd
    assert np.allclose(FIGURE.ols_upper, centre + margin, rtol=0, atol=1e-12)
    assert np.allclose(FIGURE.ols_lower, centre - margin, rtol=0, atol=1e-12)


def test_scikit_learn_finds_the_same_vertex() -> None:
    """``QuantileRegressor`` with HiGHS on the first thousand deliveries, at each quantile."""
    x, y = X_TRAIN[:1000], Y_TRAIN[:1000]
    for q in QUANTILES:
        model = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(x, y)
        theirs = [float(model.intercept_), *(float(c) for c in model.coef_)]
        assert quantile_regression(x, y, q) == pytest.approx(theirs, rel=1e-10)


def test_least_squares_matches_the_normal_equations() -> None:
    """The coefficients solve ``X'X b = X'y``; the residual deviation divides by n."""
    design = np.column_stack([np.ones(TRAINING), X_TRAIN])
    coefficients, residual_sd = least_squares(X_TRAIN, Y_TRAIN)
    solved = np.linalg.solve(design.T @ design, design.T @ Y_TRAIN)
    assert coefficients == pytest.approx(solved.tolist(), rel=1e-10)
    residual = Y_TRAIN - design @ solved
    assert residual_sd == pytest.approx(sqrt(float(np.mean(residual**2))), rel=1e-10)


def test_an_intercept_only_fit_is_the_sample_quantile() -> None:
    """With no predictors the minimiser is the order statistic at ``ceil(n q)``."""
    y = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0, 3.5, 8.0])
    none = np.empty((9, 0))
    assert quantile_regression(none, y, 0.5) == pytest.approx((3.5,))
    assert quantile_regression(none, y, 0.2) == pytest.approx((1.5,))
    assert quantile_regression(none, y, 0.9) == pytest.approx((9.0,))


def test_noise_free_data_are_fitted_exactly() -> None:
    """A response that is exactly linear returns its coefficients at every quantile."""
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, (40, 2))
    y = 2.0 + 0.5 * x[:, 0] - 3.0 * x[:, 1]
    for q in (0.1, 0.5, 0.9):
        assert quantile_regression(x, y, q) == pytest.approx((2.0, 0.5, -3.0), abs=1e-9)
    # A one-dimensional predictor is taken as a single column.
    assert quantile_regression(x[:, 0], 1.0 + 4.0 * x[:, 0], 0.3) == pytest.approx(
        (1.0, 4.0), abs=1e-9
    )


def test_the_conditional_quantiles_are_the_models() -> None:
    """At the closed-form quantile the gamma distribution function returns ``q``."""
    for q in (0.05, 0.3, 0.5, 0.95):
        for distance, load in ((1.0, 0.0), (5.0, 0.4), (10.0, 1.0)):
            value = conditional_quantile(q, distance, load)
            shock = (value - 20.0 - 3.0 * distance) / (2.0 + 12.0 * load)
            assert stats.gamma.cdf(2.0 + sqrt(2.0) * shock, 2.0) == pytest.approx(q, rel=1e-10)
    assert shock_quantile(0.5) == pytest.approx((stats.gamma.median(2.0) - 2.0) / sqrt(2.0))
    assert population_coefficients() == (20.0, 3.0, 0.0)
    # Averaged over every quantile, the load effect is the mean effect, zero.
    average, _ = integrate.quad(lambda q: population_coefficients(q)[2], 0, 1, limit=200)
    assert average == pytest.approx(0.0, abs=1e-7)
    assert late_share_at_mean() == pytest.approx(3 * exp(-2), rel=1e-12)
    assert late_share_at_mean(1.0) == pytest.approx(exp(-1), rel=1e-12)


def test_band_coverage_closed_forms() -> None:
    """The true quantile band covers 90 percent everywhere; a flat band matches a 1-D integral."""
    lower, upper = population_coefficients(0.05), population_coefficients(0.95)
    for segment in LOAD_SEGMENTS:
        assert band_coverage(lower, upper, load_range=segment) == pytest.approx(0.9, abs=1e-8)
    margin = 10.0

    def flat(load: float) -> float:
        scale = (2.0 + 12.0 * load) / sqrt(2.0)
        return float(
            stats.gamma.cdf(2.0 + margin / scale, 2.0) - stats.gamma.cdf(2.0 - margin / scale, 2.0)
        )

    band = band_coverage((20.0 - margin, 3.0, 0.0), (20.0 + margin, 3.0, 0.0), load_range=(0.5, 1))
    assert band == pytest.approx(2 * integrate.quad(flat, 0.5, 1)[0], rel=1e-8)
    assert band_coverage(upper, lower) == 0.0


def test_the_segment_coverage_matches_its_expectation() -> None:
    """Each segment's test coverage lies within three binomial errors of its expected value."""
    for row in SUMMARY.segments:
        for simulated, expected in (
            (row.ols_coverage, row.expected_ols_coverage),
            (row.quantile_coverage, row.expected_quantile_coverage),
        ):
            se = sqrt(expected * (1 - expected) / row.deliveries)
            assert abs(simulated - expected) < 3 * se
    # The least-squares band's failure is not noise: 99.5 falling to 81.5 percent in expectation.
    trend = [row.expected_ols_coverage for row in SUMMARY.segments]
    assert trend == sorted(trend, reverse=True)
    assert (round(100 * trend[0], 1), round(100 * trend[-1], 1)) == (99.5, 81.5)
    assert all(abs(row.expected_quantile_coverage - 0.9) < 0.012 for row in SUMMARY.segments)


def test_the_fits_are_within_sampling_error_of_the_model() -> None:
    """Every coefficient lies within 2.3 of Koenker's asymptotic standard errors of the truth."""
    design = np.column_stack([np.ones(TRAINING), X_TRAIN])
    load = X_TRAIN[:, 1]
    d0 = design.T @ design / TRAINING
    for q, beta in FITS.items():
        d1 = (design * _density_at_quantile(q, load)[:, None]).T @ design / TRAINING
        inverse = np.linalg.inv(d1)
        cov = q * (1 - q) * inverse @ d0 @ inverse / TRAINING
        z = (np.asarray(beta) - np.asarray(population_coefficients(q))) / np.sqrt(np.diag(cov))
        assert np.all(np.abs(z) < 2.3)
    # Least squares under the model's variance, (2 + 12 L)^2 for each delivery.
    bread = np.linalg.inv(design.T @ design)
    sandwich = bread @ (design * ((2.0 + 12.0 * load) ** 2)[:, None]).T @ design @ bread
    z = (np.array([OLS.intercept, OLS.distance, OLS.load]) - [20.0, 3.0, 0.0]) / np.sqrt(
        np.diag(sandwich)
    )
    assert np.all(np.abs(z) < 2.4)
    # The 1.17 minutes of load on the mean is 2.3 standard errors from its true value, zero.
    assert round(float(z[2]), 1) == 2.3


def test_the_statements_that_do_not_hold() -> None:
    """Load does nothing to the mean, and lowers the median."""
    assert OLS.population[2] == 0.0
    assert round(Q50.load, 2) == -1.75
    assert round(Q50.population[2], 2) == -2.73
    assert Q50.population[2] < 0 < Q95.population[2]


def test_the_figure_claims() -> None:
    """The least-squares band is flat; the quantile band widens with load and is asymmetric."""
    ols_width = np.array(FIGURE.ols_upper) - np.array(FIGURE.ols_lower)
    assert np.allclose(ols_width, 2 * NORMAL_MARGIN * SUMMARY.residual_sd, rtol=1e-12)
    width = np.array(FIGURE.upper) - np.array(FIGURE.lower)
    assert np.all(np.diff(width) > 0)
    assert width[-1] / width[0] > 8
    above = FIGURE.upper[-1] - FIGURE.median[-1]
    below = FIGURE.median[-1] - FIGURE.lower[-1]
    assert above > 2 * below
    # The quantile band is narrower than least squares at low load and wider at high load.
    assert width[0] < ols_width[0] / 2 and width[-1] > ols_width[-1]


def test_pinball_loss_and_coverage_by_hand() -> None:
    """Three residuals of -1, 0 and 1 at q = 0.25; two of four inside their intervals."""
    assert pinball_loss(0.25, [1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(1 / 3)
    assert pinball_loss(0.5, [1.0, 3.0], [2.0, 2.0]) == pytest.approx(0.5)
    assert coverage([1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 3.5, 5.0], [2.0, 2.0, 4.0, 6.0]) == 0.5
    assert promise_quantile(4.0, 1.0) == pytest.approx(0.8)
    assert promise_quantile(1.0, 3.0) == pytest.approx(0.25)


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the deliveries; another seed does not."""
    first = simulate_deliveries(np.random.default_rng(4), 50)
    again = simulate_deliveries(np.random.default_rng(4), 50)
    other = simulate_deliveries(np.random.default_rng(5), 50)
    assert np.array_equal(first.minutes, again.minutes)
    assert not np.array_equal(first.minutes, other.minutes)


def test_invalid_inputs_are_rejected() -> None:
    """Bad quantiles, shapes, counts and bands are refused."""
    x, y = X_TRAIN[:20], Y_TRAIN[:20]
    with pytest.raises(ValueError):
        quantile_regression(x, y, 0.0)
    with pytest.raises(ValueError):
        quantile_regression(x, y, 1.0)
    with pytest.raises(ValueError):
        quantile_regression(x[:10], y, 0.5)
    with pytest.raises(ValueError):
        quantile_regression(x[:2], y[:2], 0.5)
    with pytest.raises(ValueError):
        quantile_regression(x, np.append(y[:-1], np.nan), 0.5)
    with pytest.raises(ValueError):
        least_squares(x, [])
    with pytest.raises(ValueError):
        pinball_loss(0.5, [1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        coverage([1.0], [0.0, 1.0], [2.0])
    with pytest.raises(ValueError):
        band_coverage((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), load_range=(0.5, 0.5))
    with pytest.raises(ValueError):
        band_coverage((1.0, 0.0), (2.0, 0.0, 0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        promise_quantile(0.0, 1.0)
    with pytest.raises(ValueError):
        late_share_at_mean(0.0)
    with pytest.raises(ValueError):
        simulate_deliveries(np.random.default_rng(0), 0)
    with pytest.raises(TypeError):
        simulate_fixed_distance(np.random.default_rng(0), True)
    with pytest.raises(TypeError):
        conditional_quantile(0.5, True, 0.5)
