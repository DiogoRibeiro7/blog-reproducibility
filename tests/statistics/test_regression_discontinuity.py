"""Check the regression discontinuity model against the website, closed forms, article and figure.

The figure's 6,000 samples (600 at each of ten bandwidths, one generator seeded
at 0) run once here, in about three seconds, and every plotted bias, standard
deviation and error is pinned as the website's generator computes it. The port
is checked draw for draw against a transcription of the website's loop.

The closed forms are checked independently: the bias ``-0.0005 h^2`` and the
spread ``1.6 / sqrt(h)`` against least squares on a dense grid, the polynomial
spreads ``(p + 1)^2 sigma^2 / m`` likewise, the population limit under
manipulation against a sample of two million units, and the naive difference of
8.5 against a large sample. The figure's claims hold: the noise falls at every
step, the bias is within noise up to a bandwidth of 7 and many standard errors
out from 30, and the error is smallest at 20, next to the closed-form optimum
of 19.1.

The article's tables come from 2,000 samples per row (500 for manipulation and
sample sizes) drawn in another order, so they are not pinned; run separately,
its code reproduces every printed value.
Here each printed number is checked against the closed forms at its own noise
level: the naive 8.50 is the closed form exactly, and the manipulation rows sit
on the population limits of 1.95, 2.96 and 3.54 and density ratios of 1.0, 1.86
and 4.0. Two remarks in the prose do not hold as written: the naive comparison
sets units 50 points apart on average, not 25 (each side's mean is 25 from the
cutoff), and the degree-6 polynomial uses two and a half times the data of the
local fit at a bandwidth of 20, not five times.
"""

from math import sqrt

import numpy as np
import pytest
from numpy.typing import NDArray

from blog_reproducibility.statistics.regression_discontinuity import (
    BANDWIDTHS,
    NOISE_SD,
    REPLICATIONS,
    TRUE_EFFECT,
    UNITS,
    bandwidth_row,
    density_ratio,
    draw_sample,
    example_payload,
    local_linear,
    local_linear_bias,
    local_linear_limit,
    local_linear_sd,
    naive_difference,
    naive_difference_limit,
    optimal_bandwidth,
    polynomial_sd,
    units_in_window,
)

SUMMARY = example_payload()
ROWS = {row.bandwidth: row for row in SUMMARY.rows}

# The website generator's plotted values, to four decimals, at bandwidths 2 to 50.
WEBSITE_FIGURE = {
    "absolute_bias": (
        0.0749, 0.0609, 0.0455, 0.0138, 0.0441, 0.1266, 0.2230, 0.4504, 0.7823, 1.2520,
    ),
    "standard_deviation": (
        1.1322, 0.9125, 0.7038, 0.6116, 0.5109, 0.4024, 0.3506, 0.2895, 0.2482, 0.2340,
    ),
    "root_mean_squared_error": (
        1.1347, 0.9146, 0.7052, 0.6117, 0.5128, 0.4218, 0.4155, 0.5354, 0.8207, 1.2737,
    ),
}  # fmt: skip
ARTICLE_REPLICATIONS = 2000


def _website_draw(
    r: np.random.Generator, n: int = 5000
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transcribe the website generator's sample."""
    x = r.uniform(-50, 50, n)
    d = (x >= 0).astype(float)
    curve = np.where(x >= 0, 0.002, -0.001) * x**2
    return x, 10 + 0.08 * x + curve + 2.0 * d + r.normal(0, 4, n)


def _website_local_linear(x: NDArray[np.float64], y: NDArray[np.float64], h: float) -> float:
    """Transcribe the website generator's estimator."""
    est = []
    for side in (x >= 0, x < 0):
        m = side & (np.abs(x) <= h)
        design = np.column_stack([np.ones(m.sum()), x[m]])
        b, *_ = np.linalg.lstsq(design, y[m], rcond=None)
        est.append(b[0])
    return float(est[0] - est[1])


def _standard_error(row_sd: float, replications: int) -> float:
    return row_sd / sqrt(replications)


def _dense_intercept_variance(degree: int, points: int = 20_000) -> float:
    """Variance of a degree-p fit at the end of a uniform design, per unit noise, times m."""
    grid = (np.arange(points) + 0.5) / points
    design = np.vander(grid, degree + 1, increasing=True)
    inverse = np.linalg.inv(design.T @ design)
    return float(inverse[0, 0]) * points


def test_the_figure_values_are_reproduced() -> None:
    """Every plotted value, at every bandwidth, as the website's generator computes it."""
    assert tuple(ROWS) == BANDWIDTHS
    for row in SUMMARY.rows:
        assert row.replications == REPLICATIONS
    for field, printed in WEBSITE_FIGURE.items():
        assert tuple(round(getattr(row, field), 4) for row in SUMMARY.rows) == printed


def test_the_draws_and_estimates_match_the_website_loop() -> None:
    """Same generator, same order: identical samples and estimates at several bandwidths."""
    ours, theirs = np.random.default_rng(3), np.random.default_rng(3)
    for h in (2.0, 10.0, 50.0):
        x, d, y = draw_sample(ours)
        expected_x, expected_y = _website_draw(theirs)
        assert np.array_equal(x, expected_x)
        assert np.array_equal(y, expected_y)
        assert np.array_equal(d, (expected_x >= 0).astype(float))
        assert local_linear(x, y, h) == _website_local_linear(expected_x, expected_y, h)
    row = bandwidth_row(np.random.default_rng(4), 7, replications=6, units=800)
    generator = np.random.default_rng(4)
    website = np.array([_website_local_linear(*_website_draw(generator, 800), 7) for _ in range(6)])
    assert row.mean_estimate == float(website.mean())
    assert row.standard_deviation == float(website.std())


def test_the_bias_is_the_line_through_a_quadratic() -> None:
    """A line fitted to k x^2 on a dense uniform grid over [0, h] meets zero at -k h^2 / 6."""
    grid = (np.arange(200_000) + 0.5) / 200_000
    for h in (2.0, 20.0, 50.0):
        right = np.polynomial.polynomial.polyfit(h * grid, 0.002 * (h * grid) ** 2, 1)[0]
        left = np.polynomial.polynomial.polyfit(-h * grid, -0.001 * (h * grid) ** 2, 1)[0]
        assert right - left == pytest.approx(local_linear_bias(h), rel=1e-6)
        assert local_linear_limit(h) == pytest.approx(TRUE_EFFECT + local_linear_bias(h))
    assert local_linear_bias(50) == pytest.approx(-1.25)


def test_the_spreads_are_endpoint_variances() -> None:
    """A degree-p fit at the end of m uniform points has variance (p + 1)^2 sigma^2 / m."""
    for degree in (0, 1, 2, 4, 6):
        assert _dense_intercept_variance(degree) == pytest.approx((degree + 1) ** 2, rel=1e-3)
    assert local_linear_sd(10) == pytest.approx(sqrt(2 * 4 * NOISE_SD**2 / 500))
    assert local_linear_sd(50) == pytest.approx(polynomial_sd(1))
    for h in BANDWIDTHS:
        assert local_linear_sd(h) == pytest.approx(1.6 / sqrt(h))


def test_the_closed_forms_match_large_samples() -> None:
    """Two million units: the naive difference, and the estimate and density under manipulation."""
    rng = np.random.default_rng(5)
    x, _, y = draw_sample(rng, 2_000_000)
    assert naive_difference(x, y) == pytest.approx(naive_difference_limit(), abs=0.03)
    x, _, y = draw_sample(rng, 2_000_000, manipulation=0.3)
    spread = local_linear_sd(10, 2_000_000)
    assert abs(local_linear(x, y, 10) - local_linear_limit(10, manipulation=0.3)) < 4 * spread
    ratio = np.count_nonzero((x >= 0) & (x < 5)) / np.count_nonzero((x < 0) & (x > -5))
    assert ratio == pytest.approx(density_ratio(0.3), abs=0.04)


def test_the_simulation_matches_the_closed_forms() -> None:
    """Each bandwidth's mean within four standard errors, and its spread within ten percent."""
    for h, row in ROWS.items():
        error = _standard_error(row.expected_standard_deviation, REPLICATIONS)
        assert abs(row.mean_estimate - TRUE_EFFECT - row.expected_bias) < 4 * error
        assert row.standard_deviation == pytest.approx(row.expected_standard_deviation, rel=0.1)
        assert row.units_in_window == units_in_window(h)


def test_narrow_windows_are_noisy_and_wide_ones_biased() -> None:
    """The alt text: bias within noise up to 7, beyond it from 30; the noise falls throughout."""
    noise = [row.standard_deviation for row in SUMMARY.rows]
    assert noise == sorted(noise, reverse=True)
    assert noise[0] > 4 * noise[-1]
    for h, row in ROWS.items():
        error = _standard_error(row.standard_deviation, REPLICATIONS)
        if h <= 7:
            assert row.absolute_bias < 3 * error
        if h >= 30:
            assert row.absolute_bias > 10 * error
        if h >= 40:
            assert row.absolute_bias > row.standard_deviation


def test_the_error_is_smallest_at_an_intermediate_bandwidth() -> None:
    """The title: the error bottoms out at 20, next to the closed-form optimum of 19.1."""
    errors = [row.root_mean_squared_error for row in SUMMARY.rows]
    assert BANDWIDTHS[int(np.argmin(errors))] == 20
    expected = [row.expected_root_mean_squared_error for row in SUMMARY.rows]
    assert BANDWIDTHS[int(np.argmin(expected))] == 20
    assert SUMMARY.optimal_bandwidth == pytest.approx(19.13, abs=0.01)
    assert min(errors) < 0.5 * min(errors[0], errors[-1])


def test_the_article_bandwidth_table_agrees_with_the_closed_forms() -> None:
    """Units used exactly; mean, spread and error within the noise of 2,000 samples."""
    printed = {
        2: (200, 2.05, 1.14, 1.14),
        5: (500, 1.98, 0.70, 0.70),
        10: (1000, 1.95, 0.51, 0.51),
        20: (2000, 1.80, 0.36, 0.41),
        35: (3500, 1.38, 0.27, 0.67),
        50: (5000, 0.75, 0.22, 1.27),
    }
    for h, (units, mean, sd, rmse) in printed.items():
        bias, spread = local_linear_bias(h), local_linear_sd(h)
        assert round(units_in_window(h)) == units
        error = _standard_error(spread, ARTICLE_REPLICATIONS)
        assert abs(mean - TRUE_EFFECT - bias) < 0.005 + 4 * error
        assert abs(sd - spread) < 0.005 + 4 * spread / sqrt(2 * ARTICLE_REPLICATIONS)
        assert abs(rmse - sqrt(bias**2 + spread**2)) < 0.005 + 4 * error
    assert round(naive_difference_limit(), 2) == 8.50
    assert naive_difference_limit() == pytest.approx(8.5)


def test_the_article_polynomials_placebos_and_sample_sizes() -> None:
    """Global polynomials, placebo cutoffs and the spread by total units, within noise."""
    polynomials = {row.degree: row for row in SUMMARY.polynomials}
    for degree, (mean, sd) in {
        1: (0.75, 0.23),
        2: (1.98, 0.34),
        4: (2.02, 0.57),
        6: (2.04, 0.78),
    }.items():
        row = polynomials[degree]
        error = _standard_error(row.expected_standard_deviation, ARTICLE_REPLICATIONS)
        assert abs(mean - TRUE_EFFECT - row.expected_bias) < 0.005 + 4 * error
        assert abs(sd - row.expected_standard_deviation) < 0.005 + 4 * sd / sqrt(4000)
    # Worse than a local fit at 20, on two and a half times the data (the article says five).
    assert polynomials[6].expected_standard_deviation > ROWS[20].expected_root_mean_squared_error
    assert units_in_window(50) / units_in_window(20) == 2.5

    placebos = {placebo.cutoff: placebo for placebo in SUMMARY.placebos}
    printed = {-25.0: (-0.02, 0.49), -10.0: (0.02, 0.51), 10.0: (0.00, 0.50), 25.0: (0.00, 0.51)}
    for cutoff, (mean, sd) in printed.items():
        placebo = placebos[cutoff]
        assert placebo.expected_estimate == pytest.approx(0.0, abs=1e-10)
        error = _standard_error(placebo.expected_standard_deviation, ARTICLE_REPLICATIONS)
        assert abs(mean) < 0.005 + 4 * error
        assert abs(sd - placebo.expected_standard_deviation) < 0.005 + 4 * sd / sqrt(4000)

    sizes = {size.units: size for size in SUMMARY.sample_sizes}
    for units, (inside, sd) in {1000: (200, 1.11), 5000: (1000, 0.55), 20000: (4000, 0.26)}.items():
        assert sizes[units].units_in_window == inside
        spread = sizes[units].expected_standard_deviation
        assert abs(sd - spread) < 0.005 + 4 * spread / sqrt(2 * 500)
    # A thousand units leave a standard error about half the effect.
    assert sizes[1000].expected_standard_deviation / TRUE_EFFECT == pytest.approx(0.57, abs=0.01)


def test_the_article_manipulation_table() -> None:
    """Estimates on the population limits and density ratios near (1 + s) / (1 - s)."""
    rows = {row.share: row for row in SUMMARY.manipulation}
    printed = {0.0: (1.94, 1.01), 0.3: (2.93, 1.85), 0.6: (3.54, 4.05)}
    error = _standard_error(local_linear_sd(10), 500)
    for share, (estimate, ratio) in printed.items():
        row = rows[share]
        assert abs(estimate - row.expected_estimate) < 0.005 + 4 * error
        above = 0.05 * (1 + share) * UNITS
        below = 0.05 * (1 - share) * UNITS
        ratio_sd = row.density_ratio * sqrt(1 / above + 1 / below)
        # The mean of ratios sits above the ratio of means by about ratio / below.
        centre = row.density_ratio * (1 + 1 / below)
        assert abs(ratio - centre) < 0.005 + 4 * ratio_sd / sqrt(500)
    assert rows[0.0].expected_estimate == pytest.approx(1.95)
    assert round(rows[0.3].expected_estimate - TRUE_EFFECT, 2) == 0.96
    assert rows[0.6].expected_estimate - TRUE_EFFECT > 0.75 * TRUE_EFFECT
    assert rows[0.3].density_ratio == pytest.approx(13 / 7)
    assert rows[0.6].density_ratio == pytest.approx(4.0)


def test_a_noiseless_linear_trend_is_recovered_exactly() -> None:
    """With straight lines on both sides and no noise, the estimate is the jump itself."""
    x = np.linspace(-30, 30, 241)
    y = 3.0 + 0.5 * x + 1.7 * (x >= 0) - 0.2 * x * (x >= 0)
    for h in (2.0, 10.0, 30.0):
        assert local_linear(x, y, h) == pytest.approx(1.7, abs=1e-10)
    assert local_linear(x, y, 5.0, cutoff=-10.0) == pytest.approx(0.0, abs=1e-10)


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a row; another seed does not."""

    def small(seed: int) -> object:
        return bandwidth_row(np.random.default_rng(seed), 10, replications=5, units=500)

    assert small(6) == small(6)
    assert small(6) != small(7)


def test_invalid_inputs_are_rejected() -> None:
    """Bad bandwidths, shares, arrays and windows without two units a side are refused."""
    rng = np.random.default_rng(0)
    x, _, y = draw_sample(rng, 300)
    with pytest.raises(ValueError):
        local_linear(x, y, 0.0)
    with pytest.raises(ValueError):
        local_linear(x, y, 0.01)
    with pytest.raises(ValueError):
        local_linear(x, y[:-1], 5.0)
    with pytest.raises(ValueError):
        local_linear(np.append(x, np.nan), np.append(y, 1.0), 5.0)
    with pytest.raises(ValueError):
        naive_difference(x, y, cutoff=60.0)
    with pytest.raises(ValueError):
        draw_sample(rng, 0)
    with pytest.raises(ValueError):
        draw_sample(rng, 10, manipulation=1.5)
    with pytest.raises(TypeError):
        bandwidth_row(rng, True)
    with pytest.raises(ValueError):
        bandwidth_row(rng, 500.0)
    with pytest.raises(ValueError):
        local_linear_bias(60.0)
    with pytest.raises(ValueError):
        density_ratio(1.0)
    with pytest.raises(ValueError):
        polynomial_sd(-1)
    with pytest.raises(ValueError):
        local_linear_limit(10.0, manipulation=-0.1)
    assert optimal_bandwidth(UNITS * 32) == pytest.approx(optimal_bandwidth() / 2)
