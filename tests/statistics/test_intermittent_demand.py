"""Check the intermittent demand forecasts against the website, closed forms and the article.

The figure is its own simulation (seeds 900 to 939, every fourth week forecast),
so the website's forecasting functions are transcribed and matched bit for bit
on the first series, the bars are pinned (0.796, 0.800, 0.828 and 0.787 against
a true rate of 0.8), and the single-pass forecasts are compared with the
website's recomputation from the start of the series. The article's tables come
from other seeds and are not reproduced; they are checked against the closed
forms instead, within their sampling error: 82 percent of weeks without demand
against 80 in the population, errors of the forecast of zero (0.795 and 1.942
against 0.8 and 1.949), the smoothing forecasts' squared errors (1.789 and
1.818 against 1.795 and 1.824), and the absolute errors of the others, which
for a forecast of level ``F`` below one unit are ``0.6 F + 0.8``.

The figure's claims hold except in one respect. The forecast of zero is far
below the rate and has the lowest absolute error (and the highest squared
error), and the three usable methods are within two percent of the truth.
But "only one method is wrong about the rate" is not quite what the bars show:
undebiased Croston is 3.5 percent above the truth, 2.8 standard errors, with an
interval (0.809 to 0.848) that excludes 0.8; it agrees with the second-order
closed form of 0.834. The article's own table gives it as 3.6 percent, and its
debiased form as 1.5 percent below, against 1.0 percent below in closed form.
"""

from collections.abc import Callable
from math import sqrt

import numpy as np
import pytest
from numpy.typing import NDArray

from blog_reproducibility.statistics.intermittent_demand import (
    METHODS,
    SERIES,
    closed_forms,
    constant_forecast_mae,
    croston_expectation,
    croston_forecasts,
    demand_series,
    example_payload,
    forecast_weeks,
    method_forecasts,
    rolling_mean_forecasts,
    smoothing_forecasts,
)

SUMMARY = example_payload()
ZERO, MEAN, SMOOTHING, CROSTON, DEBIASED = SUMMARY.rows
CLOSED = SUMMARY.closed_forms
WEEKS = forecast_weeks()

# The article's evaluation table: 200 series of 936 forecast weeks, seeds 100 to 299.
ARTICLE_WEEKS = 200 * 936
ARTICLE_TABLE = {
    # method: (average forecast, bias, mean absolute error, root mean squared error)
    "Always zero": (0.000, -0.7951, 0.795, 1.942),
    "52-week mean": (0.795, -0.0006, 1.274, 1.789),
    "Exponential smoothing": (0.795, -0.0002, 1.276, 1.818),
    "Croston": (0.829, 0.0340, 1.295, 1.783),
    "Croston, debiased": (0.788, -0.0074, 1.270, 1.781),
}


def _website_forecasts(y: NDArray[np.float64]) -> list[list[float]]:
    """Transcribe the website generator's five forecasting functions, recomputed at each week."""
    warmup = 104

    def f_zero(y: NDArray[np.float64], t: int) -> float:
        return 0.0

    def f_mean(y: NDArray[np.float64], t: int) -> float:
        return float(y[max(0, t - 52) : t].mean())

    def f_ses(y: NDArray[np.float64], t: int, alpha: float = 0.1) -> float:
        f = y[:warmup].mean()
        for v in y[warmup:t]:
            f = alpha * v + (1 - alpha) * f
        return float(f)

    def f_croston(
        y: NDArray[np.float64], t: int, alpha: float = 0.1, debias: bool = False
    ) -> float:
        nz = np.flatnonzero(y[:t])
        if nz.size < 2:
            return float(y[:t].mean())
        z, x, last = y[nz[0]], float(nz[0] + 1), nz[0]
        for i in nz[1:]:
            z = alpha * y[i] + (1 - alpha) * z
            x = alpha * (i - last) + (1 - alpha) * x
            last = i
        rate = z / x
        return float(rate * (1 - alpha / 2) if debias else rate)

    def f_debiased(y: NDArray[np.float64], t: int) -> float:
        return f_croston(y, t, debias=True)

    methods: list[Callable[[NDArray[np.float64], int], float]] = [
        f_zero,
        f_mean,
        f_ses,
        f_croston,
        f_debiased,
    ]
    return [[fn(y, t) for t in range(warmup, 1040, 4)] for fn in methods]


def _website_series(seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    occurs = rng.random(1040) < 0.20
    size = 1 + rng.poisson(4.0 - 1, 1040)
    return np.where(occurs, size, 0).astype(float)


def test_the_forecasts_are_the_websites() -> None:
    """The first series and every method's forecasts on it, bit for bit."""
    y = demand_series(np.random.default_rng(900))
    assert np.array_equal(y, _website_series(900))
    theirs = _website_forecasts(y)
    for ours, website, row in zip(method_forecasts(y, WEEKS), theirs, SUMMARY.rows, strict=True):
        assert np.array_equal(ours, website)
        assert row.series_averages[0] == np.mean(website)
    assert WEEKS.tolist() == list(range(104, 1040, 4))


def test_the_figure_bars() -> None:
    """Average forecasts 0, 0.796, 0.800, 0.828 and 0.787, with the website's standard errors."""
    assert [row.method for row in SUMMARY.rows] == list(METHODS)
    assert (SUMMARY.series, SUMMARY.forecasts_per_series) == (40, 234)
    expected = [0.0, 0.7961764464168309, 0.8001524174626695, 0.8280843272974406, 0.7866801109325687]
    errors = [
        0.0,
        0.009764100551968184,
        0.009767542233706314,
        0.009966640349721672,
        0.00946830833223559,
    ]
    for row, average, error in zip(SUMMARY.rows, expected, errors, strict=True):
        assert row.average_forecast == pytest.approx(average, rel=1e-12, abs=1e-15)
        assert row.standard_error == pytest.approx(error, rel=1e-12, abs=1e-15)
        assert len(row.series_averages) == SERIES
        assert row.average_forecast == np.mean(row.series_averages)
        assert row.standard_error == np.std(row.series_averages) / np.sqrt(SERIES)


def test_the_figure_claims() -> None:
    """Zero is far below and wins on absolute error; three methods sit within two percent."""
    truth = CLOSED.true_rate
    assert ZERO.average_forecast == 0.0
    for row in (MEAN, SMOOTHING, DEBIASED):
        assert abs(row.average_forecast / truth - 1) < 0.02
        assert abs(row.average_forecast - truth) < 1.96 * row.standard_error
    # Croston is above the truth by 3.5 percent, and its interval excludes it.
    assert round(CROSTON.average_forecast / truth - 1, 3) == 0.035
    assert CROSTON.average_forecast - 1.96 * CROSTON.standard_error > truth
    # "It wins on error": the lowest absolute error, and the highest squared error.
    maes = [row.mean_absolute_error for row in SUMMARY.rows]
    rmses = [row.root_mean_squared_error for row in SUMMARY.rows]
    assert min(maes) == ZERO.mean_absolute_error
    assert max(rmses) == ZERO.root_mean_squared_error
    assert min(rmses) == DEBIASED.root_mean_squared_error
    # For the forecast of zero the absolute error is the demand, and minus the bias.
    assert ZERO.mean_absolute_error == pytest.approx(-ZERO.bias, rel=1e-12)


def test_debiasing_scales_every_forecast() -> None:
    """The debiased series averages are 0.95 of Croston's."""
    assert np.allclose(DEBIASED.series_averages, 0.95 * np.array(CROSTON.series_averages))


def test_the_simulation_agrees_with_the_closed_forms() -> None:
    """The unbiased methods centre on 0.8, Croston on 0.834 and debiased Croston on 0.792."""
    for row, expected in (
        (MEAN, CLOSED.true_rate),
        (SMOOTHING, CLOSED.true_rate),
        (CROSTON, CLOSED.croston),
        (DEBIASED, CLOSED.croston_debiased),
    ):
        assert abs(row.average_forecast - expected) < 3 * row.standard_error
    # Absolute error of a forecast below one unit: 0.6 times its level plus the demand.
    # Forecasts above one unit after bursts of demand add a little, most for smoothing.
    demand = -ZERO.bias
    for row in (MEAN, SMOOTHING, CROSTON, DEBIASED):
        linear = (1 - 2 * 0.2) * row.average_forecast + demand
        assert 0 < row.mean_absolute_error - linear < 0.02


def test_the_closed_forms() -> None:
    """Rate 0.8, four weeks in five empty, weekly variance 3.16, and the error formulas."""
    assert CLOSED.true_rate == pytest.approx(0.8)
    assert CLOSED.zero_share == pytest.approx(0.8)
    assert CLOSED.weekly_variance == pytest.approx(3.16)
    assert CLOSED.zero_forecast_mae == pytest.approx(0.8)
    assert CLOSED.rate_forecast_mae == pytest.approx(1.28)
    assert CLOSED.zero_forecast_rmse == pytest.approx(sqrt(3.8))
    assert CLOSED.rolling_mean_rmse == pytest.approx(sqrt(3.16 * 53 / 52))
    assert CLOSED.smoothing_rmse == pytest.approx(sqrt(3.16 * (1 + 0.1 / 1.9)))
    assert CLOSED.croston == pytest.approx(0.8 * (1 + 0.08 / 1.9))
    assert CLOSED.croston_debiased == pytest.approx(0.95 * CLOSED.croston)
    assert croston_expectation(alpha=1e-9) == pytest.approx(0.8)


def test_the_demand_distribution() -> None:
    """A large sample has the closed-form share of empty weeks, mean, variance and errors."""
    y = demand_series(np.random.default_rng(12), 400_000)
    n = y.size
    assert abs(np.mean(y == 0) - 0.8) < 3 * sqrt(0.16 / n)
    assert abs(y.mean() - 0.8) < 3 * sqrt(3.16 / n)
    assert y.var() == pytest.approx(3.16, rel=0.02)
    assert np.all(y[y > 0] >= 1)
    for c in (0.0, 0.5, 1.0, 2.5, 6.0):
        assert np.mean(np.abs(y - c)) == pytest.approx(constant_forecast_mae(c), rel=0.01)


def test_constant_forecast_mae_by_hand() -> None:
    """``(1 - p) c + p (m - c)`` below one unit; the median, zero, minimises it."""
    assert constant_forecast_mae(0.0) == pytest.approx(0.8)
    assert constant_forecast_mae(0.5) == pytest.approx(0.8 * 0.5 + 0.2 * 3.5)
    assert constant_forecast_mae(1.0) == pytest.approx(0.8 + 0.2 * 3.0)
    # Above one unit the demands of one unit are overshot: 2 p (c - 1) P(S = 1) more.
    extra = 2 * 0.2 * 1.0 * np.exp(-3.0)
    assert constant_forecast_mae(2.0) == pytest.approx(0.8 * 2 + 0.2 * 2 + extra)
    levels = np.linspace(0, 8, 161)
    assert levels[np.argmin([constant_forecast_mae(float(c)) for c in levels])] == 0.0
    # With demand in most weeks the median is positive and zero no longer wins.
    busy = [constant_forecast_mae(float(c), demand_probability=0.9) for c in levels]
    assert levels[int(np.argmin(busy))] > 0.0


def test_the_article_table_against_the_closed_forms() -> None:
    """The article's errors lie within their sampling error of the population values."""
    se_mean = sqrt(CLOSED.weekly_variance / ARTICLE_WEEKS)
    average, bias, mae, rmse = ARTICLE_TABLE["Always zero"]
    assert average == 0.0 and mae == -round(bias, 3)
    assert abs(-bias - CLOSED.true_rate) < 3 * se_mean
    # Per-series root mean squared errors average a little below the root of the mean square.
    assert rmse == pytest.approx(CLOSED.zero_forecast_rmse, rel=0.01)
    assert ARTICLE_TABLE["52-week mean"][3] == pytest.approx(CLOSED.rolling_mean_rmse, rel=0.01)
    assert ARTICLE_TABLE["Exponential smoothing"][3] == pytest.approx(
        CLOSED.smoothing_rmse, rel=0.01
    )
    for name in METHODS[1:]:
        average, bias, mae, rmse = ARTICLE_TABLE[name]
        assert mae == pytest.approx(0.6 * average + 0.8, abs=0.006)
        assert rmse < ARTICLE_TABLE["Always zero"][3]
    # Croston's printed forecasts against the second-order closed forms, within a percent.
    assert ARTICLE_TABLE["Croston"][0] == pytest.approx(CLOSED.croston, rel=0.01)
    assert ARTICLE_TABLE["Croston, debiased"][0] == pytest.approx(CLOSED.croston_debiased, rel=0.01)
    # The article's arithmetic: 3.6 percent over, 1.5 percent under, two percent apart.
    assert abs(100 * (0.829 / 0.8 - 1) - 3.6) < 0.05
    assert abs(100 * (1 - 0.788 / 0.8) - 1.5) < 0.05
    assert round(100 * (1.818 / 1.781 - 1)) == 2
    # Its sample series (seed 1): 82 percent empty, mean 0.721, within three standard errors.
    assert abs(0.82 - CLOSED.zero_share) < 3 * sqrt(0.16 / 1040)
    assert abs(0.721 - CLOSED.true_rate) < 3 * sqrt(3.16 / 1040)
    # Its stock table: nine and eighteen percent less stock; multipliers in the ratio 0.95.
    assert round(100 * (1 - 7.08 / 7.81)) == 9
    assert round(100 * (1 - 7.08 / 8.60)) == 18
    assert abs(3.39 / 3.57 - 0.95) < 0.002


def test_the_forecasting_rules_by_hand() -> None:
    """A short series worked through each rule."""
    y = np.array([0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 3.0, 0.0])
    weeks = np.array([2, 5, 7, 8])
    assert rolling_mean_forecasts(y, weeks, window=3).tolist() == pytest.approx(
        [1.0, 4 / 3, 7 / 3, 1.0]
    )
    # Smoothing from the mean of the first four weeks, 0.5, with alpha one half.
    assert smoothing_forecasts(y, np.array([4, 5, 6]), alpha=0.5, warmup=4).tolist() == (
        pytest.approx([0.5, 2.25, 1.125])
    )
    # Croston: the mean until a second demand; then size 3 over interval 2.5 (a gap of
    # three after an initial interval of two), then 3 over 2.25 (a gap of two).
    croston = croston_forecasts(y, weeks, alpha=0.5)
    assert croston.tolist() == pytest.approx([1.0, 3 / 2.5, 3 / 2.25, 3 / 2.25])
    debiased = croston_forecasts(y, weeks, alpha=0.5, debias=True)
    assert debiased[1:].tolist() == pytest.approx((0.75 * croston[1:]).tolist())
    assert debiased[0] == croston[0]


def test_limiting_cases() -> None:
    """Every week a demand: Croston's interval is one and it smooths the sizes."""
    y = np.array([2.0, 4.0, 2.0, 4.0, 2.0, 4.0])
    rates = croston_forecasts(y, np.array([3, 6]), alpha=1.0 - 1e-12)
    assert rates.tolist() == pytest.approx([2.0, 4.0])
    constant = np.full(200, 3.0)
    zero, *level, debiased = method_forecasts(constant, np.array([150, 199]))
    assert zero.tolist() == [0.0, 0.0]
    for forecasts in level:
        assert forecasts.tolist() == pytest.approx([3.0, 3.0])
    assert debiased.tolist() == pytest.approx([2.85, 2.85])


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a series; another seed does not."""
    first = demand_series(np.random.default_rng(4), 200)
    again = demand_series(np.random.default_rng(4), 200)
    other = demand_series(np.random.default_rng(5), 200)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_invalid_inputs_are_rejected() -> None:
    """Bad probabilities, sizes, weeks and smoothing constants are refused."""
    rng = np.random.default_rng(0)
    y = demand_series(rng, 120)
    with pytest.raises(ValueError):
        demand_series(rng, 0)
    with pytest.raises(ValueError):
        demand_series(rng, 10, demand_probability=1.0)
    with pytest.raises(ValueError):
        demand_series(rng, 10, mean_size=0.5)
    with pytest.raises(TypeError):
        demand_series(rng, True)
    with pytest.raises(ValueError):
        forecast_weeks(100, warmup=100)
    with pytest.raises(ValueError):
        rolling_mean_forecasts(y, np.array([0]))
    with pytest.raises(ValueError):
        rolling_mean_forecasts(y, np.array([121]))
    with pytest.raises(ValueError):
        smoothing_forecasts(y, np.array([50]), warmup=104)
    with pytest.raises(ValueError):
        croston_forecasts(y, np.array([110]), alpha=0.0)
    with pytest.raises(ValueError):
        croston_forecasts([], np.array([1]))
    with pytest.raises(ValueError):
        constant_forecast_mae(-1.0)
    with pytest.raises(ValueError):
        closed_forms(window=0)
