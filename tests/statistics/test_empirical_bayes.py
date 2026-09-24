"""Check the empirical Bayes model against closed forms, simulation, the article and the figure.

The figure's programme (seed 71) is transcribed from the website generator and
matched draw for draw, and its summary is pinned. Its description does not hold
as written in three respects, and the tests pin what is true instead:

- It calls the programme "two hundred experiments"; the generator draws 4,000
  (the article's programme has 200), which is why 328 winners are plotted.
- "The raw points sit well above the line, most of them measured by the least
  precise tests": 292 of the 328 raw points are above the line, and of the 89
  more than two percentage points above it, 60 come from the least precise
  band (standard errors of 0.014 to 0.020). Of all 328 winners, though, only 72
  come from that band; most winners are precise tests with modest overstatement.
- "The shrunk points land close to it": on average they do (their mean is
  within 0.03 points of the true mean, and their root mean squared error is 0.66
  points against 1.89 raw), but individually they are pulled to the programme's
  typical effect, as the article's own caption says: every winner with a true
  effect below 0.5 percent is shrunk to above the line, and 45 of the 47 above
  2 percent to below it.

The article's tables use other seeds and a programme of 200 and are not pinned.
They are compared with the closed forms instead: the raw error (0.01284 against
0.01286), the shrunk error (0.00744 against 0.00739 with the true spread), the
42.1 percent reduction (42.5), the precision-band table and the forecast table.
"""

from math import sqrt

import numpy as np
import pytest
from numpy.typing import NDArray

from blog_reproducibility.statistics.empirical_bayes import (
    EXPERIMENTS,
    FORECAST_SPREADS,
    PRECISION_BANDS,
    PROGRAMME_SIZE,
    SE_HIGH,
    SE_LOW,
    draw_programme,
    estimate_spread,
    example_payload,
    figure_programme,
    figure_summary,
    mean_shrinkage_factor,
    oracle_rmse,
    raw_rmse,
    shrink,
    shrink_programme,
    shrinkage_factors,
    significant_winners,
    winner_expectation,
)

SUMMARY = example_payload()
FIGURE = SUMMARY.figure
CLOSED = SUMMARY.closed_form
RESULT = figure_programme()
WIN = RESULT.winners
TRUTH = RESULT.programme.effects[WIN]
RAW = RESULT.programme.estimates[WIN]
SHRUNK = RESULT.shrunk[WIN]
ERRORS = RESULT.programme.standard_errors[WIN]


def _website_figure() -> tuple[NDArray[np.float64], ...]:
    """The website generator's draws and shrinkage, transcribed."""
    rng = np.random.default_rng(71)
    k, tau = 4000, 0.010
    truth = rng.normal(0.0, tau, k)
    se = rng.uniform(0.004, 0.020, k)
    est = truth + rng.normal(0, se)
    tau_hat = np.sqrt(max(float(est.var(ddof=1) - np.mean(se**2)), 1e-12))
    b = tau_hat**2 / (tau_hat**2 + se**2)
    shrunk = b * est
    win = est > 1.96 * se
    return truth, se, est, shrunk, win.astype(np.float64), np.array([tau_hat])


def test_the_figure_programme_matches_the_website_draw_for_draw() -> None:
    """Same seed, same draw order: identical effects, estimates, shrinkage and winners."""
    truth, se, est, shrunk, win, tau_hat = _website_figure()

    assert np.array_equal(RESULT.programme.effects, truth)
    assert np.array_equal(RESULT.programme.standard_errors, se)
    assert np.array_equal(RESULT.programme.estimates, est)
    assert RESULT.estimated_spread == tau_hat[0]
    assert np.array_equal(RESULT.shrunk, shrunk)
    assert np.array_equal(RESULT.winners, win.astype(bool))


def test_the_figure_summary_is_pinned() -> None:
    """4,000 experiments, a spread estimated at 0.00985, and 328 positive significant winners."""
    assert FIGURE.experiments == EXPERIMENTS == 20 * PROGRAMME_SIZE
    assert round(FIGURE.estimated_spread, 5) == 0.00985
    assert (round(FIGURE.raw_rmse, 5), round(FIGURE.shrunk_rmse, 5)) == (0.01300, 0.00721)
    assert FIGURE.winners == 328
    assert (round(FIGURE.winner_raw_mean, 4), round(FIGURE.winner_true_mean, 4)) == (0.0254, 0.0118)
    assert round(FIGURE.winner_shrunk_mean, 4) == 0.0121
    assert [band.winners for band in FIGURE.bands] == [151, 105, 72]
    assert [round(band.overstatement, 2) for band in FIGURE.bands] == [1.38, 2.18, 4.08]
    assert [(band.lower, band.upper) for band in FIGURE.bands] == list(PRECISION_BANDS)


def test_the_raw_winners_sit_well_above_the_line() -> None:
    """The alt text: raw winners overstate, and the largest misses come from imprecise tests."""
    assert int(np.count_nonzero(RAW > TRUTH)) == 292
    assert FIGURE.winner_raw_mean > 2 * FIGURE.winner_true_mean
    overstatements = [band.overstatement for band in FIGURE.bands]
    assert overstatements == sorted(overstatements)

    far = RAW - TRUTH > 0.02
    lower_edge = PRECISION_BANDS[-1][0]
    least_precise = lower_edge <= ERRORS
    assert (int(np.count_nonzero(far)), int(np.count_nonzero(far & least_precise))) == (89, 60)
    # Of all the winners, most are precise tests: the claim holds for the points furthest out.
    assert int(np.count_nonzero(least_precise)) == 72


def test_the_shrunk_winners_land_close_to_the_line_on_average() -> None:
    """Mean within 0.03 points of the truth, a third of the raw error, but pulled to the middle."""
    assert abs(FIGURE.winner_shrunk_mean - FIGURE.winner_true_mean) < 0.0003
    assert FIGURE.winner_shrunk_rmse < 0.36 * FIGURE.winner_raw_rmse
    assert FIGURE.shrunk_rmse < 0.56 * FIGURE.raw_rmse
    assert int(np.count_nonzero(SHRUNK > TRUTH)) == 172

    small, large = TRUTH < 0.005, TRUTH > 0.02
    assert np.all(SHRUNK[small] > TRUTH[small])
    assert (int(np.count_nonzero(large)), int(np.count_nonzero(SHRUNK[large] < TRUTH[large]))) == (
        47,
        45,
    )


def test_the_article_error_tables_agree_with_the_closed_forms() -> None:
    """Raw error 0.01284 and 0.01278, shrunk 0.00744 and a 42.1 percent reduction."""
    assert CLOSED.raw_rmse == pytest.approx(sqrt((0.004**2 + 0.004 * 0.020 + 0.020**2) / 3))
    assert round(CLOSED.raw_rmse, 5) == 0.01286
    assert CLOSED.raw_rmse == pytest.approx(0.01284, rel=0.005)
    assert CLOSED.raw_rmse == pytest.approx(0.01278, rel=0.01)
    # Estimating the spread costs a little against shrinking with the true one.
    assert round(CLOSED.oracle_rmse, 5) == 0.00739
    assert CLOSED.oracle_rmse < 0.00744
    assert CLOSED.oracle_rmse == pytest.approx(0.00744, rel=0.01)
    assert round(100 * CLOSED.rmse_reduction, 1) == 42.5
    assert CLOSED.rmse_reduction == pytest.approx(0.421, abs=0.005)


def test_the_article_shrinkage_factors() -> None:
    """At the article's estimated spread of 0.0090: 83 percent kept at best, 17 at worst."""
    best, worst = shrinkage_factors([SE_LOW, SE_HIGH], 0.009)
    assert (round(float(best), 3), round(float(worst), 3)) == (0.835, 0.168)
    assert float(best) == pytest.approx(0.83, abs=0.01)
    assert round(float(worst), 2) == 0.17
    assert mean_shrinkage_factor(0.009) == pytest.approx(0.40, abs=0.015)
    # A 4.2 percent lift measured with a one-point standard error is half noise.
    assert float(shrink([0.042], [0.01], 0.01)[0]) == pytest.approx(0.021)


def test_the_article_precision_bands_agree_with_the_closed_forms() -> None:
    """8.2, 5.5 and 3.5 winners per programme overstating by 1.4, 2.2 and 3.8 times."""
    printed = [
        (8.2, 0.0173, 0.0128, 0.0126, 1.4),
        (5.5, 0.0273, 0.0126, 0.0125, 2.2),
        (3.5, 0.0416, 0.0108, 0.0105, 3.8),
    ]
    for band, (winners, raw, truth, shrunk, ratio) in zip(CLOSED.bands, printed, strict=True):
        assert band.spread == 0.01
        assert band.winners == pytest.approx(winners, abs=0.3)
        assert band.raw_mean == pytest.approx(raw, abs=0.0006)
        assert band.true_mean == pytest.approx(truth, abs=0.0003)
        assert band.overstatement == pytest.approx(ratio, abs=0.06)
        # Shrinkage with the true spread recovers the truth exactly in expectation.
        assert abs(shrunk - truth) <= 0.0003
    assert [round(band.overstatement, 1) for band in CLOSED.bands] == [1.3, 2.2, 3.8]


def test_the_article_forecast_agrees_with_the_closed_forms() -> None:
    """10, 21 and 40 winners shipped; the raw total overstates less as real effects grow."""
    assert tuple(row.spread for row in CLOSED.forecasts) == FORECAST_SPREADS
    for row, shipped in zip(CLOSED.forecasts, (10, 21, 40), strict=True):
        assert (row.lower, row.upper) == (SE_LOW, SE_HIGH)
        assert abs(shipped - row.winners) < 3 * sqrt(row.winners)
    ratios = [row.overstatement for row in CLOSED.forecasts]
    assert [round(ratio, 1) for ratio in ratios] == [7.5, 2.1, 1.3]
    # The article's single programmes: raw over delivered is 9.6, 2.4 and 1.2.
    printed = [(0.278, 0.029), (0.549, 0.226), (1.358, 1.088)]
    assert [raw / delivered for raw, delivered in printed] == sorted(
        (raw / delivered for raw, delivered in printed), reverse=True
    )


def test_the_bands_add_up_to_the_whole_range() -> None:
    """Expected winners and their totals over the three bands equal those over all tests."""
    whole = winner_expectation()
    assert sum(band.winners for band in CLOSED.bands) == pytest.approx(whole.winners, rel=1e-9)
    assert sum(band.raw_total for band in CLOSED.bands) == pytest.approx(whole.raw_total, rel=1e-9)
    assert sum(band.true_total for band in CLOSED.bands) == pytest.approx(
        whole.true_total, rel=1e-9
    )
    assert whole == CLOSED.forecasts[1]


def test_the_closed_forms_match_a_large_simulation() -> None:
    """400,000 experiments: winner counts and means and both errors agree with the closed forms."""
    rng = np.random.default_rng(5)
    k = 400_000
    programme = draw_programme(rng, k)
    shrunk = shrink(programme.estimates, programme.standard_errors, 0.01)
    win = significant_winners(programme.estimates, programme.standard_errors)
    expected = winner_expectation(experiments=k)

    count = int(np.count_nonzero(win))
    p = expected.winners / k
    assert count == pytest.approx(expected.winners, abs=4 * sqrt(k * p * (1 - p)))
    assert float(np.mean(programme.estimates[win])) == pytest.approx(expected.raw_mean, rel=0.01)
    assert float(np.mean(programme.effects[win])) == pytest.approx(expected.true_mean, rel=0.02)
    # Selection on the estimate leaves the posterior mean unbiased.
    assert float(np.mean(shrunk[win])) == pytest.approx(
        float(np.mean(programme.effects[win])), rel=0.01
    )
    raw_error = sqrt(float(np.mean((programme.estimates - programme.effects) ** 2)))
    shrunk_error = sqrt(float(np.mean((shrunk - programme.effects) ** 2)))
    assert raw_error == pytest.approx(raw_rmse(), rel=0.005)
    assert shrunk_error == pytest.approx(oracle_rmse(), rel=0.005)
    assert estimate_spread(programme.estimates, programme.standard_errors) == pytest.approx(
        0.01, rel=0.01
    )


def test_the_shrinkage_average_matches_a_numerical_mean() -> None:
    """The arctangent formula is the average of the factor over the uniform standard errors."""
    points = 200_000
    grid = SE_LOW + (np.arange(points) + 0.5) * (SE_HIGH - SE_LOW) / points
    for spread in (0.004, 0.009, 0.01, 0.02):
        numerical = float(np.mean(shrinkage_factors(grid, spread)))
        assert mean_shrinkage_factor(spread) == pytest.approx(numerical, rel=1e-6)
    assert raw_rmse() ** 2 == pytest.approx(float(np.mean(grid**2)), rel=1e-6)


def test_the_spread_estimate_on_hand_worked_cases() -> None:
    """Variance of the estimates less the mean squared error, floored at 1e-12."""
    assert estimate_spread([0.03, -0.01, 0.01], [0.01, 0.01, 0.01]) == pytest.approx(sqrt(0.0003))
    assert estimate_spread([0.02, -0.02], [0.01, 0.03]) == pytest.approx(sqrt(0.0008 - 0.0005))
    assert estimate_spread([0.01, 0.01, 0.01], [0.01, 0.02, 0.03]) == pytest.approx(1e-6)


def test_limiting_cases() -> None:
    """Precise tests keep their estimate, noisy ones lose it, and equal variances halve it."""
    factors = shrinkage_factors([1e-6, 0.01, 10.0], 0.01)
    assert float(factors[0]) == pytest.approx(1.0)
    assert float(factors[1]) == 0.5
    assert float(factors[2]) == pytest.approx(0.0, abs=1e-5)
    narrow = mean_shrinkage_factor(0.01, se_low=0.01, se_high=0.01 + 1e-9)
    assert narrow == pytest.approx(0.5, rel=1e-6)
    assert oracle_rmse(1e-6) == pytest.approx(1e-6, rel=1e-3)
    assert oracle_rmse(10.0) == pytest.approx(raw_rmse(), rel=1e-3)
    # With almost no real spread the winners are nearly all noise.
    assert winner_expectation(spread=1e-4).overstatement > 100


def test_shrink_programme_uses_the_estimated_spread() -> None:
    """Shrinking a drawn programme is the three steps in turn."""
    programme = draw_programme(np.random.default_rng(3), 300)
    result = shrink_programme(programme)
    spread = estimate_spread(programme.estimates, programme.standard_errors)

    assert result.estimated_spread == spread
    assert np.array_equal(
        result.shrunk, shrink(programme.estimates, programme.standard_errors, spread)
    )
    assert figure_summary(result).experiments == 300


def test_the_programme_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the draws; another seed does not."""
    first, again, other = figure_programme(4, 50), figure_programme(4, 50), figure_programme(5, 50)
    assert np.array_equal(first.programme.estimates, again.programme.estimates)
    assert not np.array_equal(first.programme.estimates, other.programme.estimates)


def test_invalid_inputs_are_rejected() -> None:
    """Bad spreads, ranges, bands, shapes, and non-finite or non-positive errors are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        draw_programme(rng, 1)
    with pytest.raises(TypeError):
        draw_programme(rng, 200.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        draw_programme(rng, spread=0.0)
    with pytest.raises(ValueError):
        draw_programme(rng, se_low=0.02, se_high=0.004)
    with pytest.raises(ValueError):
        estimate_spread([0.01], [0.01])
    with pytest.raises(ValueError):
        estimate_spread([0.01, 0.02], [0.01])
    with pytest.raises(ValueError):
        estimate_spread([0.01, np.nan], [0.01, 0.01])
    with pytest.raises(ValueError):
        estimate_spread([0.01, 0.02], [0.01, 0.0])
    with pytest.raises(ValueError):
        shrinkage_factors([0.01, -0.01], 0.01)
    with pytest.raises(ValueError):
        shrink([0.01, 0.02], [0.01], 0.01)
    with pytest.raises(ValueError):
        significant_winners([0.01, 0.02], [0.01, 0.01], critical=0.0)
    with pytest.raises(ValueError):
        mean_shrinkage_factor(-0.01)
    with pytest.raises(ValueError):
        winner_expectation(0.002, 0.008)
    with pytest.raises(ValueError):
        winner_expectation(0.010, 0.010)
    with pytest.raises(ValueError):
        winner_expectation(experiments=0)
    with pytest.raises(TypeError):
        figure_programme(True)
