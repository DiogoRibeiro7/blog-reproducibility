"""Check the week-over-week noise model against the website, closed forms and the article.

The figure is its own simulation (seed 83, 250 series of 400 days compared on
days 60 to 399), so the website's loop is transcribed and matched bit for bit,
and the spreads its legend prints (4.9 and 2.9 percent) are pinned. The closed
forms are independent of the simulation: every simulated statistic lies within
three between-series standard errors of its closed form, which in turn agrees
with direct integration, a Monte Carlo draw of the underlying normals and a
covariance matrix built by hand.

The article's tables come from other seeds and are not reproduced. They agree
with the closed forms within three standard errors of the article's design,
estimated by rerunning that design (400 series of 560 days, days 400 to 559)
with twenty other seeds: spreads of 4.93, 2.86 and 4.93 percent against 4.93,
2.85 and 4.97; five-percent movements on 30.8, 8.0 and 30.8 percent of days
against 30.9, 7.9 and 31.2; thresholds of 9.63 and 12.86 percent for a single
day against 9.66 and 12.80, and 5.58 and 7.36 for weekly averages against 5.59
and 7.36. The article's year-over-year row sits 1.7 to 2.2 standard errors
below its closed form throughout, as one sample can. Its day-100 comparison (a
ratio of 1.75 against 2.65 for independent days) is 1.75 in closed form.

The figure's claim holds loosely: 31 percent of single-day comparisons fall
outside five percent (30.9 in closed form), "roughly a third".
"""

from math import sqrt

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate, stats

from blog_reproducibility.statistics.week_over_week import (
    BIN_EDGES,
    NOISE_SD,
    PERSISTENCE,
    SERIES,
    WEEKDAY_FACTORS,
    YEAR,
    RatioLaw,
    example_payload,
    exceedance,
    expected_stats,
    false_alarm_threshold,
    phase_weights,
    ratio_moments,
    same_weekday_changes,
    same_weekday_law,
    seven_day_changes,
    seven_day_law,
    simulate_metric,
    simulated_stats,
)

SUMMARY = example_payload()
SINGLE, WEEKLY = SUMMARY.same_weekday, SUMMARY.seven_day
METRIC = simulate_metric(np.random.default_rng(83))
SINGLE_CHANGES = same_weekday_changes(METRIC)
WEEKLY_CHANGES = seven_day_changes(METRIC)
WEEKLY_LAWS = [seven_day_law(phase) for phase in range(7)]

# The article's first two tables, in percent: spread, share over 5 and over 10 percent,
# and the thresholds for false-alarm rates of 5 and 1 percent.
ARTICLE = {
    "same weekday": (4.93, 30.8, 4.3, 9.63, 12.86),
    "seven-day": (2.86, 8.0, 0.1, 5.58, 7.36),
    "last year": (4.93, 30.8, 4.2, 9.65, 12.84),
}
# Their standard errors in percentage points, from twenty reruns of the article's design.
ARTICLE_ERRORS = {
    "same weekday": (0.014, 0.18, 0.061, 0.032, 0.070),
    "seven-day": (0.015, 0.17, 0.012, 0.028, 0.071),
    "last year": (0.022, 0.25, 0.099, 0.046, 0.072),
}
# Each printed value is rounded: half its last digit.
ARTICLE_ROUNDING = (0.005, 0.05, 0.05, 0.005, 0.005)


def _website_changes() -> tuple[list[float], list[float]]:
    """Transcribe the website generator's loop."""
    days = 400
    weekday = np.array([1.05, 1.08, 1.06, 1.04, 1.10, 0.82, 0.85])
    rho, sd = 0.55, 0.035
    rng = np.random.default_rng(83)

    def series() -> NDArray[np.float64]:
        eps = rng.normal(0, sd * np.sqrt(1 - rho**2), days)
        z = np.zeros(days)
        for i in range(1, days):
            z[i] = rho * z[i - 1] + eps[i]
        return 1000.0 * weekday[np.arange(days) % 7] * (1 + z)

    single: list[float] = []
    weekly: list[float] = []
    for _ in range(250):
        y = series()
        for t in range(60, days):
            single.append(y[t] / y[t - 7] - 1)
            weekly.append(y[t - 6 : t + 1].mean() / y[t - 13 : t - 6].mean() - 1)
    return single, weekly


def _per_series_error(values: NDArray[np.float64]) -> float:
    """Standard error of a pooled mean from the spread of the series' own means."""
    means = values.mean(axis=1)
    return float(means.std(ddof=1) / sqrt(means.size))


def test_the_comparisons_are_the_websites() -> None:
    """Both comparisons for every series and day, and the legend's spreads, bit for bit."""
    single, weekly = _website_changes()
    assert np.array_equal(SINGLE_CHANGES.ravel(), single)
    assert np.array_equal(WEEKLY_CHANGES.ravel(), weekly)
    assert SINGLE.simulated.spread == np.std(single)
    assert WEEKLY.simulated.spread == np.std(weekly)
    assert SUMMARY.comparisons == 85_000


def test_the_figure_legend_and_histograms() -> None:
    """Spreads of 4.9 and 2.9 percent; each histogram integrates to one over its bins."""
    assert f"{SINGLE.simulated.spread:.1%}" == "4.9%"
    assert f"{WEEKLY.simulated.spread:.1%}" == "2.9%"
    edges = np.array(SUMMARY.bin_edges)
    assert np.array_equal(edges, np.linspace(-0.20, 0.20, 81))
    assert SUMMARY.bin_edges == BIN_EDGES
    widths = np.diff(edges)
    for density, changes in (
        (SUMMARY.same_weekday_density, SINGLE_CHANGES),
        (SUMMARY.seven_day_density, WEEKLY_CHANGES),
    ):
        assert float(np.sum(np.array(density) * widths)) == pytest.approx(1.0)
        counts, _ = np.histogram(changes.ravel(), bins=edges)
        assert np.allclose(density, counts / counts.sum() / widths, rtol=1e-12)


def test_the_figure_claim() -> None:
    """Roughly a third of quiet single days move by more than five percent; a twelfth of weeks."""
    assert round(SINGLE.simulated.exceeds_small, 2) == 0.31
    assert round(SINGLE.expected.exceeds_small, 3) == 0.309
    assert round(WEEKLY.simulated.exceeds_small, 2) == 0.08
    assert WEEKLY.simulated.spread < 0.6 * SINGLE.simulated.spread


def test_the_simulation_agrees_with_the_closed_forms() -> None:
    """Shares, mean squares and thresholds within three between-series standard errors."""
    for row, changes in ((SINGLE, SINGLE_CHANGES), (WEEKLY, WEEKLY_CHANGES)):
        moves = np.abs(changes)
        simulated, expected = row.simulated, row.expected
        for threshold, share, exact in (
            (0.05, simulated.exceeds_small, expected.exceeds_small),
            (0.10, simulated.exceeds_large, expected.exceeds_large),
        ):
            error = _per_series_error((moves > threshold).astype(float))
            assert abs(share - exact) < 3 * error
        error = _per_series_error(changes)
        assert abs(simulated.mean - expected.mean) < 3 * error
        error = _per_series_error(changes**2)
        square = simulated.spread**2 + simulated.mean**2
        assert abs(square - (expected.spread**2 + expected.mean**2)) < 3 * error
        # A simulated threshold is exceeded, in closed form, about as often as it should be.
        weights = None if row is SINGLE else phase_weights()
        laws = same_weekday_law() if row is SINGLE else WEEKLY_LAWS
        for threshold, rate in (
            (simulated.threshold_five_percent, 0.05),
            (simulated.threshold_one_percent, 0.01),
        ):
            error = _per_series_error((moves > threshold).astype(float))
            assert abs(exceedance(threshold, laws, weights) - rate) < 3 * error


def test_the_article_tables_against_the_closed_forms() -> None:
    """Every printed spread, share and threshold within three standard errors of its closed form."""
    article_days = phase_weights(400, 560)
    expected = {
        "same weekday": expected_stats(same_weekday_law()),
        "seven-day": expected_stats(WEEKLY_LAWS, article_days),
        "last year": expected_stats(same_weekday_law(YEAR)),
    }
    for name, printed in ARTICLE.items():
        exact = expected[name]
        values = (
            exact.spread,
            exact.exceeds_small,
            exact.exceeds_large,
            exact.threshold_five_percent,
            exact.threshold_one_percent,
        )
        for shown, value, error, rounding in zip(
            printed, values, ARTICLE_ERRORS[name], ARTICLE_ROUNDING, strict=True
        ):
            assert abs(shown - 100 * value) < 3 * error + rounding
    # "Three days in ten", "about once a month", and ten and five and a half percent.
    assert round(expected["same weekday"].exceeds_small, 1) == 0.3
    assert 1 / expected["same weekday"].exceeds_large == pytest.approx(30, rel=0.25)
    assert round(100 * expected["same weekday"].threshold_five_percent) == 10
    assert round(100 * expected["seven-day"].threshold_five_percent, 1) == 5.6


def test_the_article_day_one_hundred() -> None:
    """Spreads of 4.925 and 2.821 percent on day 100, a ratio of 1.75 against 2.65."""
    single = ratio_moments(same_weekday_law())[1]
    weekly = ratio_moments(seven_day_law(100 % 7))[1]
    runs = 2000
    assert abs(4.925 - 100 * single) < 3 * 100 * single / sqrt(2 * runs)
    assert abs(2.821 - 100 * weekly) < 3 * 100 * weekly / sqrt(2 * runs)
    assert round(single / weekly, 2) == 1.75
    assert round(sqrt(7), 2) == 2.65
    # Independent days would buy the square root of seven; persistence loses about half of it.
    assert sqrt(7) / (single / weekly) == pytest.approx(1.5, abs=0.02)


def test_ratio_moments_match_direct_integration() -> None:
    """The Gauss-Hermite moments against adaptive quadrature over ``V``."""
    for law in (same_weekday_law(), seven_day_law(4), same_weekday_law(YEAR)):
        s = sqrt(law.variance)
        slope = law.covariance / law.variance
        spread = law.variance * (1 - slope**2)

        def moment(
            v: float,
            power: int,
            law: RatioLaw = law,
            s: float = s,
            slope: float = slope,
            spread: float = spread,
        ) -> float:
            centre = (slope - 1) * v
            conditional = centre if power == 1 else centre**2 + spread
            return float(stats.norm.pdf(v, scale=s) * conditional / (law.scale + v) ** power)

        first = integrate.quad(moment, -12 * s, 12 * s, args=(1,), epsabs=1e-15)[0]
        second = integrate.quad(moment, -12 * s, 12 * s, args=(2,), epsabs=1e-15)[0]
        mean, sd = ratio_moments(law)
        assert mean == pytest.approx(first, rel=1e-9)
        assert sd == pytest.approx(sqrt(second - first**2), rel=1e-9)
        # To first order the spread is sd(U - V) / W.
        first_order = sqrt(2 * (law.variance - law.covariance)) / law.scale
        assert sd == pytest.approx(first_order, rel=0.01)


def test_exceedance_matches_a_monte_carlo_draw() -> None:
    """Draw the two normals directly and count the comparisons beyond each threshold."""
    rng = np.random.default_rng(21)
    for law in (same_weekday_law(), seven_day_law(0)):
        cov = [[law.variance, law.covariance], [law.covariance, law.variance]]
        u, v = rng.multivariate_normal([0.0, 0.0], cov, size=400_000).T
        ratio = (u - v) / (law.scale + v)
        for threshold in (0.02, 0.05, 0.08):
            share = float(np.mean(np.abs(ratio) > threshold))
            exact = exceedance(threshold, law)
            assert abs(share - exact) < 3 * sqrt(exact * (1 - exact) / u.size)


def test_the_seven_day_law_by_hand() -> None:
    """Weighted sums of fourteen days' noise under the AR(1) covariance matrix."""
    days = np.arange(14)
    sigma = NOISE_SD**2 * PERSISTENCE ** np.abs(np.subtract.outer(days, days))
    for phase in range(7):
        factors = np.array(WEEKDAY_FACTORS)[(np.arange(-13, 1) + phase) % 7]
        previous = np.where(days < 7, factors, 0.0)
        current = np.where(days >= 7, factors, 0.0)
        law = seven_day_law(phase)
        assert law.scale == pytest.approx(sum(WEEKDAY_FACTORS))
        assert law.variance == pytest.approx(current @ sigma @ current, rel=1e-12)
        assert law.variance == pytest.approx(previous @ sigma @ previous, rel=1e-12)
        assert law.covariance == pytest.approx(current @ sigma @ previous, rel=1e-12)
    single = same_weekday_law()
    assert (single.scale, single.variance) == (1.0, NOISE_SD**2)
    assert single.covariance == pytest.approx(NOISE_SD**2 * PERSISTENCE**7)


def test_limiting_cases() -> None:
    """Independent days and equal factors buy the square root of seven; thresholds invert."""
    flat = (1.0,) * 7
    single = ratio_moments(same_weekday_law(persistence=0.0))[1]
    weekly = ratio_moments(seven_day_law(3, factors=flat, persistence=0.0))[1]
    assert single / weekly == pytest.approx(sqrt(7), rel=0.01)
    law = same_weekday_law()
    assert exceedance(0.0, law) == pytest.approx(1.0)
    assert exceedance(0.5, law) < 1e-12
    for rate in (0.05, 0.01):
        assert exceedance(false_alarm_threshold(rate, law), law) == pytest.approx(rate, rel=1e-9)
    # A mixture of one law is that law.
    assert expected_stats([law], [1.0]) == expected_stats(law)
    weights = phase_weights()
    assert sum(weights) == pytest.approx(1.0)
    # Days 60 to 399: 48 full weeks from a Friday, then Friday to Monday once more.
    assert [round(340 * w) for w in weights] == [49, 48, 48, 48, 49, 49, 49]


def test_simulated_stats_by_hand() -> None:
    """Four comparisons: their mean, spread, shares and quantiles."""
    values = np.array([[0.01, -0.06], [0.12, -0.03]])
    result = simulated_stats(values)
    assert result.mean == pytest.approx(0.01)
    assert result.spread == pytest.approx(float(np.std(values)))
    assert (result.exceeds_small, result.exceeds_large) == (0.5, 0.25)
    moves = [0.01, 0.06, 0.12, 0.03]
    assert result.threshold_five_percent == pytest.approx(np.quantile(moves, 0.95))


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the series; another seed does not."""
    first = simulate_metric(np.random.default_rng(4), 3, 30)
    again = simulate_metric(np.random.default_rng(4), 3, 30)
    other = simulate_metric(np.random.default_rng(5), 3, 30)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert METRIC.shape[0] == SERIES


def test_invalid_inputs_are_rejected() -> None:
    """Bad persistence, factors, days, laws and rates are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_metric(rng, 2, 30, persistence=1.0)
    with pytest.raises(ValueError):
        simulate_metric(rng, 2, 30, factors=(1.0,) * 6)
    with pytest.raises(ValueError):
        simulate_metric(rng, 0, 30)
    with pytest.raises(TypeError):
        simulate_metric(rng, 2, True)
    with pytest.raises(ValueError):
        same_weekday_changes(METRIC, 5)
    with pytest.raises(ValueError):
        seven_day_changes(METRIC, 10)
    with pytest.raises(ValueError):
        seven_day_changes(METRIC, 400)
    with pytest.raises(ValueError):
        same_weekday_changes(-METRIC)
    with pytest.raises(ValueError):
        exceedance(0.05, [])
    with pytest.raises(ValueError):
        exceedance(0.05, [same_weekday_law()], [0.5])
    with pytest.raises(ValueError):
        exceedance(-0.05, same_weekday_law())
    with pytest.raises(ValueError):
        false_alarm_threshold(0.0, same_weekday_law())
    with pytest.raises(ValueError):
        simulated_stats(np.array([]))
    with pytest.raises(ValueError):
        phase_weights(60, 60)
