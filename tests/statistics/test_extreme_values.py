"""Check the peaks-over-threshold model against the website, closed forms and the article.

The figure and the article share one generator seeded at 0, so the hours, the
normal and Pareto fits, the figure's curves, the threshold sweep and the
dependent series (which continues the same generator) are reproduced and pinned
at the article's precision: true levels 326 and 502, a sample maximum of 265 and
318 with the margin, a normal fit of 159 and 166, 526 exceedances with a shape of
0.19 and return levels of 302 and 425, the five rows of the threshold table, and
526 exceedances in 194 clusters with a naive ten-year level of 303. The fitted
values come from SciPy's optimiser and are pinned at the printed precision only.

The bootstrap resamples the same hours from a generator seeded at 1. Its 500
refits take about twelve seconds, so it runs here on 40 resamples, whose first
draws are checked against a transcription of the article's loop. The script's
full run gives 255 to 355 and 319 to 563, as the article prints. The replication
study over 200 records has no code; its sample-maximum rows agree with the closed
form for the largest of 26,280 hours to within their sampling noise (median 283,
quartiles 253 and 328, 74 percent below the truth, against 291, 255 to 333 and 72
percent; with the margin 339, 304 to 394 and 41 percent against 350, 307 to 400
and 38), and its normal-fit median of 160 is the population limit, 159.9.

Some statements do not hold as written, and the tests pin what is true:

* the ten-year estimate "is 8 percent under the truth": 301.7 against 325.7 is
  7.4 percent under;
* a three-year record contains a ten-year event with probability "1 - 0.9^3,
  about 27 percent": with the article's own definition (an hourly exceedance
  probability of ``1 / (10 H)``) it is ``1 - (1 - 1 / (10 H))^(3 H)``, 25.9
  percent; ``1 - 0.9^3`` treats each year as one trial. Both are one in four;
* going from the ten- to the hundred-year level "multiplies the excess over the
  threshold by about 10^0.25, a factor of 1.8": that is the limit for long
  periods; at the 98th percentile the factor is 1.92 for a shape of 0.25 and 1.90
  on the true tail;
* "the shape climbs toward its true value of 0.25 as the threshold rises": it
  rises overall, from 0.16 to 0.28, but dips from 0.20 to 0.19 at the 98th
  percentile and ends above 0.25.

The figure's claims hold: no observed hour reaches the ten-year line, the normal
fit gives the true ten-year level a probability 1e-54 times too small, and the
Pareto fit's probabilities at the true ten- and hundred-year levels are within a
factor of three of the truth.
"""

from math import log

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from blog_reproducibility.statistics.extreme_values import (
    CLUSTER_GAP,
    GRID_POINTS,
    HOURS_PER_YEAR,
    THRESHOLD_QUANTILES,
    PotFit,
    bootstrap_return_levels,
    count_clusters,
    example_payload,
    fit_peaks_over_threshold,
    maximum_distribution,
    normal_return_level,
    pareto_exceedance,
    return_level,
    simulate_dependent_load,
    simulate_load,
    true_exceedance,
    true_return_level,
)

H = HOURS_PER_YEAR
N = 3 * H
# The article's bootstrap is 500 refits, about twelve seconds; 40 exercise the same code.
SUMMARY = example_payload(bootstrap_replicates=40)
RECORD = SUMMARY.record
FIT = SUMMARY.fit
_RNG = np.random.default_rng(0)
LOAD = simulate_load(_RNG)
DEPENDENT, LATENT = simulate_dependent_load(_RNG)


def _article_pot(x: NDArray[np.float64], q: float) -> tuple[float, int, float, float, float, float]:
    """Transcribe the article's peaks-over-threshold fit and its two return levels."""
    u = np.quantile(x, q)
    exc = x[x > u] - u
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    zeta = np.mean(x > u)

    def level(t: int) -> float:
        return float(u + sigma / xi * ((t * H * zeta) ** xi - 1))

    return float(u), len(exc), float(xi), float(sigma), level(10), level(100)


def _matches(fit: PotFit, article: tuple[float, int, float, float, float, float]) -> bool:
    ours = (fit.threshold, fit.exceedances, fit.shape, fit.scale, fit.ten_year, fit.hundred_year)
    return ours == article


def test_the_draws_are_the_articles() -> None:
    """The hours, then the dependent series from the same generator, as the article draws them."""
    rng = np.random.default_rng(0)
    x = 100.0 + 10.0 * rng.standard_t(4, N)
    assert np.array_equal(LOAD, x)

    phi = 0.8
    z = np.empty(N)
    z[0] = rng.normal()
    eps = rng.normal(size=N)
    for t in range(1, N):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi**2) * eps[t]
    xd = 100.0 + 10.0 * stats.t.ppf(stats.norm.cdf(z), 4)
    assert np.array_equal(LATENT, z)
    assert np.array_equal(DEPENDENT, xd)

    # The article's run rule, hour by hour.
    over = xd > np.quantile(xd, 0.98)
    clusters, gap = 0, 10**9
    for t in range(N):
        if over[t]:
            if gap >= 24:
                clusters += 1
            gap = 0
        else:
            gap += 1
    assert count_clusters(over) == clusters == SUMMARY.dependent.clusters


def test_the_figure_curves_are_the_websites() -> None:
    """The grid, the three curves and the observed tail, bit for bit."""
    x = LOAD
    u = np.quantile(x, 0.98)
    exc = x[x > u] - u
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    zeta = np.mean(x > u)
    grid = np.linspace(u, 620, 400)
    top = np.sort(x[x > u])[::-1]

    curves = SUMMARY.curves
    assert len(curves.levels) == GRID_POINTS
    assert np.array_equal(curves.levels, grid)
    assert np.array_equal(curves.true, stats.t.sf((grid - 100.0) / 10.0, 4))
    assert np.array_equal(curves.normal, stats.norm.sf(grid, x.mean(), x.std()))
    assert np.array_equal(curves.pareto, zeta * (1 + xi * (grid - u) / sigma) ** (-1 / xi))
    assert np.array_equal(curves.observed_levels, top)
    assert np.array_equal(curves.observed_probabilities, (np.arange(len(top)) + 1) / N)


def test_the_record_matches_the_article() -> None:
    """True 326 and 502; maximum 265 and 318; normal 159 and 166; Pareto 302 and 425."""
    assert RECORD.hours == 26_280
    assert (round(RECORD.true_ten_year), round(RECORD.true_hundred_year)) == (326, 502)
    assert (round(RECORD.sample_maximum), round(RECORD.maximum_with_margin)) == (265, 318)
    assert (round(RECORD.normal_ten_year), round(RECORD.normal_hundred_year)) == (159, 166)
    assert (FIT.quantile, FIT.exceedances, round(FIT.shape, 2)) == (0.98, 526, 0.19)
    assert (round(FIT.ten_year), round(FIT.hundred_year)) == (302, 425)
    assert _matches(FIT, _article_pot(LOAD, 0.98))
    # The true level exceeded at the normal's ten-year level is about 180 times its claim.
    assert round(RECORD.normal_understatement) == 180
    # The record has already exceeded the normal's ten-year level many times.
    assert RECORD.hours_above_normal_ten_year == int(np.count_nonzero(LOAD > 159.08)) == 50
    # The maximum sits a little below the three-year level.
    assert 0 < RECORD.true_record_level - RECORD.sample_maximum < 2
    # "8 percent under the truth" is 7.4 percent.
    assert round(100 * (1 - FIT.ten_year / RECORD.true_ten_year), 1) == 7.4


def test_the_threshold_table_matches_the_article() -> None:
    """Exceedances, shape and hundred-year level at five threshold quantiles."""
    printed = [
        (2628, "0.16", 373),
        (1314, "0.20", 440),
        (526, "0.19", 425),
        (263, "0.22", 471),
        (132, "0.28", 564),
    ]
    rows = SUMMARY.thresholds
    assert tuple(row.quantile for row in rows) == THRESHOLD_QUANTILES
    for row, expected in zip(rows, printed, strict=True):
        assert (row.exceedances, f"{row.shape:.2f}", round(row.hundred_year)) == expected
    assert rows[2] == FIT
    # The range across thresholds, 373 to 564, contains the true 502.
    levels = [row.hundred_year for row in rows]
    assert min(levels) < RECORD.true_hundred_year < max(levels)
    # The shape rises overall toward 0.25 but not monotonically, and ends above it.
    shapes = [row.shape for row in rows]
    assert shapes[0] < shapes[-1] and shapes[2] < shapes[1] and shapes[-1] > 0.25


def test_the_dependent_series_matches_the_article() -> None:
    """526 exceedances in 194 clusters, an extremal index of 0.37, a naive ten-year level of 303."""
    dependent = SUMMARY.dependent
    assert (dependent.exceedances, dependent.clusters) == (526, 194)
    assert round(dependent.extremal_index, 2) == 0.37
    assert round(dependent.fit.ten_year) == 303
    assert _matches(dependent.fit, _article_pot(DEPENDENT, 0.98))
    # Close to three hours above the threshold per event, and a lag-one correlation of 0.8.
    assert 2.5 < dependent.exceedances / dependent.clusters < 3
    assert np.corrcoef(LATENT[:-1], LATENT[1:])[0, 1] == pytest.approx(0.8, abs=0.01)
    # The marginal is the same t: its quantiles match the independent record's.
    for q in (0.5, 0.9, 0.99):
        assert np.quantile(DEPENDENT, q) == pytest.approx(np.quantile(LOAD, q), rel=0.02)


def test_the_bootstrap_is_the_articles_loop() -> None:
    """The first resamples reproduce the article's loop; the reduced intervals behave as claimed."""
    ours = bootstrap_return_levels(LOAD, np.random.default_rng(1), 4)
    r = np.random.default_rng(1)
    for row in ours:
        xb = LOAD[r.integers(0, N, N)]
        *_, ten, hundred = _article_pot(xb, 0.98)
        assert (row[0], row[1]) == (ten, hundred)

    boot = SUMMARY.bootstrap
    assert boot.replicates == 40
    low, high = boot.ten_year
    assert low < FIT.ten_year < RECORD.true_ten_year < high
    # Asymmetric: the upper side is the longer one.
    assert high - FIT.ten_year > FIT.ten_year - low
    low, high = boot.hundred_year
    assert low < FIT.hundred_year < RECORD.true_hundred_year < high
    assert high - FIT.hundred_year > FIT.hundred_year - low
    # The hundred-year interval spans almost a factor of two (1.77 on the full run).
    assert 1.5 < high / low < 2
    # The article's full-run intervals are asymmetric the same way.
    assert 355 - 302 > 302 - 255 and 563 - 425 > 425 - 319


def test_the_sample_maximum_closed_form_matches_the_replication_study() -> None:
    """Median, quartiles and share below the truth, against 200 simulated records."""
    printed = {0.0: (291, 255, 333, 0.72), 0.2: (350, 307, 400, 0.38)}
    for dist in SUMMARY.maxima:
        median, lower, upper, share = printed[round(dist.margin, 1)]
        factor = 1 + dist.margin
        for simulated, level, p in (
            (median, dist.median, 0.5),
            (lower, dist.lower_quartile, 0.25),
            (upper, dist.upper_quartile, 0.75),
        ):
            # The maximum has density n F^(n-1) f; a sample quantile of 200 has this error.
            x = level / factor
            density = N * stats.t.cdf((x - 100) / 10, 4) ** (N - 1) * stats.t.pdf((x - 100) / 10, 4)
            se = factor * np.sqrt(p * (1 - p) / 200) / (density / 10)
            assert abs(simulated - level) < 3 * se + 0.5
        se = np.sqrt(dist.share_below_ten_year * (1 - dist.share_below_ten_year) / 200)
        assert abs(share - dist.share_below_ten_year) < 3 * se + 0.005
    plain, margin = SUMMARY.maxima
    assert (round(plain.median), round(plain.lower_quartile), round(plain.upper_quartile)) == (
        283,
        253,
        328,
    )
    assert round(100 * plain.share_below_ten_year) == 74
    assert margin.median == pytest.approx(1.2 * plain.median)
    # The normal fit's median of 160 is its population limit.
    assert round(SUMMARY.normal_limit_ten_year) == 160
    assert SUMMARY.normal_limit_ten_year == pytest.approx(
        100 + 10 * np.sqrt(2) * stats.norm.ppf(1 - 1 / (10 * H))
    )


def test_a_ten_year_event_in_a_three_year_record() -> None:
    """One record in four: 25.9 percent by the hourly definition, 27.1 treating years as trials."""
    exact = 1 - maximum_distribution(0.0).share_below_ten_year
    assert exact == pytest.approx(1 - (1 - 1 / (10 * H)) ** N, rel=1e-9)
    assert round(100 * exact, 1) == 25.9
    assert round(100 * (1 - 0.9**3), 1) == 27.1
    assert round(4 * exact) == 1 and round(4 * (1 - 0.9**3)) == 1


def test_how_the_levels_scale_with_the_shape() -> None:
    """54 percent from ten to a hundred years here; about 7 for an exponential tail."""
    assert round(100 * (RECORD.true_hundred_year / RECORD.true_ten_year - 1)) == 54
    # An exponential tail adds sigma log 10 instead, 7.5 percent of the ten-year level.
    assert round(100 * FIT.scale * log(10) / RECORD.true_ten_year) == 7
    # A shape of 0.25 multiplies the excess by 10^0.25 = 1.78 in the limit; 1.92 at this threshold.
    assert round(10**0.25, 1) == 1.8

    def excess_ratio(xi: float) -> float:
        periods = H * FIT.rate
        return float(((100 * periods) ** xi - 1) / ((10 * periods) ** xi - 1))

    assert round(excess_ratio(0.25), 2) == 1.92
    true_ratio = (RECORD.true_hundred_year - FIT.threshold) / (RECORD.true_ten_year - FIT.threshold)
    assert round(true_ratio, 2) == 1.90
    assert excess_ratio(0.25) == pytest.approx(10**0.25, rel=0.1)


def test_the_figure_claims() -> None:
    """The data stop short of the ten-year line; the Pareto fit extrapolates, the normal cannot."""
    curves = SUMMARY.curves
    # The empirical tail ends at the sample maximum, with probability 1 / n.
    assert curves.observed_levels[0] == RECORD.sample_maximum
    assert curves.observed_probabilities[-1] == FIT.rate
    assert min(curves.observed_probabilities) == 1 / N > 1 / (10 * H)
    # The normal fit is already below the data at the threshold.
    assert curves.normal[0] < FIT.rate
    targets = [RECORD.true_ten_year, RECORD.true_hundred_year]
    truth = np.array([1 / (10 * H), 1 / (100 * H)])
    pareto = pareto_exceedance(FIT, targets) / truth
    normal = stats.norm.sf(targets, RECORD.normal_mean, RECORD.normal_sd) / truth
    assert np.all((pareto > 1 / 3) & (pareto < 1))
    assert np.all(normal < 1e-50)
    # The Pareto return levels are within 8 and 16 percent of the truth.
    assert abs(FIT.ten_year / RECORD.true_ten_year - 1) < 0.08
    assert abs(FIT.hundred_year / RECORD.true_hundred_year - 1) < 0.16
    # And the true exceedance at every grid level is within a factor of four of the fit.
    ratio = np.array(curves.pareto) / np.array(curves.true)
    assert np.all((ratio > 0.25) & (ratio <= 1.05))


def test_closed_forms_by_hand() -> None:
    """Return levels invert the Pareto exceedance, and the t tail is the truth."""
    fit = FIT
    for years in (1, 10, 100):
        level = return_level(fit.threshold, fit.shape, fit.scale, fit.rate, years)
        assert pareto_exceedance(fit, [level])[0] == pytest.approx(1 / (years * H), rel=1e-10)
        true = true_return_level(years)
        assert true_exceedance([true])[0] == pytest.approx(1 / (years * H), rel=1e-9)
    # The exponential tail is the limit of vanishing shape.
    near = return_level(130.0, 1e-9, 10.0, 0.02, 10)
    assert near == pytest.approx(return_level(130.0, 0.0, 10.0, 0.02, 10), rel=1e-8)
    assert return_level(130.0, 0.0, 10.0, 0.02, 10) == pytest.approx(130 + 10 * log(1752))
    exponential = PotFit(0.98, 130.0, 500, 0.0, 10.0, 0.02, 0.0, 0.0)
    assert pareto_exceedance(exponential, [140.0])[0] == pytest.approx(0.02 * np.exp(-1))
    # A bounded tail has no probability above its end point.
    bounded = PotFit(0.98, 130.0, 500, -0.5, 10.0, 0.02, 0.0, 0.0)
    assert pareto_exceedance(bounded, [149.0, 150.0, 160.0]).tolist()[1:] == [0.0, 0.0]
    # A normal return level is a normal quantile.
    assert normal_return_level(0.0, 1.0, 10) == pytest.approx(stats.norm.isf(1 / (10 * H)))
    # The t tail with four degrees of freedom has shape 1/4: doubling a far level divides by 16.
    far = true_exceedance([100 + 10 * 1e4, 100 + 10 * 2e4])
    assert far[0] / far[1] == pytest.approx(16, rel=1e-3)


def test_the_fit_recovers_a_known_shape() -> None:
    """The fit recovers a known Pareto shape; the maximum's quartiles match short records."""
    rng = np.random.default_rng(7)
    sample = 10 + stats.genpareto.rvs(0.25, scale=5.0, size=40_000, random_state=rng)
    fit = fit_peaks_over_threshold(sample, 0.5)
    assert fit.shape == pytest.approx(0.25, abs=0.04)
    excess_scale = 5.0 + 0.25 * (fit.threshold - 10)
    assert fit.scale == pytest.approx(excess_scale, rel=0.05)

    records = 100 + 10 * rng.standard_t(4, (20_000, 50))
    dist = maximum_distribution(0.0, hours=50)
    maxima = records.max(axis=1)
    assert np.median(maxima) == pytest.approx(dist.median, rel=0.02)
    assert np.quantile(maxima, 0.25) == pytest.approx(dist.lower_quartile, rel=0.02)
    assert np.quantile(maxima, 0.75) == pytest.approx(dist.upper_quartile, rel=0.03)
    level = true_return_level(10)
    share = maximum_distribution(0.0, hours=50, target=level).share_below_ten_year
    assert np.mean(maxima < level) == pytest.approx(share, abs=0.002)


def test_count_clusters_by_hand() -> None:
    """A new cluster needs a full run of hours below the threshold."""
    over = np.zeros(100, dtype=bool)
    assert count_clusters(over) == 0
    over[[3, 4, 10, 40, 64, 65]] = True
    # Gaps of 5 and 29 and 23 hours below: the 29 starts a new cluster, the 23 does not.
    assert count_clusters(over) == 2
    assert count_clusters(over, gap=6) == 3
    assert count_clusters(over, gap=5) == 4
    assert count_clusters(over, gap=1) == 4
    assert count_clusters(over, gap=30) == 1
    assert CLUSTER_GAP == 24


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the hours; another does not."""
    first = simulate_load(np.random.default_rng(4), 30)
    again = simulate_load(np.random.default_rng(4), 30)
    other = simulate_load(np.random.default_rng(5), 30)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_invalid_inputs_are_rejected() -> None:
    """Bad samples, thresholds, periods and parameters are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_load(rng, 0)
    with pytest.raises(ValueError):
        simulate_load(rng, 10, df=0.0)
    with pytest.raises(ValueError):
        simulate_dependent_load(rng, 10, autocorrelation=1.0)
    with pytest.raises(TypeError):
        simulate_dependent_load(rng, 10, autocorrelation=True)
    with pytest.raises(ValueError):
        true_return_level(0.0)
    with pytest.raises(ValueError):
        true_return_level(1e-6)
    with pytest.raises(ValueError):
        fit_peaks_over_threshold([1.0, 2.0])
    with pytest.raises(ValueError):
        fit_peaks_over_threshold(LOAD, 1.0)
    with pytest.raises(ValueError):
        fit_peaks_over_threshold([1.0, 2.0, np.nan])
    with pytest.raises(ValueError):
        fit_peaks_over_threshold(np.arange(10.0), 0.95)
    with pytest.raises(ValueError):
        return_level(130.0, 0.2, 0.0, 0.02, 10)
    with pytest.raises(ValueError):
        return_level(130.0, 0.2, 10.0, 0.0, 10)
    with pytest.raises(ValueError):
        normal_return_level(100.0, -1.0, 10)
    with pytest.raises(ValueError):
        bootstrap_return_levels(LOAD, rng, 0)
    with pytest.raises(ValueError):
        count_clusters(np.array([0, 1, 1]))
    with pytest.raises(ValueError):
        count_clusters(np.array([True, False]), gap=0)
    with pytest.raises(ValueError):
        maximum_distribution(-0.1)
    with pytest.raises(ValueError):
        maximum_distribution(0.0, target=0.0)
