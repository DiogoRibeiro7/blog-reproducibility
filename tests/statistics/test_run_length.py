"""Check the monitoring charts against the website, closed forms, Markov chains and the article.

The figure's simulation (seed 23, 1,500 runs of 2,000 days, three charts over
seven shifts) runs here in full. Its batched draws and day-by-day updates are
checked against a transcription of the website's per-run loops, run by run, on a
shorter design that also exercises runs that never signal. The Shewhart curve
agrees with the exact geometric mean, and the other two with Markov-chain
approximations (Brook and Evans for the cumulative sums, a propagated grid for the
weighted average with its time-varying limits), to within simulation noise.

The article's tables come from other generators (seeds 5, 9, 11 and 13) and are
not pinned. Its closed-form numbers are: a daily signal probability of 0.2700
percent, 370 days on average, a median of 256, and 39, 106 and 256 days for a
tenth, a quarter and half of false alarms; 22 days and about 6 at two sigma. Its
simulated columns agree with the closed forms and the Markov chains to within
their noise. The quantiles of the one-sigma delay are exactly 5, 13, 31, 61 and
100 days; the article's simulation prints 99 for the last. The moving-range
standard deviation is ``sqrt(1 - rho)`` to three decimals, as printed; the false
alarm intervals it implies with the estimate held at its mean are 372, 83, 17 and
5.6 days, and the article's 307 and 76 at low correlation are shorter because the
estimate from a 200-day baseline is itself noisy, which a simulation of the
article's procedure confirms.

The figure's claims hold with one qualification. All three charts have in-control
run lengths within six percent of 370 (370, 366 and 350). The charts with memory
are 4.5 to 5.2 times faster at shifts of a half to one standard deviation, but only
2.4 times at a quarter, so "four to five times faster below one sigma" holds from a
half sigma up. Above two sigma the three are within four days of one another, and
at three sigma the cumulative sum is slower than the three-sigma rule.
"""

from collections.abc import Callable
from functools import cache

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from blog_reproducibility.statistics.run_length import (
    ARTICLE_SHIFTS,
    AUTOCORRELATIONS,
    CHARTS,
    CUSUM,
    EWMA,
    MAX_DAYS,
    REPLICATIONS,
    SHEWHART,
    SHIFTS,
    Chart,
    correlation_row,
    delay_quantiles,
    detection_curve,
    example_payload,
    false_alarms,
    first_signal_days,
    geometric_quantile,
    moving_range_sigma,
    run_lengths,
    shewhart_average_run_length,
    signal_probability,
)

SUMMARY = example_payload()
CURVES = {curve.chart: curve for curve in SUMMARY.curves}
SHEWHART_CURVE = CURVES[SHEWHART.name]
EWMA_CURVE = CURVES[EWMA.name]
CUSUM_CURVE = CURVES[CUSUM.name]


def _website_run_lengths(
    rng: np.random.Generator, shifts: tuple[float, ...], reps: int, max_days: int
) -> dict[str, list[list[int]]]:
    """Transcribe the website generator's loops, chart by chart and run by run."""

    def first_signal(sig: NDArray[np.bool_]) -> int:
        idx = int(np.argmax(sig))
        return idx + 1 if sig[idx] else max_days

    def shewhart(x: NDArray[np.float64], limit: float) -> NDArray[np.bool_]:
        return np.abs(x) > limit

    def ewma(x: NDArray[np.float64], lam: float, limit: float) -> NDArray[np.bool_]:
        acc, z = 0.0, np.zeros_like(x)
        for i, v in enumerate(x):
            acc = lam * v + (1 - lam) * acc
            z[i] = acc
        sd = np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * (np.arange(len(x)) + 1))))
        return np.abs(z) > limit * sd

    def cusum(x: NDArray[np.float64], k: float, h: float) -> NDArray[np.bool_]:
        hi = lo = 0.0
        out = np.zeros(len(x), bool)
        for i, v in enumerate(x):
            hi = max(0.0, hi + v - k)
            lo = max(0.0, lo - v - k)
            out[i] = (hi > h) or (lo > h)
        return out

    charts: dict[str, Callable[[NDArray[np.float64]], NDArray[np.bool_]]] = {
        "Three-sigma rule": lambda x: shewhart(x, 3.0),
        "Exponentially weighted": lambda x: ewma(x, 0.2, 2.86),
        "Cumulative sum": lambda x: cusum(x, 0.5, 4.72),
    }
    return {
        name: [
            [first_signal(fn(rng.normal(s, 1.0, max_days))) for _ in range(reps)] for s in shifts
        ]
        for name, fn in charts.items()
    }


def _cusum_one_sided(mean: float, k: float, h: float, states: int = 400) -> float:
    """Brook and Evans: the upper cumulative sum as a Markov chain on ``states`` cells of [0, h]."""
    w = 2 * h / (2 * states - 1)
    i = np.arange(states)
    step = (i[None, :] - i[:, None]) * w
    moves = stats.norm.cdf(step + k + w / 2 - mean) - stats.norm.cdf(step + k - w / 2 - mean)
    moves[:, 0] = stats.norm.cdf(k - i * w + w / 2 - mean)
    return float(np.linalg.solve(np.eye(states) - moves, np.ones(states))[0])


def _cusum_markov(mean: float, k: float = 0.5, h: float = 4.72) -> float:
    """Two sides combined as ``1 / ARL = 1 / ARL+ + 1 / ARL-``, an approximation."""
    return 1 / (1 / _cusum_one_sided(mean, k, h) + 1 / _cusum_one_sided(-mean, k, h))


@cache
def _ewma_markov(
    mean: float, lam: float = 0.2, limit: float = 2.86, horizon: int = MAX_DAYS, cells: int = 301
) -> float:
    """Propagate the weighted average on a grid, removing mass outside each day's limit."""
    edge = limit * np.sqrt(lam / (2 - lam))
    w = 2 * edge / cells
    z = -edge + w * (np.arange(cells) + 0.5)
    low = (z[None, :] - w / 2 - (1 - lam) * z[:, None]) / lam - mean
    high = (z[None, :] + w / 2 - (1 - lam) * z[:, None]) / lam - mean
    moves = stats.norm.cdf(high) - stats.norm.cdf(low)
    days = np.arange(1, horizon + 1)
    bounds = limit * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * days)))
    mass = stats.norm.cdf((z + w / 2) / lam - mean) - stats.norm.cdf((z - w / 2) / lam - mean)
    total = 1.0
    for day in range(1, horizon):
        mass = np.where(np.abs(z) <= bounds[day - 1], mass, 0.0)
        alive = float(mass.sum())
        total += alive
        if alive < 1e-12:
            break
        mass = mass @ moves
    return total


def _moving_range_alarms(rho: float, rng: np.random.Generator, series: int) -> float:
    """The article's autocorrelation procedure, batched: days between false alarms."""
    eps = rng.normal(0, np.sqrt(1 - rho**2), (series, 500))
    x = np.zeros((series, 500))
    for i in range(1, 500):
        x[:, i] = rho * x[:, i - 1] + eps[:, i]
    sigma = np.abs(np.diff(x[:, :200], axis=1)).mean(axis=1) / 1.128
    signals = (np.abs(x[:, 200:]) > 3 * sigma[:, None]).sum(axis=1)
    return float(300 / signals.mean())


def test_the_batched_simulation_is_the_websites_loop() -> None:
    """Run by run, on a short horizon that leaves some runs without a signal."""
    shifts, reps, days = (0.25, 1.0, 3.0), 12, 300
    website = np.random.default_rng(23)
    expected = _website_run_lengths(website, shifts, reps, days)
    ours = np.random.default_rng(23)
    censored = 0
    for chart in CHARTS:
        for s, lengths in zip(shifts, expected[chart.name], strict=True):
            got = run_lengths(ours, s, chart, replications=reps, max_days=days)
            assert got.tolist() == lengths
            censored += int(np.count_nonzero(got == days))
    assert censored > 0
    # The generator is left in the same state, so the next chart would draw the same numbers.
    assert ours.bit_generator.state == website.bit_generator.state


def test_the_figure_design() -> None:
    """Three charts in order, seven shifts, 1,500 runs of 2,000 days."""
    assert (SUMMARY.replications, SUMMARY.max_days) == (REPLICATIONS, MAX_DAYS) == (1500, 2000)
    assert tuple(curve.chart for curve in SUMMARY.curves) == tuple(c.name for c in CHARTS)
    for curve in SUMMARY.curves:
        assert curve.shifts == SHIFTS
        assert np.all(np.diff(curve.mean_days) < 0)
    # Rerunning a chart on its own gives a different stream: the order matters.
    alone = detection_curve(np.random.default_rng(23), EWMA, (0.25,), replications=200)
    assert alone.mean_days[0] != EWMA_CURVE.mean_days[0]


def test_the_shewhart_curve_is_the_geometric_run_length() -> None:
    """Simulated means within four standard errors of ``[1 - (1 - p)^M] / p``."""
    for s, mean, sd, exact in zip(
        SHIFTS,
        SHEWHART_CURVE.mean_days,
        SHEWHART_CURVE.sd_days,
        SUMMARY.shewhart_exact,
        strict=True,
    ):
        p = signal_probability(s)
        assert exact == pytest.approx((1 - (1 - p) ** MAX_DAYS) / p, rel=1e-9)
        assert abs(mean - exact) < 4 * np.sqrt((1 - p) / p**2 / REPLICATIONS) + 0.01
        # The spread of a geometric run length.
        assert sd == pytest.approx(np.sqrt(1 - p) / p, rel=0.1)


def test_the_memory_charts_match_markov_chains() -> None:
    """Simulated means within four standard errors and two percent of the chains."""
    for s, mean, sd in zip(SHIFTS, CUSUM_CURVE.mean_days, CUSUM_CURVE.sd_days, strict=True):
        exact = _cusum_markov(s)
        assert abs(mean - exact) < 4 * sd / np.sqrt(REPLICATIONS) + 0.02 * exact
    for s, mean, sd in zip(SHIFTS, EWMA_CURVE.mean_days, EWMA_CURVE.sd_days, strict=True):
        exact = _ewma_markov(s)
        assert abs(mean - exact) < 4 * sd / np.sqrt(REPLICATIONS) + 0.02 * exact
    # The weighted chart with weight one is the three-sigma rule, which the chain reproduces.
    exact = shewhart_average_run_length(1.0, max_days=MAX_DAYS)
    assert _ewma_markov(1.0, lam=1.0, limit=3.0) == pytest.approx(exact, rel=1e-6)


def test_the_charts_share_a_false_alarm_rate() -> None:
    """In control: 370 days for the rule, 366 and 350 for the calibrated charts."""
    shewhart = shewhart_average_run_length(0.0)
    ewma = _ewma_markov(0.0, horizon=4000)
    cusum = _cusum_markov(0.0)
    assert round(shewhart) == 370
    assert (round(ewma), round(cusum)) == (366, 350)
    for arl in (ewma, cusum):
        assert abs(arl / 370 - 1) < 0.06


def test_the_figure_claims() -> None:
    """Memory buys speed on small shifts, not on large ones; the curves converge above 2 sigma."""
    rule = np.array(SHEWHART_CURVE.mean_days)
    ewma = np.array(EWMA_CURVE.mean_days)
    cusum = np.array(CUSUM_CURVE.mean_days)
    small = [SHIFTS.index(s) for s in (0.5, 0.75, 1.0)]
    for memory in (ewma, cusum):
        speedup = rule / memory
        assert np.all((speedup[small] > 4.4) & (speedup[small] < 5.3))
        # Only about 2.4 times at a quarter sigma, where every chart is near its false alarms.
        assert 2.3 < speedup[0] < 2.5
        # The advantage narrows from one sigma up.
        assert np.all(np.diff(speedup[SHIFTS.index(1.0) :]) < 0)
    large = [SHIFTS.index(s) for s in (2.0, 3.0)]
    spread = np.ptp(np.vstack([rule, ewma, cusum]), axis=0)
    assert np.all(spread[large] < 4)
    assert spread[SHIFTS.index(3.0)] < 1
    assert spread[0] > 150
    # At three sigma the cumulative sum is slower than the single-day rule.
    assert cusum[-1] > rule[-1]
    # A one-sigma shift: 44 days against about nine.
    assert rule[SHIFTS.index(1.0)] == pytest.approx(44, abs=1)
    assert 8.5 < ewma[SHIFTS.index(1.0)] < 10 and 8.5 < cusum[SHIFTS.index(1.0)] < 10


def test_the_false_alarm_table_matches_the_article() -> None:
    """0.2700 percent a day, 370 days on average, a median of 256, a tenth within 39."""
    table = SUMMARY.false_alarms
    assert f"{table.signal_probability:.4%}" == "0.2700%"
    assert table.signal_probability == pytest.approx(2 * (1 - stats.norm.cdf(3)), rel=1e-12)
    assert (round(table.mean_days), round(table.median_days)) == (370, 256)
    assert [round(days) for days in table.quantile_days] == [39, 106, 256]
    assert table == false_alarms()


def test_the_article_detection_table_against_closed_forms() -> None:
    """The simulated columns (seed 9, 4,000 runs) agree with exact means and the chains."""
    printed = {
        0.0: ((378.4, 263), 359.2, 347.5),
        0.5: ((157.0, 110), 35.5, 34.7),
        1.0: ((44.0, 31), 8.8, 9.9),
        1.5: ((15.1, 11), 4.3, 5.5),
        2.0: ((6.4, 5), 2.7, 3.8),
        3.0: ((2.0, 1), 1.5, 2.5),
    }
    assert tuple(row.shift for row in SUMMARY.shewhart_table) == ARTICLE_SHIFTS
    for row in SUMMARY.shewhart_table:
        (mean, median), ewma, cusum = printed[row.shift]
        p = signal_probability(row.shift)
        se = np.sqrt(1 - p) / p / np.sqrt(4000)
        assert abs(mean - row.mean_days) < 4 * se + 0.05
        # The median of 4,000 runs: its error is about sqrt(1 / 4) / (sqrt(4000) f(m)).
        density = p * (1 - p) ** (row.median_days - 1)
        assert abs(median - row.median_days) <= 4 * 0.5 / np.sqrt(4000) / density + 1
        exact_ewma = _ewma_markov(row.shift, horizon=4000)
        exact_cusum = _cusum_markov(row.shift)
        assert abs(ewma - exact_ewma) < 4 * exact_ewma / np.sqrt(4000) + 0.05
        assert abs(cusum - exact_cusum) < 4 * exact_cusum / np.sqrt(4000) + 0.05
    # The exact medians and means behind the prose: 157 and 44 days are 155 and 44 exactly.
    exact = {row.shift: row for row in SUMMARY.shewhart_table}
    assert [exact[s].median_days for s in ARTICLE_SHIFTS] == [257, 108, 31, 11, 5, 1]
    assert (round(exact[0.5].mean_days), round(exact[1.0].mean_days)) == (155, 44)


def test_the_delay_quantiles() -> None:
    """A one-sigma shift: caught within 5, 13, 31, 61 and 100 days by a tenth to nine tenths."""
    delay = SUMMARY.one_sigma_delay
    assert delay == delay_quantiles()
    assert delay.days == (5, 13, 31, 61, 100)
    p = signal_probability(1.0)
    for q, day in zip(delay.quantiles, delay.days, strict=True):
        assert 1 - (1 - p) ** day >= q > 1 - (1 - p) ** (day - 1)
    # The article's 8,000 runs print 99 for the ninetieth percentile, within its noise.
    printed = (5, 13, 31, 61, 99)
    for q, day, shown in zip(delay.quantiles, delay.days, printed, strict=True):
        density = p * (1 - p) ** (day - 1)
        assert abs(shown - day) <= 4 * np.sqrt(q * (1 - q) / 8000) / density + 1
    # Mean 44, median 31, and one time in ten more than three months.
    assert (round(delay.mean_days), delay.days[2]) == (44, 31)
    assert delay.days[-1] > 90
    # Six weeks, as the introduction says.
    assert round(delay.mean_days / 7) == 6


def test_tightening_the_limit() -> None:
    """Two sigma: 22 days between false alarms and about 6 to catch a one-sigma shift."""
    assert round(SUMMARY.tighter_in_control) == 22
    assert round(SUMMARY.tighter_one_sigma) == 6
    assert SUMMARY.tighter_in_control == pytest.approx(1 / (2 * stats.norm.sf(2)))


def test_the_moving_range_under_autocorrelation() -> None:
    """Sigma follows sqrt(1 - rho); the plug-in misses the estimate's noise at low correlation."""
    printed = {0.0: ("1.00", 307), 0.3: ("0.84", 76), 0.6: ("0.63", 17), 0.8: ("0.45", 5.5)}
    rows = {row.autocorrelation: row for row in SUMMARY.correlations}
    assert tuple(rows) == AUTOCORRELATIONS
    for rho, (sigma, _) in printed.items():
        row = rows[rho]
        assert f"{row.sigma_estimate:.2f}" == sigma
        assert row.sigma_estimate == pytest.approx(np.sqrt(1 - rho), rel=5e-4)
        assert row == correlation_row(rho)
    # With the estimate at its mean, the intervals are 372, 83, 17 and 5.6 days.
    plug_in = [round(rows[rho].false_alarm_interval, 1) for rho in AUTOCORRELATIONS]
    assert plug_in == [371.6, 83.0, 17.3, 5.6]
    # The article's procedure, batched on another generator, gives its shorter intervals.
    rng = np.random.default_rng(2025)
    for rho, (_, interval) in printed.items():
        simulated = _moving_range_alarms(rho, rng, 4000)
        assert simulated == pytest.approx(interval, rel=0.12)
    # A mean moving range of two independent normals is 2 / sqrt(pi).
    diffs = np.abs(np.diff(np.random.default_rng(3).normal(size=400_000)))
    assert float(diffs.mean()) / 1.128 == pytest.approx(moving_range_sigma(0.0), abs=0.004)


def test_the_charts_by_hand() -> None:
    """Strict limits, both sides of the sum, time-varying weighted limits, and no signal."""
    rule = Chart("rule", "shewhart", 3.0)
    assert first_signal_days(rule, [[0.0, 3.0, -3.1, 5.0]]).tolist() == [3]
    assert first_signal_days(rule, [[0.0, 3.0, -3.0]]).tolist() == [3]
    cusum = Chart("sum", "cusum", 1.0, 0.5)
    # Upper sums 0.5, 1.0 (not above 1) and 1.3.
    assert first_signal_days(cusum, [[1.0, 1.0, 0.8, 0.0]]).tolist() == [3]
    assert first_signal_days(cusum, [[-1.0, -1.0, -0.8, 0.0]]).tolist() == [3]
    # A negative day resets the upper sum to zero: 0.5, 0, 0.5, 1.0, then 1.1 on day five.
    assert first_signal_days(cusum, [[1.0, -1.2, 1.0, 1.0, 0.6]]).tolist() == [5]
    ewma = Chart("ewma", "ewma", 1.0, 0.5)
    # Day one: z = 0.45 against 0.5; day two: z = 0.825 against sqrt(0.3125) = 0.559.
    assert first_signal_days(ewma, [[0.9, 1.2, 0.0]]).tolist() == [2]
    assert first_signal_days(ewma, [[0.9, 0.0, 0.0]]).tolist() == [3]
    rows = np.array([[1.0, 1.0, 0.8, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert first_signal_days(cusum, rows).tolist() == [3, 4]


def test_limiting_cases_and_symmetry() -> None:
    """Weight one and a zero-width sum reduce to the rule; a mirrored metric changes nothing."""
    x = np.random.default_rng(8).normal(0.5, 1.0, (60, 400))
    rule = first_signal_days(SHEWHART, x)
    assert np.array_equal(first_signal_days(Chart("w", "ewma", 3.0, 1.0), x), rule)
    assert np.array_equal(first_signal_days(Chart("s", "cusum", 0.0, 3.0), x), rule)
    for chart in CHARTS:
        assert np.array_equal(first_signal_days(chart, -x), first_signal_days(chart, x))
    # Geometric run lengths: the mean without a horizon, and a tenth within 39 days.
    assert shewhart_average_run_length(0.0) == pytest.approx(1 / signal_probability(0.0))
    assert shewhart_average_run_length(0.0, max_days=1) == 1.0
    assert geometric_quantile(0.5, 0.5) == pytest.approx(1.0)
    assert signal_probability(0.0, 1.0) == pytest.approx(2 * stats.norm.sf(1.0))


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the run lengths; another seed does not."""
    first = run_lengths(np.random.default_rng(4), 0.5, CUSUM, replications=50, max_days=200)
    again = run_lengths(np.random.default_rng(4), 0.5, CUSUM, replications=50, max_days=200)
    other = run_lengths(np.random.default_rng(5), 0.5, CUSUM, replications=50, max_days=200)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_invalid_inputs_are_rejected() -> None:
    """Bad charts, shapes, counts and probabilities are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        first_signal_days(SHEWHART, [1.0, 2.0])
    with pytest.raises(ValueError):
        first_signal_days(SHEWHART, [[1.0, np.nan]])
    with pytest.raises(ValueError):
        first_signal_days(Chart("bad", "ewma", 3.0, 0.0), [[1.0]])
    with pytest.raises(ValueError):
        first_signal_days(Chart("bad", "ewma", 3.0, 1.5), [[1.0]])
    with pytest.raises(ValueError):
        first_signal_days(Chart("bad", "shewhart", -1.0), [[1.0]])
    with pytest.raises(ValueError):
        first_signal_days(Chart("bad", "median", 3.0), [[1.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run_lengths(rng, 0.5, SHEWHART, replications=0)
    with pytest.raises(TypeError):
        run_lengths(rng, 0.5, SHEWHART, max_days=10.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        detection_curve(rng, SHEWHART, ())
    with pytest.raises(ValueError):
        geometric_quantile(1.0, 0.1)
    with pytest.raises(ValueError):
        signal_probability(0.0, 0.0)
    with pytest.raises(ValueError):
        moving_range_sigma(1.0)
    with pytest.raises(ValueError):
        shewhart_average_run_length(0.0, max_days=0)
