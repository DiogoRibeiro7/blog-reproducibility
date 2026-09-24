"""Check the latency model and the figure's precision against the quantile closed form.

The figure's simulation is checked against a direct transcription of its loop,
then pinned at the precision of its percent axis (it is the figure's own; the
article prints no table from this stream of draws). The closed forms are
checked on their own (the CDF inverts the quantile, the density is the CDF's
derivative, the moments match sampling), and the simulation is checked
against them.

The article's tables come from a generator that first draws two million
requests for each of four scenarios, so they are not pinned. Its population
values (68.4, 160.2, 225.5 and 1,203.8 ms, mean 111.8) are single draws from
this mixture and are checked to lie within three standard errors of the exact
ones (68.4, 160.3, 225.4 and 1,216.8 ms, mean 112.1). Its predicted relative
errors at 10,000 requests (0.77, 2.51 and 6.14 percent) use densities
estimated from that sample; the exact densities give 0.78, 2.56 and 6.38.

Both alt texts say the median is about "an order of magnitude" more precise
than the 99th percentile. The closed form gives a factor of 8.2 at every
sample size, and the simulation between 6.9 and 8.5: most of a tenfold gap,
not all of it. The tests pin that range.
"""

import numpy as np
import pytest

from blog_reproducibility.statistics.percentile_metrics import (
    BODY_LOG_MEAN,
    QUANTILES,
    SAMPLE_SIZES,
    PrecisionRow,
    draw_latency,
    example_payload,
    mean_relative_se,
    population_cdf,
    population_density,
    population_mean,
    population_quantile,
    population_sd,
    precision_rows,
    quantile_relative_se,
)

SUMMARY = example_payload()
SIMULATED = SUMMARY.simulated
PREDICTED = SUMMARY.predicted

FIGURE_PERCENTS = {
    500: (9.02, 3.30, 12.30, 27.95),
    1000: (7.06, 2.67, 9.21, 20.19),
    5000: (3.31, 1.12, 3.78, 9.08),
    10_000: (2.27, 0.77, 2.97, 6.52),
    50_000: (0.96, 0.39, 1.19, 2.69),
    100_000: (0.63, 0.23, 0.75, 1.79),
}


def _series(row: PrecisionRow) -> tuple[float, float, float, float]:
    return (row.mean, row.median, row.p95, row.p99)


def _figure_loop(sizes: tuple[int, ...], replications: int) -> list[tuple[float, ...]]:
    """Run a direct transcription of the figure's generator."""
    rng = np.random.default_rng(0)

    def latency(n: int) -> np.ndarray:
        body = rng.lognormal(np.log(80) - 0.18, 0.6, n)
        slow = rng.random(n) < 0.03
        return np.where(slow, rng.lognormal(np.log(900), 0.7, n), body)

    rows = []
    for n in sizes:
        acc: list[list[float]] = [[], [], [], []]
        for _ in range(replications):
            v = latency(n)
            acc[0].append(float(v.mean()))
            for series, q in zip(acc[1:], (0.5, 0.95, 0.99), strict=True):
                series.append(float(np.quantile(v, q)))
        rows.append(tuple(float(np.std(a) / np.mean(a)) for a in acc))
    return rows


def test_the_simulation_matches_the_figure_loop() -> None:
    """Same draws, same statistics, same relative errors, bit for bit."""
    fast = precision_rows(sample_sizes=(500, 1000), replications=20)
    assert [_series(row) for row in fast] == _figure_loop((500, 1000), 20)


def test_the_figure_values_are_reproduced() -> None:
    """The four series at every sample size, in percent to two decimals."""
    assert tuple(row.requests for row in SIMULATED) == SAMPLE_SIZES
    for row in SIMULATED:
        percents = tuple(round(100 * value, 2) for value in _series(row))
        assert percents == FIGURE_PERCENTS[row.requests]


def test_the_simulation_follows_the_closed_form() -> None:
    """Within 20 percent everywhere: the spread of 200 repeats is itself uncertain by about 5."""
    for simulated, predicted in zip(SIMULATED, PREDICTED, strict=True):
        assert simulated.requests == predicted.requests
        for measured, expected in zip(_series(simulated), _series(predicted), strict=True):
            assert measured == pytest.approx(expected, rel=0.2)


def test_the_median_is_most_precise_and_p99_least() -> None:
    """At every sample size: median, then mean, then p95, then p99."""
    for row in (*SIMULATED, *PREDICTED):
        assert row.median < row.mean < row.p95 < row.p99


def test_the_gap_is_most_of_an_order_of_magnitude() -> None:
    """p99 is 8.2 times less precise than the median in closed form; 6.9 to 8.5 simulated."""
    for row in PREDICTED:
        assert row.p99 / row.median == pytest.approx(8.23, abs=0.01)
    ratios = [row.p99 / row.median for row in SIMULATED]
    assert min(ratios) > 6.9
    assert max(ratios) < 8.6


def test_density_explains_the_gap() -> None:
    """The tail's sparse density, not the binomial numerator, makes p99 imprecise."""
    median, p99 = population_quantile(0.5), population_quantile(0.99)
    numerator_ratio = np.sqrt(0.5 * 0.5) / np.sqrt(0.99 * 0.01)
    density_ratio = (population_density(median) * median) / (population_density(p99) * p99)
    assert numerator_ratio == pytest.approx(5.03, abs=0.01)
    assert density_ratio == pytest.approx(41.3, abs=0.1)
    assert density_ratio / numerator_ratio == pytest.approx(8.23, abs=0.01)


def test_the_closed_forms_scale_as_one_over_root_n() -> None:
    """A hundred times the requests divides every relative error by ten."""
    for p in QUANTILES:
        assert quantile_relative_se(p, 1000) == pytest.approx(10 * quantile_relative_se(p, 100_000))
    assert mean_relative_se(1000) == pytest.approx(10 * mean_relative_se(100_000))


def test_the_quantile_and_density_are_consistent() -> None:
    """The CDF inverts the quantile, and the density is the CDF's derivative."""
    for p in (0.01, 0.5, 0.9, 0.95, 0.99, 0.999):
        assert population_cdf(population_quantile(p)) == pytest.approx(p, abs=1e-10)
    for x in (30.0, 68.4, 225.0, 1200.0):
        step = 1e-4 * x
        slope = (population_cdf(x + step) - population_cdf(x - step)) / (2 * step)
        assert population_density(x) == pytest.approx(slope, rel=1e-6)


def test_the_moments_match_the_parameters_and_sampling() -> None:
    """The body averages 80 ms by construction; the mixture's moments match a large sample."""
    assert np.exp(BODY_LOG_MEAN + 0.6**2 / 2) == pytest.approx(80.0)
    assert population_mean() == pytest.approx(0.97 * 80 + 0.03 * 900 * np.exp(0.245))
    sample = draw_latency(1_000_000, np.random.default_rng(31))
    assert sample.mean() == pytest.approx(population_mean(), rel=3 * mean_relative_se(1_000_000))
    assert sample.std() == pytest.approx(population_sd(), rel=0.03)
    assert np.mean(sample > population_quantile(0.97)) == pytest.approx(0.03, abs=0.001)


def test_the_article_population_is_a_draw_from_this_mixture() -> None:
    """The article's two-million-request values lie within three standard errors (plus rounding)."""
    printed = {0.5: 68.4, 0.9: 160.2, 0.95: 225.5, 0.99: 1203.8}
    for point in SUMMARY.population.quantiles:
        se = quantile_relative_se(point.probability, 2_000_000) * point.latency
        assert abs(printed[point.probability] - point.latency) < 3 * se + 0.05
    mean = SUMMARY.population.mean
    assert abs(111.8 - mean) < 3 * mean_relative_se(2_000_000) * mean + 0.05
    assert [round(point.latency, 1) for point in SUMMARY.population.quantiles] == [
        68.4,
        160.3,
        225.4,
        1216.8,
    ]


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the rows; another seed does not."""

    def small(seed: int) -> tuple[PrecisionRow, ...]:
        return precision_rows(seed, sample_sizes=(200,), replications=10)

    assert small(3) == small(3)
    assert small(3) != small(4)


def test_invalid_inputs_are_rejected() -> None:
    """Empty samples, impossible probabilities, non-positive latencies and booleans are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        draw_latency(0, rng)
    with pytest.raises(TypeError):
        draw_latency(True, rng)
    with pytest.raises(ValueError):
        population_quantile(1.0)
    with pytest.raises(ValueError):
        population_density(0.0)
    with pytest.raises(ValueError):
        quantile_relative_se(0.5, 0)
    with pytest.raises(ValueError):
        precision_rows(replications=1)
