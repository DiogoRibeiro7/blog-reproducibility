"""Check the block bootstrap against the website loop, closed forms, the article and the figure.

The figure's simulation and the article's coverage table are one computation
(series ``s`` from a generator seeded at ``s``, 400 series and 400 resamples),
so the table runs once here, in about three seconds, and is pinned exactly: every
count out of 400 reproduces the website loop, and the percentages the article
prints. The batched resampling is checked against a transcription of the
website's per-resample loop, interval by interval, including a block length
that does not divide the series.

The closed forms explain the table. The true standard error of the mean at an
autocorrelation of 0.7 is 0.234, as the article prints; its effective sample
sizes of about 35 and 10 are ``n (1 - phi) / (1 + phi)``; and ``n^(1/3)`` is
about six. The article's ordinary bootstrap standard error of 0.076 and its
"three times larger" come from one series (seed 1) whose sample variance is
low: the ordinary bootstrap's expected standard error is 0.098, a ratio of 2.4.
A normal interval of that expected width would cover 95, 85, 59 and 34 percent;
the simulated ordinary bootstrap covers 94, 80, 53 and 34.
"""

from math import sqrt

import numpy as np
import pytest
from numpy.typing import NDArray

from blog_reproducibility.statistics.block_bootstrap import (
    AUTOCORRELATIONS,
    BLOCK_LENGTHS,
    NOMINAL_COVERAGE,
    REPLICATIONS,
    ar1_series,
    block_bootstrap_interval,
    block_bootstrap_means,
    coverage_row,
    dependence_row,
    effective_sample_size,
    example_payload,
    mean_variance,
    normal_coverage,
    ordinary_bootstrap_variance,
    suggested_block_length,
)

SUMMARY = example_payload()
ROWS = {row.autocorrelation: row for row in SUMMARY.rows}
DEPENDENCE = {row.autocorrelation: row for row in SUMMARY.dependence}

# Series covered out of 400 at block lengths 1, 2, 5, 10, 20 and 40.
FIGURE_COUNTS = {
    0.0: (374, 373, 373, 372, 357, 334),
    0.3: (318, 342, 361, 361, 358, 334),
    0.7: (213, 271, 322, 341, 341, 325),
    0.9: (138, 169, 240, 282, 302, 306),
}
ARTICLE_TABLE = {
    0.0: (94, 93, 93, 93, 89, 84),
    0.3: (80, 86, 90, 90, 90, 84),
    0.7: (53, 68, 80, 85, 85, 81),
    0.9: (34, 42, 60, 70, 76, 76),
}


def _binomial_sd(p: float) -> float:
    return sqrt(p * (1 - p) / REPLICATIONS)


def _website_ar1(n: int, phi: float, r: np.random.Generator) -> NDArray[np.float64]:
    """Transcribe the website generator's series."""
    x = np.empty(n)
    x[0] = r.normal(0, 1 / np.sqrt(1 - phi**2))
    e = r.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _website_block_ci(
    x: NDArray[np.float64], block: int, b_count: int, r: np.random.Generator
) -> tuple[float, float]:
    """Transcribe the website generator's interval, one call per resample."""
    n = len(x)
    nb = int(np.ceil(n / block))
    means = np.empty(b_count)
    for b in range(b_count):
        starts = r.integers(0, n - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = x[idx].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def test_the_figure_counts_are_reproduced() -> None:
    """Every count out of 400, for every autocorrelation and block length, as published."""
    assert tuple(ROWS) == AUTOCORRELATIONS
    for phi, counts in FIGURE_COUNTS.items():
        assert ROWS[phi].block_lengths == BLOCK_LENGTHS
        assert ROWS[phi].replications == REPLICATIONS
        assert ROWS[phi].covered == counts
        assert ROWS[phi].coverage == tuple(hits / REPLICATIONS for hits in counts)


def test_the_article_coverage_table() -> None:
    """The article's table to the percent it prints."""
    for phi, printed in ARTICLE_TABLE.items():
        assert tuple(round(100 * value) for value in ROWS[phi].coverage) == printed


def test_the_draws_and_intervals_match_the_website_loop() -> None:
    """Same generator, same order: identical series and intervals at every block length."""
    for seed, phi in ((0, 0.7), (1, 0.9)):
        ours, theirs = np.random.default_rng(seed), np.random.default_rng(seed)
        series = ar1_series(ours, 200, phi)
        assert np.array_equal(series, _website_ar1(200, phi, theirs))
        for block in BLOCK_LENGTHS:
            expected = _website_block_ci(series, block, 400, theirs)
            assert block_bootstrap_interval(ours, series, block) == expected


def test_a_block_that_does_not_divide_the_series_is_truncated_the_same_way() -> None:
    """Joined blocks longer than the series are cut to it, as in the article's loop."""
    for n, block in ((150, 40), (23, 5), (31, 31)):
        ours, theirs = np.random.default_rng(n), np.random.default_rng(n)
        series = ar1_series(ours, n, 0.5)
        _website_ar1(n, 0.5, theirs)
        assert block_bootstrap_interval(ours, series, block, 60) == _website_block_ci(
            series, block, 60, theirs
        )


def test_the_coverage_row_counts_the_single_intervals() -> None:
    """Batched percentiles give the same coverage as one interval per series and block."""
    blocks = (1, 3, 8)
    row = coverage_row(0.6, length=60, replications=25, resamples=80, block_lengths=blocks)
    hits = [0, 0, 0]
    for seed in range(25):
        rng = np.random.default_rng(seed)
        series = ar1_series(rng, 60, 0.6)
        for j, block in enumerate(blocks):
            low, high = block_bootstrap_interval(rng, series, block, 80)
            hits[j] += low <= 0.0 <= high
    assert row.covered == tuple(hits)


def test_the_ordinary_bootstrap_breaks_under_dependence() -> None:
    """The title: 94 percent with independent data, 80 at 0.3, falling to 34 at 0.9."""
    ordinary = [ROWS[phi].coverage[0] for phi in AUTOCORRELATIONS]
    assert abs(ordinary[0] - NOMINAL_COVERAGE) < 2 * _binomial_sd(NOMINAL_COVERAGE)
    assert ordinary[1] < NOMINAL_COVERAGE - 5 * _binomial_sd(NOMINAL_COVERAGE)
    assert ordinary == sorted(ordinary, reverse=True)
    assert ordinary[-1] < 0.35


def test_any_short_block_is_fine_with_independent_data() -> None:
    """The alt text: at zero autocorrelation blocks up to 10 cover 93 percent; 40 falls to 84."""
    independent = ROWS[0.0].coverage
    for value in independent[:4]:
        assert abs(value - NOMINAL_COVERAGE) < 2 * _binomial_sd(NOMINAL_COVERAGE)
    assert independent[4] > independent[5]
    assert independent[5] < NOMINAL_COVERAGE - 5 * _binomial_sd(NOMINAL_COVERAGE)
    # Blocks of 40 leave five blocks per resample.
    assert -(-200 // 40) == 5


def test_longer_blocks_recover_most_but_not_all_of_the_coverage() -> None:
    """The alt text: the best block recovers over half the shortfall and still misses 95 percent."""
    for phi in AUTOCORRELATIONS[1:]:
        coverage = ROWS[phi].coverage
        best = max(coverage)
        recovered = (best - coverage[0]) / (NOMINAL_COVERAGE - coverage[0])
        assert recovered > 0.6
        assert best < NOMINAL_COVERAGE - 3 * _binomial_sd(best)
    # The article: 85 percent with blocks of 10 to 20 at 0.7, 76 with 20 to 40 at 0.9.
    assert ROWS[0.7].coverage[3] == ROWS[0.7].coverage[4] == max(ROWS[0.7].coverage)
    assert round(100 * max(ROWS[0.9].coverage)) == 76


def test_the_variance_of_the_mean_is_the_sum_of_autocovariances() -> None:
    """The closed form equals gamma_0 / n^2 times the sum of (n - |k|) phi^|k|."""
    for n in (1, 2, 7, 200):
        for phi in (-0.5, 0.0, 0.3, 0.7, 0.9):
            lags = np.arange(-(n - 1), n)
            direct = float(np.sum((n - np.abs(lags)) * phi ** np.abs(lags)))
            expected = direct / (n**2 * (1 - phi**2))
            assert mean_variance(n, phi) == pytest.approx(expected, rel=1e-12)
    assert mean_variance(200, 0.0) == pytest.approx(1 / 200)


def test_the_closed_forms_match_simulated_series() -> None:
    """4,000 series at 0.7: the spread of their means and their mean sample variance."""
    rng = np.random.default_rng(11)
    phi, n, series_count = 0.7, 200, 4000
    x = np.empty((series_count, n))
    x[:, 0] = rng.normal(0, 1 / sqrt(1 - phi**2), series_count)
    innovations = rng.normal(size=(series_count, n))
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + innovations[:, t]

    assert float(np.var(x.mean(axis=1))) == pytest.approx(mean_variance(n, phi), rel=0.07)
    assert float(np.mean(x.var(axis=1))) / n == pytest.approx(
        ordinary_bootstrap_variance(n, phi), rel=0.02
    )


def test_the_ordinary_bootstrap_variance_is_the_sample_variance_over_n() -> None:
    """With blocks of one and many resamples, the variance of the means is var(x) / n."""
    rng = np.random.default_rng(12)
    series = ar1_series(rng, 200, 0.7)
    means = block_bootstrap_means(rng, series, 1, 20_000)
    assert float(np.var(means)) == pytest.approx(float(np.var(series)) / 200, rel=0.03)


def test_the_article_closed_form_arithmetic() -> None:
    """A true standard error of 0.234 at 0.7, about 35 and 10 effective points, blocks near six."""
    assert round(DEPENDENCE[0.7].mean_standard_error, 3) == 0.234
    assert round(DEPENDENCE[0.7].approximate_effective_sample_size) == 35
    assert DEPENDENCE[0.9].approximate_effective_sample_size == pytest.approx(10, abs=0.6)
    assert round(DEPENDENCE[0.7].effective_sample_size, 1) == 35.8
    assert round(DEPENDENCE[0.9].effective_sample_size, 1) == 11.0
    assert round(SUMMARY.suggested_block_length) == 6
    assert suggested_block_length(1000) == pytest.approx(10.0)
    # The article's 0.076 is one series; the ordinary bootstrap expects 0.098, a ratio of 2.4.
    assert round(DEPENDENCE[0.7].ordinary_bootstrap_standard_error, 3) == 0.098
    assert round(DEPENDENCE[0.7].standard_error_ratio, 1) == 2.4


def test_the_normal_approximation_tracks_the_ordinary_bootstrap() -> None:
    """2 Phi(1.96 / r) - 1 is within six points of the simulated ordinary coverage."""
    predicted = [DEPENDENCE[phi].normal_coverage for phi in AUTOCORRELATIONS]
    assert [round(100 * value) for value in predicted] == [95, 85, 59, 34]
    for phi, value in zip(AUTOCORRELATIONS, predicted, strict=True):
        assert abs(ROWS[phi].coverage[0] - value) < 0.06


def test_limiting_cases() -> None:
    """Independent data: n effective points, a ratio of one; a single observation is itself."""
    row = dependence_row(0.0, 200)
    assert row.effective_sample_size == pytest.approx(200.0)
    assert row.standard_error_ratio == pytest.approx(sqrt(200 / 199))
    assert normal_coverage(10_000, 0.0) == pytest.approx(0.95, abs=1e-4)
    assert mean_variance(1, 0.8) == pytest.approx(1 / (1 - 0.64))
    assert effective_sample_size(200, -0.5) > 200
    sizes = [effective_sample_size(200, phi) for phi in AUTOCORRELATIONS]
    assert sizes == sorted(sizes, reverse=True)


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same first seed reproduces a row; a disjoint run of seeds does not."""

    def small(first_seed: int) -> object:
        return coverage_row(
            0.5,
            length=40,
            replications=30,
            resamples=50,
            block_lengths=(1, 4),
            first_seed=first_seed,
        )

    assert small(3) == small(3)
    assert small(3) != small(100)


def test_invalid_inputs_are_rejected() -> None:
    """Non-stationary series, oversized or zero blocks, bad series and counts are refused."""
    rng = np.random.default_rng(0)
    series = ar1_series(rng, 20, 0.5)
    with pytest.raises(ValueError):
        ar1_series(rng, 20, 1.0)
    with pytest.raises(ValueError):
        ar1_series(rng, 1, 0.5)
    with pytest.raises(TypeError):
        ar1_series(rng, 20, True)
    with pytest.raises(ValueError):
        block_bootstrap_means(rng, series, 21)
    with pytest.raises(ValueError):
        block_bootstrap_means(rng, series, 0)
    with pytest.raises(ValueError):
        block_bootstrap_means(rng, series, 2, 0)
    with pytest.raises(ValueError):
        block_bootstrap_means(rng, series.reshape(4, 5), 2)
    with pytest.raises(ValueError):
        block_bootstrap_interval(rng, np.append(series, np.nan), 2)
    with pytest.raises(ValueError):
        coverage_row(0.5, length=20, block_lengths=())
    with pytest.raises(ValueError):
        coverage_row(0.5, length=20, block_lengths=(40,))
    with pytest.raises(ValueError):
        coverage_row(0.5, replications=0)
    with pytest.raises(ValueError):
        mean_variance(0, 0.5)
    with pytest.raises(ValueError):
        normal_coverage(200, 0.5, nominal=1.0)
    with pytest.raises(ValueError):
        ordinary_bootstrap_variance(1, 0.5)
