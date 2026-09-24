"""Check the Benford spread model against the website, closed forms and the article.

The figure draws each column from a generator seeded at 3 afresh, so its curve
is reproduced draw for draw against a transcription of the website loop. The
article's spread table uses a generator seeded at 2, so its deviations are not
reproduced; they are checked against the closed-form expected deviation, which
each of them matches to within 0.0005. The table's percentile ratios and the
Benford shares are closed forms and are pinned as printed. So are the population
deviations of two columns in the article's first table, salaries uniform from
30,000 to 90,000 (0.1162, as printed) and order counts from 1 to 40 (0.0553,
printed as 0.0544 from a sample of 20,000).

One claim does not hold as written. The alt text says conformity "crosses the
conventional close-conformity threshold somewhere past a hundredfold range, so a
column confined to one or two factors of ten fails the test". It crosses 0.006
well before that: between the 28x and 44x columns of the figure (0.0087 and
0.0036), at 37x for the expected deviation. A column spanning two factors of ten
(110x) has a deviation of 0.0009, a seventh of the threshold; one spanning a
single factor of ten (11x) fails at 0.033. The article's "conformity arrives
somewhere around a hundredfold spread" is loose in the same direction.

The figure's other claims hold: the deviation falls steadily, by a factor of 170,
from 1.8x to 110x, after which it sits on the sampling floor of about 0.001 that
a column of 50,000 values cannot get below, and the range decides it.
"""

from math import log, pi, sqrt

import numpy as np
import pytest
from numpy.typing import NDArray

from blog_reproducibility.statistics.benford import (
    ACCEPTABLE_CONFORMITY,
    ARTICLE_SIGMAS,
    CLOSE_CONFORMITY,
    LOCATION,
    VALUES,
    benford_shares,
    conformity_span,
    digit_shares,
    example_payload,
    expected_mad,
    first_digits,
    lognormal_digit_shares,
    mean_absolute_deviation,
    percentile_span,
    population_mad,
    spread_curve,
    spread_row,
)

SUMMARY = example_payload()
CURVE = SUMMARY.curve
MADS = np.array(CURVE.mads)
SPANS = np.array(CURVE.spans)
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def _website_curve() -> tuple[list[float], list[float]]:
    """Transcribe the website generator's loop."""
    benford = np.log10(1 + 1 / np.arange(1, 10))

    def mad(v: NDArray[np.float64]) -> float:
        v = np.abs(v[v > 0])
        d = (v / 10 ** np.floor(np.log10(v))).astype(int)
        shares = np.array([(d == k).mean() for k in range(1, 10)])
        return float(np.abs(shares - benford).mean())

    sigmas = np.linspace(0.15, 2.6, 22)
    spans, mads = [], []
    for s in sigmas:
        v = np.random.default_rng(3).lognormal(6, s, 50_000)
        spans.append(float(np.exp(2 * 1.96 * s)))
        mads.append(mad(v))
    return spans, mads


def _fourier_shares(location: float, sigma: float, terms: int = 60) -> NDArray[np.float64]:
    """Digit shares from the Fourier series of the wrapped normal density of ``log10 X``."""
    m, s = location / log(10), sigma / log(10)
    edges = np.log10(np.arange(1, 11))
    j = np.arange(1, terms + 1)[:, None]
    damping = np.exp(-2 * pi**2 * j**2 * s**2)
    waves = np.sin(2 * pi * j * (edges[None, :] - m)) / (pi * j)
    cumulative = edges + (damping * waves).sum(axis=0)
    shares: NDArray[np.float64] = np.diff(cumulative)
    return shares


def test_the_figure_matches_the_website_loop() -> None:
    """Same seed for every column and the same digit arithmetic: identical curves."""
    spans, mads = _website_curve()
    assert list(CURVE.spans) == spans
    assert list(CURVE.mads) == mads
    assert CURVE.sigmas == tuple(float(s) for s in np.linspace(0.15, 2.6, 22))
    assert spread_curve() == CURVE


def test_the_article_benford_table() -> None:
    """30.10%, 17.61%, 12.49%, 9.69%, 7.92%, 6.69%, 5.80%, 5.12%, 4.58%."""
    assert [f"{share:.2%}" for share in SUMMARY.benford] == [
        "30.10%",
        "17.61%",
        "12.49%",
        "9.69%",
        "7.92%",
        "6.69%",
        "5.80%",
        "5.12%",
        "4.58%",
    ]
    assert sum(SUMMARY.benford) == pytest.approx(1.0, abs=1e-15)
    assert np.array_equal(benford_shares(), BENFORD)


def test_the_article_spread_table() -> None:
    """Ranges 2x to 12,185x exactly; the seed-2 deviations against the closed form."""
    rows = SUMMARY.article_rows
    assert tuple(row.sigma for row in rows) == ARTICLE_SIGMAS
    # The article writes the range as 10 ** (2 x 1.96 sigma / ln 10).
    for row in rows:
        assert row.span == pytest.approx(10 ** (2 * 1.96 * row.sigma / log(10)), rel=1e-13)
    assert [f"{row.span:,.0f}" for row in rows] == ["2", "5", "23", "110", "530", "12,185"]
    printed = (0.1348, 0.0738, 0.0131, 0.0015, 0.0013, 0.0009)
    for row, value in zip(rows, printed, strict=True):
        assert abs(value - row.expected_mad) < 0.0005
    # The three narrow columns fail both thresholds and the three wide ones conform.
    assert all(row.expected_mad > ACCEPTABLE_CONFORMITY for row in rows[:3])
    assert all(row.expected_mad < CLOSE_CONFORMITY for row in rows[3:])
    # "A column at 0.8 ... clearly not Benford, and clearly not by much."
    assert round(rows[2].population_mad, 4) == 0.0127


def test_the_article_first_table_in_closed_form() -> None:
    """Salaries from 30,000 to 90,000 deviate by 0.1162; order counts 1 to 40 by 0.0553."""
    salaries = 30_000 + 10 * np.arange(6000) + 5
    assert digit_shares(salaries).tolist() == [0, 0] + [1 / 6] * 6 + [0]
    assert round(mean_absolute_deviation(salaries), 4) == 0.1162
    orders = np.arange(1, 41)
    assert digit_shares(orders) * 40 == pytest.approx([11, 11, 11, 2, 1, 1, 1, 1, 1])
    # The article's sample of 20,000 prints 0.0544.
    assert round(mean_absolute_deviation(orders), 4) == 0.0553


def test_the_deviation_falls_with_the_range_and_then_hits_the_floor() -> None:
    """Title and alt text: steady improvement to about 110x, then sampling noise alone."""
    wide = int(np.searchsorted(SPANS, 100))
    assert round(float(SPANS[wide])) == 110
    assert np.all(np.diff(MADS[: wide + 1]) < 0)
    assert MADS[0] / MADS[wide] > 150
    floor = expected_mad(LOCATION, 50.0, VALUES)
    assert floor == pytest.approx(0.00105, abs=1e-5)
    assert np.all(np.abs(MADS[wide:] - floor) < 0.0005)
    assert not np.all(np.diff(MADS[wide:]) < 0)
    tail = MADS[SPANS > 1000]
    assert float(tail.mean()) == pytest.approx(floor, rel=0.1)


def test_where_the_thresholds_are_crossed() -> None:
    """Close conformity arrives near 37x, not past 100x as the alt text says."""
    below = int(np.argmax(MADS < CLOSE_CONFORMITY))
    assert (round(float(SPANS[below - 1])), round(float(SPANS[below]))) == (28, 44)
    assert (round(MADS[below - 1], 4), round(MADS[below], 4)) == (0.0087, 0.0036)
    assert SUMMARY.close_conformity_span == pytest.approx(36.6, abs=0.05)
    assert SUMMARY.acceptable_conformity_span == pytest.approx(23.9, abs=0.05)
    assert SPANS[below] < 100
    # One factor of ten fails even the acceptable threshold; two conform closely.
    ten = int(np.argmin(np.abs(np.log(SPANS / 10))))
    hundred = int(np.argmin(np.abs(np.log(SPANS / 100))))
    assert MADS[ten] > ACCEPTABLE_CONFORMITY
    assert MADS[hundred] < CLOSE_CONFORMITY / 6
    for threshold, span in (
        (CLOSE_CONFORMITY, SUMMARY.close_conformity_span),
        (ACCEPTABLE_CONFORMITY, SUMMARY.acceptable_conformity_span),
    ):
        sigma = log(span) / (2 * 1.96)
        assert expected_mad(LOCATION, sigma, VALUES) == pytest.approx(threshold, rel=1e-10)


def test_the_curve_follows_the_closed_form() -> None:
    """Every simulated deviation within 0.001 of its expected value.

    The deviation is a ninth of an L1 norm, so its standard deviation is at most
    the mean of ``sqrt(p (1 - p) / n)`` over the digits, about 0.0013.
    """
    expected = np.array(CURVE.expected_mads)
    assert float(np.max(np.abs(MADS - expected))) < 0.001
    population = np.array(CURVE.population_mads)
    assert np.all(expected >= population)
    assert CURVE.expected_mads == tuple(expected_mad(6.0, s, 50_000) for s in CURVE.sigmas)


def test_digit_shares_against_the_fourier_series() -> None:
    """Decade sums and the wrapped-normal Fourier series agree at every spread."""
    for sigma in (0.3, 0.6, 1.0, 1.7):
        assert lognormal_digit_shares(6.0, sigma) == pytest.approx(
            _fourier_shares(6.0, sigma), abs=1e-13
        )
    for sigma in (0.15, 0.8, 2.6):
        assert lognormal_digit_shares(6.0, sigma).sum() == pytest.approx(1.0, abs=1e-13)


def test_the_limits_of_the_digit_shares() -> None:
    """A wide lognormal follows Benford exactly; a narrow one puts everything on one digit."""
    assert lognormal_digit_shares(6.0, 3.0) == pytest.approx(BENFORD, abs=1e-13)
    assert population_mad(-4.0, 3.0) == pytest.approx(0.0, abs=1e-13)
    narrow = lognormal_digit_shares(log(3.5e6), 0.01)
    assert narrow[2] == pytest.approx(1.0, abs=1e-12)
    assert population_mad(log(3.5e6), 0.01) == pytest.approx(
        (1 - BENFORD[2] + BENFORD.sum() - BENFORD[2]) / 9, rel=1e-10
    )


def test_the_expected_deviation_against_multinomial_draws() -> None:
    """The folded-normal expectation against 4,000 multinomial columns at three spreads."""
    rng = np.random.default_rng(97)
    for sigma in (0.8, 1.0, 2.0):
        shares = lognormal_digit_shares(6.0, sigma)
        draws = rng.multinomial(VALUES, shares / shares.sum(), size=4000) / VALUES
        mads = np.abs(draws - BENFORD).mean(axis=1)
        standard_error = float(mads.std()) / sqrt(4000)
        assert abs(float(mads.mean()) - expected_mad(6.0, sigma, VALUES)) < 4 * standard_error
    # With no population deviation the floor is the mean of tau sqrt(2 / pi).
    tau = np.sqrt(BENFORD * (1 - BENFORD) / VALUES)
    assert expected_mad(6.0, 5.0, VALUES) == pytest.approx(
        float(np.mean(tau)) * sqrt(2 / pi), rel=1e-9
    )
    # And with unlimited values it is the population deviation.
    assert expected_mad(6.0, 0.8, 10**15) == pytest.approx(population_mad(6.0, 0.8), rel=1e-6)


def test_first_digits_by_hand() -> None:
    """Signs are ignored, zeros dropped, and powers of ten lead with a one."""
    assert first_digits([0.0123, 4.5, -987.0, 1e6, 0.0, 29.99]).tolist() == [1, 4, 9, 1, 2]
    assert digit_shares([1.0, 10.0, 200.0, 0.09]).tolist() == [0.5, 0.25] + [0.0] * 6 + [0.25]
    assert percentile_span(1.0) == pytest.approx(np.exp(3.92))
    assert percentile_span(1.0, z=1.0) == pytest.approx(np.exp(2.0))


def test_the_rows_and_curve_are_consistent() -> None:
    """A row is the closed form at one spread; a curve is deterministic under a seed."""
    row = spread_row(0.8)
    assert row.span == percentile_span(0.8)
    assert row.population_mad == population_mad(LOCATION, 0.8)
    small = spread_curve(5, values=500, sigmas=(0.5, 1.5))
    assert small == spread_curve(5, values=500, sigmas=(0.5, 1.5))
    assert small != spread_curve(6, values=500, sigmas=(0.5, 1.5))
    assert conformity_span(0.02) < conformity_span(0.006)


def test_invalid_inputs_are_rejected() -> None:
    """Empty columns, zeros alone, bad spreads and uncrossed thresholds are refused."""
    with pytest.raises(ValueError):
        first_digits([])
    with pytest.raises(ValueError):
        first_digits([0.0, 0.0])
    with pytest.raises(ValueError):
        first_digits([[1.0, 2.0]])
    with pytest.raises(ValueError):
        first_digits([1.0, np.inf])
    with pytest.raises(ValueError):
        lognormal_digit_shares(6.0, 0.0)
    with pytest.raises(ValueError):
        expected_mad(6.0, 1.0, 0)
    with pytest.raises(ValueError):
        conformity_span(0.5)
    with pytest.raises(ValueError):
        conformity_span(0.006, lower=2.0, upper=1.0)
    with pytest.raises(ValueError):
        spread_curve(sigmas=())
    with pytest.raises(TypeError):
        spread_curve(values=True)
    with pytest.raises(TypeError):
        percentile_span(True)
