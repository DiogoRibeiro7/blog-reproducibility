"""Check the digit heaping model against the website, closed forms and the article.

The figure is one draw of 400,000 records from a generator seeded at 23, and it
is reproduced draw for draw against a transcription of the website generator.
Both curves agree with the closed forms at every threshold to within four
standard errors, and the recorded curve equals the true one exactly at every
threshold halfway between two multiples of five.

The article's tables come from generators seeded at 2, 3 and 5 over 50,000
records, so they are not reproduced; each printed share lies within three
standard errors of its closed form. The gap of 4.7 points between "over 30" and
"30 or more" is the closed form's 4.74, and the mean moves by less than 0.002
minutes at any level of rounding, as the article says.

Some printed numbers need a caveat, and the tests pin what is true:

* the 90th percentile "drifts ... from 37 down to 35": in the population the
  recorded 90th percentile is 36 with no rounding to five, 36 at 20 percent, and
  35 at 50 and 80 percent. The printed 37 and 36 at 0 and 50 percent are sample
  outcomes on a knife edge, where the recorded distribution function sits within
  0.0007 of 0.9 at the population value, about half a standard error;
* the median at 50 percent is also on a knife edge (0.4997 of the records at or
  below 18), so 19 in one table and 18 in the fitting table are both sample
  outcomes;
* Whipple's index "reads 110 rather than exactly 100 because one value in five
  is a multiple of five by arithmetic alone, and the lognormal shape is not
  perfectly flat": its closed form is 108.8, of which 7.8 comes from the band
  itself, since the integers from 10 to 60 hold 11 multiples of five in 51, and
  only 1.0 from the shape.

The figure's claims hold: the recorded curve steps down across the true one at
every multiple of five, and within every five minutes of the grid the gap is
widest exactly on the multiple of five.
"""

from math import sqrt

import numpy as np
import pytest

from blog_reproducibility.statistics.digit_heaping import (
    ARTICLE_RECORDS,
    HEAPED_SHARE,
    QUANTILE_LEVELS,
    RECORDS,
    ROUND_THRESHOLDS,
    example_payload,
    expected_whipple_index,
    multiple_of_five_share,
    record,
    recorded_cdf,
    recorded_mean,
    recorded_quantile,
    recorded_share_above,
    share_above,
    threshold_curve,
    threshold_row,
    true_mean,
    true_quantile,
    true_share_above,
    whipple_index,
)

SUMMARY = example_payload()
CURVE = SUMMARY.curve
GRID = np.array(CURVE.thresholds)
GAP = np.array(CURVE.recorded_shares) - np.array(CURVE.true_shares)
DATA = record(np.random.default_rng(23), RECORDS, HEAPED_SHARE)


def _binomial(p: float, n: int = ARTICLE_RECORDS) -> float:
    return sqrt(p * (1 - p) / n)


def _website_curves() -> tuple[list[float], list[float]]:
    """Transcribe the website generator."""
    rng = np.random.default_rng(23)
    n = 400_000
    true = rng.lognormal(np.log(18), 0.55, n)
    heaped = rng.random(n) < 0.5
    obs = np.where(heaped, np.round(true / 5) * 5, np.round(true))
    grid = np.arange(15, 46, 0.5)
    t = [float(np.mean(true > g)) for g in grid]
    o = [float(np.mean(obs > g)) for g in grid]
    return t, o


def test_the_figure_matches_the_website_generator() -> None:
    """Same draws, same rounding, same grid: identical curves."""
    t, o = _website_curves()
    assert list(CURVE.true_shares) == t
    assert list(CURVE.recorded_shares) == o
    assert CURVE.thresholds == tuple(float(g) for g in np.arange(15, 46, 0.5))
    assert threshold_curve() == CURVE
    assert share_above(DATA.true, GRID) == CURVE.true_shares


def test_the_curves_follow_the_closed_forms() -> None:
    """Every simulated share within four binomial standard errors of its closed form."""
    for simulated, exact in (
        (CURVE.true_shares, CURVE.expected_true),
        (CURVE.recorded_shares, CURVE.expected_recorded),
    ):
        for value, p in zip(simulated, exact, strict=True):
            assert abs(value - p) < 4 * _binomial(p, RECORDS)
    assert float(np.mean(DATA.heaped)) == pytest.approx(0.5, abs=4 * _binomial(0.5, RECORDS))


def test_the_curves_agree_halfway_between_round_numbers() -> None:
    """At 17.5, 22.5, ..., 42.5 both rules put their boundary on the threshold."""
    halfway = [i for i, g in enumerate(GRID) if g % 5 == 2.5]
    assert [float(GRID[i]) for i in halfway] == [17.5, 22.5, 27.5, 32.5, 37.5, 42.5]
    for i in halfway:
        assert CURVE.recorded_shares[i] == CURVE.true_shares[i]
        assert CURVE.expected_recorded[i] == pytest.approx(CURVE.expected_true[i], abs=1e-15)


def test_the_gap_is_widest_on_the_round_numbers() -> None:
    """Title and alt text: the recorded curve steps across the truth at each multiple of five."""
    grid = GRID
    for m in ROUND_THRESHOLDS:
        before = int(np.flatnonzero(grid == m - 0.5)[0])
        assert GAP[before] > 0 > GAP[before + 1]
    for m in (15, *ROUND_THRESHOLDS):
        window = (grid >= m) & (grid < m + 5)
        widest = grid[window][np.argmax(np.abs(GAP[window]))]
        assert widest == m
    assert grid[np.argmax(np.abs(GAP))] == 15
    # In closed form too, at every multiple of five from 15 to 45.
    expected = np.array(CURVE.expected_recorded) - np.array(CURVE.expected_true)
    for m in (15, *ROUND_THRESHOLDS):
        window = (grid >= m) & (grid < m + 5)
        assert grid[window][np.argmax(np.abs(expected[window]))] == m


def test_the_article_threshold_table() -> None:
    """Over 30: 17.79% true, 15.63% strictly, 20.31% at least, 4.67% exactly on it."""
    printed = (
        (27.70, 24.34, 31.62, 7.28),
        (17.79, 15.63, 20.31, 4.67),
        (11.42, 10.16, 13.05, 2.89),
    )
    for row, values in zip(SUMMARY.on_round, printed, strict=True):
        exact = (row.truly_over, row.strictly_over, row.at_least, row.exactly_on)
        for value, p in zip(values, exact, strict=True):
            assert abs(value / 100 - p) < 3 * _binomial(p)
    thirty = SUMMARY.on_round[1]
    assert thirty.threshold == 30
    # "The gap between the two phrasings is 4.7 points", "a quarter of its value".
    assert round(100 * thirty.exactly_on, 1) == 4.7
    assert (thirty.at_least - thirty.strictly_over) / thirty.truly_over == pytest.approx(
        0.25, abs=0.03
    )


def test_the_article_off_round_table() -> None:
    """Off round numbers the error is at most a point, and a few hundredths at 28 and 33."""
    printed = (
        (23.14, 22.07, -1.07),
        (21.14, 21.10, -0.03),
        (14.94, 14.26, -0.68),
        (13.69, 13.63, -0.06),
    )
    for row, (truly, recorded, error) in zip(SUMMARY.off_round, printed, strict=True):
        assert abs(truly / 100 - row.truly_over) < 3 * _binomial(row.truly_over)
        assert abs(recorded / 100 - row.strictly_over) < 3 * _binomial(row.strictly_over)
        # A record counts differently only between the threshold and its rounding boundary.
        t = row.threshold
        differ = 0.5 * abs(true_share_above(t) - true_share_above(np.floor(t) + 0.5))
        differ += 0.5 * abs(true_share_above(t) - true_share_above(5 * np.floor(t / 5) + 2.5))
        closed_error = row.strictly_over - row.truly_over
        assert abs(error / 100 - closed_error) < 3 * sqrt(differ / ARTICLE_RECORDS)
    errors = [abs(row.strictly_over - row.truly_over) for row in SUMMARY.off_round]
    assert round(100 * max(errors), 2) == 1.00
    assert all(abs(row.strictly_over - row.truly_over) < 0.001 for row in SUMMARY.off_round[1::2])


def test_the_article_whipple_table() -> None:
    """20.3%, 36.3%, 60.1%, 84.1%, 100% on a five; Whipple 110, 193, 312, 426, 500."""
    printed = ((20.3, 110), (36.3, 193), (60.1, 312), (84.1, 426), (100.0, 500))
    in_band = recorded_cdf(60.0) - recorded_cdf(9.0)
    for row, (share, index) in zip(SUMMARY.whipple, printed, strict=True):
        assert abs(share / 100 - row.on_multiple_of_five) <= max(
            3 * _binomial(row.on_multiple_of_five), 0.0005
        )
        pi_band = row.whipple / 500
        se = 500 * sqrt(pi_band * (1 - pi_band) / (in_band * ARTICLE_RECORDS))
        assert abs(index - row.whipple) <= max(3 * se, 0.5)
    assert [round(row.whipple, 1) for row in SUMMARY.whipple] == [108.8, 192.0, 311.9, 426.5, 500.0]
    # A flat spread over the band gives 107.8: 11 multiples of five among 51 integers.
    assert whipple_index(np.arange(10, 61)) == pytest.approx(500 * 11 / 51)
    assert SUMMARY.whipple[0].on_multiple_of_five == pytest.approx(0.2, abs=1e-5)


def test_the_whipple_index_of_the_figure_records() -> None:
    """The figure's own 400,000 records against the closed forms at half heaped."""
    share = float(np.mean(np.isclose(DATA.recorded % 5, 0)))
    p = multiple_of_five_share(0.5)
    assert abs(share - p) < 4 * _binomial(p, RECORDS)
    index = whipple_index(DATA.recorded)
    expected = expected_whipple_index(0.5)
    assert abs(index - expected) < 4 * 500 * _binomial(expected / 500, int(0.85 * RECORDS))


def test_the_article_quantile_table() -> None:
    """Medians 18, 18, 19, 20 and p99 65 as printed; the 90th percentile on a knife edge."""
    rows = SUMMARY.quantiles
    assert [row.quantiles[0] for row in rows] == [18, 18, 19, 20]
    assert [row.quantiles[2] for row in rows] == [65, 65, 65, 65]
    assert [row.quantiles[1] for row in rows] == [36, 36, 35, 35]
    printed_p90 = (37, 36, 36, 35)
    knife_edges = []
    for row, value in zip(rows, printed_p90, strict=True):
        if value != row.quantiles[1]:
            # One above the population quantile, where the distribution function barely passes 0.9.
            assert value == row.quantiles[1] + 1
            assert row.at_quantiles[1] - 0.9 < _binomial(0.9)
            knife_edges.append(row.heaped_share)
    assert knife_edges == [0.0, 0.5]
    assert [round(rows[i].at_quantiles[1], 4) for i in (0, 2)] == [0.9007, 0.9003]
    # The median at half heaped: 0.4997 at or below 18, a knife edge too.
    assert rows[2].below_quantiles[0] == pytest.approx(0.49972, abs=1e-5)
    # "The median moves from 18 to 20 ..., an 11 percent error."
    assert round(100 * (rows[3].quantiles[0] / 18 - 1)) == 11


def test_the_article_means_and_true_quantiles() -> None:
    """The mean survives rounding; the true median, p90 and p99 against the lognormal's."""
    assert SUMMARY.true_mean == pytest.approx(18 * np.exp(0.55**2 / 2), rel=1e-15)
    for share in (0.0, 0.2, 0.5, 0.8, 1.0):
        assert abs(recorded_mean(share) - SUMMARY.true_mean) < 0.002
    sd = SUMMARY.true_mean * sqrt(np.exp(0.55**2) - 1)
    assert abs(20.96 - SUMMARY.true_mean) < 3 * sd / sqrt(ARTICLE_RECORDS)
    printed = (18.06, 36.54, 64.63)
    for value, exact, level in zip(printed, SUMMARY.true_quantiles, QUANTILE_LEVELS, strict=True):
        density = float(np.exp(-0.5 * (np.log(exact / 18) / 0.55) ** 2)) / (
            exact * 0.55 * sqrt(2 * np.pi)
        )
        assert abs(value - exact) < 3 * _binomial(level) / density
    assert true_quantile(0.5) == pytest.approx(18.0, rel=1e-15)


def test_the_recorded_distribution_is_consistent() -> None:
    """Masses sum to one, reproduce the mean, and the tail shares match the distribution."""
    # Beyond 1,600 minutes the true distribution holds less than 1e-15.
    values = np.arange(0, 1600)
    cdf = np.array([recorded_cdf(float(v), 0.3) for v in values])
    mass = np.diff(np.concatenate([[0.0], cdf]))
    assert mass.sum() == pytest.approx(1.0, abs=1e-12)
    assert float(values @ mass) == pytest.approx(recorded_mean(0.3), rel=1e-12)
    for share in (0.0, 0.3, 1.0):
        for t in (12.0, 25.0, 27.5, 30.0, 33.3):
            assert recorded_share_above(t, share) == pytest.approx(
                1 - recorded_cdf(t, share), abs=1e-14
            )
            below = np.ceil(t) - 1
            assert recorded_share_above(t, share, inclusive=True) == pytest.approx(
                1 - recorded_cdf(below, share), abs=1e-14
            )
    # With no heaping every value is on a five one time in five; with full heaping always.
    assert multiple_of_five_share(0.0) == pytest.approx(0.2, abs=1e-5)
    assert multiple_of_five_share(1.0) == 1.0
    assert recorded_quantile(0.5, 1.0) == 20
    assert expected_whipple_index(1.0) == pytest.approx(500.0)
    row = threshold_row(30, 0.0)
    assert row.exactly_on == pytest.approx(true_share_above(29.5) - true_share_above(30.5))


def test_recording_by_hand() -> None:
    """Rounding rules, Whipple's index and threshold shares on small inputs."""
    rng = np.random.default_rng(1)
    none = record(rng, 1000, 0.0)
    assert np.array_equal(none.recorded, np.round(none.true))
    every = record(rng, 1000, 1.0)
    assert np.all(every.recorded % 5 == 0)
    assert np.array_equal(every.recorded, 5 * np.round(every.true / 5))
    assert whipple_index([10, 15, 17, 60, 61, 3]) == 375.0
    assert share_above([1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 4.0]) == (1.0, 0.5, 0.0)
    assert true_share_above(-1.0) == 1.0
    assert true_share_above(18.0) == pytest.approx(0.5)
    assert true_mean() == SUMMARY.true_mean


def test_the_curve_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a small curve; another seed does not."""
    first = threshold_curve(4, records=2000)
    assert first == threshold_curve(4, records=2000)
    assert first != threshold_curve(5, records=2000)


def test_invalid_inputs_are_rejected() -> None:
    """Shares outside [0, 1], empty data, bad bands and non-numbers are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        record(rng, 10, 1.5)
    with pytest.raises(ValueError):
        record(rng, 0)
    with pytest.raises(ValueError):
        share_above([], [1.0])
    with pytest.raises(ValueError):
        whipple_index([1.0, 2.0])
    with pytest.raises(ValueError):
        whipple_index([10.0], band=(60, 10))
    with pytest.raises(ValueError):
        expected_whipple_index(0.5, band=(60, 10))
    with pytest.raises(ValueError):
        recorded_quantile(1.0, 0.5)
    with pytest.raises(ValueError):
        recorded_share_above(float("nan"))
    with pytest.raises(TypeError):
        recorded_cdf(True)
    with pytest.raises(TypeError):
        threshold_curve(records=True)
