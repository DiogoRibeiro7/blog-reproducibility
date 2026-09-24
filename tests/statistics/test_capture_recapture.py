"""Check the capture-recapture model against the website, closed forms and the article.

The figure reseeds a generator at 17 for every spread, and its three curves are
reproduced draw for draw against a transcription of the website loop. The
quadrature behind the closed forms is checked against the lognormal's
piecewise moments in closed form and against a million simulated defects, and
the figure's medians and means sit within four standard errors of the limits.

The article's three-pass table runs the figure's loop reseeded at 13, with
1,000 replications at spreads 0, 0.4, 0.8 and 1.2, and its seen, two-pass and
Chao columns are reproduced exactly: 418, 393, 338 and 276 seen; 497, 434, 348
and 277 from two passes; 501, 470, 419 and 359 from Chao's bound. Its other
tables use generators seeded at 3, 5, 7 and 11, and are not reproduced. Its
first table is arithmetic on its own counts and is pinned: 397 seen, 103 missed,
Lincoln-Petersen 485, Chapman 484 with a standard error of 17. Every other
printed number lies within its Monte Carlo error of the closed forms: 390, 290
and 200 seen when every defect is equally easy, the estimates of 439, 359 and
288 (12, 28 and 42 percent low) and correlations of 0.149 to 0.441 as the spread
grows, and the 110 and 210 defects left. The spread of the two-pass estimate is
the delta method's 20, 42 and 77 to within six percent (printed 20, 43, 81).

Chao's bound for three passes carries a factor ``(t - 1) / t = 2/3`` on the
correction ``f_1^2 / (2 f_2)``. The website first used the many-pass form
without it, which is 543 at zero spread, 8.6 percent high; with the factor the
limit is 501, and the figure's claims hold. The two-pass estimate falls from 499
to 245, below the truth at every spread, and ends level with the number actually
found. Chao's bound starts at the truth (500.5 in the figure), is below it from
a spread of 0.1 on, and is closer to it than the two-pass estimate at every
spread, with at most 0.7 of its error.
"""

from math import erf, exp, inf, isnan, log, sqrt

import numpy as np
import pytest
from numpy.typing import NDArray

from blog_reproducibility.statistics.capture_recapture import (
    ARTICLE_SPREADS,
    DETECTION,
    REPLICATIONS,
    TRUE_POPULATION,
    TWO_PASS_DETECTION,
    capture_frequencies,
    chao_estimate,
    chapman,
    chapman_standard_error,
    draw_passes,
    example_payload,
    expected_capture,
    heterogeneity_curve,
    homogeneous_row,
    lincoln_petersen,
    overlap,
    two_pass_standard_deviation,
)

SUMMARY = example_payload()
CURVE = SUMMARY.curve
EXPECTED = SUMMARY.expected
TWO = np.array(CURVE.two_pass)
CHAO = np.array(CURVE.chao)
SEEN = np.array(CURVE.seen)
SPREADS = np.array(CURVE.spreads)


def _phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def _website_curve() -> tuple[list[float], list[float], list[float]]:
    """Transcribe the website generator's loop, with Chao's factor for three passes."""
    true_n = 500

    def passes(
        ps: tuple[float, ...], spread: float, rng: np.random.Generator
    ) -> list[NDArray[np.bool_]]:
        ease = rng.lognormal(-(spread**2) / 2, spread, true_n) if spread else np.ones(true_n)
        return [rng.random(true_n) < np.clip(p * ease, 0, 1) for p in ps]

    def chapman_(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
        n1, n2, m = a.sum(), b.sum(), (a & b).sum()
        return float((n1 + 1) * (n2 + 1) / (m + 1) - 1)

    def chao(a: NDArray[np.bool_], b: NDArray[np.bool_], c: NDArray[np.bool_]) -> float:
        times = a.astype(int) + b.astype(int) + c.astype(int)
        f1, f2 = (times == 1).sum(), (times == 2).sum()
        # The website's loop left out the (t - 1) / t = 2/3 on the correction.
        return float((times > 0).sum() + (2 / 3 * f1**2 / (2 * f2) if f2 else np.nan))

    spreads = np.linspace(0, 1.4, 15)
    two, three, union = [], [], []
    for s in spreads:
        rng = np.random.default_rng(17)
        t2, t3, u = [], [], []
        for _ in range(200):
            a, b, c = passes((0.5, 0.45, 0.4), s, rng)
            t2.append(chapman_(a, b))
            t3.append(chao(a, b, c))
            u.append((a | b | c).sum())
        two.append(float(np.median(t2)))
        three.append(float(np.median(t3)))
        union.append(float(np.mean(u)))
    return two, three, union


def _piecewise_moments(first: float, second: float, spread: float) -> tuple[float, float, float]:
    """``E[q_1], E[q_2], E[q_1 q_2]`` from the lognormal's partial moments, ``p_1 >= p_2``."""
    s2 = spread * spread

    def partial_first(c: float) -> float:
        return _phi((log(c) - s2 / 2) / spread)

    def above(c: float) -> float:
        return _phi(-(log(c) + s2 / 2) / spread)

    def mean_q(p: float) -> float:
        return p * partial_first(1 / p) + above(1 / p)

    c1, c2 = 1 / first, 1 / second
    joint = first * second * exp(s2) * _phi((log(c1) - 1.5 * s2) / spread)
    joint += second * (partial_first(c2) - partial_first(c1)) + above(c2)
    return mean_q(first), mean_q(second), joint


def _median_error(sd: float, runs: int) -> float:
    """Large-sample standard error of a median of ``runs`` roughly normal estimates."""
    return 1.2533 * sd / sqrt(runs)


def test_the_figure_matches_the_website_loop() -> None:
    """Same reseeding, same draws and the same estimators: identical curves."""
    two, three, union = _website_curve()
    assert list(CURVE.two_pass) == two
    assert list(CURVE.chao) == three
    assert list(CURVE.seen) == union
    assert CURVE.spreads == tuple(float(s) for s in np.linspace(0, 1.4, 15))
    # Something was always seen twice, so Chao's estimate is finite in every run.
    assert CURVE.smallest_twice > 0
    assert np.all(np.isfinite(CHAO))


def test_the_curves_follow_the_closed_forms() -> None:
    """Medians and means within four standard errors of the large-population limits."""
    for i, expected in enumerate(EXPECTED):
        assert expected.spread == CURVE.spreads[i]
        assert expected.chao_limit is not None
        two_error = _median_error(CURVE.two_pass_sd[i], REPLICATIONS)
        chao_error = _median_error(CURVE.chao_sd[i], REPLICATIONS)
        assert abs(TWO[i] - expected.two_pass_limit) < 4 * two_error
        assert abs(CHAO[i] - expected.chao_limit) < 4 * chao_error
        assert abs(SEEN[i] - expected.seen) < 4 * expected.missed_sd / sqrt(REPLICATIONS)


def test_uneven_difficulty_pulls_the_two_pass_estimate_down() -> None:
    """Title and alt text: below the truth everywhere, falling to the number actually found."""
    assert np.all(TWO < TRUE_POPULATION)
    assert np.all(np.diff(TWO) < 0)
    assert (round(TWO[0]), round(TWO[-1])) == (499, 245)
    limits = [e.two_pass_limit for e in EXPECTED]
    assert limits[0] == pytest.approx(TRUE_POPULATION, rel=1e-12)
    assert np.all(np.diff(limits) < 0)
    # "Ending level with what was actually found."
    assert abs(TWO[-1] - SEEN[-1]) < 1
    assert abs(EXPECTED[-1].two_pass_limit - EXPECTED[-1].seen) < 1


def test_chaos_bound_stays_much_closer_and_errs_low() -> None:
    """Alt text: at the truth with no spread, then low, and always much closer than two passes."""
    chao_error = np.abs(CHAO - TRUE_POPULATION)
    two_error = np.abs(TWO - TRUE_POPULATION)
    assert np.all(chao_error < 0.7 * two_error)
    assert chao_error[0] < 1
    assert np.all(CHAO[SPREADS > 0.05] < TRUE_POPULATION)
    # The limits agree: closer and lower as soon as the ease varies.
    chao_limits = np.array([e.chao_limit for e in EXPECTED], dtype=float)
    two_limits = np.array([e.two_pass_limit for e in EXPECTED])
    varied = SPREADS > 0.05
    assert np.all(np.abs(chao_limits - 500)[varied] < np.abs(two_limits - 500)[varied])
    assert np.all(chao_limits[varied] < 500)


def test_chaos_factor_for_three_passes() -> None:
    """With ``(t - 1) / t`` the limit at zero spread is 501; without it, 543, 8.6% high."""
    zero = EXPECTED[0]
    f0, f1, f2, _ = zero.frequencies
    assert (f0, f1, f2) == pytest.approx((0.165, 0.41, 0.335), rel=1e-12)
    assert zero.chao_limit is not None
    assert zero.chao_limit == pytest.approx(500 * (1 - f0 + 2 / 3 * f1**2 / (2 * f2)), rel=1e-12)
    assert round(zero.chao_limit) == 501
    many_pass = 500 * (1 - f0 + f1**2 / (2 * f2))
    assert round(many_pass) == 543
    assert round(many_pass / TRUE_POPULATION - 1, 3) == 0.086


def test_chaos_bound_holds_when_the_passes_are_alike() -> None:
    """Equal detection on every pass: exact with no spread, below the truth with any."""
    alike = (0.45, 0.45, 0.45)
    assert expected_capture(0.0, alike).chao_limit == pytest.approx(TRUE_POPULATION, rel=1e-12)
    for spread in (0.2, 0.6, 1.0, 1.4):
        limit = expected_capture(spread, alike).chao_limit
        assert limit is not None and limit < TRUE_POPULATION


def test_the_quadrature_against_the_lognormal_moments() -> None:
    """Two-pass limits and correlations from the piecewise closed form of the moments."""
    for detection in (DETECTION, TWO_PASS_DETECTION):
        for spread in (0.1, 0.4, 0.8, 1.4):
            e1, e2, e12 = _piecewise_moments(detection[0], detection[1], spread)
            expected = expected_capture(spread, detection)
            assert expected.two_pass_limit == pytest.approx(500 * e1 * e2 / e12, rel=1e-9)
            correlation = (e12 - e1 * e2) / sqrt(e1 * (1 - e1) * e2 * (1 - e2))
            assert expected.pass_correlation == pytest.approx(correlation, rel=1e-8)
            assert sum(expected.frequencies) == pytest.approx(1.0, abs=1e-12)
            # The mean number of captures is the sum of the mean detection probabilities.
            captures = sum(k * f for k, f in enumerate(expected.frequencies))
            assert captures == pytest.approx(
                sum(_piecewise_moments(p, p, spread)[0] for p in detection), rel=1e-9
            )


def test_the_frequencies_against_a_million_defects() -> None:
    """Shares seen zero to three times at a spread of 0.8, within four standard errors."""
    n = 1_000_000
    passes = draw_passes(np.random.default_rng(41), DETECTION, 0.8, n)
    observed = np.array(capture_frequencies(passes)) / n
    expected = np.array(expected_capture(0.8).frequencies)
    assert np.all(np.abs(observed - expected) < 4 * np.sqrt(expected * (1 - expected) / n))


def test_equally_easy_defects_in_closed_form() -> None:
    """With no spread the counts are binomial and both limits are exact."""
    zero = expected_capture(0.0)
    q = np.array(DETECTION)
    missed = float(np.prod(1 - q))
    assert zero.frequencies[0] == pytest.approx(missed, rel=1e-14)
    assert zero.frequencies[3] == pytest.approx(float(np.prod(q)), rel=1e-14)
    assert zero.seen == pytest.approx(500 * (1 - missed), rel=1e-14)
    assert zero.pass_correlation == pytest.approx(0.0, abs=1e-15)
    two = expected_capture(0.0, TWO_PASS_DETECTION)
    assert two.chao_limit is None
    assert two.seen == pytest.approx(390.0, rel=1e-14)


def test_the_article_first_table() -> None:
    """313 and 237 found, 153 by both: 397 seen, 103 missed, 485, and 484 +- 17."""
    first, second, both = 313, 237, 153
    seen = first + second - both
    assert (seen, TRUE_POPULATION - seen) == (397, 103)
    assert round(lincoln_petersen(first, second, both)) == 485
    assert round(chapman(first, second, both)) == 484
    assert round(chapman_standard_error(first, second, both)) == 17
    assert round(chapman(first, second, both) - seen) == 87


def test_the_article_homogeneous_table() -> None:
    """390, 290 and 200 seen; a spread of 20, 43 and 81 against the delta method's."""
    rows = SUMMARY.homogeneous
    assert [round(row.seen) for row in rows] == [390, 290, 200]
    printed = ((499, 20), (501, 43), (501, 81))
    for row, (mean, sd) in zip(rows, printed, strict=True):
        assert sd == pytest.approx(row.estimate_sd, rel=0.06)
        # Chapman's mean over 2,000 runs, within three standard errors of the truth.
        assert abs(mean - TRUE_POPULATION) < 3 * sd / sqrt(2000) + 0.5
    assert [round(row.estimate_sd) for row in rows] == [20, 42, 77]


def test_the_article_heterogeneity_table() -> None:
    """Two passes at 60 and 45 percent: 501, 439, 359, 288; seen 390, 367, 313, 254."""
    rows = SUMMARY.two_pass
    assert tuple(row.spread for row in rows) == ARTICLE_SPREADS
    printed = (
        (501, 0.1, 390, -0.001),
        (439, -12.2, 367, 0.149),
        (359, -28.1, 313, 0.328),
        (288, -42.4, 254, 0.441),
    )
    for row, (estimate, bias, seen, correlation) in zip(rows, printed, strict=True):
        # The mean of 2,000 estimates, each spread by at most 24, has a standard error under 0.55.
        assert abs(estimate - row.two_pass_limit) <= 1.5
        assert abs(bias - 100 * (row.two_pass_limit / TRUE_POPULATION - 1)) < 0.3
        assert round(row.seen) == seen
        assert abs(correlation - row.pass_correlation) < 0.003


def test_the_article_remaining_table() -> None:
    """110.2 +- 9.3 and 209.9 +- 11.0 actually left: binomial, 110 +- 9.26 and 210 +- 11.04."""
    first, second = SUMMARY.homogeneous[:2]
    for row, (mean, sd, estimated) in zip(
        (first, second), ((110.2, 9.3, 109.5), (209.9, 11.0, 209.8)), strict=True
    ):
        assert abs(mean - row.missed) < 3 * row.missed_sd / sqrt(2000)
        assert abs(sd - row.missed_sd) < 3 * row.missed_sd / sqrt(2 * 2000) + 0.05
        assert abs(estimated - row.missed) < 3 * row.estimate_sd / sqrt(2000)
    assert (round(first.missed_sd, 1), round(second.missed_sd, 1)) == (9.3, 11.0)


def test_the_article_three_pass_table() -> None:
    """Seen 418, 393, 338, 276; two passes 497, 434, 348, 277; Chao 501, 470, 419, 359."""
    printed = ((418, 497, 501), (393, 434, 470), (338, 348, 419), (276, 277, 359))
    # The article's loop: reseeded at 13 for each spread, 1,000 runs of three passes.
    table = heterogeneity_curve(13, replications=1000, spreads=ARTICLE_SPREADS)
    assert [round(x) for x in table.seen] == [row[0] for row in printed]
    assert [round(x) for x in table.two_pass] == [row[1] for row in printed]
    assert [round(x) for x in table.chao] == [row[2] for row in printed]
    # And the closed forms, within the Monte Carlo error of a median of 1,000 runs.
    for row, (seen, two, chao) in zip(SUMMARY.three_pass, printed, strict=True):
        i = int(np.argmin(np.abs(SPREADS - row.spread)))
        assert round(row.seen) == seen
        assert row.chao_limit is not None
        assert abs(two - row.two_pass_limit) < 4 * _median_error(CURVE.two_pass_sd[i], 1000) + 0.5
        assert abs(chao - row.chao_limit) < 4 * _median_error(CURVE.chao_sd[i], 1000) + 0.5
        # Chao's bound is the closest of the two, and low once the ease varies.
        assert abs(chao - TRUE_POPULATION) <= abs(two - TRUE_POPULATION)
        assert chao < TRUE_POPULATION or row.spread == 0.0
    assert round(SUMMARY.three_pass[0].chao_limit or 0) == 501


def test_estimators_by_hand() -> None:
    """Small counts, empty overlaps and nothing seen twice."""
    assert lincoln_petersen(10, 8, 4) == 20.0
    assert lincoln_petersen(10, 8, 0) == inf
    assert chapman(10, 8, 0) == 98.0
    assert chapman(10, 8, 4) == pytest.approx(11 * 9 / 5 - 1)
    assert chapman_standard_error(10, 8, 8) == 0.0
    assert chao_estimate(10, 4, 2, occasions=2) == 12.0
    assert chao_estimate(10, 4, 2, occasions=3) == pytest.approx(10 + 2 / 3 * 4)
    assert isnan(chao_estimate(10, 4, 0, occasions=3))
    a = np.array([True, True, False, False, True])
    b = np.array([True, False, False, True, True])
    c = np.array([False, False, False, True, True])
    assert overlap(a, b) == (3, 3, 2)
    assert capture_frequencies((a, b, c)) == (1, 1, 2, 1)
    assert two_pass_standard_deviation(0.6, 0.45) == pytest.approx(sqrt(500 * 0.4 * 0.55 / 0.27))
    row = homogeneous_row((0.5, 0.5), 100)
    assert (row.seen, row.missed) == (75.0, 25.0)


def test_the_curve_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a small curve; another seed does not."""
    first = heterogeneity_curve(3, replications=5, spreads=(0.0, 0.7))
    assert first == heterogeneity_curve(3, replications=5, spreads=(0.0, 0.7))
    assert first != heterogeneity_curve(4, replications=5, spreads=(0.0, 0.7))


def test_invalid_inputs_are_rejected() -> None:
    """Impossible counts, bad probabilities, too few passes and non-numbers are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        chapman(10, 8, 9)
    with pytest.raises(ValueError):
        chao_estimate(5, 4, 2, occasions=3)
    with pytest.raises(ValueError):
        chao_estimate(10, 4, 2, occasions=1)
    with pytest.raises(ValueError):
        draw_passes(rng, (0.5, 1.5))
    with pytest.raises(ValueError):
        draw_passes(rng, (0.5,), spread=-0.1)
    with pytest.raises(ValueError):
        expected_capture(0.5, (0.5,))
    with pytest.raises(ValueError):
        heterogeneity_curve(detection=(0.5, 0.4))
    with pytest.raises(ValueError):
        heterogeneity_curve(spreads=())
    with pytest.raises(ValueError):
        overlap(np.ones(3, dtype=bool), np.ones(4, dtype=bool))
    with pytest.raises(ValueError):
        capture_frequencies(())
    with pytest.raises(ValueError):
        two_pass_standard_deviation(1.0, 0.5)
    with pytest.raises(TypeError):
        chapman(10.0, 8, 4)  # type: ignore[arg-type]
