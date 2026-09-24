"""Check the negative control model against the website loop, closed forms and the article.

The figure's simulation (seed 67, 12 populations of 30,000 users at each of 17
exposures) is transcribed from the website generator and matched draw for draw,
and its claims hold: the control's signal stays within 0.008 of its expected
0.498 at every exposure, the bias follows ``c_y g``, and the two cross at the
control's own exposure of 0.6, the control overstating the bias to the left and
understating it to the right.

The closed forms are checked independently: with ``a = 1`` and an offset of one
half, ``phi(u) expit(u - 1/2)`` is symmetric about one half, so adopters'
mean engagement is exactly 0.5 and the gap is ``1 / (2 (1 - p))``; Stein's lemma
gives the same first moment from a different integral; and a large simulation
agrees with the moments.

The article's tables use other designs and generators and are not pinned. They
agree with the closed forms: adoption 39.8 percent, standard errors 0.013 and
0.012, the correction table to every printed digit (0.964, 0.466, 0.715, 0.217),
and the rest within three standard errors of their simulations. Its sentence
"the ratio is exactly the ratio of the two confounding strengths" holds in
expectation; the printed ratios of 1.34 and 2.01, against 4/3 and 2, are that
table's sampling noise.
"""

from math import sqrt

import numpy as np
import pytest
from scipy import integrate, special, stats

from blog_reproducibility.statistics.negative_controls import (
    CONTROL_EXPOSURE,
    RUNS_PER_STRENGTH,
    STRENGTHS,
    USERS,
    TrackingCurve,
    aa_flag_rate,
    article_numbers,
    bias_and_signal,
    detection_rate,
    difference_standard_error,
    draw_population,
    engagement_selection,
    example_payload,
    naive_difference,
    tracking_curve,
    tracking_point,
)

SUMMARY = example_payload()
CURVE = SUMMARY.curve
SELECTION = SUMMARY.selection
ARTICLE = SUMMARY.article

# The figure's points, bias and signal, at exposures 0, 0.1, ..., 1.6.
FIGURE_BIASES = (
    0.006990,
    0.087780,
    0.167050,
    0.250422,
    0.334350,
    0.414465,
    0.496852,
    0.578664,
    0.666479,
    0.744676,
    0.831947,
    0.900280,
    0.992933,
    1.086574,
    1.162091,
    1.245566,
    1.324374,
)
FIGURE_SIGNALS = (
    0.498034,
    0.499844,
    0.502250,
    0.503464,
    0.498624,
    0.492500,
    0.493817,
    0.500090,
    0.501398,
    0.498339,
    0.504844,
    0.493990,
    0.505945,
    0.491412,
    0.499620,
    0.499676,
    0.502801,
)


def _website_curve() -> tuple[list[float], list[float]]:
    """Transcribe the website generator's loop."""
    rng = np.random.default_rng(67)
    n, effect, conf_n = 30000, 0.30, 0.60

    def run(conf_y: float) -> tuple[float, float]:
        u = rng.normal(0, 1, n)
        adopt = rng.random(n) < 1 / (1 + np.exp(-(u - 0.5)))
        y = 5 + conf_y * u + effect * adopt + rng.normal(0, 1, n)
        nc = 5 + conf_n * u + rng.normal(0, 1, n)
        bias = (y[adopt].mean() - y[~adopt].mean()) - effect
        signal = nc[adopt].mean() - nc[~adopt].mean()
        return float(bias), float(signal)

    biases, signals = [], []
    for s in np.linspace(0.0, 1.6, 17):
        b, g = np.mean([run(float(s)) for _ in range(12)], axis=0)
        biases.append(float(b))
        signals.append(float(g))
    return biases, signals


def _point_error(exposure: float) -> float:
    """Standard error of one of the figure's points: 12 populations of 30,000 averaged."""
    return difference_standard_error(USERS, exposure, SELECTION) / sqrt(RUNS_PER_STRENGTH)


def test_the_figure_matches_the_website_loop_draw_for_draw() -> None:
    """Same generator, same order: every bias and signal is identical."""
    biases, signals = _website_curve()
    assert CURVE.strengths == STRENGTHS
    assert list(CURVE.biases) == biases
    assert list(CURVE.signals) == signals


def test_the_figure_points_are_pinned() -> None:
    """The 17 points, to six decimals."""
    assert CURVE.biases == pytest.approx(FIGURE_BIASES, abs=1e-6)
    assert CURVE.signals == pytest.approx(FIGURE_SIGNALS, abs=1e-6)
    assert (CURVE.users, CURVE.runs, CURVE.control_exposure) == (30_000, 12, 0.6)


def test_the_control_signal_stays_flat() -> None:
    """The alt text: its own exposure is fixed, so its signal does not follow the outcome's."""
    for signal in CURVE.signals:
        assert abs(signal - CURVE.expected_signal) < 3 * _point_error(CONTROL_EXPOSURE)
    assert max(CURVE.signals) - min(CURVE.signals) < 0.015
    assert CURVE.expected_signal == pytest.approx(0.6 * SELECTION.gap)


def test_the_bias_follows_the_outcome_exposure() -> None:
    """Each point's bias is within 3.5 standard errors of ``c_y g``."""
    for exposure, bias, expected in zip(
        CURVE.strengths, CURVE.biases, CURVE.expected_biases, strict=True
    ):
        assert expected == pytest.approx(exposure * SELECTION.gap)
        assert abs(bias - expected) < 3.5 * _point_error(exposure)
    slope = np.polyfit(CURVE.strengths, CURVE.biases, 1)[0]
    assert slope == pytest.approx(SELECTION.gap, abs=0.01)


def test_the_two_agree_only_where_the_exposures_coincide() -> None:
    """Overstated to the left of 0.6, understated to the right, equal to within 0.005 at it."""
    gaps = [signal - bias for bias, signal in zip(CURVE.biases, CURVE.signals, strict=True)]
    crossing = CURVE.strengths.index(STRENGTHS[6])
    assert CURVE.strengths[crossing] == pytest.approx(CONTROL_EXPOSURE)
    assert all(gap > 0.05 for gap in gaps[:crossing])
    assert all(gap < -0.05 for gap in gaps[crossing + 1 :])
    assert abs(gaps[crossing]) < 0.005
    assert int(np.argmin(np.abs(gaps))) == crossing


def test_the_control_announces_bias_that_is_not_there() -> None:
    """The title: with no confounding of the outcome the control still reads about 0.5."""
    assert abs(CURVE.biases[0]) < 3 * _point_error(0.0)
    assert CURVE.signals[0] > 0.45


def test_adopters_engagement_is_symmetric_about_one_half() -> None:
    """``phi(u) expit(u - 1/2)`` is symmetric about 1/2, so ``E[u | adopt] = 1/2`` exactly."""
    for u in (-2.0, 0.0, 0.3, 1.7):
        left = stats.norm.pdf(0.5 - u) * special.expit(-u)
        right = stats.norm.pdf(0.5 + u) * special.expit(u)
        assert float(left) == pytest.approx(float(right), rel=1e-12)
    assert SELECTION.adopter_mean == pytest.approx(0.5, abs=1e-12)
    p = SELECTION.adoption_rate
    assert SELECTION.gap == pytest.approx(1 / (2 * (1 - p)), rel=1e-12)
    assert SELECTION.other_mean == pytest.approx(-p / (2 * (1 - p)), rel=1e-12)


def test_steins_lemma_gives_the_same_gap() -> None:
    """``E[u expit(u - c)] = E[expit'(u - c)]``, a different integrand for the same moment."""

    def derivative(u: float) -> float:
        s = float(special.expit(u - 0.5))
        return s * (1 - s) * float(stats.norm.pdf(u))

    first, _ = integrate.quad(derivative, -np.inf, np.inf, epsabs=1e-13)
    p = SELECTION.adoption_rate
    assert SELECTION.gap == pytest.approx(first / (p * (1 - p)), rel=1e-9)


def test_the_moments_match_a_large_simulation() -> None:
    """A million users: adoption rate, group means and variances of engagement."""
    rng = np.random.default_rng(2026)
    u = rng.normal(size=1_000_000)
    adopt = rng.random(u.size) < special.expit(u - 0.5)
    assert float(adopt.mean()) == pytest.approx(SELECTION.adoption_rate, abs=0.002)
    assert float(u[adopt].mean()) == pytest.approx(SELECTION.adopter_mean, abs=0.004)
    assert float(u[~adopt].mean()) == pytest.approx(SELECTION.other_mean, abs=0.004)
    assert float(u[adopt].var()) == pytest.approx(SELECTION.adopter_variance, abs=0.005)
    assert float(u[~adopt].var()) == pytest.approx(SELECTION.other_variance, abs=0.005)


def test_the_standard_error_matches_simulated_populations() -> None:
    """The spread of 300 naive differences on the control at 5,000 users."""
    rng = np.random.default_rng(99)
    estimates = [
        naive_difference(pop.adopted, pop.control)[0]
        for pop in (draw_population(rng, 5000, outcome_exposure=0.0) for _ in range(300))
    ]
    expected = difference_standard_error(5000, CONTROL_EXPOSURE)
    assert float(np.std(estimates)) == pytest.approx(expected, rel=0.12)
    assert float(np.mean(estimates)) == pytest.approx(0.6 * SELECTION.gap, abs=0.006)


def test_the_article_first_table_against_the_closed_forms() -> None:
    """39.8 percent adopt; +0.960 +- 0.013 and +0.487 +- 0.012 against 0.964 and 0.498."""
    assert round(100 * ARTICLE.adoption_rate, 1) == 39.8
    assert round(ARTICLE.naive_standard_error, 3) == 0.013
    assert round(ARTICLE.control_standard_error, 3) == 0.012
    assert abs(0.960 - ARTICLE.naive_effect) < 2 * ARTICLE.naive_standard_error
    assert abs(0.487 - ARTICLE.control_signal) < 2 * ARTICLE.control_standard_error


def test_the_article_confounding_table_against_the_closed_forms() -> None:
    """Bias and signal within 0.002 of ``c g``; the ratio is the ratio of exposures."""
    printed = (
        (0.665, 0.497, 1.34),
        (0.665, 0.663, 1.00),
        (0.332, 0.497, 0.67),
        (0.998, 0.497, 2.01),
        (-0.001, 0.497, 0.00),
    )
    # 60 populations of 20,000: a standard error of about 0.0023 on each mean.
    for row, (bias, signal, ratio) in zip(ARTICLE.confounding, printed, strict=True):
        assert abs(row.bias - bias) < 0.002
        assert abs(row.signal - signal) < 0.002
        assert row.ratio == pytest.approx(row.outcome_exposure / row.control_exposure)
        assert abs(row.ratio - ratio) <= 0.011
    assert [round(row.ratio, 2) for row in ARTICLE.confounding] == [1.33, 1.0, 0.67, 2.0, 0.0]


def test_the_article_detection_table_against_the_closed_forms() -> None:
    """A signal of 0.208 and 88 percent power at 1,000 users against 0.211 and 89.8."""
    printed = (
        (1000, 0.211, 0.898),
        (5000, 0.206, 1.0),
        (20_000, 0.206, 1.0),
        (100_000, 0.207, 1.0),
    )
    for row, (users, signal, flagged) in zip(ARTICLE.detection, printed, strict=True):
        assert row.users == users
        assert abs(row.signal - signal) < 3 * row.standard_error / sqrt(400)
        # 400 simulated populations: a binomial standard deviation of at most 0.025.
        assert abs(row.flag_rate - flagged) < 2 * sqrt(0.9 * 0.1 / 400)
    assert round(ARTICLE.detection[0].signal, 3) == 0.208
    assert round(ARTICLE.detection[0].flag_rate, 2) == 0.88


def test_the_article_correction_table_to_every_printed_digit() -> None:
    """Raw 0.964; corrected 0.466, 0.715 and 0.217, labelled 75, 37 and 112 percent."""
    rows = ARTICLE.correction
    assert [round(row.raw, 3) for row in rows] == [0.964, 0.964, 0.964]
    assert [round(row.corrected, 3) for row in rows] == [0.466, 0.715, 0.217]
    assert [f"{row.share_of_confounding:4.0%}" for row in rows] == [" 75%", " 37%", "112%"]
    for row in rows:
        assert row.corrected - 0.3 == pytest.approx(
            (0.8 - row.control_exposure) * SELECTION.gap, rel=1e-12
        )


def test_the_article_aa_table_against_the_closed_forms() -> None:
    """Every rate within 2.5 binomial standard deviations of 2,000 simulated tests."""
    printed = {
        (2000, 0.0): 0.052,
        (2000, 0.02): 0.103,
        (2000, 0.05): 0.355,
        (10_000, 0.0): 0.044,
        (10_000, 0.02): 0.293,
        (10_000, 0.05): 0.948,
        (50_000, 0.0): 0.059,
        (50_000, 0.02): 0.889,
        (50_000, 0.05): 1.000,
    }
    for row in ARTICLE.aa_tests:
        rate = row.flag_rate
        sd = sqrt(max(rate * (1 - rate), 1e-6) / 2000)
        assert abs(printed[(row.users_per_arm, row.hidden_bias)] - rate) < 2.5 * sd + 5e-4
    # "Missed seven times in ten at ten thousand users per arm."
    assert round(1 - aa_flag_rate(10_000, 0.02), 1) == 0.7


def test_limiting_cases() -> None:
    """No selection, no bias, no exposure: the nominal five percent and a zero gap."""
    flat = engagement_selection(adoption_exposure=0.0)
    assert flat.gap == pytest.approx(0.0, abs=1e-12)
    assert flat.adoption_rate == pytest.approx(float(special.expit(-0.5)), rel=1e-12)
    assert flat.adopter_variance == pytest.approx(1.0, rel=1e-10)
    nominal = 2 * float(stats.norm.cdf(-1.96))
    assert aa_flag_rate(2000, 0.0) == pytest.approx(nominal, rel=1e-12)
    assert detection_rate(1000, 0.0) == pytest.approx(nominal, rel=1e-12)
    rates = [detection_rate(n, 0.25) for n in (200, 500, 1000, 2000)]
    assert rates == sorted(rates)
    assert aa_flag_rate(10_000, 0.05) > aa_flag_rate(10_000, 0.02) > aa_flag_rate(2000, 0.02)
    assert difference_standard_error(4000, 0.6) == pytest.approx(
        difference_standard_error(1000, 0.6) / 2
    )
    assert article_numbers(flat).correction[0].corrected == pytest.approx(0.3)


def test_the_naive_difference_matches_welch() -> None:
    """The unpooled standard error is SciPy's Welch statistic's denominator."""
    rng = np.random.default_rng(5)
    pop = draw_population(rng, 3000)
    difference, se = naive_difference(pop.adopted, pop.outcome)
    welch = stats.ttest_ind(pop.outcome[pop.adopted], pop.outcome[~pop.adopted], equal_var=False)
    assert difference / se == pytest.approx(float(welch.statistic), rel=1e-10)
    assert naive_difference([True, True, False, False], [3.0, 5.0, 1.0, 2.0]) == pytest.approx(
        (2.5, sqrt(2 / 2 + 0.5 / 2))
    )


def test_a_population_is_drawn_in_the_article_order() -> None:
    """Engagement, adoption, then the outcome's and the control's noise, from one generator."""
    rng, mirror = np.random.default_rng(8), np.random.default_rng(8)
    pop = draw_population(rng, 50, outcome_exposure=0.4, control_exposure=0.9, effect=0.2)
    u = mirror.normal(0, 1, 50)
    adopt = mirror.random(50) < 1 / (1 + np.exp(-(u - 0.5)))
    y = 5 + 0.4 * u + 0.2 * adopt + mirror.normal(0, 1, 50)
    nc = 5 + 0.9 * u + mirror.normal(0, 1, 50)
    assert np.array_equal(pop.engagement, u)
    assert np.array_equal(pop.adopted, adopt)
    assert np.array_equal(pop.outcome, y)
    assert np.array_equal(pop.control, nc)
    bias, signal = bias_and_signal(pop, 0.2)
    assert bias == float((y[adopt].mean() - y[~adopt].mean()) - 0.2)
    assert signal == float(nc[adopt].mean() - nc[~adopt].mean())


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a small curve; another seed does not."""

    def small(seed: int) -> TrackingCurve:
        return tracking_curve(seed, strengths=(0.0, 1.0), users=500, runs=2)

    assert small(4) == small(4)
    assert small(4) != small(5)
    first = tracking_point(np.random.default_rng(4), 0.0, users=500, runs=2)
    assert (small(4).biases[0], small(4).signals[0]) == first


def test_invalid_inputs_are_rejected() -> None:
    """Too few users, empty groups, bad arrays and non-numbers are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        draw_population(rng, 3)
    with pytest.raises(TypeError):
        draw_population(rng, 100, outcome_exposure=True)
    with pytest.raises(ValueError):
        draw_population(rng, 100, effect=float("nan"))
    with pytest.raises(ValueError):
        naive_difference([True, False, False], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        naive_difference([1, 0, 1, 0], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        naive_difference([True, True, False, False], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        naive_difference([True, True, False, False], [1.0, np.inf, 3.0, 4.0])
    with pytest.raises(ValueError):
        tracking_curve(strengths=())
    with pytest.raises(ValueError):
        tracking_point(rng, 0.5, runs=0)
    with pytest.raises(ValueError):
        aa_flag_rate(0, 0.02)
    with pytest.raises(ValueError):
        aa_flag_rate(2000, -0.02)
    with pytest.raises(ValueError):
        aa_flag_rate(2000, 0.02, sd=0.0)
    with pytest.raises(ValueError):
        detection_rate(1000, 0.25, critical=0.0)
    with pytest.raises(TypeError):
        engagement_selection(adoption_exposure=True)
