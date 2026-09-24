"""Check the post-stratification model against the website, closed forms and the article.

The figure is one population of 120,000 from a generator seeded at 73 and one
response draw per slope from the same generator, and its three series are
reproduced draw for draw against a transcription of the website generator. The
effective sample equals ``1 / sum_a s_a^2 / r_a`` for the realised respondent
shares ``r_a``, which lie within four standard errors of their closed forms, and
the biases before and after weighting lie within four standard errors of theirs.

The article's tables use a population of 200,000 that also carries an attitude,
and response draws from generators seeded at 5, 11 and 1000 onwards, so they are
not reproduced; each printed number lies within its sampling error of the closed
form. The age shares among respondents (15, 22, 30 and 33 percent) and the
effective samples of 97, 85, 71 and 64 percent round to the closed forms, and
with the attitude, weighting on age leaves 0.227 of an unweighted bias of 0.457
in closed form (printed 0.229 of 0.458), half of it, as the article says.

Two statements in the alt text do not hold for the figure as drawn:

* precision "falls from 97 percent": the figure's first point, at a response
  slope of 0.05, keeps 99.8 percent; 97 percent is the article's mild row, at a
  slope of 0.2, which the figure passes between its second and third points;
* the surviving bias "stays within a few percent of zero throughout": it is 7.9
  and 7.4 percent at the first two points, where the unweighted bias is only
  0.034 and 0.096 and the weighted estimate keeps its sampling error of a few
  thousandths; from the third point on it stays within 2.7 percent.

The rest holds: the effective sample ends under two thirds, at 64 percent, as
the largest weight grows to between five and six times the smallest, and the
weighted estimate is within 0.009 of the truth at every slope.
"""

from math import exp, pi, sqrt

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate

from blog_reproducibility.statistics.post_stratification import (
    AGE_SHARES,
    ARTICLE_SLOPES,
    CELL_MEANS,
    POPULATION,
    attitude_row,
    band_means,
    draw_population,
    example_payload,
    expected_effective_share,
    expected_unweighted_bias,
    imbalance_row,
    kish_effective_size,
    post_stratification_weights,
    respond,
    respondent_shares,
    response_probabilities,
    response_rate,
    weighting_curve,
)

SUMMARY = example_payload()
CURVE = SUMMARY.curve
_RNG = np.random.default_rng(73)
PEOPLE = draw_population(_RNG)
ANSWERED = tuple(respond(_RNG, PEOPLE.age, slope) for slope in CURVE.slopes)
SHARES = np.array(AGE_SHARES)
# Score variance within an age band (region and noise) and across the population.
WITHIN_SD = sqrt(0.16 * 0.61 + 1.2**2)
TOTAL_SD = sqrt(float(np.dot(SHARES, (np.array(band_means()) - 7.102) ** 2)) + WITHIN_SD**2)


def _website_curves() -> tuple[list[float], list[float], list[float]]:
    """Transcribe the website generator."""
    rng = np.random.default_rng(73)
    pop = 120_000
    ages = np.array([0.28, 0.27, 0.25, 0.20])
    regions = np.array([0.50, 0.30, 0.20])
    score = np.array([[6.0, 6.4, 6.8], [6.6, 7.0, 7.4], [7.2, 7.6, 8.0], [7.8, 8.2, 8.6]])
    age = rng.choice(4, pop, p=ages)
    region = rng.choice(3, pop, p=regions)
    y = score[age, region] + rng.normal(0, 1.2, pop)
    truth = y.mean()
    eff, left, ratio = [], [], []
    for slope in np.linspace(0.05, 1.6, 16):
        s = rng.random(pop) < 1 / (1 + np.exp(-(-1.6 + slope * age)))
        a_s = age[s]
        w = np.zeros(s.sum())
        for c, share in enumerate(ages):
            m = a_s == c
            if m.sum():
                w[m] = share / (m.sum() / m.size)
        raw_bias = y[s].mean() - truth
        eff.append(float(w.sum() ** 2 / np.sum(w**2) / s.sum()))
        left.append(float((np.average(y[s], weights=w) - truth) / raw_bias))
        ratio.append(float(w.max() / w.min()))
    return eff, left, ratio


def _band_shares(answered: NDArray[np.bool_]) -> NDArray[np.float64]:
    ages = PEOPLE.age[answered]
    shares: NDArray[np.float64] = np.bincount(ages, minlength=4) / ages.size
    return shares


def test_the_figure_matches_the_website_generator() -> None:
    """Same population, same response draws, same weights: identical series."""
    eff, left, ratio = _website_curves()
    assert list(CURVE.effective_share) == eff
    assert list(CURVE.surviving_bias) == left
    assert list(CURVE.weight_ratio) == ratio
    assert CURVE.slopes == tuple(float(b) for b in np.linspace(0.05, 1.6, 16))
    assert weighting_curve() == CURVE


def test_the_effective_sample_is_the_closed_form_at_the_realised_shares() -> None:
    """``n_eff / n = 1 / sum s^2 / r`` and the weight ratio is ``max s / r`` over ``min s / r``."""
    for i, answered in enumerate(ANSWERED):
        realised = _band_shares(answered)
        assert CURVE.respondents[i] == int(answered.sum())
        assert CURVE.effective_share[i] == pytest.approx(
            1 / float(np.sum(SHARES**2 / realised)), rel=1e-12
        )
        weights = SHARES / realised
        assert CURVE.weight_ratio[i] == pytest.approx(weights.max() / weights.min(), rel=1e-12)
        expected = np.array(respondent_shares(CURVE.slopes[i]))
        standard_error = np.sqrt(expected * (1 - expected) / answered.sum())
        assert np.all(np.abs(realised - expected) < 4 * standard_error)


def test_the_biases_follow_the_closed_forms() -> None:
    """Unweighted bias within four standard errors of its closed form; weighted, of zero."""
    for i, answered in enumerate(ANSWERED):
        n = int(answered.sum())
        # Respondents are a subsample of the population whose mean is the truth.
        unweighted = TOTAL_SD * sqrt(1 / n - 1 / POPULATION)
        assert abs(CURVE.unweighted_bias[i] - CURVE.expected_unweighted_bias[i]) < 4 * unweighted
        counts = np.bincount(PEOPLE.age[answered], minlength=4)
        sizes = np.bincount(PEOPLE.age, minlength=4)
        weighted = WITHIN_SD * sqrt(float(np.sum(SHARES**2 * (1 / counts - 1 / sizes))))
        # The population's band shares differ from the targets by about 1 / sqrt(120,000).
        composition = 0.6 * sqrt(0.75 / POPULATION)
        assert abs(CURVE.weighted_bias[i]) < 4 * (weighted + composition)
    assert max(abs(b) for b in CURVE.weighted_bias) < 0.009
    assert CURVE.population_mean == pytest.approx(SUMMARY.population_mean, abs=4 * TOTAL_SD / 346)


def test_precision_falls_as_the_weights_spread() -> None:
    """From 99.8 percent (not 97) to 64 percent, under two thirds, as the ratio passes five."""
    effective = np.array(CURVE.effective_share)
    assert round(100 * effective[0], 1) == 99.8
    assert round(100 * effective[-1]) == 64
    assert effective[-1] < 2 / 3
    assert np.all(np.diff(effective[:-1]) < 0)
    expected = np.array(CURVE.expected_effective_share)
    assert np.all(np.diff(expected) < 0)
    assert (round(100 * expected[0], 1), round(100 * expected[-1], 1)) == (99.8, 63.3)
    # 97 percent is the closed form at slope 0.2, between the figure's second and third points.
    assert round(100 * expected_effective_share(0.2)) == 97
    assert effective[1] > 0.97 > effective[2]
    ratio = np.array(CURVE.weight_ratio)
    assert (round(ratio[0], 2), round(ratio.max(), 1)) == (1.13, 5.7)
    assert np.all(np.diff(CURVE.expected_weight_ratio) > 0)


def test_the_surviving_bias() -> None:
    """Near zero from the third point on; about 8 percent where the unweighted bias is tiny."""
    surviving = np.abs(np.array(CURVE.surviving_bias))
    assert (round(100 * surviving[0], 1), round(100 * surviving[1], 1)) == (7.9, 7.4)
    assert float(surviving[2:].max()) < 0.027
    assert (round(CURVE.unweighted_bias[0], 3), round(CURVE.unweighted_bias[1], 3)) == (
        0.034,
        0.096,
    )


def test_the_article_survey_numbers() -> None:
    """Mean 7.097 and 62,706 of 200,000 answering; shares 15, 22, 30 and 33 percent."""
    survey = SUMMARY.survey
    population = 200_000
    assert abs(7.097 - SUMMARY.population_mean) < 3 * TOTAL_SD / sqrt(population)
    rate = survey.response_rate
    assert abs(62_706 - population * rate) < 3 * sqrt(population * rate * (1 - rate))
    assert [round(100 * r) for r in survey.respondent_shares] == [15, 22, 30, 33]
    # "The oldest band is over-represented by two thirds."
    assert survey.respondent_shares[-1] / AGE_SHARES[-1] == pytest.approx(5 / 3, abs=0.05)
    n = 62_706
    # The article's score has noise of 1.0 and the attitude's 0.7 in place of noise of 1.2.
    within_sd = sqrt(WITHIN_SD**2 - 1.2**2 + 1.0 + 0.7**2)
    total_sd = sqrt(TOTAL_SD**2 - WITHIN_SD**2 + within_sd**2)
    unweighted_error = total_sd * sqrt(1 / n - 1 / population)
    assert abs(0.2639 - survey.unweighted_bias) < 3 * unweighted_error
    weighted_error = within_sd * sqrt(1 / (0.85 * n) - 1 / population)
    for printed in (0.0036, 0.0041, 0.0042):
        assert printed < 3 * weighted_error
    assert abs(53_248 / n - survey.effective_share) < 0.005


def test_the_article_imbalance_table() -> None:
    """16.6% youngest; 26.5, 50.8, 79.8, 94.6% oldest; ratios 1.59 to 5.65; 97 to 64%."""
    printed = (
        (26.5, 1.59, 97, 0.0034),
        (50.8, 3.04, 85, 0.0063),
        (79.8, 4.77, 71, 0.0045),
        (94.6, 5.65, 64, 0.0010),
    )
    population = 200_000
    young, old = population * AGE_SHARES[0], population * AGE_SHARES[-1]
    for row, (oldest, ratio, effective, bias) in zip(SUMMARY.imbalance, printed, strict=True):
        p0, p3 = row.youngest_response, row.oldest_response
        assert abs(0.166 - p0) < 3 * sqrt(p0 * (1 - p0) / young)
        assert abs(oldest / 100 - p3) < 3 * sqrt(p3 * (1 - p3) / old)
        # The ratio's relative variance: response and band size noise in both extreme bands.
        relative = sqrt(
            (1 - p0) / (young * p0)
            + (1 - p3) / (old * p3)
            + (1 - AGE_SHARES[0]) / young
            + (1 - AGE_SHARES[-1]) / old
        )
        assert abs(ratio / row.weight_ratio - 1) < 3 * relative
        assert round(100 * row.effective_share) == effective
        respondents = population * row.response_rate
        error = sqrt(WITHIN_SD**2 - 1.2**2 + 1.0 + 0.7**2) * sqrt(
            1 / (row.effective_share * respondents) - 1 / population
        )
        assert bias < 3 * error
    assert tuple(row.slope for row in SUMMARY.imbalance) == ARTICLE_SLOPES
    assert round(SUMMARY.imbalance[0].youngest_response, 3) == 0.168


def test_the_article_attitude_table() -> None:
    """Weighting on age removes all the bias, or half of it when the attitude drives answering."""
    age_only, attitude = SUMMARY.attitude
    assert age_only.unweighted_bias == pytest.approx(expected_unweighted_bias(0.55), rel=1e-9)
    assert age_only.weighted_bias == 0.0
    # 300 runs on one fixed population of 200,000, whose own attitude and noise do not average out.
    tolerance = 0.006
    assert abs(0.2604 - age_only.unweighted_bias) < tolerance
    for printed in (0.0005, 0.0014, 0.0014):
        assert abs(printed) < tolerance
    assert abs(0.4582 - attitude.unweighted_bias) < tolerance
    for printed in (0.2291, 0.2297, 0.2297):
        assert abs(printed - attitude.weighted_bias) < tolerance
    assert (round(attitude.unweighted_bias, 3), round(attitude.weighted_bias, 3)) == (0.457, 0.227)
    assert round(attitude.weighted_bias / attitude.unweighted_bias, 1) == 0.5


def test_the_attitude_integrals_by_steins_lemma() -> None:
    """``E[k sigma(eta + b k)] = b E[sigma'(eta + b k)]``: the lean from a second integral."""
    b = 0.5
    answer, lean = [], []
    for eta in (-1.6 + 0.55 * a for a in range(4)):

        def sigma(k: float, e: float = eta) -> float:
            return 1 / (1 + exp(-(e + b * k)))

        def normal(k: float) -> float:
            return exp(-k * k / 2) / sqrt(2 * pi)

        answer.append(integrate.quad(lambda k: sigma(k) * normal(k), -12, 12)[0])
        slope = integrate.quad(lambda k: sigma(k) * (1 - sigma(k)) * normal(k), -12, 12)[0]
        lean.append(b * slope / answer[-1])
    assert attitude_row(b).weighted_bias == pytest.approx(0.7 * float(np.dot(SHARES, lean)))


def test_the_attitude_biases_against_a_million_people() -> None:
    """Unweighted and age-weighted biases from a simulated population, within four errors."""
    rng = np.random.default_rng(29)
    n = 1_000_000
    age = rng.choice(4, n, p=list(AGE_SHARES))
    region = rng.choice(3, n, p=[0.5, 0.3, 0.2])
    keen = rng.normal(0, 1, n)
    y = np.asarray(CELL_MEANS)[age, region] + 0.7 * keen + rng.normal(0, 1, n)
    answered = rng.random(n) < 1 / (1 + np.exp(-(-1.6 + 0.55 * age + 0.5 * keen)))
    band = [float(y[answered & (age == a)].mean()) for a in range(4)]
    unweighted = float(y[answered].mean() - y.mean())
    weighted = float(np.dot(SHARES, band) - y.mean())
    row = attitude_row(0.5)
    error = 1.6 / sqrt(answered.sum())
    assert abs(unweighted - row.unweighted_bias) < 4 * error
    assert abs(weighted - row.weighted_bias) < 4 * error


def test_the_closed_forms_by_hand() -> None:
    """Equal response rates cost nothing; Kish's formula on an idealised respondent set."""
    assert response_probabilities(0.0) == (1 / (1 + exp(1.6)),) * 4
    assert expected_effective_share(0.0) == pytest.approx(1.0, rel=1e-15)
    assert expected_unweighted_bias(0.0) == pytest.approx(0.0, abs=1e-14)
    assert imbalance_row(0.0).weight_ratio == pytest.approx(1.0)
    assert sum(respondent_shares(0.9)) == pytest.approx(1.0, rel=1e-15)
    assert response_rate(0.0) == pytest.approx(1 / (1 + exp(1.6)))
    assert band_means() == pytest.approx((6.28, 6.88, 7.48, 8.08))
    counts = np.round(1_000_000 * np.array(respondent_shares(1.2))).astype(int)
    groups = np.repeat(np.arange(4), counts)
    weights = post_stratification_weights(groups)
    assert kish_effective_size(weights) / groups.size == pytest.approx(
        expected_effective_share(1.2), rel=1e-5
    )


def test_weights_by_hand() -> None:
    """One of four respondents in a half-share group counts two; the rest two thirds."""
    weights = post_stratification_weights([0, 1, 1, 1], (0.5, 0.5))
    assert weights == pytest.approx([2.0, 2 / 3, 2 / 3, 2 / 3])
    assert kish_effective_size(weights) == pytest.approx(3.0)
    assert kish_effective_size([1.0, 1.0, 1.0]) == pytest.approx(3.0)
    # A group with no respondents keeps no weight.
    assert post_stratification_weights([0, 0], (0.5, 0.5)).tolist() == [0.5, 0.5]


def test_the_curve_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a small curve; another seed does not."""
    first = weighting_curve(3, size=4000, slopes=(0.5, 1.0))
    assert first == weighting_curve(3, size=4000, slopes=(0.5, 1.0))
    assert first != weighting_curve(4, size=4000, slopes=(0.5, 1.0))


def test_invalid_inputs_are_rejected() -> None:
    """Empty groups, negative weights, missing slopes and non-numbers are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        post_stratification_weights([])
    with pytest.raises(ValueError):
        post_stratification_weights([0, 1], (0.5, -0.5))
    with pytest.raises(ValueError):
        kish_effective_size([1.0, -1.0])
    with pytest.raises(ValueError):
        kish_effective_size([0.0, 0.0])
    with pytest.raises(ValueError):
        weighting_curve(slopes=())
    with pytest.raises(ValueError):
        draw_population(rng, 0)
    with pytest.raises(ValueError):
        attitude_row(-0.5)
    with pytest.raises(TypeError):
        respond(rng, [0, 1], True)
    with pytest.raises(ValueError):
        response_probabilities(float("inf"))
