"""Check the factorial design model against the website, closed forms and the article.

The figure is one computation from a generator seeded at 0, so the website's
loop is transcribed and every one of its 15,000 experiments ends at the same
setting here; the counts behind both curves are pinned (129 of 1,500 one factor
at a time with ten runs, down to 3 with a hundred; 1,040 for the half fraction
and 1,078, 1,184 and 1,283 for the full factorial at 16, 32 and 64 runs). The
closed forms are checked against Monte Carlo draws of the means and of the
coefficients' exact sampling law, and against direct two-dimensional
integration. The article's tables use the same seed in a different order with
4,000 experiments per row and are not reproduced; every value in them agrees
with its closed form within three standard errors and its printed rounding.

The figure's claims hold: one factor at a time ends at the best setting in
under one experiment in ten at every budget (9.3 percent at ten runs in closed
form), and less often the more it runs; the half fraction does so 69 percent of
the time ("about two thirds") and the sixteen-run factorial 70 percent.

Some statements in the article do not hold as written, and the tests pin what is true:

* "its t statistic exceeds 2 in every one of the four thousand simulations": it
  does so with probability 0.99865, so about five of four thousand fall short
  and all four thousand clear it with probability 0.005 (the article's own
  code, run with its seed, has six short);
* "the eight-run half fraction ... set A and B correctly every time" across
  three thousand experiments: the half fraction gets A or B wrong about once in
  six hundred, and did so in one of the figure's 1,500 experiments;
* the choice of D "has about even odds of the wrong sign": it is wrong about
  three times in ten, as the article's own 70 percent says;
* the half fraction "finds the optimum ten times more often than one at a time
  with a hundred" runs: about 1,600 times as often (0.69 against 0.0004); ten
  times is the comparison with fifteen runs;
* raising temperature and feed rate together "lifts the yield from 8.5 to
  16.5": it lifts it to 15.5, and the longer mixing time adds the last point.
"""

import itertools
from math import exp, pi, sqrt

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate, special, stats

from blog_reproducibility.statistics.factorial_designs import (
    BASELINE,
    COEFFICIENTS,
    CORNERS,
    EXPERIMENTS,
    FACTORIAL_RUNS,
    HALF_FRACTION,
    NOISE_SD,
    OFAT_RUNS,
    design_for_runs,
    effect_standard_error,
    ends_at_best,
    example_payload,
    factorial_choices,
    factorial_d_probability,
    interaction_detection_probability,
    mean_yield,
    model_columns,
    ofat_choices,
    ofat_effect_sd,
    ofat_success_probability,
)

SUMMARY = example_payload()
CLOSED = SUMMARY.closed_forms
HALF, FULL, TWICE, FOUR_TIMES = SUMMARY.factorial
ARTICLE_EXPERIMENTS = 4000


def _website_choices() -> NDArray[np.float64]:
    """Transcribe the website generator, recording each experiment's chosen setting."""
    r = np.random.default_rng(0)
    sigma = 2.0
    beta = {"A": 2.0, "B": 1.5, "C": 0.0, "D": 0.5, "AB": 2.5}
    corners = np.array(list(itertools.product([-1, 1], repeat=4)), dtype=float)
    chosen: list[NDArray[np.float64]] = []

    def response(x: NDArray[np.float64]) -> NDArray[np.float64]:
        a, b, c, d = x.T
        mean = 10 + beta["A"] * a + beta["B"] * b + beta["C"] * c + beta["D"] * d
        mean = mean + beta["AB"] * a * b
        noisy: NDArray[np.float64] = mean + r.normal(0, sigma, len(x))
        return noisy

    def ofat_best(per_setting: int) -> None:
        base = -np.ones(4)
        settings = [base.copy()]
        for col in range(4):
            x = base.copy()
            x[col] = 1
            settings.append(x)
        y = np.array([response(np.repeat(s[None], per_setting, 0)).mean() for s in settings])
        pick = base.copy()
        pick[y[1:] - y[0] > 0] = 1
        chosen.append(pick)

    def factorial_best(design: NDArray[np.float64]) -> None:
        pairs = (
            [(0, 1), (0, 2), (0, 3)]
            if len(design) % 16
            else list(itertools.combinations(range(4), 2))
        )

        def cols(d: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.column_stack(
                [np.ones(len(d))]
                + [d[:, i] for i in range(4)]
                + [d[:, i] * d[:, j] for i, j in pairs]
            )

        coef, *_ = np.linalg.lstsq(cols(design), response(design), rcond=None)
        chosen.append(corners[(cols(corners) @ coef).argmax()])

    half = corners[np.prod(corners, axis=1) == 1]
    for budget in (10, 15, 25, 40, 60, 100):
        for _ in range(1500):
            ofat_best(budget // 5)
    for design in (half, corners, np.vstack([corners] * 2), np.vstack([corners] * 4)):
        for _ in range(1500):
            factorial_best(design)
    return np.array(chosen)


def _within(simulated: float, exact: float, experiments: int, rounding: float = 0.0) -> bool:
    """Within three binomial standard errors, plus the printed rounding."""
    return abs(simulated - exact) < 3 * sqrt(exact * (1 - exact) / experiments) + rounding


def test_every_choice_is_the_websites() -> None:
    """All 15,000 experiments end at the website's setting."""
    rng = np.random.default_rng(0)
    ours = [ofat_choices(rng, runs // 5) for runs in OFAT_RUNS]
    ours += [factorial_choices(rng, design_for_runs(runs)) for runs in FACTORIAL_RUNS]
    assert np.array_equal(np.vstack(ours), _website_choices())


def test_the_figure_counts() -> None:
    """The successes out of 1,500 behind each point of the figure."""
    assert [row.runs for row in SUMMARY.ofat] == list(OFAT_RUNS)
    assert [row.successes for row in SUMMARY.ofat] == [129, 101, 54, 18, 5, 3]
    assert [row.runs for row in SUMMARY.factorial] == list(FACTORIAL_RUNS)
    assert [row.successes for row in SUMMARY.factorial] == [1040, 1078, 1184, 1283]
    for row in SUMMARY.ofat + SUMMARY.factorial:
        assert row.experiments == EXPERIMENTS
        assert row.share == row.successes / EXPERIMENTS


def test_the_figure_claims() -> None:
    """Under one in ten one at a time, worse with more runs; two thirds and seven in ten."""
    assert all(row.share < 0.1 for row in SUMMARY.ofat)
    expected = [row.expected_share for row in SUMMARY.ofat]
    assert max(expected) < 0.1
    assert expected == sorted(expected, reverse=True)
    assert round(expected[0], 3) == 0.093
    assert round(HALF.share, 2) == 0.69 and round(HALF.expected_share, 2) == 0.69
    assert round(FULL.share, 1) == 0.7 and round(FULL.expected_share, 2) == 0.70
    shares = [row.share for row in SUMMARY.factorial]
    assert shares == sorted(shares)


def test_the_simulation_agrees_with_the_closed_forms() -> None:
    """Every count is a plausible binomial draw from its closed-form probability."""
    for row in SUMMARY.ofat + SUMMARY.factorial:
        test = stats.binomtest(row.successes, row.experiments, row.expected_share)
        assert test.pvalue > 0.01
    # The factorial designs miss only on D; one half fraction got A or B wrong as well.
    assert HALF.share_a_and_b_raised == 1499 / 1500
    for row in (FULL, TWICE, FOUR_TIMES):
        assert row.share_a_and_b_raised == 1.0
        assert row.share == row.share_d_raised


def test_the_ofat_probability_by_monte_carlo() -> None:
    """Draw the five setting means directly and count the experiments that end at the best."""
    rng = np.random.default_rng(31)
    settings = np.vstack([BASELINE, BASELINE + 2 * np.eye(4)])
    means = mean_yield(settings)
    draws = 400_000
    for per_setting in (2, 3, 8):
        observed = means + rng.normal(0, NOISE_SD / sqrt(per_setting), (draws, 5))
        raised = observed[:, 1:] > observed[:, :1]
        share = float(np.mean(raised[:, 0] & raised[:, 1] & raised[:, 3]))
        assert _within(share, ofat_success_probability(per_setting), draws)


def test_the_d_probability_by_direct_integration() -> None:
    """The one-dimensional reduction against the two-dimensional integral over ``u`` and ``w``."""
    for runs in (16, 32, 64):
        variance = NOISE_SD**2 / runs
        s_u = s_v = sqrt(3 * variance)
        s_w = sqrt(variance)

        def integrand(
            w: float, u: float, s_u: float = s_u, s_v: float = s_v, s_w: float = s_w
        ) -> float:
            # D is raised when v exceeds (|u - w| - |u + w|) / 2, with v centred on 0.5.
            threshold = (abs(u - w) - abs(u + w)) / 2
            weight = exp(-0.5 * ((u / s_u) ** 2 + (w / s_w) ** 2)) / (2 * pi * s_u * s_w)
            return weight * float(special.ndtr((0.5 - threshold) / s_v))

        value, _ = integrate.dblquad(
            integrand, -8 * s_u, 8 * s_u, -8 * s_w, 8 * s_w, epsabs=1e-10, epsrel=1e-10
        )
        assert factorial_d_probability(runs) == pytest.approx(value, rel=1e-6)
    assert factorial_d_probability(8) == pytest.approx(stats.norm.cdf(0.5), rel=1e-14)


def test_the_factorial_success_by_the_coefficients_sampling_law() -> None:
    """Draw least-squares coefficients from their exact normal law and pick the best corner."""
    rng = np.random.default_rng(41)
    draws = 200_000
    for runs, expected_ab in ((8, 0.998), (16, 0.99993)):
        design = design_for_runs(runs)
        half = runs == 8
        columns = model_columns(design, half=half)
        centre = columns.T @ mean_yield(design) / runs
        coefficients = centre + rng.normal(0, NOISE_SD / sqrt(runs), (draws, columns.shape[1]))
        chosen = CORNERS[(coefficients @ model_columns(CORNERS, half=half).T).argmax(axis=1)]
        share = float(np.mean(ends_at_best(chosen)))
        both = float(np.mean((chosen[:, 0] == 1) & (chosen[:, 1] == 1)))
        d_right = factorial_d_probability(runs)
        # Success is D right with A and B right: a little below the closed form for D.
        assert _within(share, d_right * both, draws)
        assert both == pytest.approx(expected_ab, abs=5e-4)


def test_the_article_effect_tables() -> None:
    """Effects -1, -2, 0, 1 one at a time (sd 1.6) against 4, 3, 0, 1 and 5 (se 1.0)."""
    assert CLOSED.ofat_effects == (-1.0, -2.0, 0.0, 1.0)
    assert CLOSED.main_effects == (4.0, 3.0, 0.0, 1.0)
    assert CLOSED.interaction_effect == 5.0
    assert round(CLOSED.ofat_effect_sd, 1) == 1.6
    assert round(ofat_effect_sd(3), 2) == 1.63
    assert [round(se, 2) for se in CLOSED.factorial_effect_se] == [1.00, 0.71, 0.50]
    assert round(CLOSED.half_fraction_effect_se, 1) == 1.4
    # "Two and a half times the runs" for the same precision on the main effects.
    assert CLOSED.ofat_runs_for_equal_precision == pytest.approx(2.5)
    # One at a time with N / 5 runs per setting: 1.63 at fifteen runs, and at sixteen
    # "sigma sqrt(2 / (N / 5)), which is 1.58", against 1.0 for the factorial.
    assert round(NOISE_SD * sqrt(2 / (15 / 5)), 2) == 1.63
    assert round(NOISE_SD * sqrt(2 / (16 / 5)), 2) == 1.58
    # Detection of AB: 99.9, and effectively 100, percent.
    assert [round(100 * p) for p in CLOSED.interaction_detection] == [100, 100, 100]
    # Yields: 8.5 at the baseline, 16.5 at the best setting, eight points apart.
    assert (CLOSED.baseline_yield, CLOSED.best_yield) == (8.5, 16.5)


def test_the_article_share_table() -> None:
    """7, 2 and 0 percent one at a time; 69, 70 and 77 for the factorial designs."""
    printed = {3: 0.07, 8: 0.02, 20: 0.0}
    for per_setting, share in printed.items():
        exact = ofat_success_probability(per_setting)
        assert _within(share, exact, ARTICLE_EXPERIMENTS, rounding=0.005)
    for runs, share in ((8, 0.69), (16, 0.70), (32, 0.77)):
        exact = factorial_d_probability(runs)
        assert _within(share, exact, ARTICLE_EXPERIMENTS, rounding=0.005)


def test_the_statements_that_do_not_hold() -> None:
    """Every one of four thousand; A and B every time; even odds; ten times; 8.5 to 16.5."""
    detected = CLOSED.interaction_detection[0]
    assert round(ARTICLE_EXPERIMENTS * (1 - detected), 1) == 5.4
    assert detected**ARTICLE_EXPERIMENTS < 0.005
    # D is wrong three times in ten, not half the time.
    assert round(1 - factorial_d_probability(16), 1) == 0.3
    assert round(1 - factorial_d_probability(8), 1) == 0.3
    # The half fraction against a hundred runs one at a time, and against fifteen.
    assert HALF.expected_share / ofat_success_probability(20) > 1000
    assert round(HALF.expected_share / ofat_success_probability(3)) == 11
    # Raising A and B alone from the baseline reaches 15.5; D adds the last point.
    raised = mean_yield(np.array([[1.0, 1.0, -1.0, -1.0], [1.0, 1.0, -1.0, 1.0]]))
    assert raised.tolist() == [15.5, 16.5]


def test_the_designs_by_hand() -> None:
    """Orthogonal columns, the half fraction's aliases, and the yields at the corners."""
    for runs in (8, 16, 32):
        design = design_for_runs(runs)
        columns = model_columns(design, half=runs == 8)
        assert np.array_equal(columns.T @ columns, runs * np.eye(columns.shape[1]))
    assert len(HALF_FRACTION) == 8
    assert np.all(np.prod(HALF_FRACTION, axis=1) == 1)
    # AB and CD share a column in the half fraction.
    assert np.array_equal(
        HALF_FRACTION[:, 0] * HALF_FRACTION[:, 1], HALF_FRACTION[:, 2] * HALF_FRACTION[:, 3]
    )
    yields = mean_yield(CORNERS)
    best = CORNERS[yields == yields.max()]
    assert best.tolist() == [[1.0, 1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    assert ends_at_best(best).all()
    assert not ends_at_best(BASELINE[None]).any()
    assert CORNERS.tolist()[:2] == [[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, 1.0]]


def test_limiting_cases() -> None:
    """Without the interaction one at a time works; with no D effect D is a coin toss."""
    no_interaction = dict(COEFFICIENTS, AB=0.0)
    assert ofat_success_probability(200, coefficients=no_interaction) > 0.99
    assert ofat_success_probability(200) < 1e-12
    assert factorial_d_probability(16, d_coefficient=0.0) == pytest.approx(0.5)
    assert factorial_d_probability(8, d_coefficient=0.0) == pytest.approx(0.5)
    assert factorial_d_probability(16, d_coefficient=5.0) > 0.9999
    assert interaction_detection_probability(16, coefficients=no_interaction) == pytest.approx(
        2 * stats.norm.sf(2)
    )
    assert effect_standard_error(16) == 1.0
    assert ofat_success_probability(1) > ofat_success_probability(2)


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the choices; another seed does not."""
    first = factorial_choices(np.random.default_rng(4), CORNERS, 30)
    again = factorial_choices(np.random.default_rng(4), CORNERS, 30)
    other = ofat_choices(np.random.default_rng(4), 1, 200)
    assert np.array_equal(first, again)
    assert not np.array_equal(other, ofat_choices(np.random.default_rng(5), 1, 200))


def test_invalid_inputs_are_rejected() -> None:
    """Bad designs, budgets, coefficients and noise levels are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        design_for_runs(24)
    with pytest.raises(ValueError):
        design_for_runs(4)
    with pytest.raises(ValueError):
        factorial_choices(rng, np.zeros((8, 4)))
    with pytest.raises(ValueError):
        ofat_choices(rng, 0)
    with pytest.raises(ValueError):
        ofat_choices(rng, 3, sigma=0.0)
    with pytest.raises(ValueError):
        mean_yield(np.ones((2, 3)))
    with pytest.raises(ValueError):
        ofat_success_probability(3, coefficients={"A": 1.0})
    with pytest.raises(TypeError):
        factorial_d_probability(True)
    with pytest.raises(ValueError):
        CORNERS[0, 0] = 5.0
