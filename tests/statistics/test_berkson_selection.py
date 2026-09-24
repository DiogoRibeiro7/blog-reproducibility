"""Check the Berkson selection model against the website, the article and closed forms.

The figure and the article are one computation from a generator seeded at 0,
and the article's later code blocks continue that generator, so the figure's
loop and the article's blocks are transcribed and matched exactly, and every
number the article prints is pinned at its printed precision: the correlations
of -0.003 and -0.40 to -0.65, both regression tables, the model comparison, the
either-or rule and the hiring correlations. The figure's labels include -0.64 at
2 percent, which the article's table skips.

The closed forms are independent of the simulation. Truncating a joint normal on
the score gives the induced correlation, the regression limits and the selected
model's errors: the simulated values sit within three standard errors of them
(the correlations within two), and the selected model's root mean squared error
of 4.649 hours is 4.649 in closed form. The claims all hold: zero correlation in
the population, more negative the more selective the rule, -0.65 at one
percent; severity's coefficient understated by a third and value's close to -1
once the selection used difficulty; an error two thirds larger for the model
trained on the escalated tickets, and no worse when the selection used only
recorded inputs.
"""

from math import exp, pi, sqrt

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate, stats

from blog_reproducibility.statistics.berkson_selection import (
    ESCALATED_SHARE,
    SELECTION_SHARES,
    TICKETS,
    SelectedFit,
    correlation,
    draw_tickets,
    either_rule_correlation,
    either_rule_share,
    example_payload,
    least_squares_slopes,
    selected_correlation,
    selected_least_squares,
    selection_curve,
    top_share,
    transfer_errors,
    truncation_factor,
)

SUMMARY = example_payload()
CURVE = SUMMARY.curve
TRANSFER = SUMMARY.transfer


def _correlation_error(rho: float, selected: int) -> float:
    """Large-sample standard error of a correlation estimated from ``selected`` pairs."""
    return (1 - rho**2) / sqrt(selected)


def _website_figure() -> list[float]:
    """Transcribe the website generator's loop."""
    r = np.random.default_rng(0)
    n = 200000
    severity, value = r.normal(0, 1, n), r.normal(0, 1, n)
    score = severity + value + r.normal(0, 0.5, n)
    shares = [0.5, 0.3, 0.15, 0.05, 0.02, 0.01]
    corr = []
    for sh in shares:
        sel = score > np.quantile(score, 1 - sh)
        corr.append(float(np.corrcoef(severity[sel], value[sel])[0, 1]))
    return corr


def _ols(x: NDArray[np.float64], y: NDArray[np.float64]) -> list[float]:
    design = np.column_stack([np.ones(len(x)), x])
    return [float(c) for c in np.linalg.lstsq(design, y, rcond=None)[0][1:]]


def _article_blocks() -> dict[str, list[float]]:
    """Transcribe the article's code blocks, which share one generator."""
    rng = np.random.default_rng(0)
    n = 200000
    severity = rng.normal(0, 1, n)
    value = rng.normal(0, 1, n)
    score = severity + value + rng.normal(0, 0.5, n)
    escalated = score > np.quantile(score, 0.85)
    out: dict[str, list[float]] = {"all": [float(np.corrcoef(severity, value)[0, 1])]}
    out["table"] = []
    for q in (0.5, 0.7, 0.85, 0.95, 0.99):
        sel = score > np.quantile(score, q)
        out["table"].append(float(np.corrcoef(severity[sel], value[sel])[0, 1]))

    hours = 4 + 3 * severity + rng.normal(0, 2, n)
    both = np.column_stack([severity, value])
    out["population"] = _ols(both, hours)
    out["escalated"] = _ols(both[escalated], hours[escalated])
    out["severity alone"] = _ols(severity[escalated, None], hours[escalated])
    out["value alone"] = _ols(value[escalated, None], hours[escalated])
    out["with score"] = _ols(np.column_stack([both, score])[escalated], hours[escalated])

    difficulty = rng.normal(0, 1, n)
    score2 = severity + value + difficulty + rng.normal(0, 0.5, n)
    esc2 = score2 > np.quantile(score2, 0.85)
    hours2 = 4 + 3 * severity + 2 * difficulty + rng.normal(0, 2, n)
    out["hidden population"] = _ols(both, hours2)
    out["hidden escalated"] = _ols(both[esc2], hours2[esc2])

    def fit_predict(mask: NDArray[np.bool_], y: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.column_stack([np.ones(n), severity, value])
        b = np.linalg.lstsq(x[mask], y[mask], rcond=None)[0]
        predictions: NDArray[np.float64] = x @ b
        return predictions

    pred_sel = fit_predict(esc2, hours2)
    pred_all = fit_predict(np.ones(n, bool), hours2)
    hi, lo = value > 1.5, severity < -1
    out["transfer"] = [
        float(np.sqrt(np.mean((hours2 - pred_sel) ** 2))),
        float(np.sqrt(np.mean((hours2 - pred_all) ** 2))),
        float(np.mean(pred_sel[hi] - hours2[hi])),
        float(np.mean(pred_all[hi] - hours2[hi])),
        float(np.mean(pred_sel[lo] - hours2[lo])),
        float(np.mean(pred_all[lo] - hours2[lo])),
    ]

    sel = (severity > 1) | (value > 1)
    out["either"] = [float(sel.mean()), float(np.corrcoef(severity[sel], value[sel])[0, 1])]
    talent, polish = rng.normal(0, 1, n), rng.normal(0, 1, n)
    out["hiring"] = []
    for q in (0.9, 0.99):
        hired = talent + polish > np.quantile(talent + polish, q)
        out["hiring"].append(float(np.corrcoef(talent[hired], polish[hired])[0, 1]))
    return out


def test_the_figure_matches_the_website_loop() -> None:
    """Same draws and quantiles: every correlation in the figure is identical."""
    assert CURVE.shares == SELECTION_SHARES
    assert list(CURVE.correlations) == _website_figure()
    assert selection_curve() == CURVE


def test_the_payload_matches_the_article_code_blocks() -> None:
    """The article's blocks, run in turn from one generator, give exactly the payload."""
    blocks = _article_blocks()
    rows = SUMMARY.regressions
    hidden = SUMMARY.hidden_regressions
    assert blocks["all"] == [CURVE.population_correlation]
    assert blocks["table"] == [CURVE.correlations[i] for i in (0, 1, 2, 3, 5)]
    for key, row in zip(
        ("population", "escalated", "severity alone", "value alone", "with score"),
        rows,
        strict=True,
    ):
        assert blocks[key] == list(row.coefficients)
    assert blocks["hidden population"] == list(hidden[0].coefficients)
    assert blocks["hidden escalated"] == list(hidden[1].coefficients)
    assert blocks["transfer"] == [
        TRANSFER.rmse_selected,
        TRANSFER.rmse_population,
        TRANSFER.high_value_error_selected,
        TRANSFER.high_value_error_population,
        TRANSFER.low_severity_error_selected,
        TRANSFER.low_severity_error_population,
    ]
    assert blocks["either"] == [SUMMARY.either.share, SUMMARY.either.correlation]
    assert blocks["hiring"] == [row.correlation for row in SUMMARY.hiring]


def test_the_article_correlation_table_and_figure_labels() -> None:
    """-0.003 in all tickets; -0.40, -0.48, -0.55, -0.61, (-0.64,) -0.65 as selection tightens."""
    assert round(CURVE.population_correlation, 3) == -0.003
    assert [f"{value:+.2f}" for value in CURVE.correlations] == [
        "-0.40",
        "-0.48",
        "-0.55",
        "-0.61",
        "-0.64",
        "-0.65",
    ]
    assert CURVE.selected == (100_000, 60_000, 30_000, 10_000, 4000, 2000)
    assert CURVE.tickets == TICKETS


def test_the_article_regression_tables() -> None:
    """+3.00 and -0.01; +2.97 and -0.04; +2.99; -1.66; +2.93 and -0.07; then +3.00, +0.00 and
    +2.00, -0.96 once the selection used difficulty."""
    printed = ((3.00, -0.01), (2.97, -0.04), (2.99,), (-1.66,), (2.93, -0.07))
    for row, values in zip(SUMMARY.regressions, printed, strict=True):
        assert tuple(round(c, 2) for c in row.coefficients[: len(values)]) == values
    hidden = SUMMARY.hidden_regressions
    assert tuple(round(c, 2) for c in hidden[0].coefficients) == (3.00, 0.00)
    assert tuple(round(c, 2) for c in hidden[1].coefficients) == (2.00, -0.96)


def test_the_article_model_comparison() -> None:
    """RMSE 4.65 against 2.83; +1.53 and -0.02 on high value; +4.96 and +0.02 on low severity."""
    assert round(TRANSFER.rmse_selected, 2) == 4.65
    assert round(TRANSFER.rmse_population, 2) == 2.83
    assert round(TRANSFER.high_value_error_selected, 2) == 1.53
    assert round(TRANSFER.high_value_error_population, 2) == -0.02
    assert round(TRANSFER.low_severity_error_selected, 2) == 4.96
    assert round(TRANSFER.low_severity_error_population, 2) == 0.02


def test_the_article_either_rule_and_hiring() -> None:
    """29 percent kept at -0.56; hiring a tenth gives -0.71 and a hundredth -0.83."""
    assert f"{SUMMARY.either.share:.0%}" == "29%"
    assert round(SUMMARY.either.correlation, 2) == -0.56
    assert [round(row.correlation, 2) for row in SUMMARY.hiring] == [-0.71, -0.83]


def test_selection_makes_independent_attributes_look_opposed() -> None:
    """Title and alt text: zero overall, more negative with selectivity, -0.65 at one percent."""
    assert abs(CURVE.population_correlation) < 3 / sqrt(TICKETS)
    correlations = list(CURVE.correlations)
    assert correlations == sorted(correlations, reverse=True)
    assert all(value < -0.35 for value in correlations)
    assert CURVE.shares[-1] == 0.01
    assert round(correlations[-1], 2) == -0.65


def test_the_curve_follows_the_closed_form() -> None:
    """Each simulated correlation within two standard errors of ``-k / (sigma^2 - k)``."""
    for value, expected, selected in zip(
        CURVE.correlations, CURVE.expected_correlations, CURVE.selected, strict=True
    ):
        assert abs(value - expected) < 2 * _correlation_error(expected, selected)
    assert CURVE.expected_correlations == tuple(
        selected_correlation(share) for share in SELECTION_SHARES
    )


def test_the_truncation_factor_matches_scipy() -> None:
    """``1 - k`` is the variance of a standard normal truncated to its top share."""
    for share in (0.5, 0.15, 0.01):
        a = float(stats.norm.isf(share))
        remaining = float(stats.truncnorm.var(a, np.inf))
        assert 1 - truncation_factor(share) == pytest.approx(remaining, rel=1e-10)
    assert truncation_factor(0.5) == pytest.approx(2 / np.pi, rel=1e-12)


def test_the_closed_form_correlation_on_a_fresh_sample() -> None:
    """A million new tickets at five percent: within three standard errors of the closed form."""
    tickets = draw_tickets(np.random.default_rng(314), 1_000_000)
    kept = top_share(tickets.score, 0.05)
    expected = selected_correlation(0.05)
    value = correlation(tickets.severity[kept], tickets.value[kept])
    assert abs(value - expected) < 3 * _correlation_error(expected, int(kept.sum()))


def test_the_limits_of_the_induced_correlation() -> None:
    """Almost no selection gives zero; a strict rule approaches -0.8 with noise, -1 without."""
    assert selected_correlation(0.999999) == pytest.approx(0.0, abs=1e-4)
    # k approaches one only like 1 - 1 / a^2, so the limit needs a far tail.
    assert truncation_factor(1e-300) == pytest.approx(1.0, abs=0.002)
    assert selected_correlation(1e-300) == pytest.approx(-1 / 1.25, abs=0.002)
    assert selected_correlation(1e-300, noise_sd=0.0) == pytest.approx(-1.0, abs=0.002)
    # Half the tickets: k = 2 / pi.
    k = 2 / np.pi
    assert selected_correlation(0.5) == pytest.approx(-k / (2.25 - k), rel=1e-12)
    strictness = [selected_correlation(s) for s in (0.5, 0.1, 0.01, 0.001)]
    assert strictness == sorted(strictness, reverse=True)


def test_the_regressions_against_their_limits() -> None:
    """Every coefficient within three standard errors of its limit under selection.

    Among the escalated, severity and value keep at least 0.44 of their variance
    given the other predictor, and 0.2 given the other and the score; resolution
    hours vary by at most 3 hours given the predictors, or by 2 once they include
    severity, and by at most ``sqrt(8)`` once difficulty drives them too.
    """
    escalated = ESCALATED_SHARE * TICKETS
    population, *selected, with_score = SUMMARY.regressions
    for coefficient, limit in zip(population.coefficients, population.expected, strict=True):
        assert abs(coefficient - limit) < 3 * 2 / sqrt(TICKETS)
    for row in selected:
        for coefficient, limit in zip(row.coefficients, row.expected, strict=True):
            assert abs(coefficient - limit) < 3 * 3 / sqrt(0.44 * escalated)
    for coefficient, limit in zip(
        with_score.coefficients[:2], with_score.expected[:2], strict=True
    ):
        assert abs(coefficient - limit) < 3 * 2 / sqrt(0.2 * escalated)
    assert SUMMARY.regressions[3].expected[0] == pytest.approx(
        3 * selected_correlation(ESCALATED_SHARE)
    )
    hidden_population, hidden = SUMMARY.hidden_regressions
    for coefficient, limit in zip(
        hidden_population.coefficients, hidden_population.expected, strict=True
    ):
        assert abs(coefficient - limit) < 3 * sqrt(8) / sqrt(TICKETS)
    for coefficient, limit in zip(hidden.coefficients, hidden.expected, strict=True):
        assert abs(coefficient - limit) < 3 * sqrt(8) / sqrt(0.44 * escalated)


def test_selection_on_recorded_inputs_leaves_the_conditional_mean_alone() -> None:
    """With no hidden input the limits are the truth: intercept 4, severity 3, value 0."""
    fit = selected_least_squares(0.15)
    assert (fit.intercept, fit.severity, fit.value) == pytest.approx((4.0, 3.0, 0.0), abs=1e-12)
    other = selected_least_squares(0.01, hidden_hours=2.0)
    assert (other.intercept, other.severity, other.value) == pytest.approx((4.0, 3.0, 0.0))
    hidden = selected_least_squares(0.15, hidden_sd=1.0, hidden_hours=2.0)
    assert (round(hidden.severity, 2), round(hidden.value, 2)) == (2.02, -0.98)


def test_the_hidden_limits_solve_the_normal_equations() -> None:
    """Build the selected covariance matrix directly and solve for the slopes."""
    share, sigma2 = 0.15, 3.25
    q = truncation_factor(share) / sigma2
    covariance = np.array([[1 - q, -q], [-q, 1 - q]])
    cross = np.array([3 * (1 - q) - 2 * q, -3 * q - 2 * q])
    slopes = np.linalg.solve(covariance, cross)
    fit = selected_least_squares(share, hidden_sd=1.0, hidden_hours=2.0)
    assert (fit.severity, fit.value) == pytest.approx(tuple(slopes), rel=1e-12)


def test_the_model_comparison_against_the_closed_form() -> None:
    """The selected model's errors on all tickets, and the claims made about them."""
    assert TRANSFER.expected_rmse_population == pytest.approx(sqrt(8), rel=1e-12)
    assert TRANSFER.rmse_selected == pytest.approx(TRANSFER.expected_rmse_selected, abs=0.01)
    assert TRANSFER.rmse_population == pytest.approx(TRANSFER.expected_rmse_population, abs=0.01)
    # About 13,000 high-value and 32,000 low-severity tickets, errors spread by about 3 hours.
    assert TRANSFER.high_value_error_selected == pytest.approx(
        TRANSFER.expected_high_value_error, abs=4 * 3 / sqrt(13_000)
    )
    assert TRANSFER.low_severity_error_selected == pytest.approx(
        TRANSFER.expected_low_severity_error, abs=4 * 3 / sqrt(32_000)
    )
    # "Two thirds larger overall" and "overpredicts ... by five hours".
    assert 1.6 < TRANSFER.rmse_selected / TRANSFER.rmse_population < 1.7
    assert round(TRANSFER.low_severity_error_selected) == 5
    # "In the first version ... as good as one trained on everything."
    assert TRANSFER.recorded_only_rmse_selected == pytest.approx(
        TRANSFER.recorded_only_rmse_population, rel=0.002
    )
    assert TRANSFER.recorded_only_rmse_population == pytest.approx(2.0, abs=0.01)
    assert transfer_errors(SelectedFit(4.0, 3.0, 0.0), hidden_sd=0.0) == pytest.approx(
        (2.0, 0.0, 0.0)
    )


def test_the_either_rule_against_numerical_integration() -> None:
    """Share ``1 - Phi(1)^2`` and the correlation from direct integrals over the kept region."""
    t = 1.0
    assert either_rule_share(t) == pytest.approx(1 - float(stats.norm.cdf(t)) ** 2, rel=1e-12)

    def moment(power_x: int, power_y: int) -> float:
        def integrand(y: float, x: float) -> float:
            return x**power_x * y**power_y * exp(-(x * x + y * y) / 2) / (2 * pi)

        # The kept region is x > t, together with x <= t and y > t.
        right, _ = integrate.dblquad(integrand, t, 10, -10, 10, epsabs=1e-12)
        above, _ = integrate.dblquad(integrand, -10, t, t, 10, epsabs=1e-12)
        return float(right) + float(above)

    p = moment(0, 0)
    mean = moment(1, 0) / p
    variance = moment(2, 0) / p - mean**2
    covariance = moment(1, 1) / p - mean**2
    assert p == pytest.approx(either_rule_share(t), rel=1e-8)
    assert covariance / variance == pytest.approx(either_rule_correlation(t), rel=1e-7)
    assert SUMMARY.either.expected_correlation == either_rule_correlation(t)
    assert SUMMARY.either.expected_share == either_rule_share(t)
    kept = SUMMARY.either.share * TICKETS
    assert abs(SUMMARY.either.correlation - SUMMARY.either.expected_correlation) < 2 * (
        _correlation_error(SUMMARY.either.expected_correlation, int(kept))
    )
    assert either_rule_correlation(-8.0) == pytest.approx(0.0, abs=1e-9)


def test_hiring_against_the_closed_form() -> None:
    """Selecting on talent plus polish with no noise: -0.71 and -0.82 in closed form."""
    rows = SUMMARY.hiring
    assert [round(row.expected_correlation, 2) for row in rows] == [-0.71, -0.82]
    for row in rows:
        selected = row.share * TICKETS
        assert abs(row.correlation - row.expected_correlation) < 3 * _correlation_error(
            row.expected_correlation, int(selected)
        )


def test_least_squares_matches_a_hand_worked_fit() -> None:
    """An exact line is recovered, and two predictors are solved jointly."""
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert least_squares_slopes(x, 1 + 2 * x[:, 0]) == pytest.approx((2.0,))
    both = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 3.0], [3.0, 1.0]])
    y = 0.5 + 1.5 * both[:, 0] - 2.0 * both[:, 1]
    assert least_squares_slopes(both, y) == pytest.approx((1.5, -2.0))


def test_the_curve_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a small curve; another seed does not."""
    assert selection_curve(3, tickets=2000) == selection_curve(3, tickets=2000)
    assert selection_curve(3, tickets=2000) != selection_curve(4, tickets=2000)
    small = selection_curve(3, tickets=2000, shares=(0.5, 0.1))
    assert small.selected == (1000, 200)


def test_invalid_inputs_are_rejected() -> None:
    """Shares outside (0, 1), bad arrays, mismatched lengths and non-numbers are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        draw_tickets(rng, 1)
    with pytest.raises(ValueError):
        draw_tickets(rng, 10, judgement_sd=-0.5)
    with pytest.raises(ValueError):
        top_share([1.0, 2.0, 3.0], 1.0)
    with pytest.raises(ValueError):
        top_share([1.0, 2.0, 3.0], 0.0)
    with pytest.raises(ValueError):
        top_share([[1.0, 2.0], [3.0, 4.0]], 0.5)
    with pytest.raises(ValueError):
        correlation([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        correlation([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        least_squares_slopes([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        selection_curve(shares=())
    with pytest.raises(TypeError):
        selected_correlation(True)
    with pytest.raises(ValueError):
        selected_least_squares(0.15, hidden_sd=-1.0)
    with pytest.raises(ValueError):
        either_rule_share(float("inf"))
    with pytest.raises(ValueError):
        truncation_factor(1.5)
