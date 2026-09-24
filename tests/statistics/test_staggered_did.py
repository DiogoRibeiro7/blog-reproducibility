"""Check the staggered rollout model against the website, closed forms, the article and the figure.

The figure's 200 panels (one generator seeded at 0) run here in a fraction of a
second, and every plotted estimate and both horizontal lines are pinned as the
website's generator computes them. The port is checked draw for draw against a
transcription of the website's loop, and against the article's own loop, which
draws the noise one cell at a time.

The closed forms are checked independently: the event-study and overall
group-time estimators equal fixed weights whose rows and columns sum to zero,
so a noise-free panel returns the truth exactly, and the regression's
expectation ``sum(D~ tau) / sum(D~^2)`` is what it returns on a noise-free
panel. The spreads match simulation. The figure's claims hold: every group-time
estimate lies within two standard errors of the true effect, and the regression
averages 0.58 against a true average effect of 1.27, below half, as its
expectation of 0.60 is.

The article's single panel from seed 1 is the figure's computation and is pinned
(1.27, 0.39, 1.14 and 1.13), as are its noise-free comparisons (1.00 and 0.00)
and its scenario table's true effects (1.00, 1.27, 1.42 and 0.97, closed
forms). The rest of its tables (500 panels per pattern from a generator seeded
at 0, then 200 more for its event study) are other draws and are not pinned;
run separately, its code reproduces them. They are checked
against the closed forms at their noise level, with two remarks. The constant
effect's regression prints 0.98 where its expectation is exactly 1.00: that run
of 500 panels sits four standard errors low. And the event study is not "within
two hundredths at every exposure": the article's own table prints 0.97 at an
exposure of four, where the truth is 1.00.
"""

from math import sqrt

import numpy as np
import pytest
from numpy.typing import NDArray

from blog_reproducibility.statistics.staggered_did import (
    EXPOSURES,
    NEVER_TREATED,
    PATTERNS,
    REPLICATIONS,
    cohorts,
    did_2x2,
    effect_matrix,
    event_study,
    event_study_weights,
    example_payload,
    expected_twfe,
    group_time_att,
    group_time_average,
    group_time_average_weights,
    simulate_event_study,
    simulate_panel,
    treatment_matrix,
    true_att,
    twfe,
    twfe_sd,
)

SUMMARY = example_payload()
SCENARIOS = {row.pattern: row for row in SUMMARY.scenarios}
ARTICLE_REPLICATIONS = 500

# The website generator's group-time estimates, to four decimals, at exposures 0 to 9.
WEBSITE_ESTIMATES = (
    0.1827, 0.3739, 0.5842, 0.7731, 0.9826, 1.1777, 1.3964, 1.6280, 1.8044, 2.0393,
)  # fmt: skip
G_OF = np.array([5] * 15 + [10] * 15 + [15] * 15 + [-1] * 15)


def _website_panel(
    r: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Transcribe the website generator's panel."""
    unit_fe = r.normal(0, 1, 60)
    time_fe = np.linspace(0, 2, 20) + r.normal(0, 0.2, 20)
    t = np.arange(20)[None, :]
    g = G_OF[:, None]
    treated = (g >= 0) & (t >= g)
    tau = np.where(treated, 0.2 * (t - g + 1), 0.0)
    y = unit_fe[:, None] + time_fe[None, :] + tau + r.normal(0, 1, (60, 20))
    return y, treated.astype(float), tau


def _website_twfe(y: NDArray[np.float64], d: NDArray[np.float64]) -> float:
    yd = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
    dd = d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()
    return float((dd * yd).sum() / (dd**2).sum())


def _website_event_study(y: NDArray[np.float64]) -> list[float]:
    """Transcribe the website generator's group-time estimates for one panel."""
    out = []
    for k in range(10):
        vals = []
        for g in (5, 10, 15):
            t = g + k
            if t >= 20:
                continue
            tr = g == G_OF
            co = (G_OF == -1) | (t < G_OF)
            vals.append((y[tr, t] - y[tr, g - 1]).mean() - (y[co, t] - y[co, g - 1]).mean())
        out.append(float(np.mean(vals)))
    return out


def _article_panel(r: np.random.Generator) -> NDArray[np.float64]:
    """Transcribe the article's loop, which draws the noise one cell at a time."""
    unit_fe = r.normal(0, 1, 60)
    time_fe = np.linspace(0, 2, 20) + r.normal(0, 0.2, 20)
    y = np.empty((60, 20))
    for i in range(60):
        g = int(G_OF[i])
        for t in range(20):
            tau = 0.2 * (t - g + 1) if g >= 0 and t >= g else 0.0
            y[i, t] = unit_fe[i] + time_fe[t] + tau + r.normal(0, 1.0)
    return y


def test_the_figure_values_are_reproduced() -> None:
    """Every plotted estimate and both lines, as the website's generator computes them."""
    assert tuple(row.exposure for row in SUMMARY.rows) == EXPOSURES
    assert SUMMARY.replications == REPLICATIONS
    assert tuple(round(row.estimate, 4) for row in SUMMARY.rows) == WEBSITE_ESTIMATES
    assert round(SUMMARY.twfe, 4) == 0.5829
    assert SUMMARY.true_att == pytest.approx(19 / 15)
    assert (f"{SUMMARY.true_att:.2f}", f"{SUMMARY.twfe:.2f}") == ("1.27", "0.58")


def test_the_panels_match_the_website_and_article_loops() -> None:
    """Same generator, same order: identical panels, regressions and event studies."""
    ours, theirs = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(3):
        y, d, tau = simulate_panel(ours)
        expected_y, expected_d, expected_tau = _website_panel(theirs)
        assert np.array_equal(y, expected_y)
        assert np.array_equal(d, expected_d)
        assert np.array_equal(tau, expected_tau)
        assert twfe(y, d) == _website_twfe(expected_y, expected_d)
        assert list(event_study(y)) == _website_event_study(expected_y)
    article = _article_panel(np.random.default_rng(1))
    assert np.array_equal(simulate_panel(np.random.default_rng(1))[0], article)
    assert np.array_equal(cohorts(), G_OF)


def test_the_simulation_loop_matches_the_website() -> None:
    """A short run of the figure's loop equals the website's accumulation."""
    estimates, coefficient, effect = simulate_event_study(5, replications=3)
    generator = np.random.default_rng(5)
    expected = np.zeros(10)
    coefficients, effects = [], []
    for _ in range(3):
        y, d, tau = _website_panel(generator)
        coefficients.append(_website_twfe(y, d))
        effects.append(tau[d == 1].mean())
        for k, value in enumerate(_website_event_study(y)):
            expected[k] += value / 3
    assert np.array_equal(estimates, expected)
    assert coefficient == float(np.mean(coefficients))
    assert effect == float(np.mean(effects))


def test_the_group_time_weights_cancel_the_fixed_effects() -> None:
    """Rows and columns of the weights sum to zero, and they reproduce the estimators."""
    y = simulate_panel(np.random.default_rng(6))[0]
    for control in ("not_yet", "never"):
        for k in EXPOSURES:
            weights = event_study_weights(k, control=control)
            assert np.allclose(weights.sum(axis=0), 0.0, atol=1e-12)
            assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-12)
            assert event_study(y, (k,), control=control)[0] == pytest.approx(
                float((weights * y).sum()), abs=1e-12
            )
        weights = group_time_average_weights(control=control)
        assert group_time_average(y, control=control) == pytest.approx(
            float((weights * y).sum()), abs=1e-12
        )


def test_the_group_time_estimators_are_unbiased() -> None:
    """A noise-free panel returns the true dynamic effect and the true average, for any pattern."""
    for pattern in PATTERNS:
        panel = simulate_panel(np.random.default_rng(7), pattern, noise_sd=0.0)[0]
        tau = effect_matrix(pattern)
        for control in ("not_yet", "never"):
            assert group_time_average(panel, control=control) == pytest.approx(true_att(pattern))
            weights = group_time_average_weights(control=control)
            assert float((weights * tau).sum()) == pytest.approx(true_att(pattern))
    growing = simulate_panel(np.random.default_rng(8), noise_sd=0.0)[0]
    assert event_study(growing) == pytest.approx([0.2 * (k + 1) for k in EXPOSURES])


def test_the_regression_expectation_is_what_a_noise_free_panel_gives() -> None:
    """sum(D~ tau) / sum(D~^2) for every pattern; exactly one for a constant effect."""
    d = treatment_matrix()
    for pattern in PATTERNS:
        panel = simulate_panel(np.random.default_rng(9), pattern, noise_sd=0.0)[0]
        assert twfe(panel, d) == pytest.approx(expected_twfe(pattern))
    assert expected_twfe("constant") == pytest.approx(1.0)
    assert expected_twfe("growing") == pytest.approx(0.60)


def test_the_spreads_match_simulation() -> None:
    """1,000 fresh panels: the regression's and the overall estimate's spreads within 7 percent."""
    rng = np.random.default_rng(10)
    coefficients, overall = [], []
    for _ in range(1000):
        y, d, _ = simulate_panel(rng)
        coefficients.append(twfe(y, d))
        overall.append(group_time_average(y))
    growing = SCENARIOS["growing"]
    assert float(np.std(coefficients)) == pytest.approx(twfe_sd(), rel=0.07)
    assert float(np.std(overall)) == pytest.approx(growing.group_time_sd, rel=0.07)
    assert abs(float(np.mean(coefficients)) - 0.60) < 4 * twfe_sd() / sqrt(1000)
    assert abs(float(np.mean(overall)) - 19 / 15) < 4 * growing.group_time_sd / sqrt(1000)


def test_the_group_time_estimates_track_the_true_effect() -> None:
    """The alt text: every event-study point within two standard errors of 0.2 (k + 1)."""
    for row in SUMMARY.rows:
        assert row.true_effect == pytest.approx(0.2 * (row.exposure + 1))
        assert abs(row.estimate - row.true_effect) < 2 * row.standard_error


def test_the_regression_reports_less_than_half_the_true_average() -> None:
    """The alt text: 0.58 against 1.27, as the expectation of 0.60 is below half of 1.27."""
    assert SUMMARY.twfe < SUMMARY.true_att / 2
    assert SUMMARY.expected_twfe < SUMMARY.true_att / 2
    assert abs(SUMMARY.twfe - SUMMARY.expected_twfe) < 4 * SUMMARY.twfe_standard_error


def test_the_article_single_panel() -> None:
    """True effect 1.27, regression 0.39, group-time 1.14 and 1.13; noise-free 1.00 and 0.00."""
    panel = SUMMARY.single_panel
    assert panel.seed == 1
    printed = (panel.true_att, panel.twfe, panel.never_treated, panel.not_yet_treated)
    assert tuple(round(value, 2) for value in printed) == (1.27, 0.39, 1.14, 1.13)
    assert panel.clean_comparison == pytest.approx(1.0)
    assert panel.forbidden_comparison == pytest.approx(0.0, abs=1e-12)


def test_the_article_scenario_table() -> None:
    """True effects exactly; the regression and group-time columns within their noise."""
    printed = {
        "constant": (1.00, 0.98, 0.99),
        "growing": (1.27, 0.60, 1.26),
        "cohort": (1.42, 1.15, 1.41),
        "fading": (0.97, 1.24, 0.97),
    }
    for pattern, (truth, regression, group_time) in printed.items():
        row = SCENARIOS[pattern]
        assert round(row.true_att, 2) == truth
        twfe_error = row.twfe_sd / sqrt(ARTICLE_REPLICATIONS)
        assert abs(regression - row.expected_twfe) < 0.005 + 4 * twfe_error
        group_error = row.group_time_sd / sqrt(ARTICLE_REPLICATIONS)
        assert abs(group_time - row.true_att) < 0.005 + 4 * group_error
    assert SCENARIOS["growing"].true_att == pytest.approx(19 / 15)
    assert SCENARIOS["cohort"].true_att == pytest.approx(17 / 12)
    # Less than half; a fifth low; a quarter (28 percent) high.
    assert SCENARIOS["growing"].expected_twfe / SCENARIOS["growing"].true_att < 0.5
    assert 1 - SCENARIOS["cohort"].expected_twfe / SCENARIOS["cohort"].true_att == pytest.approx(
        0.19, abs=0.01
    )
    assert SCENARIOS["fading"].expected_twfe / SCENARIOS["fading"].true_att == pytest.approx(
        1.28, abs=0.01
    )


def test_the_article_event_study_table() -> None:
    """Within noise of the truth, but three hundredths off at an exposure of four."""
    printed = (0.19, 0.39, 0.59, 0.79, 0.97, 1.18, 1.40, 1.60, 1.79, 1.99)
    gaps = []
    for row, value in zip(SUMMARY.rows, printed, strict=True):
        assert abs(value - row.true_effect) < 0.005 + 4 * row.standard_error
        gaps.append(round(abs(value - row.true_effect), 2))
    assert max(gaps) == 0.03
    assert gaps.index(0.03) == 4


def test_not_yet_treated_controls_are_more_precise() -> None:
    """The article: the not-yet-treated version uses more data and has the smaller spread."""
    never = float((group_time_average_weights(control="never") ** 2).sum())
    not_yet = float((group_time_average_weights(control="not_yet") ** 2).sum())
    assert not_yet < never
    # So is every event-study point but one: at exposure 4 the never-treated comparisons
    # chain through periods 9 and 14, and the shared noise partly cancels.
    for k in EXPOSURES:
        never_k = float((event_study_weights(k, control="never") ** 2).sum())
        not_yet_k = float((event_study_weights(k) ** 2).sum())
        assert (not_yet_k < never_k) == (k != 4)


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a run; another seed does not."""
    first = simulate_event_study(11, replications=4)
    assert np.array_equal(first[0], simulate_event_study(11, replications=4)[0])
    assert not np.array_equal(first[0], simulate_event_study(12, replications=4)[0])


def test_invalid_inputs_are_rejected() -> None:
    """Unknown patterns and controls, bad panels, cells and comparisons are refused."""
    y, d, _ = simulate_panel(np.random.default_rng(0))
    with pytest.raises(ValueError):
        effect_matrix("sudden")
    with pytest.raises(ValueError):
        simulate_panel(np.random.default_rng(0), noise_sd=-1.0)
    with pytest.raises(ValueError):
        twfe(y[:, :-1], d[:, :-1])
    with pytest.raises(ValueError):
        twfe(y, np.ones_like(d))
    with pytest.raises(ValueError):
        group_time_att(y, 7, 10)
    with pytest.raises(ValueError):
        group_time_att(y, 10, 9)
    with pytest.raises(ValueError):
        event_study(y, control="everyone")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        event_study(y, (20,))
    with pytest.raises(ValueError):
        event_study_weights(20)
    with pytest.raises(ValueError):
        did_2x2(y, 10, 10, 9, 14)
    with pytest.raises(ValueError):
        did_2x2(y, 10, NEVER_TREATED, 14, 9)
    with pytest.raises(TypeError):
        simulate_event_study(0.5)  # type: ignore[arg-type]


def test_a_group_time_effect_by_hand() -> None:
    """Cohort 10 at period 14: rows 15 to 29 against rows 30 to 59 (or 45 to 59), from period 9."""
    y = simulate_panel(np.random.default_rng(13))[0]

    def change(rows: slice) -> float:
        return float((y[rows, 14] - y[rows, 9]).mean())

    treated = change(slice(15, 30))
    assert group_time_att(y, 10, 14) == pytest.approx(treated - change(slice(30, 60)))
    assert group_time_att(y, 10, 14, control="never") == pytest.approx(
        treated - change(slice(45, 60))
    )
    assert did_2x2(y, 10, NEVER_TREATED, 9, 14) == pytest.approx(
        group_time_att(y, 10, 14, control="never")
    )
