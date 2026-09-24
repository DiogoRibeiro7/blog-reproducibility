"""Check the sample ratio mismatch model against closed forms, the article and the figure.

The chi-square check is compared with SciPy's goodness-of-fit test and with the
equivalent two-sided z-test, and the article's table of detectable deviations
and the arithmetic of its opening example are pinned. The large-sample
expectations of the measured lift, the logged treated share, the alarm rate and
the relative bias are checked against a simulation of 50,000-user experiments.

The full figure run (360 experiments of a million users) takes 7 seconds or
more, so it is not repeated here. Its first experiment is pinned exactly,
the experiment is checked against a direct transcription of the website's, and
the figure's claims are checked with the closed forms at a million users. The
full run gives relative biases of -0.017, -0.013, 0.093, 0.255, 0.480 and 0.954,
and alarms in 0, 0, 1, 8, 58 and 60 of 60 experiments.

The figure plots the bias, the mean signed error of the measured lift, so the
alt text's claim is tested as it reads. The check is nearly silent (alarm
probability 0.1 to 1.1 percent) at drops up to 0.2 percent, where the bias is a
tenth of the effect or less, and it fires reliably (96 percent, then certainly)
at 1 and 2 percent, where the bias is 51 and 103 percent of the effect. The
lift's own sampling error, 0.6 of the effect at a million users, averages out of
the bias; it would dominate a mean absolute error, which sits near half the
effect even when nothing is lost.

The article's own table (200 replications, drops 0 to 2 percent) runs from a
different stream and is not pinned; its measured lifts (1.03, 1.12, 1.27, 1.52,
2.07 percent), treated shares and alarm rates agree with the closed forms here
(1.00, 1.10, 1.25, 1.51, 2.03 percent) within its simulation noise.
"""

from math import sqrt

import numpy as np
import pytest
from scipy import stats

from blog_reproducibility.statistics.sample_ratio_mismatch import (
    DROP_SHARES,
    REPLICATIONS,
    SEED,
    TRUE_LIFT,
    Design,
    alarm_probability,
    article_numbers,
    detectable_deviation,
    drop_theory,
    expected_lift,
    expected_treated_share,
    run_experiment,
    simulate_drops,
    srm_p_value,
)

THEORY = {drop: drop_theory(drop) for drop in DROP_SHARES}


def _website_experiment(r: np.random.Generator, n: int, drop: float) -> tuple[float, float]:
    """The website generator's experiment, transcribed, returning the lift and the p-value."""
    lift, slow_share, slow_ratio = 0.01, 0.20, 0.5
    z = r.integers(0, 2, n)
    slow = r.random(n) < slow_share
    base = np.where(
        slow, 0.10 * slow_ratio, 0.10 * (1 + slow_share * (1 - slow_ratio) / (1 - slow_share))
    )
    y = r.random(n) < base * (1 + lift * z)
    keep = np.ones(n, bool)
    if drop > 0:
        keep = ~((z == 1) & slow & (r.random(n) < min(1.0, drop / slow_share)))
    za, ya = z[keep], y[keep]
    measured = ya[za == 1].mean() / ya[za == 0].mean() - 1
    a, b = np.sum(za == 0), np.sum(za == 1)
    tot = a + b
    p = stats.chi2.sf((a - tot / 2) ** 2 / (tot / 2) + (b - tot / 2) ** 2 / (tot / 2), 1)
    return float(measured), float(p)


def test_the_experiment_matches_the_website() -> None:
    """Same draws in the same order give the same lift and p-value, with and without drops."""
    ours, theirs = np.random.default_rng(12), np.random.default_rng(12)
    design = Design(users=50_000)
    for drop in (0.0, 0.004, 0.02, 0.0, 0.3):
        outcome = run_experiment(ours, drop, design)
        assert (outcome.measured_lift, outcome.p_value) == _website_experiment(theirs, 50_000, drop)


def test_the_figure_starts_from_the_website_stream() -> None:
    """The figure's first million-user experiment, as the website's seed-0 generator draws it."""
    outcome = run_experiment(np.random.default_rng(SEED), DROP_SHARES[0])

    assert outcome.measured_lift == 0.020286404420354787
    assert outcome.treated_share == 500_418 / 1_000_000
    assert outcome.p_value == 0.4031549030561953


def test_the_check_is_the_chi_square_goodness_of_fit_test() -> None:
    """Equal to SciPy's test and, for an even split, to a two-sided z-test on the difference."""
    assert srm_p_value(5100, 4900) == pytest.approx(stats.chisquare([5100, 4900]).pvalue)
    assert srm_p_value(5100, 4900) == pytest.approx(2 * stats.norm.sf(200 / sqrt(10_000)))
    assert srm_p_value(9100, 900, treated_share=0.1) == pytest.approx(
        stats.chisquare([9100, 900], f_exp=[9000, 1000]).pvalue
    )
    assert srm_p_value(5000, 5000) == 1.0


def test_the_detectable_deviation_table() -> None:
    """5.20, 1.65, 0.52, 0.16 and 0.05 percent; 52, 165, 520, 1,645 and 5,203 users."""
    rows = article_numbers().detectable

    assert [row.users for row in rows] == [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    assert [round(100 * row.deviation, 2) for row in rows] == [5.20, 1.65, 0.52, 0.16, 0.05]
    assert [round(row.users_off_even) for row in rows] == [52, 165, 520, 1645, 5203]
    # The deviation shrinks with the square root of the sample.
    assert detectable_deviation(400) == pytest.approx(2 * detectable_deviation(1600))


def test_the_detectable_deviation_is_where_the_check_fires() -> None:
    """At a million users, 1,646 users off even trips the alarm and 1,645 does not."""
    assert detectable_deviation(1_000_000) * 1_000_000 == pytest.approx(1645.26, abs=0.01)
    assert srm_p_value(500_000 + 1646, 500_000 - 1646) < 0.001
    assert srm_p_value(500_000 + 1645, 500_000 - 1645) > 0.001


def test_the_opening_example() -> None:
    """502,470 against 497,530: 2,470 short, 4.94 standard deviations, about one in a million."""
    numbers = article_numbers()

    assert numbers.arm_size_sd == 500.0
    assert numbers.shortfall == 2470
    assert numbers.shortfall_in_sd == pytest.approx(4.94)
    assert 5e-7 < numbers.shortfall_p_value < 2e-6
    assert round(100 * 497_530 / 1_000_000, 2) == 49.75


def test_the_closed_forms_at_no_drop_and_without_selection() -> None:
    """Nothing lost: no bias, an even split and the nominal alarm rate. Random loss: no bias."""
    assert expected_lift(0.0) == pytest.approx(TRUE_LIFT)
    assert expected_treated_share(0.0) == 0.5
    assert alarm_probability(0.0) == pytest.approx(0.001, abs=1e-12)
    assert drop_theory(0.0).relative_bias == pytest.approx(0.0, abs=1e-12)
    # When the lost users convert like everyone else, losing them biases nothing.
    unselective = Design(slow_ratio=1.0)
    for drop in (0.01, 0.1):
        assert expected_lift(drop, unselective) == pytest.approx(TRUE_LIFT)
    # The loss comes only from slow users, so it is capped at the slow share.
    assert expected_lift(0.5) == expected_lift(0.2)


def test_one_percent_lost_inflates_the_lift_by_half() -> None:
    """The prose: a 1.0 percent lift measured as 1.5, with 49.75 percent of users treated."""
    assert round(100 * expected_lift(0.01), 1) == 1.5
    assert round(100 * expected_treated_share(0.01), 2) == 49.75


def test_the_closed_forms_match_a_simulation() -> None:
    """Mean lift, treated share, alarm rate and relative bias at 50,000 users, 150 runs."""
    design = Design(users=50_000)
    rows = simulate_drops(7, drops=(0.0, 0.03), design=design, replications=150)
    for row in rows:
        theory = drop_theory(row.drop, design)
        share_se = sqrt(0.25 / (design.users * (1 - row.drop / 2))) / sqrt(150)
        alarm_se = sqrt(max(theory.alarm_probability * (1 - theory.alarm_probability), 1e-3) / 150)
        bias_se = theory.lift_standard_error / TRUE_LIFT / sqrt(150)
        assert row.mean_lift == pytest.approx(
            theory.expected_lift, abs=4 * theory.lift_standard_error / sqrt(150)
        )
        assert row.mean_treated_share == pytest.approx(theory.treated_share, abs=4 * share_se)
        assert row.alarm_rate == pytest.approx(theory.alarm_probability, abs=4 * alarm_se)
        assert row.relative_bias == pytest.approx(theory.relative_bias, abs=4 * bias_se)


def test_the_plotted_series_is_the_signed_bias() -> None:
    """The mean signed relative error is the mean lift's error, so noise cancels in it."""
    design = Design(users=20_000)
    rows = simulate_drops(5, drops=(0.0, 0.05), design=design, replications=30)
    for row in rows:
        assert row.relative_bias == pytest.approx((row.mean_lift - TRUE_LIFT) / TRUE_LIFT)
    # One experiment's error is 0.6 of the effect; the mean of the figure's 60 is 0.08.
    assert THEORY[0.0].lift_standard_error / TRUE_LIFT == pytest.approx(0.6, abs=0.01)
    assert THEORY[0.0].lift_standard_error / TRUE_LIFT / sqrt(REPLICATIONS) < 0.08


def test_the_alarm_is_quiet_while_the_bias_is_small() -> None:
    """The alt text: up to 0.2 percent lost the check almost never fires; bias a tenth or less."""
    for drop in (0.0, 0.001, 0.002):
        assert THEORY[drop].alarm_probability < 0.012
        assert round(THEORY[drop].relative_bias, 2) <= 0.10
    assert THEORY[0.001].relative_bias == pytest.approx(0.05, abs=0.005)
    assert round(THEORY[0.002].relative_bias, 1) == 0.1


def test_the_alarm_fires_once_the_bias_is_half_the_effect() -> None:
    """The alt text and title: 96 percent at 1 percent lost, bias half the effect; certain at 2."""
    assert THEORY[0.01].alarm_probability > 0.95
    assert THEORY[0.01].relative_bias == pytest.approx(0.51, abs=0.01)
    assert THEORY[0.02].alarm_probability > 0.9999
    assert THEORY[0.02].relative_bias == pytest.approx(1.03, abs=0.01)
    alarms = [THEORY[drop].alarm_probability for drop in DROP_SHARES]
    biases = [THEORY[drop].relative_bias for drop in DROP_SHARES]
    assert alarms == sorted(alarms)
    assert biases == sorted(biases)


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the rows; another seed does not."""
    design = Design(users=20_000)

    def small(seed: int) -> object:
        return simulate_drops(seed, drops=(0.0, 0.01), design=design, replications=10)

    assert small(3) == small(3)
    assert small(3) != small(4)


def test_invalid_inputs_are_rejected() -> None:
    """Impossible designs, counts, shares and thresholds are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        Design(users=1)
    with pytest.raises(ValueError):
        Design(lift=0.0)
    with pytest.raises(ValueError):
        Design(slow_share=0.0)
    with pytest.raises(ValueError):
        Design(slow_ratio=10.0)
    with pytest.raises(ValueError):
        Design(base_rate=0.95)
    with pytest.raises(ValueError):
        srm_p_value(-1, 5)
    with pytest.raises(ValueError):
        srm_p_value(0, 0)
    with pytest.raises(TypeError):
        srm_p_value(True, 5)
    with pytest.raises(ValueError):
        srm_p_value(5, 5, treated_share=1.0)
    with pytest.raises(ValueError):
        run_experiment(rng, 1.5)
    with pytest.raises(ValueError):
        run_experiment(np.random.default_rng(0), 0.0, Design(users=2))
    with pytest.raises(ValueError):
        simulate_drops(replications=0)
    with pytest.raises(ValueError):
        simulate_drops(threshold=0.0)
    with pytest.raises(ValueError):
        detectable_deviation(0)
    with pytest.raises(ValueError):
        alarm_probability(0.01, threshold=1.0)
    with pytest.raises(ValueError):
        expected_lift(-0.1)
