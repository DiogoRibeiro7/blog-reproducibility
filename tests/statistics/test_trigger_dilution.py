"""Check the dilution closed forms against the mixture, simulation and the article's tables.

The article's sample-size and visibility tables are the closed forms the
figure draws, so both are pinned at their printed precision, and the figure's
three ratio labels are pinned too. The mixture variance is checked against the
law of total variance and against sampling, and the two sample sizes against
simulated power on the article's population. The article's simulated tables
(seeds 3, 101 and 5) are separate streams of draws and are not pinned.

Two numbers in the article's prose do not follow from its model. The excerpt
says the diluted analysis "needs thirteen times the traffic" at an 8 percent
trigger rate; the article's own table, reproduced here, gives 6.4 times.
Thirteen is ``1 / p = 12.5``, the ratio if both analyses faced the same
variance, which the article itself says is not the case. And the closing
section calls a 6 percent effect on 8 percent of users "worth 0.5 percent
overall"; that is the naive product ``0.08 x 0.06 = 0.48`` percent, while the
model's diluted effect, as the visibility table prints, is 0.76 percent of the
population mean because triggered users have the higher outcome.
"""

from math import log

import numpy as np
import pytest

from blog_reproducibility.statistics.trigger_dilution import (
    CURVE_POINTS,
    LIFT,
    TABLE_RATES,
    TRUE_EFFECT,
    annotated_gaps,
    draw_arm,
    example_payload,
    population_mean,
    population_variance,
    sample_size_curve,
    smallest_detectable_effect,
    users_all,
    users_triggered,
    visibility_row,
)

SUMMARY = example_payload()


def _welch_z(control: np.ndarray, treated: np.ndarray) -> float:
    difference = treated.mean() - control.mean()
    se = np.sqrt(control.var(ddof=1) / control.size + treated.var(ddof=1) / treated.size)
    return float(difference / se)


def test_the_sample_size_table_matches_the_article() -> None:
    """Users per arm for each analysis and their ratio at four trigger rates."""
    printed = {
        0.50: (3_382, 1_899, 1.8),
        0.20: (15_388, 4_748, 3.2),
        0.08: (76_307, 11_870, 6.4),
        0.02: (1_043_155, 47_481, 22.0),
    }
    assert tuple(row.trigger_rate for row in SUMMARY.sample_sizes) == TABLE_RATES
    for row in SUMMARY.sample_sizes:
        everyone, triggered, ratio = printed[row.trigger_rate]
        assert round(row.users_all) == everyone
        assert round(row.users_triggered) == triggered
        assert round(row.ratio, 1) == ratio


def test_the_visibility_table_matches_the_article() -> None:
    """Diluted and smallest detectable effects as percentages of the metric, and the verdict."""
    printed = {
        0.50: (3.75, 0.49, True),
        0.20: (1.76, 0.49, True),
        0.08: (0.76, 0.47, True),
        0.02: (0.20, 0.45, False),
    }
    for row in SUMMARY.visibility:
        diluted, detectable, visible = printed[row.trigger_rate]
        assert round(100 * row.diluted_effect, 2) == diluted
        assert round(100 * row.detectable_effect, 2) == detectable
        assert row.visible is visible


def test_the_figure_labels() -> None:
    """21x, 6x and 3x, at the grid points nearest 2, 8 and 20 percent."""
    labels = [(round(100 * gap.trigger_rate, 2), f"{gap.ratio:.0f}x") for gap in SUMMARY.gap_labels]
    assert labels == [(2.09, "21x"), (8.16, "6x"), (21.0, "3x")]


def test_the_mixture_variance() -> None:
    """Law of total variance, the quadratic 81 + 259p - 144p^2, and the limits."""
    for p in (0.02, 0.2, 0.5, 0.9):
        within = p * 14.0**2 + (1 - p) * 9.0**2
        between = p * (1 - p) * (30.0 - 18.0) ** 2
        assert population_variance(p) == pytest.approx(within + between, rel=1e-12)
        assert population_variance(p) == pytest.approx(81 + 259 * p - 144 * p**2, rel=1e-12)
    assert population_variance(1e-9) == pytest.approx(81.0, rel=1e-6)
    assert population_mean(0.5) == 24.0


def test_the_population_matches_sampling() -> None:
    """A large control arm has the mixture's mean and variance and the stated trigger rate."""
    outcome, triggered = draw_arm(400_000, 0.2, np.random.default_rng(13), treated=False)
    assert triggered.mean() == pytest.approx(0.2, abs=0.003)
    assert outcome.mean() == pytest.approx(population_mean(0.2), abs=0.05)
    assert outcome.var() == pytest.approx(population_variance(0.2), rel=0.01)


def test_dilution_divides_the_effect_by_the_trigger_rate() -> None:
    """All users see p times 1.8; triggered users see 1.8, within simulation noise."""
    rng = np.random.default_rng(17)
    control, control_triggered = draw_arm(200_000, 0.2, rng, treated=False)
    treated, treated_triggered = draw_arm(200_000, 0.2, rng, treated=True)
    everyone = treated.mean() - control.mean()
    only = treated[treated_triggered].mean() - control[control_triggered].mean()
    assert everyone == pytest.approx(0.2 * TRUE_EFFECT, abs=3 * 0.036)
    assert only == pytest.approx(TRUE_EFFECT, abs=3 * 0.1)
    assert round(TRUE_EFFECT, 12) == 1.8
    assert LIFT == 0.06


def test_the_sample_sizes_deliver_their_power() -> None:
    """At the closed-form sizes, both analyses reject about 80 percent of the time."""
    rng = np.random.default_rng(23)
    reps = 400
    for p, analysis in ((0.5, "all"), (0.5, "triggered"), (0.2, "triggered")):
        users = round(users_all(p) if analysis == "all" else users_triggered(p))
        hits = 0
        for _ in range(reps):
            control, control_triggered = draw_arm(users, p, rng, treated=False)
            treated, treated_triggered = draw_arm(users, p, rng, treated=True)
            if analysis == "triggered":
                control, treated = control[control_triggered], treated[treated_triggered]
            hits += abs(_welch_z(control, treated)) > 1.959964
        assert 0.72 < hits / reps < 0.88


def test_the_curves_scale_as_stated() -> None:
    """Triggered-only scales exactly as 1/p; all-user as 1/p^2 times a slowly varying variance."""
    rates, needed_all, needed_triggered = sample_size_curve()
    assert rates.size == CURVE_POINTS
    assert (rates[0], rates[-1]) == pytest.approx((0.01, 0.6))
    np.testing.assert_allclose(needed_triggered * rates, needed_triggered[0] * rates[0])
    variances = np.array([population_variance(float(p)) for p in rates])
    scaled = needed_all * rates**2 / variances
    np.testing.assert_allclose(scaled, scaled[0])
    # Local log-log slopes: close to -2 at low rates, flattening towards 50 percent.
    slope_low = log(users_all(0.02) / users_all(0.01)) / log(2)
    slope_high = log(users_all(0.5) / users_all(0.25)) / log(2)
    assert -2.0 < slope_low < -1.95
    assert -1.8 < slope_high < -1.6
    for rate, value in zip(rates[::7], needed_all[::7], strict=True):
        assert value == pytest.approx(users_all(float(rate)), rel=1e-12)


def test_the_gap_widens_from_under_two_to_over_twenty() -> None:
    """The alt text: under twofold at 50 percent, more than twentyfold at 2 percent."""
    assert users_all(0.5) / users_triggered(0.5) < 2
    assert users_all(0.02) / users_triggered(0.02) > 20
    _, needed_all, needed_triggered = sample_size_curve()
    ratio = needed_all / needed_triggered
    assert np.all(np.diff(ratio) < 0)


def test_the_excerpt_ratio_is_one_over_p_not_the_model() -> None:
    """At 8 percent the model needs 6.4 times the traffic; 'thirteen times' is 1/p = 12.5."""
    ratio = users_all(0.08) / users_triggered(0.08)
    assert round(ratio, 1) == 6.4
    assert 1 / 0.08 == 12.5
    assert ratio < 0.6 * 12.5


def test_the_diluted_share_is_not_the_naive_product() -> None:
    """0.76 percent of the population mean, against the naive 8% x 6% = 0.48 percent."""
    row = visibility_row(0.08)
    assert round(100 * row.diluted_effect, 2) == 0.76
    assert round(100 * 0.08 * LIFT, 2) == 0.48
    # A 2 percent trigger rate is invisible at 200,000 users per arm.
    assert smallest_detectable_effect(0.02) > 0.02 * TRUE_EFFECT


def test_invalid_inputs_are_rejected() -> None:
    """Trigger rates outside (0, 1), tiny arms, negative lifts and booleans are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        users_all(0.0)
    with pytest.raises(ValueError):
        users_triggered(1.0)
    with pytest.raises(TypeError):
        population_variance(True)
    with pytest.raises(ValueError):
        smallest_detectable_effect(0.1, users_per_arm=1)
    with pytest.raises(ValueError):
        users_all(0.1, power=0.0)
    with pytest.raises(ValueError):
        draw_arm(0, 0.1, rng, treated=False)
    with pytest.raises(ValueError):
        draw_arm(10, 0.1, rng, treated=True, lift=-0.1)
    with pytest.raises(ValueError):
        annotated_gaps((1.5,))
    with pytest.raises(ValueError):
        sample_size_curve(points=1)
