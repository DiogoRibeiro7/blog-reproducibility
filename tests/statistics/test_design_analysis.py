"""Check the Type S and Type M closed forms against integration, simulation and the article.

The exaggeration ratio is computed in closed form and checked against the
article's numerical integral; the sign error, the exaggeration and the share of
replications that shrink are checked against simulation. Every closed-form
number in the article (the design table without its simulated columns, the
concrete experiment, the post-hoc power table, and the precision table) is
pinned at its printed precision, and all of them match. The post-hoc table
reads power at the simulated p-value nearest each target; the closed form at
the target itself gives the same printed values.

The article's simulated columns (a generator seeded at 37, which the figure
does not use) are not pinned. They are checked against the closed forms: the
simulated sign error and exaggeration agree to within their noise, and the
replication table's 19.7, 50.0 and 80.2 percent and 90.0, 75.0 and 59.9 percent
agree with the closed forms 19.8, 50.0 and 80.0 and exactly 90, 75 and 60, the
last being ``1 - power / 2``.

The figure is deterministic, and its labels (3.79x, 2.30x, 1.42x and 1.13x at
the grid points nearest 10, 20, 50 and 80 percent power) are pinned. Its claims
hold: all three curves are below 1.34x above sixty percent power, and at equal
power the looser threshold exaggerates more over the whole common range.
"""

import numpy as np
import pytest
from scipy import integrate, stats

from blog_reproducibility.statistics.design_analysis import (
    ANNOTATED_POWERS,
    CURVE_POINTS,
    FIGURE_SIGNIFICANCES,
    INTERVAL_WIDTHS,
    P_VALUES,
    TABLE_POWERS,
    USERS_PER_ARM,
    critical_value,
    effect_for_power,
    exaggeration_curve,
    example_payload,
    experiment_row,
    post_hoc_power,
    power,
    precision_row,
    replication_rate,
    shrinkage_rate,
    type_m,
    type_s,
)

SUMMARY = example_payload()
CURVES = {curve.significance: curve for curve in SUMMARY.curves}


def _article_type_m(z: float, significance: float) -> float:
    """The article's numerical integral for the exaggeration ratio, transcribed."""
    zc = stats.norm.ppf(1 - significance / 2)

    def f(x: float) -> float:
        return float(abs(x) * stats.norm.pdf(x, z, 1))

    lo = integrate.quad(f, -np.inf, -zc)[0]
    hi = integrate.quad(f, zc, np.inf)[0]
    return float((lo + hi) / power(z, significance=significance) / z)


def test_the_design_table_matches_the_article() -> None:
    """Power, true effect, wrong sign and exaggeration for seven target powers."""
    printed = [
        ("10%", "0.65", "4.50%", "3.71"),
        ("15%", "0.91", "1.35%", "2.70"),
        ("20%", "1.11", "0.53%", "2.26"),
        ("35%", "1.57", "0.06%", "1.67"),
        ("50%", "1.96", "0.01%", "1.41"),
        ("80%", "2.80", "0.00%", "1.12"),
        ("95%", "3.60", "0.00%", "1.03"),
    ]
    assert tuple(row.target_power for row in SUMMARY.design) == TABLE_POWERS
    for row, expected in zip(SUMMARY.design, printed, strict=True):
        shown = (f"{row.power:.0%}", f"{row.effect:.2f}", f"{row.wrong_sign:.2%}")
        assert (*shown, f"{row.exaggeration:.2f}") == expected


def test_the_design_table_prose() -> None:
    """Twelve percent at 80% power, twice at 20%, nearly four times and 1 in 22 at 10%."""
    rows = {row.target_power: row for row in SUMMARY.design}
    assert round(100 * (rows[0.80].exaggeration - 1)) == 12
    assert rows[0.20].exaggeration > 2
    assert 3.5 < rows[0.10].exaggeration < 4
    assert round(1 / rows[0.10].wrong_sign) == 22


def test_the_concrete_experiment_matches_the_article() -> None:
    """Standard error, power, wrong sign, exaggeration and what a significant result reads."""
    printed = [
        ("0.0318", "9.6%", "5.00%", "3.85", "7.69%"),
        ("0.0142", "29.0%", "0.13%", "1.84", "3.68%"),
        ("0.0064", "88.2%", "0.00%", "1.07", "2.14%"),
        ("0.0028", "100.0%", "0.00%", "1.00", "2.00%"),
    ]
    assert tuple(row.users_per_arm for row in SUMMARY.experiments) == USERS_PER_ARM
    for row, expected in zip(SUMMARY.experiments, printed, strict=True):
        shown = (f"{row.standard_error:.4f}", f"{row.power:.1%}", f"{row.wrong_sign:.2%}")
        assert (*shown, f"{row.exaggeration:.2f}", f"{row.significant_estimate:.2%}") == expected


def test_the_introduction() -> None:
    """A standard error of 3.2%, a bar of 6.2% and a reported 7.7% against a true 2%."""
    row = experiment_row(400)
    assert round(100 * row.standard_error, 1) == 3.2
    assert round(100 * row.smallest_significant_estimate, 1) == 6.2
    assert round(100 * row.significant_estimate, 1) == 7.7
    # Nothing smaller than three times the truth can clear the bar.
    assert row.smallest_significant_estimate / 0.02 > 3
    assert row.smallest_significant_estimate == pytest.approx(critical_value() * row.standard_error)


def test_type_m_matches_the_article_integral() -> None:
    """The closed form agrees with the article's numerical integration."""
    for significance in FIGURE_SIGNIFICANCES:
        for z in (0.3, 0.65, 1.1, 2.0, 2.8, 4.2):
            expected = _article_type_m(z, significance)
            assert type_m(z, significance=significance) == pytest.approx(expected, rel=1e-8)


def test_type_s_and_type_m_match_a_simulation() -> None:
    """At ten percent power, simulated significant estimates give 4.5% and 3.71x."""
    z = effect_for_power(0.10)
    estimates = np.random.default_rng(11).normal(z, 1.0, 400_000)
    significant = estimates[np.abs(estimates) >= critical_value()]
    share = significant.size / estimates.size
    wrong = float(np.mean(significant < 0))
    ratio = float(np.mean(np.abs(significant))) / z

    assert share == pytest.approx(power(z), abs=0.002)
    assert wrong == pytest.approx(type_s(z), abs=4 * np.sqrt(0.045 * 0.955 / significant.size))
    assert ratio == pytest.approx(type_m(z), rel=0.01)


def test_the_bisection_finds_the_target_power() -> None:
    """The effect for a target power reproduces it, and 80% needs about 1.96 + 0.84."""
    for target in TABLE_POWERS:
        assert power(effect_for_power(target)) == pytest.approx(target, abs=1e-12)
    assert effect_for_power(0.8) == pytest.approx(critical_value() + stats.norm.ppf(0.8), abs=1e-4)
    assert effect_for_power(0.2, significance=0.10) < effect_for_power(0.2, significance=0.01)


def test_limiting_cases() -> None:
    """No effect leaves power at alpha, a coin-flip sign and no bound on the exaggeration."""
    assert power(1e-9) == pytest.approx(0.05, abs=1e-9)
    assert type_s(1e-6) == pytest.approx(0.5, abs=1e-5)
    assert type_m(1e-3) > 1000
    assert type_m(8.0) == pytest.approx(1.0, abs=1e-8)
    assert type_s(8.0) < 1e-15
    # Exaggeration is at least one: a significant estimate is never smaller on average.
    for z in np.linspace(0.05, 6, 40):
        assert type_m(float(z)) >= 1.0


def test_post_hoc_power_matches_the_article() -> None:
    """Power computed from the observed effect at five p-values, and a rank correlation of -1."""
    printed = ["90.8%", "73.1%", "50.0%", "24.9%", "10.4%"]
    assert tuple(row.p_value for row in SUMMARY.post_hoc) == P_VALUES
    assert [f"{row.post_hoc_power:.1%}" for row in SUMMARY.post_hoc] == printed

    # A p-value of alpha puts the observed effect on the critical value: power one half.
    c = critical_value()
    assert post_hoc_power(0.05) == pytest.approx(0.5 + stats.norm.cdf(-2 * c), abs=1e-12)
    p_values = np.linspace(0.0005, 0.9995, 400)
    post_hoc = [post_hoc_power(float(p)) for p in p_values]
    assert stats.spearmanr(p_values, post_hoc).statistic == pytest.approx(-1.0)
    assert np.all(np.diff(post_hoc) < 0)


def test_replication_matches_the_article() -> None:
    """The replication rate is about the power; the shrinkage rate is exactly 1 - power / 2."""
    printed = {0.20: (19.7, 90.0), 0.50: (50.0, 75.0), 0.80: (80.2, 59.9)}
    for row in SUMMARY.replication:
        replicates, shrinks = printed[row.target_power]
        assert row.power == pytest.approx(row.target_power, abs=1e-12)
        assert row.replicates == pytest.approx(row.power, abs=0.003)
        assert row.shrinks == pytest.approx(1 - row.target_power / 2, abs=1e-12)
        # The article simulated 200,000 first studies; a significant subset of 200,000 x power.
        noise = 4 * np.sqrt(row.replicates * (1 - row.replicates) / (200_000 * row.power))
        assert replicates / 100 == pytest.approx(row.replicates, abs=noise)
        noise = 4 * np.sqrt(row.shrinks * (1 - row.shrinks) / (200_000 * row.power))
        assert shrinks / 100 == pytest.approx(row.shrinks, abs=noise)
    assert [f"{row.replicates:.1%}" for row in SUMMARY.replication] == ["19.8%", "50.0%", "80.0%"]


def test_replication_closed_forms_match_integration_and_simulation() -> None:
    """Both rates agree with direct integration over the first estimate and with simulation."""
    z = effect_for_power(0.35)
    c = critical_value()

    def smaller(x: float) -> float:
        t = abs(x)
        return float(stats.norm.pdf(x - z) * (stats.norm.cdf(t - z) - stats.norm.cdf(-t - z)))

    integral = integrate.quad(smaller, c, np.inf)[0] + integrate.quad(smaller, -np.inf, -c)[0]
    assert shrinkage_rate(z) == pytest.approx(integral / power(z), rel=1e-7)

    rng = np.random.default_rng(12)
    first, second = rng.normal(z, 1.0, (2, 400_000))
    significant = np.abs(first) >= c
    same = (np.abs(second) >= c) & (np.sign(second) == np.sign(first))
    shrinks = np.abs(second) < np.abs(first)
    assert float(np.mean(same[significant])) == pytest.approx(replication_rate(z), abs=0.006)
    assert float(np.mean(shrinks[significant])) == pytest.approx(shrinkage_rate(z), abs=0.006)


def test_the_precision_table_matches_the_article() -> None:
    """Standard error, users per arm and exaggeration at a 2% effect for four interval widths."""
    printed = [
        ("0.0102", 3889, "1.41"),
        ("0.0051", 15_558, "1.02"),
        ("0.0026", 62_232, "1.00"),
        ("0.0013", 248_927, "1.00"),
    ]
    assert tuple(row.width for row in SUMMARY.precision) == INTERVAL_WIDTHS
    for row, (se, users, exaggeration) in zip(SUMMARY.precision, printed, strict=True):
        assert f"{row.standard_error:.4f}" == se
        assert round(row.users_per_arm) == users
        assert f"{row.exaggeration:.2f}" == exaggeration
    # 15,558 users per arm buy an interval of 1.0 to 3.0 percent around a 2 percent effect.
    row = precision_row(0.02)
    half_width = critical_value() * 0.45 * np.sqrt(2 / round(row.users_per_arm))
    assert (round(0.02 - half_width, 3), round(0.02 + half_width, 3)) == (0.010, 0.030)


def test_the_figure_curves_and_labels() -> None:
    """Three thresholds over 70 effects from 0.3 to 4.2; labels 3.79x, 2.30x, 1.42x and 1.13x."""
    assert tuple(CURVES) == FIGURE_SIGNIFICANCES
    for curve in SUMMARY.curves:
        assert len(curve.effects) == len(curve.powers) == len(curve.exaggerations) == CURVE_POINTS
        assert curve.effects[0] == 0.3 and curve.effects[-1] == pytest.approx(4.2)
    assert tuple(point.target_power for point in SUMMARY.annotated) == ANNOTATED_POWERS
    assert [f"{point.exaggeration:.2f}x" for point in SUMMARY.annotated] == [
        "3.79x",
        "2.30x",
        "1.42x",
        "1.13x",
    ]
    grid = CURVES[0.05].effects
    assert [grid.index(point.effect) for point in SUMMARY.annotated] == [6, 14, 29, 44]


def test_the_figure_claims() -> None:
    """Near one above 60% power, steep below 30%, and the looser threshold exaggerates most."""
    for curve in SUMMARY.curves:
        powers, ratios = np.asarray(curve.powers), np.asarray(curve.exaggerations)
        # Power rises with the effect and the exaggeration falls with it.
        assert np.all(np.diff(powers) > 0) and np.all(np.diff(ratios) < 0)
        assert ratios[powers >= 0.6].max() < 1.34
        assert ratios[0] > 6
        at_sixty = effect_for_power(0.6, significance=curve.significance)
        at_thirty = effect_for_power(0.3, significance=curve.significance)
        assert type_m(at_sixty, significance=curve.significance) < 1.34
        assert 1.5 < type_m(at_thirty, significance=curve.significance) < 2.1

    lowest = max(min(curve.powers) for curve in SUMMARY.curves)
    highest = min(max(curve.powers) for curve in SUMMARY.curves)
    common = np.linspace(lowest, highest, 400)
    loose, usual, strict = (
        np.interp(common, CURVES[alpha].powers, CURVES[alpha].exaggerations)
        for alpha in FIGURE_SIGNIFICANCES
    )
    assert np.all(loose > usual) and np.all(usual > strict)


def test_invalid_inputs_are_rejected() -> None:
    """Non-positive effects, impossible levels and targets, and bad grids are refused."""
    with pytest.raises(ValueError):
        power(0.0)
    with pytest.raises(ValueError):
        type_m(-1.0)
    with pytest.raises(TypeError):
        type_s(True)
    with pytest.raises(ValueError):
        critical_value(0.0)
    with pytest.raises(ValueError):
        effect_for_power(0.04)
    with pytest.raises(ValueError):
        effect_for_power(1.0)
    with pytest.raises(ValueError):
        post_hoc_power(0.0)
    with pytest.raises(TypeError):
        experiment_row(400.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        precision_row(0.0)
    with pytest.raises(ValueError):
        exaggeration_curve(0.05, start=1.0, stop=0.5)
    with pytest.raises(ValueError):
        exaggeration_curve(0.05, points=1)
