"""Check the drift-monitoring model against SciPy and brute force, then the article's numbers.

The batched KS computation is checked against ``scipy.stats.ks_2samp`` called
feature by feature, and both simulations are checked on small designs against a
literal transcription of the article's loops. The seeded runs then reproduce the
power table (all four procedures, including the Benjamini-Yekutieli column the
figure does not draw) and the burst statistics exactly at the printed precision.

The article's other numbers (9.8 alerts a day and 293 in a month with no
drift, the 0.2-shift table with 4.67 true alerts, the persistence run and the
large-batch runs) come from later positions of the seed-7 generator or other
seeds, not from the figure simulations, so they are not pinned here.
"""

from itertools import product

import numpy as np
import pytest
from scipy import stats

from blog_reproducibility.machine_learning.monitoring_alerts import (
    ALPHA,
    FEATURES,
    SHIFTS,
    alert_bursts,
    benjamini_hochberg,
    benjamini_yekutieli,
    example_payload,
    false_alert_probability,
    ks_lattice_statistics,
    ks_pvalues,
    persistence_false_alerts,
    power_curves,
)

SUMMARY = example_payload()


def _percent(value: float) -> int:
    return round(100 * value)


def test_ks_matches_scipy_feature_by_feature() -> None:
    """Statistics and p-values equal SciPy's, including unequal sizes and ties."""
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(30, 120))
    batch = rng.normal(loc=0.3, size=(30, 45))
    tied_reference = rng.integers(0, 6, size=(10, 40)).astype(float)
    tied_batch = rng.integers(0, 6, size=(10, 25)).astype(float)

    for first, second in ((reference, batch), (tied_reference, tied_batch)):
        results = [stats.ks_2samp(a, b) for a, b in zip(first, second, strict=True)]
        step = np.lcm(first.shape[1], second.shape[1])
        expected = np.round(np.array([r.statistic for r in results]) * step)
        assert np.array_equal(ks_lattice_statistics(first, second), expected)
        assert np.array_equal(ks_pvalues(first, second), [r.pvalue for r in results])


def test_power_curves_follow_the_article_loop() -> None:
    """A small design reproduces a literal transcription of the article's generator."""
    shifts, days, m, moved, n_ref, n_batch = (0.1, 0.4), 3, 12, 3, 150, 60
    rows = power_curves(
        5,
        shifts=shifts,
        days=days,
        features=m,
        drifting=moved,
        reference_size=n_ref,
        batch_size=n_batch,
    )

    rng = np.random.default_rng(5)
    for row, delta in zip(rows, shifts, strict=True):
        raw, bonferroni, bh = [], [], []
        for _ in range(days):
            p = np.array(
                [
                    stats.ks_2samp(
                        rng.normal(size=n_ref),
                        rng.normal(loc=delta if j < moved else 0.0, size=n_batch),
                    ).pvalue
                    for j in range(m)
                ]
            )
            raw.append(np.mean(p[:moved] < ALPHA))
            bonferroni.append(np.mean(p[:moved] < ALPHA / m))
            bh.append(np.mean(benjamini_hochberg(p, 0.05)[:moved]))
        assert row.uncorrected == pytest.approx(np.mean(raw))
        assert row.bonferroni == pytest.approx(np.mean(bonferroni))
        assert row.benjamini_hochberg == pytest.approx(np.mean(bh))


def test_alert_bursts_follow_the_article_loop() -> None:
    """A small design reproduces the article's independent and correlated loops."""
    days, m, k, n_ref, n_batch = 4, 15, 3, 120, 50
    result = alert_bursts(
        9, days=days, features=m, factors=k, reference_size=n_ref, batch_size=n_batch
    )

    rng = np.random.default_rng(9)
    load = rng.normal(size=(m, k)) * np.sqrt(0.8 / k)

    def batch(n: int) -> np.ndarray:
        f = rng.normal(size=(n, k))
        return np.asarray(f @ load.T + rng.normal(scale=np.sqrt(0.2), size=(n, m)))

    independent, correlated = [], []
    for _ in range(days):
        p_ind = [
            stats.ks_2samp(rng.normal(size=n_ref), rng.normal(size=n_batch)).pvalue
            for _ in range(m)
        ]
        ref, new = batch(n_ref), batch(n_batch)
        p_cor = [stats.ks_2samp(ref[:, j], new[:, j]).pvalue for j in range(m)]
        independent.append(int(np.sum(np.array(p_ind) < ALPHA)))
        correlated.append(int(np.sum(np.array(p_cor) < ALPHA)))

    assert result.independent.counts == tuple(independent)
    assert result.correlated.counts == tuple(correlated)


def test_benjamini_hochberg_by_hand() -> None:
    """Thresholds k q / m, and the step-up rule rejects below the largest crossing."""
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 8

    # p_(1) = 0.04 misses its threshold of 0.025, but p_(2) meets 0.05, so both go.
    assert benjamini_hochberg([0.04, 0.04], 0.05).tolist() == [True, True]
    assert not benjamini_hochberg([0.3, 0.9, 0.5], 0.05).any()


def test_procedures_are_nested() -> None:
    """Bonferroni rejects a subset of BH, and BY a subset of BH, on any p-values."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = np.concatenate([rng.uniform(size=40), rng.uniform(0, 0.002, size=5)])
        bh = benjamini_hochberg(p, 0.05)
        assert np.all(bh[p < 0.05 / p.size])
        assert np.all(bh[benjamini_yekutieli(p, 0.05)])


def test_closed_forms_match_the_article() -> None:
    """40%, 87% and 99.996%; ten a day; about 300 a month; H_200 near 5.9; 0.7 a month."""
    table = dict(SUMMARY.false_alert_probability)
    assert _percent(table[10]) == 40
    assert _percent(table[40]) == 87
    assert round(100 * table[200], 3) == 99.996
    assert SUMMARY.expected_false_alerts_per_day == pytest.approx(10.0)
    assert 30 * SUMMARY.expected_false_alerts_per_day == pytest.approx(300.0)
    assert round(SUMMARY.yekutieli_divisor, 1) == 5.9
    assert SUMMARY.persistence_false_alerts_per_month == pytest.approx(0.7)


def test_persistence_formula_by_enumeration() -> None:
    """(d - 2) m alpha^3 equals the expectation over every pass/fail sequence."""
    days, alpha = 6, 0.3
    expected = 0.0
    for fails in product((False, True), repeat=days):
        weight = float(np.prod([alpha if f else 1 - alpha for f in fails]))
        windows = sum(all(fails[t - 2 : t + 1]) for t in range(2, days))
        expected += weight * windows
    assert persistence_false_alerts(days, 1, alpha) == pytest.approx(expected)
    assert persistence_false_alerts(days, 7, alpha) == pytest.approx(7 * expected)


def test_power_table_matches_the_article() -> None:
    """The four procedure columns of the article's power table, in percent."""
    table = [
        (
            row.shift,
            _percent(row.uncorrected),
            _percent(row.bonferroni),
            _percent(row.benjamini_hochberg),
            _percent(row.benjamini_yekutieli),
        )
        for row in SUMMARY.power
    ]
    assert table == [
        (0.05, 18, 1, 1, 1),
        (0.1, 37, 1, 1, 1),
        (0.15, 71, 9, 13, 6),
        (0.2, 91, 35, 44, 24),
        (0.3, 100, 95, 97, 92),
        (0.5, 100, 100, 100, 100),
    ]
    assert tuple(row.shift for row in SUMMARY.power) == SHIFTS


def test_corrections_cost_power_only_for_subtle_shifts() -> None:
    """The title: corrected procedures trail badly below 0.3 and nearly match above."""
    for row in SUMMARY.power:
        corrected = min(row.bonferroni, row.benjamini_hochberg)
        if row.shift >= 0.3:
            assert corrected >= 0.9 * row.uncorrected
        else:
            assert corrected <= 0.5 * row.uncorrected


def test_uncorrected_power_is_paid_for_in_false_alerts() -> None:
    """Two false alerts per true one at best, and an FDR above 80% below 0.15 sd."""
    for row in SUMMARY.power:
        assert row.uncorrected_false_per_day >= 1.8 * row.uncorrected_true_per_day
        assert row.uncorrected_true_per_day == pytest.approx(5 * row.uncorrected)
        if row.shift < 0.15:
            assert row.uncorrected_fdr > 0.8
        if row.shift >= 0.2:
            assert round(row.uncorrected_fdr, 1) in {0.6, 0.7}


def test_bursts_match_the_article() -> None:
    """sd 3.2 against 6.9, maximum 36, days over 20 and quiet days as printed."""
    independent, correlated = SUMMARY.bursts.independent, SUMMARY.bursts.correlated

    assert len(independent.counts) == len(correlated.counts) == 60
    assert round(independent.sd, 1) == 3.2
    assert round(correlated.sd, 1) == 6.9
    assert correlated.maximum == 36
    assert (independent.days_over_threshold, correlated.days_over_threshold) == (1, 4)
    assert (independent.quiet_days, correlated.quiet_days) == (0, 4)


def test_same_average_but_bursts() -> None:
    """The alt text: both average about ten a day; correlation doubles the spread."""
    independent, correlated = SUMMARY.bursts.independent, SUMMARY.bursts.correlated

    assert abs(independent.mean - ALPHA * FEATURES) < 1
    assert abs(correlated.mean - ALPHA * FEATURES) < 1
    assert correlated.sd > 2 * independent.sd
    assert independent.mean == pytest.approx(np.mean(independent.counts))


def test_simulations_are_deterministic() -> None:
    """The same seed gives the same counts; another seed does not."""
    kwargs = {"days": 3, "features": 20, "reference_size": 100, "batch_size": 40}
    assert alert_bursts(3, **kwargs) == alert_bursts(3, **kwargs)
    assert alert_bursts(3, **kwargs) != alert_bursts(4, **kwargs)


def test_invalid_inputs_are_rejected() -> None:
    """Bad levels, shapes, p-values and designs are refused."""
    with pytest.raises(ValueError):
        false_alert_probability(10, alpha=1.5)
    with pytest.raises(TypeError):
        false_alert_probability(True)
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, 1.2])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1], q=0.0)
    with pytest.raises(ValueError):
        ks_pvalues(np.zeros((3, 5)), np.zeros((2, 5)))
    with pytest.raises(ValueError):
        ks_lattice_statistics(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError):
        power_curves(features=3, drifting=5)
    with pytest.raises(ValueError):
        alert_bursts(noise_variance=0.0)
