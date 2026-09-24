"""Check the acceptance sampling closed forms against exact sums, simulation and the article.

The figure is deterministic, and every table in the article that is the same
closed form is pinned at its printed precision: the clean-sample bounds, the
operating characteristic of the figure's four plans at five rates, the plan
search, the zero-tolerance comparison, the single plan in the two-stage table
and the finite-lot table. All of them match. The binomial probabilities are
checked against exact sums of binomial terms, the plan search against a
transcription of the article's loop, and the absence of any plan allowing zero,
one or two defects is proved from the monotonicity of the operating
characteristic in the sample size.

The two-stage plan's acceptance and average number inspected, and the audit's
chance of finding nothing, are simulated in the article with generators seeded
at 3 and 5 that the figure does not use, so they are not pinned; they agree with
the exact values to within their simulation noise. To the article's precision the
exact two-stage acceptances are 99.8, 97.7, 80.1 and 13.6 percent (printed 99.8,
97.8, 79.8 and 13.8), with 85, 95, 112 and 107 items inspected on average (all as
printed), and the audits find nothing 81.9, 36.7, 36.8, 0.7, 0.7 and 0.0 percent
of the time (printed 81.7, 37.1, 37.1, 0.6, 0.6 and 0.0). The prose's "13.8 percent
acceptance instead of 9.9" is therefore 13.6 exactly; its "95 items instead of 132,
a 28 percent saving" holds exactly.

The figure's claims hold: every plan accepts over 99 percent of near-perfect lots
and under 2 percent at 8 percent defective, and each larger plan accepts more lots
at the agreed 1 percent and fewer at 5 percent, so its gap between the two widens
from 53 to 93 points.
"""

from math import comb, log, prod

import numpy as np
import pytest
from scipy import stats

from blog_reproducibility.statistics.acceptance_sampling import (
    ACCEPTABLE_QUALITY,
    AUDIT_ERROR_RATES,
    AUDIT_SAMPLE_SIZES,
    CLEAN_SAMPLE_SIZES,
    DOUBLE_PLAN,
    FIGURE_PLANS,
    LOT_SIZES,
    LOT_TOLERANCE,
    RATE_POINTS,
    TABLE_RATES,
    TWO_STAGE_RATES,
    ZERO_TOLERANCE_PLANS,
    DoublePlan,
    acceptance_probability,
    audit_row,
    clean_sample_bound,
    double_sampling,
    example_payload,
    finite_lot_acceptance,
    finite_lot_row,
    operating_characteristic,
    operating_curve,
    plan_search_row,
    smallest_sample,
    two_stage_row,
)

SUMMARY = example_payload()


def _exact_acceptance(n: int, c: int, p: float) -> float:
    """Sum the binomial terms directly."""
    return sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(c + 1))


def _article_search(c: int) -> int | None:
    """Transcribe the article's search for the smallest plan meeting both risks."""
    n = c + 1
    while n < 5000:
        if stats.binom.cdf(c, n, 0.01) >= 0.95 and stats.binom.cdf(c, n, 0.05) <= 0.10:
            return n
        n += 1
    return None


def _within(simulated: float, exact: float, standard_error: float) -> bool:
    return abs(simulated - exact) < 4 * standard_error


def test_the_clean_sample_table_matches_the_article() -> None:
    """Bounds of 13.91, 5.82, 2.95, 0.99 and 0.30 percent; half of each passes 22 to 24%."""
    printed = [
        ("13.91%", "23.6%"),
        ("5.82%", "22.9%"),
        ("2.95%", "22.6%"),
        ("0.99%", "22.4%"),
        ("0.30%", "22.4%"),
    ]
    assert tuple(row.sample_size for row in SUMMARY.clean_samples) == CLEAN_SAMPLE_SIZES
    for row, expected in zip(SUMMARY.clean_samples, printed, strict=True):
        assert (f"{row.rate_bound:.2%}", f"{row.half_bound_acceptance:.1%}") == expected
        # The bound is the rate at which a clean sample happens exactly 5 percent of the time.
        assert acceptance_probability(row.sample_size, 0, row.rate_bound) == pytest.approx(0.05)
    # Fifty clean items rule out about six percent and say nothing below three.
    fifty = SUMMARY.clean_samples[1]
    assert round(100 * fifty.rate_bound) == 6
    assert fifty.rate_bound / 2 < 0.03


def test_the_rule_of_three() -> None:
    """The bound is below 3 / n and n times it rises to -log 0.05 = 2.996."""
    scaled = [n * clean_sample_bound(n) for n in (20, 50, 100, 1000, 100_000)]
    assert np.all(np.diff(scaled) > 0)
    assert scaled[-1] == pytest.approx(-log(0.05), rel=1e-4)
    for n in CLEAN_SAMPLE_SIZES:
        assert clean_sample_bound(n) < 3 / n


def test_the_operating_characteristic_table_matches_the_article() -> None:
    """The figure's four plans at 0.1, 0.5, 1, 2 and 5 percent defective."""
    printed = [
        ("95.1%", "77.8%", "60.5%", "36.4%", "7.7%"),
        ("99.5%", "91.0%", "73.6%", "40.3%", "3.7%"),
        ("100.0%", "98.1%", "85.8%", "43.1%", "0.9%"),
        ("100.0%", "99.9%", "93.3%", "33.1%", "0.0%"),
    ]
    assert SUMMARY.table_rates == TABLE_RATES
    assert tuple((row.sample_size, row.accept_number) for row in SUMMARY.operating) == FIGURE_PLANS
    for row, expected in zip(SUMMARY.operating, printed, strict=True):
        assert tuple(f"{value:.1%}" for value in row.acceptance) == expected
    # The excerpt: inspect fifty, find none, and a 2% lot passes 36%, a 5% lot 8%.
    smallest = SUMMARY.operating[0].acceptance
    assert (round(100 * smallest[3]), round(100 * smallest[4])) == (36, 8)
    # The first plan rejects four good lots in ten at 1 percent; the last accepts nine in ten.
    assert round(10 * (1 - smallest[2])) == 4
    assert round(10 * SUMMARY.operating[-1].acceptance[2]) == 9
    # The bottom row is kinder at 1 percent and harsher at 2 and 5 percent than the top one.
    largest = SUMMARY.operating[-1].acceptance
    assert largest[2] > smallest[2] and largest[3] < smallest[3] and largest[4] < smallest[4]


def test_scipy_agrees_with_the_exact_binomial_sum() -> None:
    """The operating characteristic against a direct sum of binomial terms."""
    for n, c in (*FIGURE_PLANS, (132, 3), (45, 0), (258, 8)):
        for p in (0.0001, 0.003, *TABLE_RATES, 0.08):
            expected = _exact_acceptance(n, c, p)
            assert acceptance_probability(n, c, p) == pytest.approx(expected, rel=1e-10)


def test_the_plan_search_matches_the_article() -> None:
    """Accept numbers 3, 4, 5, 6, 8 and 11 need 132, 158, 184, 209, 258 and 330 items."""
    printed = {
        3: (132, "95.6%", "9.9%"),
        4: (158, "97.8%", "10.0%"),
        5: (184, "98.9%", "9.8%"),
        6: (209, "99.5%", "9.8%"),
        8: (258, "99.9%", "9.9%"),
        11: (330, "100.0%", "9.8%"),
    }
    rows = {row.accept_number: row for row in SUMMARY.plan_search}
    assert tuple(rows) == tuple(range(12))
    for c, (n, good, bad) in printed.items():
        row = rows[c]
        assert row.acceptable_acceptance is not None and row.tolerance_acceptance is not None
        shown = (f"{row.acceptable_acceptance:.1%}", f"{row.tolerance_acceptance:.1%}")
        assert (row.sample_size, *shown) == (n, good, bad)
    # Zero, one and two defects allowed appear nowhere: no sample size meets both risks.
    for c in (0, 1, 2):
        assert rows[c] == plan_search_row(c)
        assert rows[c].sample_size is None and rows[c].acceptable_acceptance is None
    # The article's loop finds the same plans; the table omits 7, 9 and 10 (234, 282, 306).
    for c in range(3, 12):
        assert rows[c].sample_size == _article_search(c)
    assert [rows[c].sample_size for c in (7, 9, 10)] == [234, 282, 306]
    # The smallest plan meeting both risks inspects 132 and tolerates three defects.
    assert min(
        (row.sample_size, row.accept_number) for row in rows.values() if row.sample_size
    ) == (
        132,
        3,
    )


def test_no_plan_with_few_allowed_defects_meets_both_risks() -> None:
    """The acceptable-quality condition caps n below the floor the tolerance condition sets."""
    n = np.arange(1, 5000)
    for c in (0, 1, 2):
        good = stats.binom.cdf(c, n, ACCEPTABLE_QUALITY)
        bad = stats.binom.cdf(c, n, LOT_TOLERANCE)
        # Both fall with n, so the passing sizes are an initial run and a tail.
        assert np.all(np.diff(good[c:]) < 0) and np.all(np.diff(bad[c:]) < 0)
        largest_good = int(n[good >= 0.95].max())
        smallest_bad = int(n[bad <= 0.10].min())
        assert largest_good < smallest_bad
    assert smallest_sample(0, tolerance=0.95) == 1


def test_the_zero_tolerance_table_matches_the_article() -> None:
    """Equally severe at 5 percent, the plans reject 36.4, 18.0, 8.9 and 4.4% of good lots."""
    printed = [
        ("63.6%", "9.9%", "36.4%"),
        ("82.0%", "9.7%", "18.0%"),
        ("91.1%", "9.9%", "8.9%"),
        ("95.6%", "9.9%", "4.4%"),
    ]
    rows = SUMMARY.zero_tolerance
    assert tuple((row.sample_size, row.accept_number) for row in rows) == ZERO_TOLERANCE_PLANS
    for row, expected in zip(rows, printed, strict=True):
        shown = (f"{row.acceptable_acceptance:.1%}", f"{row.tolerance_acceptance:.1%}")
        assert (*shown, f"{row.producer_risk:.1%}") == expected
        # Each is the smallest plan with its accept number catching 90% of 5% lots.
        assert row.tolerance_acceptance <= 0.10
        assert acceptance_probability(row.sample_size - 1, row.accept_number, 0.05) > 0.10
    # More than a third rejected by zero tolerance, one in twenty-three by the tolerant plan.
    assert rows[0].producer_risk > 1 / 3
    assert round(1 / rows[-1].producer_risk) == 23


def test_the_two_stage_table() -> None:
    """The single plan is pinned; the article's simulated two-stage columns match exact values."""
    single = [("99.5%", 132), ("95.6%", 132), ("72.8%", 132), ("9.9%", 132)]
    simulated = [(0.998, 85), (0.978, 95), (0.798, 112), (0.138, 107)]
    assert tuple(row.rate for row in SUMMARY.two_stage) == TWO_STAGE_RATES
    for row, printed, (accepted, inspected) in zip(
        SUMMARY.two_stage, single, simulated, strict=True
    ):
        assert (f"{row.single_acceptance:.1%}", row.single_inspected) == printed
        # The exact average inspected rounds to the article's figure.
        assert round(row.double_average_inspected) == inspected
        # 20,000 simulated lots per rate.
        se = np.sqrt(row.double_acceptance * (1 - row.double_acceptance) / 20_000)
        assert _within(accepted, row.double_acceptance, se)
        undecided = (row.double_average_inspected - 80) / 80
        se = 80 * np.sqrt(undecided * (1 - undecided) / 20_000)
        assert _within(inspected, row.double_average_inspected, max(se, 0.5))
    exact = [f"{row.double_acceptance:.1%}" for row in SUMMARY.two_stage]
    assert exact == ["99.8%", "97.7%", "80.1%", "13.6%"]
    # 95 instead of 132 at 1 percent is a 28 percent saving, accepting more often.
    at_one = SUMMARY.two_stage[1]
    assert round(100 * (1 - at_one.double_average_inspected / 132)) == 28
    assert at_one.double_acceptance > at_one.single_acceptance
    # Less discrimination at the bad end.
    assert SUMMARY.two_stage[-1].double_acceptance > SUMMARY.two_stage[-1].single_acceptance


def test_double_sampling_by_enumeration_and_simulation() -> None:
    """The exact sums against every pair of counts, and against the article's simulation."""
    plan = DOUBLE_PLAN
    for p in (0.003, *TWO_STAGE_RATES, 0.12):
        first = stats.binom.pmf(np.arange(plan.first_size + 1), plan.first_size, p)
        second = stats.binom.pmf(np.arange(plan.second_size + 1), plan.second_size, p)
        k1, k2 = np.meshgrid(np.arange(first.size), np.arange(second.size), indexing="ij")
        decided = (k1 <= plan.first_accept) | (k1 >= plan.first_reject)
        accepts = np.where(decided, k1 <= plan.first_accept, k1 + k2 <= plan.second_accept)
        weights = np.outer(first, second)
        inspected = np.where(decided, plan.first_size, plan.first_size + plan.second_size)
        accept, average = double_sampling(p, plan)
        assert accept == pytest.approx(float(np.sum(weights * accepts)), rel=1e-12)
        assert average == pytest.approx(float(np.sum(weights * inspected)), rel=1e-12)

    # The article's simulation, with its own generator and more lots.
    rng = np.random.default_rng(2024)
    reps = 400_000
    first_draw = rng.binomial(80, 0.02, reps)
    decided_draw = (first_draw <= 1) | (first_draw >= 4)
    total = first_draw + rng.binomial(80, 0.02, reps)
    accepted = np.where(decided_draw, first_draw <= 1, total <= 4)
    accept, average = double_sampling(0.02)
    assert float(accepted.mean()) == pytest.approx(accept, abs=4 * np.sqrt(0.16 / reps))
    inspected_draw = np.where(decided_draw, 80, 160)
    assert float(inspected_draw.mean()) == pytest.approx(average, abs=4 * 80 * np.sqrt(0.25 / reps))


def test_the_finite_lot_table_matches_the_article() -> None:
    """Exact 31.32, 34.52, 35.96 and 36.41 percent against a binomial 36.42."""
    printed = [
        ("31.32%", "36.42%", "-5.10%"),
        ("34.52%", "36.42%", "-1.90%"),
        ("35.96%", "36.42%", "-0.46%"),
        ("36.41%", "36.42%", "-0.01%"),
    ]
    assert tuple(row.lot_size for row in SUMMARY.finite_lot) == LOT_SIZES
    for row, expected in zip(SUMMARY.finite_lot, printed, strict=True):
        shown = (f"{row.exact:.2%}", f"{row.binomial:.2%}", f"{row.difference:+.2%}")
        assert shown == expected
        assert row.defects == round(0.02 * row.lot_size)
        # With none allowed, the exact chance is a product over the fifty draws.
        clean = prod((row.lot_size - row.defects - i) / (row.lot_size - i) for i in range(50))
        assert row.exact == pytest.approx(clean, rel=1e-10)
    # The binomial errs in the safe direction and the gap shrinks with the lot.
    differences = [row.difference for row in SUMMARY.finite_lot]
    assert all(d < 0 for d in differences) and np.all(np.diff(differences) > 0)
    assert finite_lot_row(10_000_000).difference == pytest.approx(0.0, abs=1e-4)


def test_the_audit_table() -> None:
    """Expected and actual error counts are pinned; the chance of finding nothing is exact."""
    simulated = [0.817, 0.371, 0.371, 0.006, 0.006, 0.000]
    expected = [(0.2, 2000), (1.0, 10_000), (1.0, 2000), (5.0, 10_000), (5.0, 2000), (25.0, 10_000)]
    rows = SUMMARY.audit
    grid = [(n, p) for n in AUDIT_SAMPLE_SIZES for p in AUDIT_ERROR_RATES]
    assert [(row.sample_size, row.error_rate) for row in rows] == grid
    for row, found, (in_sample, in_table) in zip(rows, simulated, expected, strict=True):
        assert row.clean_probability == pytest.approx(
            (1 - row.error_rate) ** row.sample_size, rel=1e-10
        )
        assert (round(row.expected_in_sample, 1), round(row.errors_in_table)) == (
            in_sample,
            in_table,
        )
        se = np.sqrt(row.clean_probability * (1 - row.clean_probability) / 20_000)
        assert _within(found, row.clean_probability, max(se, 0.0005))
    exact = [f"{row.clean_probability:.1%}" for row in rows]
    assert exact == ["81.9%", "36.7%", "36.8%", "0.7%", "0.7%", "0.0%"]
    # Two hundred rows against two thousand bad ones come back clean four times in five.
    assert round(5 * rows[0].clean_probability) == 4
    assert audit_row(200, 0.001, rows=1000).errors_in_table == pytest.approx(1.0)


def test_the_figure_curves() -> None:
    """Four plans over 250 rates from 0.01 to 8 percent, in the figure's order."""
    assert len(SUMMARY.curves) == len(FIGURE_PLANS)
    for curve, (n, c) in zip(SUMMARY.curves, FIGURE_PLANS, strict=True):
        assert (curve.sample_size, curve.accept_number) == (n, c)
        assert len(curve.rates) == len(curve.acceptance) == RATE_POINTS
        assert curve.rates == tuple(np.linspace(0.0001, 0.08, 250).tolist())
        assert np.array_equal(curve.acceptance, stats.binom.cdf(c, n, curve.rates))
        # Acceptance falls as the lot gets worse.
        assert np.all(np.diff(curve.acceptance) < 0)


def test_the_figure_claims() -> None:
    """All four pass near-perfect lots and stop bad ones; bigger plans separate them more."""
    for curve in SUMMARY.curves:
        assert curve.acceptance[0] > 0.99
        assert curve.acceptance[-1] < 0.02
    good = [acceptance_probability(n, c, ACCEPTABLE_QUALITY) for n, c in FIGURE_PLANS]
    bad = [acceptance_probability(n, c, LOT_TOLERANCE) for n, c in FIGURE_PLANS]
    gaps = np.subtract(good, bad)
    # Kinder to good lots and harsher on bad ones, not just harsher.
    assert np.all(np.diff(good) > 0) and np.all(np.diff(bad) < 0)
    assert np.all(np.diff(gaps) > 0)
    assert (round(100 * gaps[0]), round(100 * gaps[-1])) == (53, 93)
    # The largest plan accepts over nine in ten at 1 percent and almost none at 5.
    assert good[-1] > 0.9 and bad[-1] < 0.001


def test_limiting_cases() -> None:
    """Perfect lots pass, fully defective ones fail unless every defect is allowed."""
    assert acceptance_probability(50, 0, 0.0) == 1.0
    assert acceptance_probability(50, 3, 1.0) == 0.0
    assert acceptance_probability(50, 50, 1.0) == 1.0
    assert acceptance_probability(1, 0, 0.3) == pytest.approx(0.7)
    assert finite_lot_acceptance(200, 0, 50, 0) == 1.0
    assert finite_lot_acceptance(50, 1, 50, 0) == 0.0
    # Inspecting the whole of a small lot sees every defect.
    assert finite_lot_acceptance(60, 3, 60, 2) == 0.0
    assert finite_lot_acceptance(60, 3, 60, 3) == 1.0
    # A second stage that is never needed leaves a single plan.
    accept, average = double_sampling(0.02, DoublePlan(50, 0, 1, 10, 0))
    assert accept == pytest.approx(acceptance_probability(50, 0, 0.02))
    assert average == 50
    # The operating characteristic at the table's rates is the scalar function.
    curve = operating_characteristic(100, 1, TABLE_RATES)
    assert curve.tolist() == [acceptance_probability(100, 1, p) for p in TABLE_RATES]
    assert two_stage_row(0.0).double_acceptance == 1.0


def test_invalid_inputs_are_rejected() -> None:
    """Impossible plans, rates, lots and grids are refused."""
    with pytest.raises(ValueError):
        acceptance_probability(0, 0, 0.1)
    with pytest.raises(ValueError):
        acceptance_probability(10, 11, 0.1)
    with pytest.raises(ValueError):
        acceptance_probability(10, 1, 1.5)
    with pytest.raises(TypeError):
        acceptance_probability(10.0, 1, 0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        acceptance_probability(10, True, 0.1)
    with pytest.raises(ValueError):
        operating_characteristic(10, 1, [0.1, -0.1])
    with pytest.raises(ValueError):
        operating_characteristic(10, 1, [])
    with pytest.raises(ValueError):
        operating_curve(50, 0, start=0.05, stop=0.01)
    with pytest.raises(ValueError):
        operating_curve(50, 0, points=1)
    with pytest.raises(ValueError):
        clean_sample_bound(50, pass_rate=0.0)
    with pytest.raises(ValueError):
        smallest_sample(3, acceptable=0.05, tolerance=0.01)
    with pytest.raises(ValueError):
        smallest_sample(3, producer_risk=1.0)
    with pytest.raises(ValueError):
        double_sampling(0.02, DoublePlan(80, 4, 1, 80, 4))
    with pytest.raises(ValueError):
        double_sampling(0.02, DoublePlan(80, 1, 81, 80, 4))
    with pytest.raises(ValueError):
        finite_lot_acceptance(40, 2, 50, 0)
    with pytest.raises(ValueError):
        finite_lot_acceptance(40, 41, 10, 0)
    with pytest.raises(ValueError):
        audit_row(200, 1.2)
