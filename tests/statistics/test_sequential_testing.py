"""Check the sequential boundaries against exact probabilities, published constants and the article.

The recursion is checked against the exact one- and two-look crossing
probabilities, against a simulated random walk, and through its calibrations
against the Pocock and O'Brien-Fleming constants tabulated by Jennison and
Turnbull (2000): 2.178 and 1.977 for two looks, 2.485 and 2.063 for seven. The
mixture boundary is checked by substituting it back into the likelihood ratio,
and the ratio's mean of one under the null by integration.

The figure is deterministic, and its boundaries match the article's table at
the two decimals printed. The article's recursion results (the peeking and
monitoring tables and the 5.000 and 4.995 percent crossing probabilities of its
calibrated boundaries) are reproduced exactly on the article's own grid. On the
figure's fixed grid the O'Brien-Fleming calibration lands on a grid jump at a
first critical value of exactly 5.46 and spends 5.02 percent rather than 5; the
crossing probability is a step function of the scale on any grid, so bisection
can only get within one jump of the target. The same steps make the third
printed decimal platform-dependent: a last-bit difference in the normal CDF can
move the calibration across a jump, and macOS gives 4.997 percent where Linux
and Windows give the article's 5.000, so those values are compared within 0.01
percentage points.
"""

import numpy as np
import pytest
from scipy import integrate, stats

from blog_reproducibility.statistics.sequential_testing import (
    DAYS,
    PER_DAY,
    SIGMA,
    TAU,
    FigureBoundaries,
    calibrate,
    crossing_probability,
    example_payload,
    mixture_boundary,
    mixture_likelihood_ratio,
    monitoring_rate,
    naive_critical_value,
    obrien_fleming_shape,
    pocock_shape,
)

SUMMARY = example_payload()
FIGURE = SUMMARY.figure
OBF_TABLE = (5.46, 3.86, 3.15, 2.73, 2.44, 2.23, 2.06)
OBF_LEVELS = (0.000, 0.000, 0.002, 0.006, 0.015, 0.026, 0.039)


def _two_look_exact(bound: float) -> float:
    """Chance that either of two equally spaced z statistics reaches ``bound``."""
    correlation = np.sqrt(0.5)
    joint = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, correlation], [correlation, 1]])
    inside = (
        joint.cdf([bound, bound])
        - joint.cdf([-bound, bound])
        - joint.cdf([bound, -bound])
        + joint.cdf([-bound, -bound])
    )
    return float(1 - inside)


def _simulated_crossing(bounds: np.ndarray, paths: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal((paths, bounds.size)), axis=1)
    z = walk / np.sqrt(np.arange(1, bounds.size + 1))
    return float((np.abs(z) >= bounds).any(axis=1).mean())


def test_one_look_matches_the_normal_tail() -> None:
    """One look is a single test; the grid adds a small upward bias that shrinks with the step."""
    z = naive_critical_value()
    coarse = crossing_probability([z])
    fine = crossing_probability([z], step=0.001)

    assert 0.05 < coarse < 0.051
    assert abs(fine - 0.05) < abs(coarse - 0.05)
    assert fine == pytest.approx(0.05, abs=1e-4)


def test_two_looks_match_the_bivariate_normal() -> None:
    """Two looks: z statistics with correlation sqrt(1/2), integrated exactly."""
    for bound in (1.96, 2.178, 2.5):
        assert crossing_probability([bound, bound]) == pytest.approx(
            _two_look_exact(bound), abs=0.0015
        )


def test_the_recursion_matches_a_simulated_random_walk() -> None:
    """Seven naive looks, and the calibrated boundaries, against 200,000 simulated walks."""
    naive = np.full(7, naive_critical_value())
    assert crossing_probability(naive) == pytest.approx(
        _simulated_crossing(naive, 200_000, 3), abs=0.004
    )
    for boundary in (FIGURE.pocock, FIGURE.obrien_fleming):
        bounds = np.array(boundary.critical_values)
        assert _simulated_crossing(bounds, 200_000, 4) == pytest.approx(0.05, abs=0.002)


def test_calibrations_match_published_constants() -> None:
    """Jennison and Turnbull's Pocock and O'Brien-Fleming constants at 5 percent."""
    assert calibrate(pocock_shape(2))[0] == pytest.approx(2.178, abs=0.003)
    assert calibrate(obrien_fleming_shape(2))[-1] == pytest.approx(1.977, abs=0.003)
    assert FIGURE.pocock.critical_values[0] == pytest.approx(2.485, abs=0.003)
    assert FIGURE.obrien_fleming.critical_values[-1] == pytest.approx(2.063, abs=0.002)


def test_boundaries_have_the_shapes_they_claim() -> None:
    """Pocock is flat; O'Brien-Fleming is a constant in sum units, so falls as 1 / sqrt(k)."""
    pocock = FIGURE.pocock.critical_values
    obf = np.array(FIGURE.obrien_fleming.critical_values)

    assert len(set(pocock)) == 1
    np.testing.assert_allclose(obf * np.sqrt(np.arange(1, 8)), obf[0], rtol=1e-12)


def test_the_boundary_table_matches_the_article() -> None:
    """Pocock 2.49 at a nominal 0.013 per look; O'Brien-Fleming from 5.46 down to 2.06."""
    for boundary in (FIGURE.pocock, SUMMARY.article_pocock):
        assert [round(value, 2) for value in boundary.critical_values] == [2.49] * 7
        assert [round(level, 3) for level in boundary.nominal_levels] == [0.013] * 7
    for boundary in (FIGURE.obrien_fleming, SUMMARY.article_obrien_fleming):
        assert tuple(round(value, 2) for value in boundary.critical_values) == OBF_TABLE
        assert tuple(round(level, 3) for level in boundary.nominal_levels) == OBF_LEVELS


def test_both_boundaries_spend_five_percent() -> None:
    """The article's grid gives 5.000 and 4.995 percent; the figure's is within a grid jump."""
    assert 100 * SUMMARY.article_pocock.crossing_probability == pytest.approx(5.000, abs=0.01)
    assert 100 * SUMMARY.article_obrien_fleming.crossing_probability == pytest.approx(
        4.995, abs=0.01
    )
    assert FIGURE.pocock.crossing_probability == pytest.approx(0.05, abs=1e-4)
    assert FIGURE.obrien_fleming.crossing_probability == pytest.approx(0.05, abs=3e-4)
    for figure, article in (
        (FIGURE.pocock, SUMMARY.article_pocock),
        (FIGURE.obrien_fleming, SUMMARY.article_obrien_fleming),
    ):
        np.testing.assert_allclose(figure.critical_values, article.critical_values, atol=2e-3)


def test_the_peeking_table_matches_the_article() -> None:
    """The recursion column: 5.1 percent for one look up to 27.5 percent for 28."""
    printed = {1: 5.1, 2: 8.3, 3: 10.8, 4: 12.6, 7: 16.6, 14: 22.0, 28: 27.5}

    assert {row.looks: round(100 * row.crossing_probability, 1) for row in SUMMARY.peeking} == (
        printed
    )


def test_the_monitoring_table_matches_the_article() -> None:
    """Daily, six-hourly, hourly and ten-minute checks: 27.5, 33.5, 38.6 and 40.8 percent."""
    printed = [("Daily", 28, 27.5), ("Every 6 hours", 112, 33.5), ("Hourly", 672, 38.6)]
    printed.append(("Every 10 minutes", 4032, 40.8))

    assert [
        (row.cadence, row.looks, round(100 * row.crossing_probability, 1))
        for row in SUMMARY.monitoring
    ] == printed
    rates = [row.crossing_probability for row in SUMMARY.monitoring]
    assert rates == sorted(rates)


def test_looks_that_cannot_stop_add_nothing() -> None:
    """Infinite bounds never cross; a zero bound at the first look always does."""
    assert crossing_probability([np.inf, np.inf], span=10.0) == 0.0
    assert crossing_probability([0.0, 1.96]) == pytest.approx(1.0, abs=1e-9)
    assert monitoring_rate(1, step=0.01, days=1) == crossing_probability([naive_critical_value()])


def test_the_mixture_boundary_is_where_the_ratio_reaches_twenty() -> None:
    """Substituting the boundary back into the likelihood ratio gives exactly 1 / alpha."""
    n = PER_DAY * np.arange(1, DAYS + 1)
    se = SIGMA * np.sqrt(2 / n)
    z = np.array(FIGURE.mixture)

    np.testing.assert_allclose(mixture_likelihood_ratio(z * se, se, TAU), 20.0, rtol=1e-10)
    np.testing.assert_allclose(mixture_boundary(), z, rtol=0)


def test_the_mixture_ratio_has_mean_one_under_the_null() -> None:
    """E[Lambda] = 1 when the estimate is centred on zero, whatever the variance.

    The standard errors span the design's first and last days. Beyond |z| = 35
    the integrand is below exp(-60), and the ratio alone would overflow.
    """
    for se in (0.178, 0.5, 0.943, 3.0):
        mean, _ = integrate.quad(
            lambda z, s=se: float(mixture_likelihood_ratio(z * s, s)[()]) * stats.norm.pdf(z),
            -35.0,
            35.0,
            points=[0.0],
            limit=200,
        )
        assert mean == pytest.approx(1.0, rel=1e-8)


def test_the_mixture_rule_holds_its_level_under_daily_looks() -> None:
    """Ville's inequality: over 28 daily looks the null rejection rate stays below 5 percent."""
    rng = np.random.default_rng(5)
    days = np.arange(1, DAYS + 1)
    z = np.cumsum(rng.standard_normal((20_000, DAYS)), axis=1) / np.sqrt(days)
    se = SIGMA * np.sqrt(2 / (PER_DAY * days))
    rate = float((mixture_likelihood_ratio(z * se, se) >= 20).any(axis=1).mean())

    assert 0 < rate < 0.05


def test_the_figure_claims() -> None:
    """1.96 flat, Pocock 2.49 flat, O'Brien-Fleming above 5 down to 2.06, mixture 5 down to 3."""
    mixture = np.array(FIGURE.mixture)
    obf = FIGURE.obrien_fleming.critical_values

    assert round(FIGURE.naive, 2) == 1.96
    assert obf[0] > 5 and round(obf[-1], 2) == 2.06
    assert list(obf) == sorted(obf, reverse=True)
    assert mixture[0] > 5
    assert list(mixture) == sorted(mixture, reverse=True)
    # Steep through the first week, then flat near three for the remaining three weeks.
    assert mixture[0] - mixture[6] > 5 * (mixture[6] - mixture[-1])
    assert np.all((mixture[7:] > 3.0) & (mixture[7:] < 3.25))


def test_invalid_inputs_are_rejected() -> None:
    """Empty, negative or undefined bounds, bad grids and impossible levels are refused."""
    with pytest.raises(ValueError):
        crossing_probability([])
    with pytest.raises(ValueError):
        crossing_probability([1.96, -1.0])
    with pytest.raises(ValueError):
        crossing_probability([np.nan])
    with pytest.raises(ValueError):
        crossing_probability([np.inf])
    with pytest.raises(ValueError):
        crossing_probability([1.96], step=0.0)
    with pytest.raises(ValueError):
        calibrate(pocock_shape(3), target=1.0)
    with pytest.raises(ValueError):
        pocock_shape(0)
    with pytest.raises(TypeError):
        obrien_fleming_shape(True)
    with pytest.raises(ValueError):
        naive_critical_value(0.0)
    with pytest.raises(ValueError):
        mixture_likelihood_ratio(0.1, 0.0)
    with pytest.raises(ValueError):
        mixture_boundary(tau=0.0)
    with pytest.raises(ValueError):
        monitoring_rate(0, step=0.01)


def test_figure_boundaries_type() -> None:
    """The payload carries the figure's boundaries in the renderer's input type."""
    assert isinstance(FIGURE, FigureBoundaries)
    assert len(FIGURE.mixture) == DAYS
