"""Check the post-selection intervals against the article, closed forms and simulation.

Every table in the article comes from the same computations as the figures, so
all four are pinned at their printed precision: the exact intervals and their
tail approximations, the width quantiles given selection, the coverage of the
ordinary interval, and the procedure table, whose randomised columns follow the
figure's simulation seeded at 20260918. The article's own coverage simulations
(20,000 selected draws per row) and its lasso stress test use generators it
does not give; coverage is checked here with separate seeded draws instead.

The exact interval is checked by inverting the pivot at known quantiles, the
tail approximation by its convergence as the distance shrinks, the width
quantiles against the 1 / w tail they imply, and the randomised pivot against
numerical integration of phi(x - mu) Phi(x - c).

One sentence does not hold as written. The article says the randomised interval
"was never wider than 5.51 over the whole range of observations". The widest
tabulated interval is 5.51, but it sits at the lower edge of the table
(x = -4), and below it the width keeps rising towards the splitting width:
5.52 at x = -6 and 5.54 at x = -12. What holds is the bound it cites, that the
randomised interval is never wider than the split one (5.5437).

The prose also says the exact width follows 3.69 / d "over two orders of
magnitude". That is loose: it stays within 10 percent of it only up to
d = 0.2, about 1.3 decades of distance (widths from 367 down to 20).

A finer point about the 1 / w tail: 3.69 / d is the tail approximation to the
lower limit, but the upper limit recedes as well, by ln(1 / 0.975) / d, so the
width itself is ln(39) / d = 3.66 / d close to the threshold. The tail of the
width is therefore h ln(39) / w = 8.69 / w at a mean of zero, not the article's
8.75 / w. The two differ by 0.7 percent, and the article presents its form as an
approximation, so both are checked: 8.75 as the approximation, 8.69 as the limit.
"""

from math import exp, log

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from blog_reproducibility.statistics.post_selection_intervals import (
    ALPHA,
    DISTANCE_POINTS,
    FIGURE_MEANS,
    OBSERVED,
    TABLE_OBSERVATIONS,
    THRESHOLD,
    conditional_cdf,
    conditional_density,
    example_payload,
    ordinary_coverage,
    procedure_curves,
    randomised_conditional_cdf,
    randomised_interval,
    randomised_selection_probability,
    randomised_width_table,
    selection_probability,
    selective_interval,
    split_width,
    tail_constant,
    tail_limit,
    threshold_hazard,
    width_exceedance,
    width_quantile,
)

SUMMARY = example_payload()
TABLE = randomised_width_table()
PROCEDURES = {row.mean: row for row in SUMMARY.procedures}


def test_the_interval_table_matches_the_article() -> None:
    """Ordinary and exact selective intervals, and the width, at seven observations."""
    printed = {
        2.01: (0.01, (0.05, 3.97), (-366.88, -0.17), 366.7),
        2.05: (0.05, (0.09, 4.01), (-71.74, 2.53), 74.3),
        2.20: (0.20, (0.24, 4.16), (-16.29, 3.66), 19.9),
        2.50: (0.50, (0.54, 4.46), (-4.99, 4.31), 9.3),
        3.00: (1.00, (1.04, 4.96), (-0.93, 4.93), 5.9),
        3.50: (1.50, (1.54, 5.46), (0.66, 5.46), 4.8),
        5.00: (3.00, (3.04, 6.96), (2.96, 6.96), 4.0),
    }
    assert tuple(row.observed for row in SUMMARY.observations) == TABLE_OBSERVATIONS
    for row in SUMMARY.observations:
        distance, ordinary, selective, width = printed[row.observed]
        assert round(row.distance, 2) == distance
        assert (round(row.ordinary.lower, 2), round(row.ordinary.upper, 2)) == ordinary
        assert (round(row.selective.lower, 2), round(row.selective.upper, 2)) == selective
        assert round(row.width, 1) == width


def test_the_tail_approximation_matches_the_article() -> None:
    """c - 3.69 / d: -71.78, -366.89 and -16.44 in the prose, and the code block's output."""
    assert round(SUMMARY.tail_constant, 2) == 3.69
    assert tail_constant() == pytest.approx(log(40.0))
    approximations = {row.observed: round(row.tail_limit, 2) for row in SUMMARY.observations}
    assert approximations[2.05] == -71.78
    assert approximations[2.01] == -366.89
    assert approximations[2.20] == -16.44
    # The code block prints the exact interval and the approximation at four observations.
    printed = {
        2.05: (-71.74, 2.53, -71.78),
        2.50: (-4.99, 4.31, -5.38),
        3.00: (-0.93, 4.93, -1.69),
        3.50: (0.66, 5.46, -0.46),
    }
    for row in SUMMARY.observations:
        if row.observed in printed:
            low, high, approx = printed[row.observed]
            assert (round(row.selective.lower, 2), round(row.selective.upper, 2)) == (low, high)
            assert round(row.tail_limit, 2) == approx


def test_the_tail_approximation_improves_as_the_distance_shrinks() -> None:
    """The gap to the exact lower limit, relative to c - mu_L, falls towards zero."""
    errors = []
    for distance in (0.5, 0.2, 0.05, 0.01, 0.002):
        exact = selective_interval(THRESHOLD + distance).lower
        approx = tail_limit(THRESHOLD + distance)
        errors.append(abs(exact - approx) / (THRESHOLD - exact))
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-3


def test_the_prose_numbers() -> None:
    """[-71.7, 2.53]; selection at 10^-1183; 2.01 outside its own interval; far above, no change."""
    rows = {row.observed: row for row in SUMMARY.observations}
    assert round(rows[2.05].selective.lower, 1) == -71.7
    assert round(SUMMARY.log10_selection_at_lower_limit) == -1183
    assert not rows[2.01].selective.lower <= 2.01 <= rows[2.01].selective.upper
    # The upper limits agree from about x = 3; at x = 5 the intervals are nearly identical.
    for x in (3.0, 3.5, 5.0):
        assert abs(rows[x].selective.upper - rows[x].ordinary.upper) < 0.03
    assert abs(rows[5.0].selective.lower - rows[5.0].ordinary.lower) < 0.1


def test_the_observation_is_typical_under_all_three_means() -> None:
    """At -20 the excess is exponential with mean 0.045 and 2.05 sits at its 67th percentile."""
    positions = {row.mean: row for row in SUMMARY.positions}
    assert round(positions[-20.0].exponential_mean, 3) == 0.045
    assert round(100 * positions[-20.0].percentile) == 67
    assert round(100 * positions[1.0].percentile) == 7
    # The exponential approximation is close at -20 and poor at 1, as the article says.
    distance = OBSERVED - THRESHOLD
    assert 1 - exp(-22 * distance) == pytest.approx(positions[-20.0].percentile, abs=0.002)
    assert 1 - exp(-1 * distance) < positions[1.0].percentile - 0.02
    # Typical: every one of the three means lies inside the exact interval.
    interval = selective_interval(OBSERVED)
    for position in SUMMARY.positions:
        assert ALPHA / 2 < position.percentile < 1 - ALPHA / 2
        assert interval.lower < position.mean < interval.upper


def test_the_density_piles_up_at_the_threshold() -> None:
    """The density integrates to one and starts at the hazard, higher for lower means."""
    starts = []
    for mean in (1.0, -5.0, -20.0):
        grid = np.linspace(THRESHOLD, THRESHOLD + 40.0, 400_001)
        density = conditional_density(grid, mean)
        assert float(np.trapezoid(density, grid)) == pytest.approx(1.0, abs=1e-6)
        assert density[0] == pytest.approx(threshold_hazard(mean), rel=1e-12)
        starts.append(float(density[0]))
    assert starts == sorted(starts)
    assert starts[-1] < 23  # inside the left panel's axis
    assert conditional_density([THRESHOLD - 0.1], 0.0)[0] == 0.0


def test_the_pivot_inverts_exactly() -> None:
    """At the 2.5% and 97.5% conditional quantiles of X, the limits land on the mean."""
    for mean in (-3.0, 0.0, 1.5, 4.0):
        tail = norm.sf(THRESHOLD - mean)
        lowest = mean + norm.isf((1 - ALPHA / 2) * tail)  # F_mu(x) = 0.025
        highest = mean + norm.isf((ALPHA / 2) * tail)  # F_mu(x) = 0.975
        assert float(conditional_cdf(mean, lowest)) == pytest.approx(ALPHA / 2, rel=1e-9)
        assert selective_interval(lowest).upper == pytest.approx(mean, abs=1e-8)
        assert selective_interval(highest).lower == pytest.approx(mean, abs=1e-8)


def test_conditional_cdf_is_exact_where_the_textbook_form_fails() -> None:
    """The textbook ratio agrees at moderate means and is 0 / 0 at the article's lower limit."""
    for mean, x in ((0.0, 2.3), (1.5, 3.0), (-2.0, 2.1)):
        textbook = (norm.cdf(x - mean) - norm.cdf(THRESHOLD - mean)) / norm.sf(THRESHOLD - mean)
        assert float(conditional_cdf(mean, x)) == pytest.approx(textbook, rel=1e-9)
    with np.errstate(invalid="ignore"):
        numerator = np.float64(norm.cdf(OBSERVED + 71.74) - norm.cdf(THRESHOLD + 71.74))
        textbook = numerator / np.float64(1 - norm.cdf(THRESHOLD + 71.74))
    assert np.isnan(textbook)
    assert float(conditional_cdf(-71.74, OBSERVED)) == pytest.approx(0.975, abs=1e-3)
    assert float(conditional_cdf(0.0, THRESHOLD)) == 0.0


def test_the_width_quantile_table_matches_the_article() -> None:
    """Selection probability, median, 90th and 99th percentile widths, share wider than 20."""
    printed = {
        0.0: (0.023, 14.96, 84.1, 865.9, 38.8),
        1.0: (0.159, 10.85, 55.1, 557.1, 27.4),
        2.0: (0.500, 7.51, 30.6, 292.7, 15.8),
        3.0: (0.841, 5.32, 13.9, 108.0, 6.3),
        4.0: (0.977, 4.31, 6.6, 25.1, 1.3),
    }
    for row in SUMMARY.widths:
        selected, median, p90, p99, wide = printed[row.mean]
        assert round(row.selected, 3) == selected
        assert round(row.median, 2) == median
        assert round(row.p90, 1) == p90
        assert round(row.p99, 1) == p99
        assert round(100 * row.wider_than_20, 1) == wide


def test_the_width_quantiles_are_quantiles() -> None:
    """The share of selected draws past each quantile's observation matches its level."""
    rng = np.random.default_rng(3)
    mean = 2.0
    x = mean + rng.standard_normal(200_000)
    selected = x[x > THRESHOLD]
    for level in (0.5, 0.9):
        cut = mean + norm.isf(level * norm.sf(THRESHOLD - mean))
        assert float(np.mean(selected > cut)) == pytest.approx(level, abs=0.006)
        assert selective_interval(cut).width == pytest.approx(width_quantile(mean, level))
    # The width is wider than 20 exactly when the draw lands within 0.1994 of the threshold.
    assert selective_interval(THRESHOLD + 0.1994182).width == pytest.approx(20.0, abs=1e-5)
    share = float(np.mean(selected < THRESHOLD + 0.1994182))
    assert share == pytest.approx(SUMMARY.widths[2].wider_than_20, abs=0.006)


def test_the_width_has_a_one_over_w_tail() -> None:
    """w P(width > w) settles at h(c - mu) ln(39), within one percent of the article's 8.75."""
    assert round(SUMMARY.null_hazard, 2) == 2.37
    approximation = SUMMARY.null_hazard * SUMMARY.tail_constant
    assert round(approximation, 2) == 8.75
    # The upper limit recedes too, as ln(1 / 0.975) / d, so the width is ln(39) / d.
    limit = SUMMARY.null_hazard * log((1 - ALPHA / 2) / (ALPHA / 2))
    assert round(limit, 2) == 8.69
    scaled = [w * width_exceedance(0.0, w) for w in (100.0, 1000.0, 10_000.0)]
    gaps = [abs(value / limit - 1) for value in scaled]
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 5e-4
    assert abs(scaled[-1] / approximation - 1) < 0.01
    # The 99th percentile reads the same tail the other way round.
    assert 0.01 * SUMMARY.widths[0].p99 == pytest.approx(limit, rel=0.005)


def test_the_coverage_table_matches_the_article() -> None:
    """Coverage of X +/- 1.96 among reported observations; zero for any mean below 0.04."""
    printed = {0.0: 0.0, 0.5: 62.6, 1.0: 84.2, 1.5: 91.9, 2.0: 95.0, 3.0: 97.0}
    for row in SUMMARY.coverage:
        assert round(100 * row.coverage, 1) == printed[row.mean]
    assert round(SUMMARY.zero_coverage_below, 2) == 0.04
    for mean in (-50.0, -1.0, 0.0, 0.0399):
        assert ordinary_coverage(mean) == 0.0
    assert ordinary_coverage(0.05) > 0.0


def test_ordinary_coverage_matches_a_simulation() -> None:
    """Seeded draws kept above 2 are covered by X +/- 1.96 at the closed-form rate."""
    rng = np.random.default_rng(11)
    for mean in (0.5, 1.0, 3.0):
        x = mean + rng.standard_normal(200_000)
        kept = x[x > THRESHOLD]
        simulated = float(np.mean(np.abs(kept - mean) <= 1.96))
        assert simulated == pytest.approx(ordinary_coverage(mean), abs=0.012)


def test_the_procedure_table_matches_the_article() -> None:
    """Selection rates, hard and randomised widths given selection, and the split width."""
    printed = {
        0.0: (2.3, 7.9, 14.96, 84.1, 5.13, 5.31),
        1.0: (15.9, 24.0, 10.85, 55.1, 4.96, 5.22),
        2.0: (50.0, 50.0, 7.51, 30.6, 4.74, 5.08),
        3.0: (84.1, 76.0, 5.32, 13.9, 4.47, 4.87),
        4.0: (97.7, 92.1, 4.31, 6.6, 4.22, 4.60),
    }
    for mean, (hard, soft, median, p90, soft_median, soft_p90) in printed.items():
        row = PROCEDURES[mean]
        assert round(100 * row.hard_selected, 1) == hard
        assert round(100 * row.randomised_selected, 1) == soft
        assert (round(row.hard_median, 2), round(row.hard_p90, 1)) == (median, p90)
        assert (round(row.randomised_median, 2), round(row.randomised_p90, 2)) == (
            soft_median,
            soft_p90,
        )
        assert round(row.split, 2) == 5.54


def test_what_randomisation_costs_in_selection() -> None:
    """Over three times as many false leads under the null; 8% of real effects missed at 4."""
    null, strong = PROCEDURES[0.0], PROCEDURES[4.0]
    assert null.randomised_selected / null.hard_selected > 3
    assert round(100 * (1 - strong.randomised_selected)) == 8
    assert round(100 * (1 - strong.hard_selected)) == 2


def test_the_randomised_interval_at_the_observation() -> None:
    """[-1.31, 3.60], width 4.92, where the hard threshold gave 74.3."""
    interval = SUMMARY.randomised_at_observed
    assert (round(interval.lower, 2), round(interval.upper, 2)) == (-1.31, 3.60)
    assert round(interval.width, 2) == 4.92
    assert interval == randomised_interval(OBSERVED)


def test_the_randomised_width_is_bounded_by_the_split_width() -> None:
    """The widest tabulated interval is 5.51, at the table's edge; below it the width nears 5.54."""
    assert split_width() == pytest.approx(2 * 1.96 * np.sqrt(2.0))
    widths = TABLE.widths
    assert round(SUMMARY.widest_randomised, 2) == 5.51
    assert int(np.argmax(widths)) == 0 and TABLE.observations[0] == -4.0
    assert np.all(widths < split_width())
    far_below = randomised_interval(-12.0).width
    assert 5.53 < far_below < split_width()


def test_randomised_cdf_matches_numerical_integration() -> None:
    """The grid sum agrees with quadrature to 2e-3, and its normaliser has a closed form."""
    for mean, x in ((0.0, 1.0), (2.0, 2.05), (-3.0, 0.0), (4.0, 5.0)):
        numerator = integrate.quad(
            lambda t, m=mean: norm.pdf(t - m) * norm.cdf(t - THRESHOLD), -np.inf, x
        )[0]
        total = integrate.quad(
            lambda t, m=mean: norm.pdf(t - m) * norm.cdf(t - THRESHOLD), -np.inf, np.inf
        )[0]
        assert total == pytest.approx(randomised_selection_probability(mean), rel=1e-7)
        exact = numerator / total
        assert randomised_conditional_cdf(mean, x) == pytest.approx(exact, abs=2e-3)


def test_randomised_intervals_cover_given_selection() -> None:
    """Seeded draws selected on X + omega > 2 are covered about 95 percent of the time."""
    rng = np.random.default_rng(17)
    for mean in (0.0, 2.0, 4.0):
        x = mean + rng.standard_normal(200_000)
        kept = x[x + rng.standard_normal(x.size) > THRESHOLD]
        lower = np.interp(kept, TABLE.observations, TABLE.lower)
        upper = np.interp(kept, TABLE.observations, TABLE.upper)
        coverage = float(np.mean((lower <= mean) & (mean <= upper)))
        assert coverage == pytest.approx(0.95, abs=0.008)


def test_the_simulated_selection_rates_match_the_closed_form() -> None:
    """The figure's 400,000 draws per mean select at P(X + omega > 2) = sf((2 - mu) / sqrt 2)."""
    curves = SUMMARY.procedure_curves
    for mean, share in zip(curves.means, curves.randomised_selected, strict=True):
        assert share == pytest.approx(randomised_selection_probability(mean), abs=0.003)


def test_the_width_follows_one_over_the_distance() -> None:
    """Within 1% of 3.69 / d up to d = 0.05, 10% up to 0.2, and within 0.01 of 3.92 at d = 4."""
    curve = SUMMARY.width_curve
    distances, widths = np.array(curve.distances), np.array(curve.widths)
    assert distances.size == DISTANCE_POINTS
    assert (distances[0], distances[-1]) == pytest.approx((0.01, 4.0))
    assert np.all(np.diff(widths) < 0)
    ratio = widths * distances / tail_constant()
    assert np.all(np.abs(ratio[distances <= 0.05] - 1) < 0.01)
    assert np.all(np.abs(ratio[distances <= 0.2] - 1) < 0.1)
    assert np.any(np.abs(ratio[distances <= 0.3] - 1) > 0.1)
    assert widths[-1] == pytest.approx(3.92, abs=0.01)
    # Very close to the threshold the width itself is ln(39) / d, the lower limit ln(40) / d.
    closest = 1e-4
    assert selective_interval(THRESHOLD + closest).width * closest == pytest.approx(
        log((1 - ALPHA / 2) / (ALPHA / 2)), rel=1e-3
    )


def test_the_figure_claims() -> None:
    """Hard: about 15 and 84 at zero, 4.3 at four. Randomised: 4.2 to 5.3. Splitting: 5.54."""
    curves = SUMMARY.procedure_curves
    assert curves.means == FIGURE_MEANS
    assert tuple(np.linspace(0.0, 4.0, 17)) == FIGURE_MEANS
    assert round(curves.hard_median[0]) == 15
    assert round(curves.hard_p90[0]) == 84
    assert round(curves.hard_median[-1], 1) == 4.3
    assert min(curves.randomised_median) >= 4.2
    assert max(curves.randomised_median) <= 5.3
    assert round(max(curves.randomised_p90), 1) == 5.3
    assert round(curves.split, 2) == 5.54
    # What the hard threshold costs: wider than randomised selection at every mean.
    for hard, soft in zip(curves.hard_median, curves.randomised_median, strict=True):
        assert hard > soft
    assert min(curves.hard_p90) > curves.split
    assert all(width > 2 * 1.96 for width in curves.randomised_median)


def test_procedure_curves_are_deterministic() -> None:
    """The same seed gives the same randomised widths; another seed does not."""
    first = procedure_curves(TABLE, means=(0.0, 2.0), draws=20_000)
    again = procedure_curves(TABLE, means=(0.0, 2.0), draws=20_000)
    other = procedure_curves(TABLE, means=(0.0, 2.0), draws=20_000, seed=1)
    assert first == again
    assert first.randomised_median != other.randomised_median
    assert first.hard_median == other.hard_median


def test_invalid_inputs_are_rejected() -> None:
    """Observations at or below the threshold, bad levels, spreads, draws and widths."""
    with pytest.raises(ValueError, match="exceed"):
        selective_interval(THRESHOLD)
    with pytest.raises(ValueError, match="exceed"):
        tail_limit(1.5)
    with pytest.raises(ValueError):
        selective_interval(2.5, alpha=0.0)
    with pytest.raises(TypeError):
        selective_interval(True)
    with pytest.raises(ValueError):
        width_quantile(0.0, 1.0)
    with pytest.raises(ValueError, match="width"):
        width_exceedance(0.0, 3.0)
    with pytest.raises(ValueError):
        randomised_interval(2.0, sd=0.0)
    with pytest.raises(ValueError):
        procedure_curves(TABLE, draws=0)
    with pytest.raises(ValueError):
        selection_probability(float("nan"))
