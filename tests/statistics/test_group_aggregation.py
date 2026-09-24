"""Check the group aggregation model against the website loop, closed forms and the article.

The figure's simulation (600 groups, a generator seeded at 31 at every point)
runs once here, in under a second, and its 66 correlations are pinned; a
transcription of the website loop confirms that drawing each group size once
and combining it with every intraclass correlation gives the same values
exactly. They lie within three standard errors of the closed form on Fisher's
``z`` scale (the largest gap is 2.9), and the figure's claims hold: each curve
starts near the individual correlation, all three end within about one standard
error of 0.9, the smaller share lies below the larger at every size, and the
individual correlation of the same populations stays near its closed form at
every size. The measured curves are not monotone: at 5 and 10 percent they dip
between 2 and 3 people and between 75 and 104, because every group size draws
new person parts, and the tests say so.

The article's tables come from other seeds and 1,000 groups and are not pinned.
Their "predicted" columns are the closed form and are reproduced at the printed
precision, and every simulated value in all five tables lies within three
standard errors of its closed form. The Fisher interval follows from the printed
correlation of 0.767 on 1,000 groups. One statement does not hold as written: the
article calls that interval "half a percentage point wide either side", but
0.740 to 0.791 is 2.5 points either side of 0.767; the tests pin what is true.
"""

from math import atanh, sqrt

import numpy as np
import pytest

from blog_reproducibility.statistics.group_aggregation import (
    BETWEEN_CORRELATION,
    GROUPS,
    INTRACLASS_CORRELATIONS,
    SIZE_TABLE_GROUP_SIZES,
    aggregation_curves,
    combine_parts,
    draw_parts,
    example_payload,
    fisher_interval,
    group_mean_correlation,
    group_size_for_correlation,
    group_sizes,
    individual_correlation,
    levels_row,
    population,
    predicted_group_mean_correlation,
    predicted_group_mean_slope,
    predicted_individual_correlation,
)

SUMMARY = example_payload()
CURVES = {curve.intraclass_correlation: curve for curve in SUMMARY.curves}

FIGURE_SIZES: tuple[int, ...] = (
    2, 3, 4, 5, 7, 10, 14, 20, 28, 39, 54, 75, 104, 144, 200, 278, 386, 537, 746, 1036, 1439, 2000,
)  # fmt: skip
# The measured correlation between group means, to four decimals.
FIGURE_MEASURED = {
    0.05: (
        0.1604, 0.0918, 0.1073, 0.2444, 0.2683, 0.3562, 0.3933, 0.4833, 0.5910, 0.6196, 0.7153,
        0.7696, 0.7685, 0.7946, 0.8370, 0.8592, 0.8777, 0.8868, 0.8906, 0.8971, 0.8990, 0.9012,
    ),
    0.10: (
        0.2414, 0.2033, 0.2365, 0.3756, 0.4231, 0.5212, 0.5604, 0.6385, 0.7201, 0.7486, 0.8049,
        0.8414, 0.8370, 0.8521, 0.8754, 0.8864, 0.8956, 0.9003, 0.9007, 0.9044, 0.9047, 0.9055,
    ),
    0.25: (
        0.4332, 0.4463, 0.4990, 0.6054, 0.6579, 0.7292, 0.7506, 0.7932, 0.8341, 0.8508, 0.8722,
        0.8899, 0.8839, 0.8897, 0.8994, 0.9029, 0.9062, 0.9079, 0.9069, 0.9085, 0.9082, 0.9082,
    ),
}  # fmt: skip

# The article's simulated tables, as printed (1,000 groups, seeds 2 to 10).
ARTICLE_SIZE_TABLE = {
    5: (0.095, 0.338),
    20: (0.099, 0.637),
    100: (0.091, 0.823),
    1000: (0.093, 0.898),
}
ARTICLE_SIGN_TABLE = {
    (0.80, -0.30): (-0.130, 0.773),
    (-0.60, 0.40): (0.246, -0.591),
    (0.00, 0.50): (0.423, 0.013),
    (0.70, 0.70): (0.700, 0.708),
}
ARTICLE_SLOPE_TABLE = {
    0.05: (-0.147, 0.697),
    0.10: (-0.098, 0.740),
    0.25: (0.048, 0.769),
    0.50: (0.292, 0.780),
}
# Single slope, within-group slope and between-group slope.
ARTICLE_TWO_LEVEL = {(0.80, -0.30): (-0.135, -0.299, 0.770), (0.00, 0.50): (0.424, 0.499, 0.002)}
ARTICLE_GROUPS = 1000


def _website_curve(icc: float, sizes: tuple[int, ...]) -> list[float]:
    """Transcribe the website generator's loop for one intraclass correlation."""
    rho_b, rho_w = 0.90, 0.0
    measured = []
    for m in sizes:
        rng = np.random.default_rng(31)
        gb = rng.multivariate_normal([0, 0], [[1, rho_b], [rho_b, 1]], 600)
        wi = rng.multivariate_normal([0, 0], [[1, rho_w], [rho_w, 1]], (600, int(m)))
        x = np.sqrt(icc) * gb[:, None, 0] + np.sqrt(1 - icc) * wi[:, :, 0]
        y = np.sqrt(icc) * gb[:, None, 1] + np.sqrt(1 - icc) * wi[:, :, 1]
        measured.append(float(np.corrcoef(x.mean(axis=1), y.mean(axis=1))[0, 1]))
    return measured


def _fisher_gap(measured: float, expected: float, groups: int) -> float:
    """Distance between two correlations in standard errors on Fisher's z scale."""
    return abs(atanh(measured) - atanh(expected)) * sqrt(groups - 3)


def _person_se(share: float, between: float, within: float, groups: int, size: int) -> float:
    """Approximate standard error of a correlation or slope across people.

    The sample covariance is a group term from ``groups`` pairs of group parts, a
    person term from ``groups * size`` pairs and two cross terms; both variables
    have unit variance, so the correlation and the slope are close to it.
    """
    people = groups * size
    return sqrt(
        share**2 * (1 + between**2) / groups
        + (1 - share) ** 2 * (1 + within**2) / people
        + 2 * share * (1 - share) / people
    )


def _slope_se(correlation: float, groups: int) -> float:
    """Standard error of a slope between group means with equal variances."""
    return sqrt((1 - correlation**2) / (groups - 2))


def test_the_group_sizes_are_the_websites() -> None:
    """Twenty-two rounded sizes from 2 to 2,000, all distinct."""
    assert group_sizes() == FIGURE_SIZES
    assert group_sizes(1, 10, 30) == tuple(range(1, 11))


def test_the_figure_correlations_are_reproduced() -> None:
    """Every measured correlation between group means, to four decimals."""
    assert tuple(CURVES) == INTRACLASS_CORRELATIONS
    for share, values in FIGURE_MEASURED.items():
        assert CURVES[share].group_sizes == FIGURE_SIZES
        assert CURVES[share].measured == pytest.approx(values, abs=5e-5)


def test_one_draw_per_size_matches_the_website_loop() -> None:
    """Drawing each size once and reusing it for every share gives the loop's values exactly."""
    sizes = (2, 7, 104, 2000)
    curves = aggregation_curves(sizes=sizes)
    for curve in curves:
        assert list(curve.measured) == _website_curve(curve.intraclass_correlation, sizes)


def test_the_prediction_is_the_websites() -> None:
    """The dotted curves are the website's ``rho_B lambda / (lambda + (1 - lambda) / m)``."""
    for curve in SUMMARY.curves:
        share = curve.intraclass_correlation
        for size, value in zip(curve.group_sizes, curve.predicted, strict=True):
            website = (BETWEEN_CORRELATION * share) / (share + (1 - share) / size)
            assert value == pytest.approx(website, rel=1e-12)


def test_the_article_closed_forms() -> None:
    """The article's predicted columns: 0.090 for individuals, 0.321 to 0.892 for group means."""
    for row in SUMMARY.size_rows:
        assert round(row.individual_correlation, 3) == 0.090
    assert tuple(row.group_size for row in SUMMARY.size_rows) == SIZE_TABLE_GROUP_SIZES
    assert tuple(round(row.group_mean_correlation, 3) for row in SUMMARY.size_rows) == (
        0.321,
        0.621,
        0.826,
        0.892,
    )


def test_the_article_simulations_agree_with_the_closed_forms() -> None:
    """Every simulated value in the article lies within three standard errors of its closed form."""
    for size, (person, means) in ARTICLE_SIZE_TABLE.items():
        row = levels_row(0.90, 0.0, 0.10, size)
        assert abs(person - row.individual_correlation) < 3 * _person_se(
            0.10, 0.90, 0.0, ARTICLE_GROUPS, size
        )
        assert _fisher_gap(means, row.group_mean_correlation, ARTICLE_GROUPS) < 3

    for (between, within), (person, means) in ARTICLE_SIGN_TABLE.items():
        row = levels_row(between, within, 0.15, 200)
        se = _person_se(0.15, between, within, ARTICLE_GROUPS, 200)
        assert abs(person - row.individual_correlation) < 3 * se
        assert _fisher_gap(means, row.group_mean_correlation, ARTICLE_GROUPS) < 3

    for share, (person, means) in ARTICLE_SLOPE_TABLE.items():
        row = levels_row(0.80, -0.20, share, 200)
        assert abs(person - row.individual_correlation) < 3 * _person_se(
            share, 0.80, -0.20, ARTICLE_GROUPS, 200
        )
        assert abs(means - row.group_mean_slope) < 3 * _slope_se(means, ARTICLE_GROUPS)

    for (between, within), (single, inside, across) in ARTICLE_TWO_LEVEL.items():
        row = levels_row(between, within, 0.15, 200)
        people = ARTICLE_GROUPS * 200
        assert abs(single - row.individual_correlation) < 3 * _person_se(
            0.15, between, within, ARTICLE_GROUPS, 200
        )
        assert abs(inside - within) < 3 * sqrt((1 - within**2) / (people - ARTICLE_GROUPS))
        assert abs(across - row.group_mean_slope) < 3 * _slope_se(across, ARTICLE_GROUPS)


def test_the_signs_the_article_describes() -> None:
    """Aggregation reverses the first two populations, erases the third and spares the fourth."""
    reversed_, mirrored, no_signal, agreeing = SUMMARY.sign_rows
    assert reversed_.individual_correlation < 0 < reversed_.group_mean_correlation
    assert mirrored.group_mean_correlation < 0 < mirrored.individual_correlation
    assert abs(no_signal.group_mean_correlation) < 0.02 < 0.4 < no_signal.individual_correlation
    assert agreeing.group_mean_correlation == pytest.approx(agreeing.individual_correlation)
    # The slope table: the individual slope changes sign, the group-mean slope stays near 0.75.
    slopes = [row.individual_correlation for row in SUMMARY.slope_rows]
    assert slopes[0] < slopes[1] < 0 < slopes[2] < slopes[3]
    for row in SUMMARY.slope_rows:
        assert 0.70 < row.group_mean_slope < 0.80


def test_the_article_fisher_interval() -> None:
    """0.740 to 0.791 from 0.767 on 1,000 groups: 2.5 points either side, not half a point."""
    low, high = fisher_interval(0.767, ARTICLE_GROUPS)
    assert (round(low, 3), round(high, 3)) == (0.740, 0.791)
    assert not low <= -0.133 <= high
    for side in (0.767 - low, high - 0.767):
        assert 0.024 < side < 0.028
        assert side > 4 * 0.005


def test_every_curve_climbs_towards_the_group_correlation() -> None:
    """The alt text: from near the individual correlation towards 0.9 as groups grow."""
    for curve in SUMMARY.curves:
        first, last = curve.measured[0], curve.measured[-1]
        assert first - curve.individual < BETWEEN_CORRELATION - first
        assert last - first > 0.45
        assert _fisher_gap(last, BETWEEN_CORRELATION, GROUPS) < 1.2
        assert list(curve.predicted) == sorted(curve.predicted)
        for measured, predicted in zip(curve.measured, curve.predicted, strict=True):
            assert _fisher_gap(measured, predicted, GROUPS) < 3
    # Every size draws new person parts, so the measured curves dip in places.
    dips = {
        share: [
            curve.group_sizes[j + 1]
            for j in range(len(curve.measured) - 1)
            if curve.measured[j + 1] < curve.measured[j]
        ]
        for share, curve in CURVES.items()
    }
    assert dips == {0.05: [3, 104], 0.10: [3, 104], 0.25: [104, 746, 1439]}


def test_smaller_group_shares_need_larger_groups() -> None:
    """The alt text: the smaller the share, the larger the groups; 152, 72 and 24 to reach 0.8."""
    for j in range(len(FIGURE_SIZES)):
        measured = [curve.measured[j] for curve in SUMMARY.curves]
        predicted = [curve.predicted[j] for curve in SUMMARY.curves]
        assert measured == sorted(measured)
        assert predicted == sorted(predicted)
    sizes = [group_size_for_correlation(0.8, share, 0.9, 0.0) for share in INTRACLASS_CORRELATIONS]
    assert sizes == pytest.approx([152, 72, 24])
    for share, size in zip(INTRACLASS_CORRELATIONS, sizes, strict=True):
        assert predicted_group_mean_correlation(share, share, 0.9, 0.0, round(size)) == (
            pytest.approx(0.8)
        )


def test_the_same_people_keep_their_correlation() -> None:
    """The title: the individual correlation of the figure's populations does not climb."""
    for size in (2, 20, 200, 2000):
        x, y = population(np.random.default_rng(31), GROUPS, size, 0.10, 0.10)
        person = individual_correlation(x, y)
        assert abs(person - 0.09) < 3 * _person_se(0.10, 0.9, 0.0, GROUPS, size)
        assert group_mean_correlation(x, y) == CURVES[0.10].measured[FIGURE_SIZES.index(size)]


def test_the_closed_form_matches_a_large_simulation() -> None:
    """Unequal shares and both correlations nonzero, on 40,000 groups of five."""
    rng = np.random.default_rng(5)
    x, y = population(rng, 40_000, 5, 0.3, 0.6, between=0.5, within=-0.4)
    expected_person = predicted_individual_correlation(0.3, 0.6, 0.5, -0.4)
    expected_means = predicted_group_mean_correlation(0.3, 0.6, 0.5, -0.4, 5)
    assert individual_correlation(x, y) == pytest.approx(expected_person, abs=0.01)
    assert _fisher_gap(group_mean_correlation(x, y), expected_means, 40_000) < 3
    slope = float(np.polyfit(x.mean(axis=1), y.mean(axis=1), 1)[0])
    assert slope == pytest.approx(predicted_group_mean_slope(0.3, 0.6, 0.5, -0.4, 5), abs=0.02)


def test_limiting_cases() -> None:
    """One person per group is the individual; large groups tend to the group parts."""
    for share in (0.0, 0.05, 0.5, 1.0):
        for between, within in ((0.9, 0.0), (0.8, -0.3), (-0.6, 0.4)):
            person = predicted_individual_correlation(share, share, between, within)
            assert predicted_group_mean_correlation(share, share, between, within, 1) == (
                pytest.approx(person)
            )
            assert predicted_group_mean_slope(share, share, between, within, 1) == (
                pytest.approx(person)
            )
            if share > 0:
                limit = predicted_group_mean_correlation(share, share, between, within, 10**9)
                assert limit == pytest.approx(between, abs=1e-6)
    # Equal correlations at both levels survive any aggregation.
    for size in (1, 5, 200, 10_000):
        assert predicted_group_mean_correlation(0.2, 0.2, 0.7, 0.7, size) == pytest.approx(0.7)
    # Symmetric in the two variables.
    assert predicted_group_mean_correlation(0.1, 0.4, 0.8, 0.2, 7) == pytest.approx(
        predicted_group_mean_correlation(0.4, 0.1, 0.8, 0.2, 7)
    )


def test_the_parts_are_combined_as_the_article_writes() -> None:
    """A share of zero leaves only the person parts; a share of one only the group parts."""
    group_parts, person_parts = draw_parts(np.random.default_rng(1), 4, 3)
    x, y = combine_parts(group_parts, person_parts, 0.0, 1.0)
    assert np.array_equal(x, person_parts[:, :, 0])
    assert np.array_equal(y, np.repeat(group_parts[:, None, 1], 3, axis=1))


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a curve; another seed does not."""

    def small(seed: int) -> object:
        return aggregation_curves(seed, groups=50, intraclass_correlations=(0.2,), sizes=(3, 9))

    assert small(31) == small(31)
    assert small(31) != small(32)


def test_invalid_inputs_are_rejected() -> None:
    """Shares outside [0, 1], correlations outside [-1, 1], bad sizes and arrays are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        predicted_individual_correlation(1.2, 0.1, 0.9, 0.0)
    with pytest.raises(ValueError):
        predicted_group_mean_correlation(0.1, 0.1, 1.5, 0.0, 5)
    with pytest.raises(ValueError):
        predicted_group_mean_correlation(0.1, 0.1, 0.9, 0.0, 0)
    with pytest.raises(TypeError):
        predicted_group_mean_correlation(0.1, 0.1, 0.9, 0.0, True)
    with pytest.raises(ValueError):
        draw_parts(rng, 1, 5)
    with pytest.raises(ValueError):
        group_size_for_correlation(0.95, 0.1, 0.9, 0.0)
    with pytest.raises(ValueError):
        group_size_for_correlation(0.5, 0.0, 0.9, 0.0)
    with pytest.raises(ValueError):
        fisher_interval(1.0, 100)
    with pytest.raises(ValueError):
        fisher_interval(0.5, 3)
    with pytest.raises(ValueError):
        individual_correlation(np.zeros((3, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        combine_parts(np.zeros((3, 2)), np.zeros((4, 2, 2)), 0.1, 0.1)
    with pytest.raises(ValueError):
        aggregation_curves(intraclass_correlations=())
    with pytest.raises(ValueError):
        aggregation_curves(sizes=())
    with pytest.raises(ValueError):
        group_sizes(10, 5)
