"""Check the winner's-curse optimism against closed forms and the figure's claims.

The expected maximum of k normals is checked against its exact values for two
and three variables and against brute-force sampling. With equal candidates the
simulated optimism matches the closed form. The figure's simulation (seed 0) is
then checked against the claims in its alt text. The article's optimism, regret
and test-set tables use 4,000 replications and a different order of draws, so
they are not pinned; the closed-form numbers its prose quotes are.
"""

from math import pi, sqrt

import numpy as np
import pytest

from blog_reproducibility.machine_learning.winners_curse import (
    CANDIDATES,
    VALIDATION_SIZES,
    OptimismCurve,
    equal_candidates_optimism,
    example_payload,
    expected_maximum_of_normals,
    optimism_curves,
    selection_optimism,
    validation_standard_error,
)

SUMMARY = example_payload()
CURVES = {curve.validation_size: curve for curve in SUMMARY.curves}


def _optimism(candidates: int, validation_size: int) -> float:
    curve = CURVES[validation_size]
    return curve.optimism[curve.candidates.index(candidates)]


def test_expected_maximum_has_its_exact_small_values() -> None:
    """E[max] is 0 for one normal, 1/sqrt(pi) for two and 3/(2 sqrt(pi)) for three."""
    assert expected_maximum_of_normals(1) == 0.0
    assert expected_maximum_of_normals(2) == pytest.approx(1 / sqrt(pi))
    assert expected_maximum_of_normals(3) == pytest.approx(3 / (2 * sqrt(pi)))


def test_expected_maximum_matches_sampling() -> None:
    """The integral agrees with the mean maximum of simulated normals."""
    draws = np.random.default_rng(1).standard_normal((200_000, 10)).max(axis=1)
    assert expected_maximum_of_normals(10) == pytest.approx(draws.mean(), abs=0.01)


def test_closed_form_numbers_match_the_article() -> None:
    """E[max] of 1.16, 1.87 and 2.51; a standard error of 1.26 points; optimism of 3.2."""
    assert [(k, round(value, 2)) for k, value in SUMMARY.expected_maxima] == [
        (5, 1.16),
        (20, 1.87),
        (100, 2.51),
    ]
    assert round(SUMMARY.standard_error_points, 2) == 1.26
    assert round(SUMMARY.equal_candidates_optimism_points, 1) == 3.2


def test_equal_candidates_match_the_closed_form() -> None:
    """With no spread in true accuracy the simulated optimism is sigma_v times E[max]."""
    rng = np.random.default_rng(7)
    simulated = selection_optimism(20, 1000, rng, replications=2000, spread=0.0)
    assert simulated == pytest.approx(equal_candidates_optimism(20, 1000), abs=0.1)


def test_a_single_candidate_has_no_optimism() -> None:
    """The alt text: with one configuration the gap is zero, up to Monte Carlo error."""
    for size in VALIDATION_SIZES:
        standard_error = 100 * validation_standard_error(size) / sqrt(3000)
        assert abs(_optimism(1, size)) < 3 * standard_error


def test_the_figures_headline_numbers() -> None:
    """Best of 100 on 1,000 items overstates by about 2.5 points, on 200 items by more than 6."""
    assert 2.2 < _optimism(100, 1000) < 2.8
    assert _optimism(100, 200) > 6.0


def test_the_best_score_is_an_overestimate() -> None:
    """The title: optimism is positive, grows with candidates and shrinks with validation size."""
    for curve in SUMMARY.curves:
        selected = curve.optimism[1:]
        assert all(value > 0 for value in selected)
        assert list(selected) == sorted(selected)
    for k in CANDIDATES[1:]:
        by_size = [_optimism(k, size) for size in VALIDATION_SIZES]
        assert by_size == sorted(by_size, reverse=True)


def test_differing_candidates_reduce_the_optimism() -> None:
    """A one-point spread keeps the optimism below the equal-candidates closed form."""
    for curve in SUMMARY.curves:
        for simulated, bound in zip(
            curve.optimism[1:], curve.equal_candidates_bound[1:], strict=True
        ):
            assert simulated < bound


def test_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces the curves; another seed does not."""

    def small(seed: int) -> tuple[OptimismCurve, ...]:
        return optimism_curves(seed, candidates=(1, 5), validation_sizes=(200,), replications=50)

    assert small(3) == small(3)
    assert small(3) != small(4)


def test_invalid_inputs_are_rejected() -> None:
    """Empty searches, negative spreads, booleans and impossible accuracies are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        selection_optimism(0, 1000, rng)
    with pytest.raises(ValueError):
        selection_optimism(5, 1000, rng, spread=-0.01)
    with pytest.raises(TypeError):
        selection_optimism(5, True, rng)
    with pytest.raises(ValueError):
        validation_standard_error(1000, accuracy=1.2)
    with pytest.raises(ValueError):
        expected_maximum_of_normals(0)
