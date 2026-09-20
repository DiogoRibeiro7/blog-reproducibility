"""Check the interferometer model against projections, search, and tensor products.

The checks reconstruct each quantity a different way: the analytic interference
formula, an explicit search over measurement directions, explicit local
projectors applied in both orders, a phase-averaged mixture built by hand, and
environment states written out as tensor products.
"""

import cmath
from itertools import product
from math import acos, cos, pi, sqrt

import pytest

from blog_reproducibility.physics.quantum_observer import (
    environment_overlap,
    example_payload,
    joint_probabilities,
    marker_basis,
    metrics,
    reduced_path_state,
    state,
)

SUMMARY = example_payload()


def test_joint_probabilities_match_the_interference_formula_and_conserve_mass() -> None:
    """Every table is a probability distribution and follows (1 + gamma cos phi)/2."""
    for gamma, phi in product((0.0, 0.2, 0.6, 0.95, 1.0), (0.0, 0.41, pi / 2, pi, 2 * pi)):
        amplitudes = state(phi, gamma)
        assert sum(abs(a) ** 2 for row in amplitudes for a in row) == pytest.approx(1)

        joint = joint_probabilities(phi, gamma)
        assert sum(sum(row) for row in joint) == pytest.approx(1)
        assert sum(joint[0]) == pytest.approx((1 + gamma * cos(phi)) / 2)
        assert all(value >= 0 for row in joint for value in row)


def test_reduced_matrix_is_a_density_matrix_with_the_reported_purity() -> None:
    """The traced state is Hermitian, unit trace, positive, and as pure as claimed."""
    for gamma, phi in product((0.0, 0.3, 0.6, 1.0), (0.2, pi / 2)):
        rho = reduced_path_state(phi, gamma)

        assert rho[0][1] == pytest.approx(gamma * cmath.exp(-1j * phi) / 2)
        assert rho[1][0] == pytest.approx(rho[0][1].conjugate())
        assert rho[0][0] + rho[1][1] == pytest.approx(1)

        determinant = (rho[0][0] * rho[1][1] - rho[0][1] * rho[1][0]).real
        assert determinant >= -1e-15
        assert determinant == pytest.approx((1 - gamma * gamma) / 4)
        assert metrics(gamma).path_purity == pytest.approx((1 + gamma * gamma) / 2)


def test_visibility_and_distinguishability_are_complementary() -> None:
    """V^2 + D^2 = 1 leaves no room for anything else to enter."""
    for gamma in (0.0, 0.2, 0.6, 0.95, 1.0):
        row = metrics(gamma)
        assert row.visibility**2 + row.distinguishability**2 == pytest.approx(1)
        assert row.port_plus_at_phase_zero - row.port_plus_at_phase_pi == pytest.approx(
            row.visibility, abs=1e-12
        )
        assert sum(row.path_eigenvalues) == pytest.approx(1)


def test_optimal_path_guess_against_an_independent_measurement_search() -> None:
    """Searching real measurement directions finds the same best guess rate.

    The two marker states are real, so an optimal measurement exists in their
    plane and a one-parameter search over it is exhaustive.
    """
    for gamma in (0.0, 0.2, 0.6, 0.95, 1.0):
        marker = (gamma, sqrt(1 - gamma * gamma))
        best = 0.0
        for index in range(8192):
            angle = pi * index / 8192
            first = (cos(angle), sqrt(1 - cos(angle) ** 2))
            second = (-first[1], first[0])
            success = 0.5 * (
                first[0] ** 2 + sum(a * b for a, b in zip(second, marker, strict=True)) ** 2
            )
            best = max(best, success)

        assert best == pytest.approx(metrics(gamma).optimal_path_guess_probability, abs=4e-8)


def test_marker_basis_does_not_change_unsorted_path_statistics() -> None:
    """Choosing how to read the marker cannot change what the path detector sees."""
    for gamma, phi, beta, eta in product(
        (0.0, 0.6, 1.0), (0.3, 1.7), (0.0, 0.23, pi / 4), (0.0, 0.71)
    ):
        table = joint_probabilities(phi, gamma, beta, eta)
        expected = (1 + gamma * cos(phi)) / 2
        assert sum(table[0]) == pytest.approx(expected)
        assert sum(table[1]) == pytest.approx(1 - expected)

    # Conditioning on the marker result does change the subset, by a lot.
    table = joint_probabilities(pi / 3, 0.0, pi / 4)
    assert table[0][0] / sum(row[0] for row in table) == pytest.approx(0.75)
    assert table[0][1] / sum(row[1] for row in table) == pytest.approx(0.25)
    assert sum(table[0]) == pytest.approx(0.5)


def test_local_projectors_commute_on_the_full_joint_state() -> None:
    """Measuring path then marker gives the same vector as the reverse order."""
    path = (1 / sqrt(2), -1 / sqrt(2))
    marker = (cos(0.37), cmath.exp(0.8j) * sqrt(1 - cos(0.37) ** 2))

    def outer(vector: tuple[complex, complex]) -> list[list[complex]]:
        return [[a * complex(b).conjugate() for b in vector] for a in vector]

    def tensor(a: object, b: object) -> list[list[complex]]:
        left, right = a, b
        return [
            [left[i // 2][j // 2] * right[i % 2][j % 2] for j in range(4)]  # type: ignore[index]
            for i in range(4)
        ]

    def apply(matrix: list[list[complex]], vector: list[complex]) -> list[complex]:
        return [sum(matrix[i][j] * vector[j] for j in range(4)) for i in range(4)]

    identity = ((1 + 0j, 0j), (0j, 1 + 0j))
    path_full = tensor(outer(path), identity)
    marker_full = tensor(identity, outer(marker))
    joint = [value for row in state(1.3, 0.6) for value in row]

    first = apply(path_full, apply(marker_full, joint))
    second = apply(marker_full, apply(path_full, joint))
    assert first == pytest.approx(second)


def test_phase_averaging_and_entanglement_can_have_identical_local_states() -> None:
    """A classical phase mixture reproduces the reduced state exactly.

    This is the article's counterexample: the local statistics alone cannot
    distinguish an entangled marker from a randomised phase.
    """
    for gamma in (0.0, 0.6, 1.0):
        phi, alpha = 0.71, acos(gamma)
        vectors = [
            (1 / sqrt(2), cmath.exp(1j * (phi + shift)) / sqrt(2)) for shift in (-alpha, alpha)
        ]
        mixture = [
            [sum(v[i] * complex(v[j]).conjugate() for v in vectors) / 2 for j in range(2)]
            for i in range(2)
        ]
        reduced = reduced_path_state(phi, gamma)
        for actual, expected in zip(mixture, reduced, strict=True):
            assert actual == pytest.approx(expected)


def test_product_environment_overlap_against_explicit_tensor_states() -> None:
    """Building the two environment states as tensor products gives the same overlap."""
    overlaps = (0.8, 0.6, 0.3)
    zero: list[float] = [1.0]
    one: list[float] = [1.0]
    for gamma in overlaps:
        zero = [a * b for a in zero for b in (1.0, 0.0)]
        one = [a * b for a in one for b in (gamma, sqrt(1 - gamma * gamma))]

    direct = sum(a * b for a, b in zip(zero, one, strict=True))
    assert environment_overlap(overlaps) == pytest.approx(direct)

    assert environment_overlap(()) == 1.0
    assert environment_overlap((0.8, 0.0, 0.9)) == 0.0


def test_visibility_collapses_as_the_environment_records_more() -> None:
    """Twenty weak interactions are enough to remove most of the interference."""
    visibility = SUMMARY.environment_visibility

    assert visibility["0"] == 1.0
    assert visibility["1"] == pytest.approx(0.95)
    assert visibility["20"] == pytest.approx(0.95**20)
    assert visibility["100"] < 0.01


def test_published_numbers() -> None:
    """The values the article's tables report should come back unchanged."""
    full, partial, none = SUMMARY.marker_cases

    assert full.overlap == 1.0
    assert full.port_plus_at_phase_zero == pytest.approx(1.0)
    assert full.port_plus_at_phase_pi == pytest.approx(0.0, abs=1e-12)

    assert partial.overlap == 0.6
    assert partial.distinguishability == pytest.approx(0.8)
    assert partial.optimal_path_guess_probability == pytest.approx(0.9)
    assert partial.port_plus_at_phase_zero == pytest.approx(0.8)
    assert partial.port_plus_at_phase_pi == pytest.approx(0.2)

    assert none.visibility == 0.0
    assert none.optimal_path_guess_probability == pytest.approx(1.0)
    assert none.path_eigenvalues == pytest.approx((0.5, 0.5))

    assert SUMMARY.trials == 10_000
    assert SUMMARY.expected_plus_count == pytest.approx(8_000)
    assert SUMMARY.plus_count_standard_deviation == pytest.approx(40.0)


def test_the_eraser_marginal_never_moves() -> None:
    """Every phase leaves the unsorted + probability at one half."""
    for row in SUMMARY.eraser:
        assert row.port_plus_marginal == pytest.approx(0.5)
        assert sum(sum(pair) for pair in row.joint_probabilities) == pytest.approx(1)
        assert sum(sum(pair) for pair in row.expected_counts) == 10_000

    # The conditional subsets are complementary and swap as the phase turns.
    first, middle, last = SUMMARY.eraser
    assert first.port_plus_given_marker == pytest.approx((1.0, 0.0))
    assert middle.port_plus_given_marker == pytest.approx((0.5, 0.5))
    assert last.port_plus_given_marker == pytest.approx((0.0, 1.0), abs=1e-12)


def test_invalid_model_parameters() -> None:
    """Overlaps outside [0, 1] and non-finite angles are rejected."""
    for gamma in (-0.1, 1.1):
        with pytest.raises(ValueError):
            state(0.0, gamma)
    for gamma in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            state(0.0, gamma)

    with pytest.raises(ValueError):
        state(float("nan"), 0.5)
    with pytest.raises(ValueError):
        marker_basis(float("inf"))
    with pytest.raises(ValueError):
        environment_overlap((0.5, 1.5))
