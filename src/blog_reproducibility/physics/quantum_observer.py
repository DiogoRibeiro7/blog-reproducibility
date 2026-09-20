"""An ideal two-path interferometer with a which-path marker.

The article this supports argues that "observation" in quantum mechanics means a
physical interaction that records which path was taken, not a conscious observer
noticing anything. The model makes that concrete: interference visibility
follows the overlap of the marker states and nothing else.

The joint state before the final recombination is four complex amplitudes,
``C[path][marker]``. The marker states are ``d0 = (1, 0)`` and
``d1 = (gamma, sqrt(1 - gamma^2))``, so ``gamma`` is their overlap: one means the
marker recorded nothing and interference is full, zero means the path is fully
recorded and interference is gone.

Everything is standard library complex arithmetic. These are model predictions
for an idealised apparatus, not measurements: there is no noise, no loss, and no
detector inefficiency anywhere in it.
"""

import cmath
from dataclasses import dataclass
from math import cos, pi, prod, sin, sqrt
from typing import Final

from blog_reproducibility.common.validation import probability, real

__all__ = [
    "EraserRow",
    "MarkerMetrics",
    "QuantumSummary",
    "environment_overlap",
    "example_payload",
    "joint_probabilities",
    "marker_basis",
    "metrics",
    "reduced_path_state",
    "state",
]

Amplitude = complex
JointState = tuple[tuple[Amplitude, Amplitude], tuple[Amplitude, Amplitude]]
DensityMatrix = list[list[Amplitude]]

TRIALS: Final[int] = 10_000


@dataclass(frozen=True, slots=True)
class MarkerMetrics:
    """What one marker overlap implies for interference and path knowledge."""

    overlap: float
    visibility: float
    distinguishability: float
    optimal_path_guess_probability: float
    path_purity: float
    path_eigenvalues: tuple[float, float]
    port_plus_at_phase_zero: float
    port_plus_at_phase_pi: float


@dataclass(frozen=True, slots=True)
class EraserRow:
    """One phase setting of the quantum-eraser example."""

    phase_over_pi: float
    joint_probabilities: tuple[tuple[float, float], tuple[float, float]]
    expected_counts: tuple[tuple[int, int], tuple[int, int]]
    port_plus_marginal: float
    port_plus_given_marker: tuple[float, float]


@dataclass(frozen=True, slots=True)
class QuantumSummary:
    """Every number the article reports."""

    marker_cases: tuple[MarkerMetrics, ...]
    trials: int
    expected_plus_count: float
    plus_count_standard_deviation: float
    eraser: tuple[EraserRow, ...]
    environment_visibility: dict[str, float]


def _overlap(value: float) -> float:
    """Validate a marker overlap, which is a real number on [0, 1]."""
    return probability(value, name="gamma")


def state(phi: float, gamma: float) -> JointState:
    """Joint amplitudes ``C[path][marker]`` before the final recombination.

    The two path alternatives have equal amplitude and a relative phase ``phi``.
    """
    overlap = _overlap(gamma)
    phase_angle = real(phi, name="phi")

    phase = cmath.exp(1j * phase_angle)
    return (
        (1 / sqrt(2) + 0j, 0j),
        (phase * overlap / sqrt(2), phase * sqrt(1 - overlap * overlap) / sqrt(2)),
    )


def reduced_path_state(phi: float, gamma: float) -> DensityMatrix:
    """Trace the joint state over the two orthogonal marker basis states."""
    amplitudes = state(phi, gamma)
    return [
        [
            sum(amplitudes[a][k] * complex(amplitudes[b][k]).conjugate() for k in range(2))
            for b in range(2)
        ]
        for a in range(2)
    ]


def marker_basis(
    beta: float = 0.0, eta: float = 0.0
) -> tuple[tuple[Amplitude, Amplitude], tuple[Amplitude, Amplitude]]:
    """An arbitrary orthonormal marker basis, including a complex relative phase."""
    angle = real(beta, name="beta")
    relative_phase = real(eta, name="eta")

    return (
        (cos(angle) + 0j, cmath.exp(1j * relative_phase) * sin(angle)),
        (-cmath.exp(-1j * relative_phase) * sin(angle), cos(angle) + 0j),
    )


def joint_probabilities(
    phi: float,
    gamma: float,
    beta: float = 0.0,
    eta: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Born probabilities ``P[path port +/-][marker result 0/1]``."""
    amplitudes = state(phi, gamma)
    paths = ((1 / sqrt(2), 1 / sqrt(2)), (1 / sqrt(2), -1 / sqrt(2)))
    markers = marker_basis(beta, eta)

    table = [
        [
            abs(
                sum(
                    complex(paths[a][p]).conjugate()
                    * complex(markers[j][k]).conjugate()
                    * amplitudes[p][k]
                    for p in range(2)
                    for k in range(2)
                )
            )
            ** 2
            for j in range(2)
        ]
        for a in range(2)
    ]
    return ((table[0][0], table[0][1]), (table[1][0], table[1][1]))


def metrics(gamma: float) -> MarkerMetrics:
    """Return what a given marker overlap implies.

    Visibility equals the overlap, and distinguishability is
    ``sqrt(1 - gamma^2)``, so ``V^2 + D^2 = 1``: interference and path knowledge
    trade off exactly, with no room for anything else to enter.
    """
    overlap = _overlap(gamma)
    distinguishability = sqrt(1 - overlap * overlap)
    density = reduced_path_state(0.73, overlap)
    purity = sum(density[i][j] * density[j][i] for i in range(2) for j in range(2)).real

    return MarkerMetrics(
        overlap=overlap,
        visibility=overlap,
        distinguishability=distinguishability,
        optimal_path_guess_probability=(1 + distinguishability) / 2,
        path_purity=purity,
        path_eigenvalues=((1 + overlap) / 2, (1 - overlap) / 2),
        port_plus_at_phase_zero=sum(joint_probabilities(0.0, overlap)[0]),
        port_plus_at_phase_pi=sum(joint_probabilities(pi, overlap)[0]),
    )


def environment_overlap(overlaps: tuple[float, ...]) -> float:
    """Product overlap for conditionally factorised pure environment records.

    Each independent interaction that records the path multiplies the overlap,
    so visibility falls away quickly once anything at all is watching. Nothing
    has to be conscious for this to happen.
    """
    values = tuple(_overlap(value) for value in overlaps)
    return float(prod(values, start=1.0))


def example_payload() -> QuantumSummary:
    """Return the numbers the article reports."""
    rows: list[EraserRow] = []
    for fraction in (0.0, 0.5, 1.0):
        joint = joint_probabilities(fraction * pi, 0.0, beta=pi / 4)
        marginal = sum(joint[0])
        rows.append(
            EraserRow(
                phase_over_pi=fraction,
                joint_probabilities=joint,
                expected_counts=(
                    (round(TRIALS * joint[0][0]), round(TRIALS * joint[0][1])),
                    (round(TRIALS * joint[1][0]), round(TRIALS * joint[1][1])),
                ),
                port_plus_marginal=marginal,
                port_plus_given_marker=(
                    joint[0][0] / sum(joint[a][0] for a in range(2)),
                    joint[0][1] / sum(joint[a][1] for a in range(2)),
                ),
            )
        )

    plus = sum(joint_probabilities(0.0, 0.6)[0])
    return QuantumSummary(
        marker_cases=tuple(metrics(overlap) for overlap in (1.0, 0.6, 0.0)),
        trials=TRIALS,
        expected_plus_count=TRIALS * plus,
        plus_count_standard_deviation=sqrt(TRIALS * plus * (1 - plus)),
        eraser=tuple(rows),
        environment_visibility={
            str(number): environment_overlap((0.95,) * number) for number in (0, 1, 20, 100)
        },
    )
