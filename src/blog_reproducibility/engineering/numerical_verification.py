"""Refinement as a check on an implementation, for the verification article.

The article's argument is that a solver converging cleanly is not a solver
computing the right thing. Refining the step size tells you whether the method
is implemented at its stated order; it does not tell you whether the equation
being solved is the one you meant.

The test problem is a two-state conversion, ``A -> B`` at rate ``k``, integrated
to ``t = 1`` from ``(1, 0)``. Its exact answer is ``(exp(-k), 1 - exp(-k))``, so
the error is known rather than estimated. Euler should show first-order
convergence and Heun second-order, and both conserve total mass exactly because
the right-hand side sums to zero by construction.

The third case is the point. Heun with the wrong rate converges at its proper
second order *against itself* — successive refinements agree better and better
— while the error against the true answer flattens out at ``exp(-1.8) -
exp(-2)``. Self-convergence is a property of the method; correctness is a
property of the model.

Everything is standard library: the state is two numbers.
"""

from dataclasses import dataclass
from math import exp, log2
from typing import Final

from blog_reproducibility.common.validation import count, positive, real

__all__ = [
    "METHODS",
    "ConvergenceRow",
    "Solution",
    "VerificationSummary",
    "example_payload",
    "exact_solution",
    "first_design_below",
    "solve_conversion",
]

METHODS: Final[tuple[str, ...]] = ("euler", "heun")
DEFAULT_RATE: Final[float] = 2.0
STEP_COUNTS: Final[tuple[int, ...]] = (5, 10, 20, 40, 80)


@dataclass(frozen=True, slots=True)
class Solution:
    """The endpoint an integration reached, and how far mass drifted on the way."""

    remaining: float
    converted: float
    mass_error: float


@dataclass(frozen=True, slots=True)
class ConvergenceRow:
    """One refinement: the step count, the error, and the order it implies."""

    method: str
    rate: float
    steps: int
    endpoint_error: float
    mass_error: float
    observed_order: float | None


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    """Every number the article reports."""

    rows: tuple[ConvergenceRow, ...]
    wrong_rate_self_convergence_order: float
    wrong_rate_limiting_error: float
    first_design_below_tolerance: dict[str, tuple[int, int]]


def exact_solution(rate: float = DEFAULT_RATE) -> tuple[float, float]:
    """Return the exact state at ``t = 1`` for the given rate."""
    constant = positive(rate, name="rate")
    return (exp(-constant), 1 - exp(-constant))


def solve_conversion(
    steps: int,
    method: str = "heun",
    rate: float = DEFAULT_RATE,
) -> Solution:
    """Integrate the two-state conversion to ``t = 1``.

    ``mass_error`` is the largest departure of the two states from summing to
    one over the whole integration. It stays at rounding level for both methods,
    because the right-hand side moves the same quantity out of one state and
    into the other.
    """
    count_of_steps = count(steps, name="steps", minimum=1)
    scheme = method if method in METHODS else None
    if scheme is None:
        raise ValueError(f"method must be one of {METHODS}")
    constant = real(rate, name="rate")

    width = 1 / count_of_steps
    remaining, converted = 1.0, 0.0
    mass_error = 0.0

    def derivative(state: tuple[float, float]) -> tuple[float, float]:
        flow = constant * state[0]
        return (-flow, flow)

    for _ in range(count_of_steps):
        first = derivative((remaining, converted))
        if scheme == "euler":
            remaining += width * first[0]
            converted += width * first[1]
        else:
            predicted = (remaining + width * first[0], converted + width * first[1])
            second = derivative(predicted)
            remaining += width * (first[0] + second[0]) / 2
            converted += width * (first[1] + second[1]) / 2
        mass_error = max(mass_error, abs(remaining + converted - 1))

    return Solution(remaining=remaining, converted=converted, mass_error=mass_error)


def endpoint_error(solution: Solution, rate: float = DEFAULT_RATE) -> float:
    """Largest component error against the exact answer at the true rate."""
    exact_remaining, exact_converted = exact_solution(rate)
    return max(
        abs(solution.remaining - exact_remaining),
        abs(solution.converted - exact_converted),
    )


def first_design_below(
    tolerance: float,
    method: str = "heun",
    rate: float = DEFAULT_RATE,
) -> tuple[int, int]:
    """Smallest power-of-two step count under a tolerance, and its cost.

    The cost is the number of right-hand-side evaluations, which is what a
    method actually charges: Heun pays two per step and Euler one, so the
    comparison of step counts alone flatters Heun.
    """
    limit = positive(tolerance, name="tolerance")
    steps = 1
    while endpoint_error(solve_conversion(steps, method, rate), rate) > limit:
        steps *= 2
    return (steps, steps if method == "euler" else 2 * steps)


def _convergence_rows(method: str, rate: float, label: str) -> tuple[ConvergenceRow, ...]:
    """Refine the step count and record the order each refinement implies."""
    rows: list[ConvergenceRow] = []
    previous: float | None = None
    for steps in STEP_COUNTS:
        solution = solve_conversion(steps, method, rate)
        error = endpoint_error(solution, DEFAULT_RATE)
        rows.append(
            ConvergenceRow(
                method=label,
                rate=rate,
                steps=steps,
                endpoint_error=error,
                mass_error=solution.mass_error,
                observed_order=None if previous is None else log2(previous / error),
            )
        )
        previous = error
    return tuple(rows)


def example_payload() -> VerificationSummary:
    """Return the numbers the article reports."""
    cases = (
        ("Euler", "euler", DEFAULT_RATE),
        ("Heun", "heun", DEFAULT_RATE),
        ("Heun, wrong rate", "heun", 1.8),
    )
    rows = tuple(
        row for label, method, rate in cases for row in _convergence_rows(method, rate, label)
    )

    # The wrong-rate run still converges at second order against itself.
    wrong = [solve_conversion(steps, "heun", 1.8).remaining for steps in (20, 40, 80)]
    self_order = log2(abs(wrong[0] - wrong[1]) / abs(wrong[1] - wrong[2]))

    return VerificationSummary(
        rows=rows,
        wrong_rate_self_convergence_order=self_order,
        wrong_rate_limiting_error=exp(-1.8) - exp(-2),
        first_design_below_tolerance={
            method: first_design_below(1e-4, method) for method in METHODS
        },
    )
