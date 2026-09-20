"""Check the convergence study, including the case that converges to the wrong answer.

The exact solution of the test problem is known, so the error is measured rather
than estimated, and the observed order can be compared with the order each
method claims.
"""

from math import exp

import pytest

from blog_reproducibility.engineering.numerical_verification import (
    METHODS,
    ConvergenceRow,
    endpoint_error,
    exact_solution,
    example_payload,
    first_design_below,
    solve_conversion,
)

SUMMARY = example_payload()


def test_mass_is_conserved_to_rounding_by_both_methods() -> None:
    """The two states always sum to one, because the flow leaves one and enters the other."""
    for method in METHODS:
        for steps in (5, 20, 80, 500):
            solution = solve_conversion(steps, method)
            assert solution.mass_error < 1e-15
            assert solution.remaining + solution.converted == pytest.approx(1.0)
            assert 0.0 < solution.remaining < 1.0


def test_each_method_converges_at_the_order_it_claims() -> None:
    """Euler is first order and Heun is second, measured against the exact answer."""
    euler = [row for row in SUMMARY.rows if row.method == "Euler"]
    heun = [row for row in SUMMARY.rows if row.method == "Heun"]

    euler_orders = [row.observed_order for row in euler[1:]]
    heun_orders = [row.observed_order for row in heun[1:]]
    assert all(order is not None for order in euler_orders)
    assert all(order is not None for order in heun_orders)

    assert all(order == pytest.approx(1.0, abs=0.06) for order in euler_orders)
    assert all(order == pytest.approx(2.0, abs=0.26) for order in heun_orders)
    # And the orders settle as the step shrinks.
    first, last = heun_orders[0], heun_orders[-1]
    assert first is not None and last is not None
    assert abs(last - 2.0) < abs(first - 2.0)


def test_heun_is_more_accurate_than_euler_at_every_step_size() -> None:
    """A second-order method should beat a first-order one on this problem."""
    for steps in (5, 10, 20, 40, 80):
        assert endpoint_error(solve_conversion(steps, "heun")) < endpoint_error(
            solve_conversion(steps, "euler")
        )


def test_the_wrong_rate_converges_cleanly_to_the_wrong_answer() -> None:
    """Self-convergence is a property of the method, not evidence of correctness."""
    wrong = [row for row in SUMMARY.rows if row.method == "Heun, wrong rate"]

    # Against the true answer the error stops improving.
    assert wrong[-1].observed_order is not None
    assert wrong[-1].observed_order < 0.05
    assert wrong[-1].endpoint_error == pytest.approx(SUMMARY.wrong_rate_limiting_error, rel=0.01)

    # Against itself it converges at its proper second order.
    assert SUMMARY.wrong_rate_self_convergence_order == pytest.approx(2.0, abs=0.1)

    # The floor is exactly the difference the wrong rate makes.
    assert SUMMARY.wrong_rate_limiting_error == pytest.approx(exp(-1.8) - exp(-2))


def test_refinement_alone_cannot_distinguish_the_two_heun_runs() -> None:
    """Both look like well-behaved second-order methods on their own output."""
    right = [solve_conversion(steps, "heun", 2.0).remaining for steps in (20, 40, 80)]
    wrong = [solve_conversion(steps, "heun", 1.8).remaining for steps in (20, 40, 80)]

    for series in (right, wrong):
        first_gap = abs(series[0] - series[1])
        second_gap = abs(series[1] - series[2])
        assert first_gap / second_gap == pytest.approx(4.0, rel=0.1)


def test_the_exact_solution_is_the_one_being_compared_against() -> None:
    """The closed form is written out and the solver approaches it."""
    remaining, converted = exact_solution(2.0)

    assert remaining == pytest.approx(exp(-2))
    assert converted == pytest.approx(1 - exp(-2))
    assert remaining + converted == pytest.approx(1.0)

    # At h = 1e-4 a second-order method leaves an error near 2e-9, not zero.
    fine = solve_conversion(10_000, "heun")
    assert fine.remaining == pytest.approx(remaining, abs=1e-8)
    assert endpoint_error(fine) == pytest.approx(
        endpoint_error(solve_conversion(5_000, "heun")) / 4, rel=0.05
    )


def test_the_cost_of_reaching_a_tolerance() -> None:
    """Heun needs 64 steps and 128 evaluations; Euler needs 4,096 of each."""
    assert SUMMARY.first_design_below_tolerance["euler"] == (4096, 4096)
    assert SUMMARY.first_design_below_tolerance["heun"] == (64, 128)

    for method in METHODS:
        steps, calls = first_design_below(1e-4, method)
        assert endpoint_error(solve_conversion(steps, method)) <= 1e-4
        assert endpoint_error(solve_conversion(steps // 2, method)) > 1e-4
        assert calls >= steps


def test_published_errors() -> None:
    """The first and last row of each method, as the article prints them."""
    by_method: dict[str, list[ConvergenceRow]] = {}
    for row in SUMMARY.rows:
        by_method.setdefault(row.method, []).append(row)

    assert by_method["Euler"][0].endpoint_error == pytest.approx(0.057575283, abs=5e-9)
    assert by_method["Euler"][-1].endpoint_error == pytest.approx(0.003397478, abs=5e-9)
    assert by_method["Heun"][0].endpoint_error == pytest.approx(0.010058074, abs=5e-9)
    assert by_method["Heun"][-1].endpoint_error == pytest.approx(0.000028732, abs=5e-9)
    assert by_method["Heun, wrong rate"][-1].endpoint_error == pytest.approx(0.029989139, abs=5e-9)
    assert SUMMARY.wrong_rate_self_convergence_order == pytest.approx(2.058, abs=5e-4)


def test_invalid_inputs() -> None:
    """Unknown methods, zero steps, and non-positive rates are rejected."""
    with pytest.raises(ValueError):
        solve_conversion(10, "rk4")
    with pytest.raises(ValueError):
        solve_conversion(0)
    with pytest.raises(ValueError):
        solve_conversion(10, "heun", float("nan"))
    with pytest.raises(ValueError):
        exact_solution(0.0)
    with pytest.raises(ValueError):
        first_design_below(0.0)
    with pytest.raises(TypeError):
        solve_conversion(True)
