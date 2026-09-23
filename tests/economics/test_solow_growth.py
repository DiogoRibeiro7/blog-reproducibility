"""Check the Solow steady state as a fixed point and as the limit of accumulation.

The closed forms are checked through the conditions that define them, not by
re-evaluating the same expressions: saving equals break-even investment at the
steady state, the golden rule maximises consumption over a grid, and the
accumulation recursion converges to the steady state from either side.
"""

import numpy as np
import pytest

from blog_reproducibility.economics.solow_growth import (
    DEFAULT_PARAMETERS,
    SolowParameters,
    break_even_investment,
    capital_path,
    example_payload,
    golden_rule_capital,
    output_per_worker,
    saving,
    steady_state_capital,
)

SUMMARY = example_payload()


def test_steady_state_is_where_saving_meets_break_even() -> None:
    """At k*, saving exactly replaces what growth and depreciation dilute."""
    kstar = steady_state_capital()

    assert float(saving(kstar)) == pytest.approx(float(break_even_investment(kstar)))
    assert kstar == pytest.approx(5.771752844662493)
    assert round(kstar, 1) == 5.8


def test_steady_state_is_stable() -> None:
    """Capital grows below k*, shrinks above it, and the recursion converges either way."""
    kstar = steady_state_capital()
    below = np.linspace(0.1, kstar * 0.99, 50)
    above = np.linspace(kstar * 1.01, 14, 50)

    assert np.all(saving(below) > break_even_investment(below))
    assert np.all(saving(above) < break_even_investment(above))

    for start in (0.5, 12.0):
        path = capital_path(start, 400)
        assert path[-1] == pytest.approx(kstar, rel=1e-6)
        steps = np.diff(path)
        assert np.all(steps > 0) if start < kstar else np.all(steps < 0)


def test_golden_rule_maximises_steady_state_consumption() -> None:
    """Scanning saving rates, steady-state consumption peaks at the golden-rule capital."""
    rates = np.linspace(0.01, 0.99, 9801)
    consumption = []
    for rate in rates:
        parameters = SolowParameters(saving_rate=float(rate))
        capital = steady_state_capital(parameters)
        consumption.append((1 - rate) * float(output_per_worker(capital, parameters)))
    best = float(rates[int(np.argmax(consumption))])

    assert best == pytest.approx(DEFAULT_PARAMETERS.capital_share, abs=1e-3)
    assert golden_rule_capital() == pytest.approx(
        steady_state_capital(SolowParameters(saving_rate=DEFAULT_PARAMETERS.capital_share))
    )
    # The article's economy saves less than the capital share, so it sits below the golden rule.
    assert SUMMARY.steady_state_capital < SUMMARY.golden_rule_capital


def test_summary_accounts_for_output() -> None:
    """Steady-state output splits into consumption and saving."""
    assert SUMMARY.steady_state_output == pytest.approx(SUMMARY.steady_state_capital**0.35)
    assert SUMMARY.steady_state_consumption == pytest.approx(0.75 * SUMMARY.steady_state_output)


def test_higher_saving_raises_the_steady_state() -> None:
    """More saving, or less dilution, supports more capital per worker."""
    base = steady_state_capital()

    assert steady_state_capital(SolowParameters(saving_rate=0.3)) > base
    assert steady_state_capital(SolowParameters(depreciation=0.1)) < base


@pytest.mark.parametrize(
    "overrides",
    [
        {"capital_share": 1.0},
        {"saving_rate": 0.0},
        {"population_growth": -0.1},
        {"depreciation": float("inf")},
    ],
)
def test_invalid_parameters_are_rejected(overrides: dict[str, float]) -> None:
    """Shares and rates outside the model's domain are refused."""
    with pytest.raises(ValueError):
        SolowParameters(**overrides)


def test_invalid_capital_is_rejected() -> None:
    """Negative capital and non-positive starting points are refused."""
    with pytest.raises(ValueError):
        output_per_worker([-1.0])
    with pytest.raises(ValueError):
        capital_path(0.0, 10)
    with pytest.raises(ValueError):
        capital_path(1.0, -1)
