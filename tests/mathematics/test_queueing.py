"""Check the queue simulation against hand-worked recursions and the article's tables.

The Lindley recursion is checked on a trace worked by hand and against a direct
event simulation of the server. The seeded simulations then reproduce the
article's two tables.
"""

import numpy as np
import pytest

from blog_reproducibility.mathematics.queueing import (
    example_payload,
    lindley_waits,
    mean_wait,
    overloaded_day,
    simulate_utilisations,
)

SUMMARY = example_payload()


def test_lindley_matches_a_hand_worked_trace() -> None:
    """Arrivals at 0, 1, 2, 10 with services 3, 1, 1, 1 wait 0, 2, 2, 0."""
    waits = lindley_waits([0.0, 1.0, 1.0, 8.0], [3.0, 1.0, 1.0, 1.0])

    assert waits.tolist() == [0.0, 2.0, 2.0, 0.0]


def test_lindley_matches_an_event_simulation() -> None:
    """Each job starts when it arrives or when the previous job ends, whichever is later."""
    rng = np.random.default_rng(4)
    gaps = rng.exponential(1.2, 500)
    services = rng.exponential(1.0, 500)
    arrivals = np.cumsum(gaps)

    free_at, expected = 0.0, []
    for arrival, service in zip(arrivals, services, strict=True):
        start = max(arrival, free_at)
        expected.append(start - arrival)
        free_at = start + service

    np.testing.assert_allclose(lindley_waits(gaps, services), expected, atol=1e-9)


def test_pollaczek_khinchine_formula() -> None:
    """Exponential service gives rho / (1 - rho); deterministic service halves it."""
    assert mean_wait(0.8) == pytest.approx(4.0)
    assert mean_wait(0.9) == pytest.approx(9.0)
    assert mean_wait(0.8, 0.0) == pytest.approx(2.0)
    assert mean_wait(0.8, 4.0) == pytest.approx(10.0)


def test_utilisation_table_matches_the_article() -> None:
    """Simulated mean and 95th percentile waits at each utilisation."""
    published = {
        0.5: (1.02, 4.7),
        0.7: (2.30, 8.6),
        0.8: (4.09, 14.3),
        0.9: (9.15, 30.0),
        0.95: (19.96, 63.1),
    }
    for row in SUMMARY.utilisation:
        mean, p95 = published[row.utilisation]
        assert round(row.simulated_mean, 2) == mean
        assert round(row.percentile_95, 1) == p95
        # The article's point: simulation and formula agree within noise to 95 percent.
        assert row.simulated_mean == pytest.approx(row.formula_mean, rel=0.06)


def test_overloaded_day_matches_the_article() -> None:
    """One hour at 130 percent leaves a backlog that lasts through midday."""
    published = {7: 1.6, 8: 17.7, 9: 18.7, 10: 14.4, 11: 8.8, 12: 4.1, 14: 2.1}

    for row in SUMMARY.overloaded_day:
        assert round(row.mean_wait, 1) == published[row.hour]
        assert row.load == (1.3 if row.hour == 8 else 0.8)


def test_overloaded_day_is_consistent() -> None:
    """Arrivals are ordered within the day and every wait is non-negative."""
    times, waits, loads = overloaded_day()

    assert np.all(np.diff(times) >= 0)
    assert times.min() >= 0 and times.max() < 24 * 60
    assert np.all(waits >= 0)
    assert loads.tolist().count(1.3) == 1


def test_short_simulation_is_deterministic() -> None:
    """The same seed reproduces the same rows."""
    first = simulate_utilisations((0.5,), jobs=1000, seed=9)

    assert first == simulate_utilisations((0.5,), jobs=1000, seed=9)
    assert first != simulate_utilisations((0.5,), jobs=1000, seed=10)


def test_invalid_inputs_are_rejected() -> None:
    """Mismatched, empty, or negative inputs and impossible utilisations are refused."""
    with pytest.raises(ValueError):
        lindley_waits([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        lindley_waits([], [])
    with pytest.raises(ValueError):
        lindley_waits([-1.0], [1.0])
    with pytest.raises(ValueError):
        mean_wait(1.0)
    with pytest.raises(ValueError):
        mean_wait(0.5, -1.0)
    with pytest.raises(ValueError):
        simulate_utilisations(jobs=5)
