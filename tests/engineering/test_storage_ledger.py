"""Check the storage dispatch conserves energy and respects every limit.

The strongest checks here are conservation: whatever arrives has to leave, be
stored, be lost, or be curtailed, in every interval and over the whole series.
"""

import pytest

from blog_reproducibility.engineering.storage_ledger import (
    DEMAND,
    GENERATION,
    example_payload,
    storage_ledger,
    summarise_ledger,
)

OUTCOMES = example_payload()


def test_energy_is_conserved_in_every_interval() -> None:
    """Every kilowatt-hour generated is served, stored, lost, or curtailed."""
    for outcome in OUTCOMES:
        for row in outcome.rows:
            stored = row.after - row.before
            served_from_generation = min(row.generation, row.demand)
            # Everything generated is accounted for.
            accounted = served_from_generation + row.charge + row.curtailment
            assert accounted == pytest.approx(row.generation)
            # Everything demanded is met from generation, the battery, or the grid.
            assert served_from_generation + row.discharge + row.imports == pytest.approx(row.demand)
            # The change in stored energy is the charge in, less the discharge out and the loss.
            assert stored == pytest.approx(row.charge - row.discharge - row.loss, abs=1e-9)


def test_the_whole_series_balances() -> None:
    """Totals over the day balance as well as each interval does."""
    for outcome in OUTCOMES:
        generated = sum(row.generation for row in outcome.rows)
        demanded = sum(row.demand for row in outcome.rows)
        left_in_battery = outcome.rows[-1].after

        assert generated + outcome.imports_kwh == pytest.approx(
            demanded + outcome.curtailment_kwh + outcome.loss_kwh + left_in_battery
        )


def test_no_limit_is_ever_exceeded() -> None:
    """The state stays inside the capacity and no flow exceeds the power rating."""
    for outcome in OUTCOMES:
        for row in outcome.rows:
            assert 0.0 <= row.before <= outcome.capacity_kwh + 1e-12
            assert 0.0 <= row.after <= outcome.capacity_kwh + 1e-12
            assert 0.0 <= row.charge <= outcome.power_kw + 1e-12
            assert 0.0 <= row.discharge <= outcome.power_kw + 1e-12
            assert row.imports >= -1e-12
            assert row.curtailment >= -1e-12
            assert row.loss >= -1e-12
            # Charging and discharging never happen in the same interval.
            assert row.charge == 0.0 or row.discharge == 0.0


def test_raising_power_then_capacity_each_buys_something_different() -> None:
    """More power cuts imports; more capacity removes the curtailment."""
    small, faster, larger = OUTCOMES

    assert small.imports_kwh > faster.imports_kwh > larger.imports_kwh
    assert small.curtailment_kwh > faster.curtailment_kwh > larger.curtailment_kwh
    assert larger.curtailment_kwh == 0.0
    # Moving more energy through the battery costs more in losses.
    assert small.loss_kwh < faster.loss_kwh < larger.loss_kwh


def test_published_totals() -> None:
    """The three configurations the draft compares."""
    small, faster, larger = OUTCOMES

    assert (small.capacity_kwh, small.power_kw) == (6.0, 2.0)
    assert small.imports_kwh == pytest.approx(4.76)
    assert small.curtailment_kwh == pytest.approx(4.0)
    assert small.loss_kwh == pytest.approx(0.76)

    assert faster.imports_kwh == pytest.approx(2.6)
    assert larger.imports_kwh == pytest.approx(2.0)
    assert larger.loss_kwh == pytest.approx(1.4667, abs=5e-4)


def test_a_perfect_battery_loses_nothing_but_cannot_import_backwards() -> None:
    """Unit efficiencies remove every loss; the empty first hour still costs.

    Generation equals demand over the whole day, so a lossless battery with
    room to spare wastes nothing. It still imports the first hour, because the
    sun has not risen yet and a battery cannot discharge what it has not stored.
    That is the draft's point about a day-total figure hiding the ordering.
    """
    rows = storage_ledger(
        GENERATION,
        DEMAND,
        capacity=20.0,
        power=10.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    outcome = summarise_ledger(20.0, 10.0, rows)

    assert outcome.loss_kwh == pytest.approx(0.0)
    assert outcome.curtailment_kwh == pytest.approx(0.0)
    assert sum(GENERATION) == pytest.approx(sum(DEMAND))
    # Only the hours before the first surplus are imported.
    assert outcome.imports_kwh == pytest.approx(DEMAND[0])
    assert rows[0].imports == pytest.approx(DEMAND[0])
    assert all(row.imports == 0.0 for row in rows[1:])

    # Starting the day charged removes even that.
    charged = storage_ledger(
        GENERATION,
        DEMAND,
        capacity=20.0,
        power=10.0,
        initial=2.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    assert summarise_ledger(20.0, 10.0, charged).imports_kwh == pytest.approx(0.0)


def test_a_battery_with_no_capacity_does_nothing() -> None:
    """Every deficit is imported and every surplus is curtailed."""
    rows = storage_ledger(GENERATION, DEMAND, capacity=0.0, power=10.0)
    outcome = summarise_ledger(0.0, 10.0, rows)

    assert all(row.charge == 0.0 and row.discharge == 0.0 for row in rows)
    assert outcome.imports_kwh == pytest.approx(
        sum(max(0.0, load - solar) for solar, load in zip(GENERATION, DEMAND, strict=True))
    )
    assert outcome.loss_kwh == 0.0


def test_the_round_trip_costs_what_the_efficiencies_say() -> None:
    """Storing one kilowatt-hour and taking it back returns 81% of it."""
    rows = storage_ledger((4.0, 0.0), (0.0, 4.0), capacity=10.0, power=4.0)

    stored = rows[0].charge * 0.9
    assert rows[0].after == pytest.approx(stored)
    returned = rows[1].discharge
    assert returned == pytest.approx(stored * 0.9)
    assert returned / rows[0].charge == pytest.approx(0.81)


def test_invalid_parameters() -> None:
    """Mismatched series, negative values, and impossible efficiencies are rejected."""
    with pytest.raises(ValueError):
        storage_ledger((1.0, 2.0), (1.0,), 5.0, 2.0)
    with pytest.raises(ValueError):
        storage_ledger(GENERATION, DEMAND, -1.0, 2.0)
    with pytest.raises(ValueError):
        storage_ledger(GENERATION, DEMAND, 5.0, 2.0, initial=6.0)
    with pytest.raises(ValueError):
        storage_ledger(GENERATION, DEMAND, 5.0, 2.0, charge_efficiency=0.0)
    with pytest.raises(ValueError):
        storage_ledger(GENERATION, DEMAND, 5.0, 2.0, discharge_efficiency=1.5)
    with pytest.raises(ValueError):
        storage_ledger(GENERATION, DEMAND, 5.0, 2.0, hours=0.0)
    with pytest.raises(ValueError):
        storage_ledger((-1.0,), (1.0,), 5.0, 2.0)
