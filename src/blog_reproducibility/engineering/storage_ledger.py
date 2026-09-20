"""A greedy battery dispatch, for an environment draft.

Storage has to balance three things at once: how much energy it can hold, how
fast it can move it, and how much it loses on the way. The draft's point is that
these are separate limits and that raising one without the others buys less than
it appears to.

The dispatch is deliberately simple. Each interval, surplus generation charges
the battery and any deficit discharges it, both capped by the power rating and
by what the state of charge allows. Shortfalls that remain are imported;
surpluses that remain are curtailed. Efficiency is applied on both directions,
so a round trip at 90% each way returns 81% of what went in.

Left out on purpose: charging from the grid, exporting, standby loss, reserve
requirements, degradation, and simultaneous charge and discharge. The draft's
argument does not need them, and each would need its own assumptions.

Energy is in kWh, power in kW, and an interval is ``hours`` long.
"""

from dataclasses import dataclass

from blog_reproducibility.common.validation import non_negative, positive, real

__all__ = [
    "LedgerRow",
    "StorageOutcome",
    "example_payload",
    "storage_ledger",
    "summarise_ledger",
]


def _efficiency(value: float, *, name: str) -> float:
    """Validate a one-way efficiency, which lies in (0, 1]."""
    fraction = positive(value, name=name)
    if fraction > 1.0:
        raise ValueError(f"{name} must not exceed one")
    return fraction


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """One interval of the dispatch."""

    generation: float
    demand: float
    before: float
    charge: float
    discharge: float
    after: float
    imports: float
    curtailment: float
    loss: float


@dataclass(frozen=True, slots=True)
class StorageOutcome:
    """One configuration and what it achieved over the whole series."""

    capacity_kwh: float
    power_kw: float
    rows: tuple[LedgerRow, ...]
    imports_kwh: float
    curtailment_kwh: float
    loss_kwh: float


def storage_ledger(
    generation: tuple[float, ...],
    demand: tuple[float, ...],
    capacity: float,
    power: float,
    *,
    initial: float = 0.0,
    charge_efficiency: float = 0.9,
    discharge_efficiency: float = 0.9,
    hours: float = 1.0,
) -> tuple[LedgerRow, ...]:
    """Run the greedy dispatch and return one row per interval.

    ``charge`` and ``discharge`` are AC power at the meter; ``before`` and
    ``after`` are energy stored in the battery. The two differ by the
    efficiencies, which is where the loss column comes from.
    """
    if len(generation) != len(demand):
        raise ValueError("generation and demand must be the same length")

    limit = non_negative(capacity, name="capacity")
    rating = non_negative(power, name="power")
    start = non_negative(initial, name="initial")
    # An efficiency of one is a lossless battery, which is a legitimate limit to
    # check against; zero would divide by nothing on discharge.
    charging = _efficiency(charge_efficiency, name="charge_efficiency")
    discharging = _efficiency(discharge_efficiency, name="discharge_efficiency")
    interval = positive(hours, name="hours")

    if start > limit:
        raise ValueError("initial must not exceed capacity")
    for value in (*generation, *demand):
        non_negative(value, name="power series value")

    state = start
    rows: list[LedgerRow] = []
    for solar, load in zip(generation, demand, strict=True):
        before = state
        surplus = max(0.0, solar - load)
        deficit = max(0.0, load - solar)

        charge = min(surplus, rating, (limit - state) / (charging * interval))
        discharge = min(deficit, rating, state * discharging / interval)
        state = before + interval * (charging * charge - discharge / discharging)
        # Clamp away rounding at an active bound, not a physical overshoot.
        state = min(limit, max(0.0, state))

        rows.append(
            LedgerRow(
                generation=real(solar, name="generation"),
                demand=real(load, name="demand"),
                before=before,
                charge=charge,
                discharge=discharge,
                after=state,
                imports=deficit - discharge,
                curtailment=surplus - charge,
                loss=interval * ((1 - charging) * charge + (1 / discharging - 1) * discharge),
            )
        )
    return tuple(rows)


def summarise_ledger(
    capacity: float,
    power: float,
    rows: tuple[LedgerRow, ...],
) -> StorageOutcome:
    """Total what one configuration imported, curtailed, and lost."""
    return StorageOutcome(
        capacity_kwh=capacity,
        power_kw=power,
        rows=rows,
        imports_kwh=sum(row.imports for row in rows),
        curtailment_kwh=sum(row.curtailment for row in rows),
        loss_kwh=sum(row.loss for row in rows),
    )


# The draft's day: no sun, two hours of strong sun, then nothing, against a
# steady two-kilowatt load.
GENERATION: tuple[float, ...] = (0.0, 6.0, 6.0, 0.0, 0.0, 0.0)
DEMAND: tuple[float, ...] = (2.0,) * 6


def example_payload() -> tuple[StorageOutcome, ...]:
    """Return the three configurations the draft compares.

    Raising the power rating alone cuts imports but wastes more in losses;
    raising capacity as well removes the curtailment entirely.
    """
    return tuple(
        summarise_ledger(
            capacity,
            power,
            storage_ledger(GENERATION, DEMAND, capacity, power),
        )
        for capacity, power in ((6.0, 2.0), (6.0, 4.0), (10.0, 4.0))
    )
