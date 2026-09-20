"""Two ways a correct number can mislead, for two science-communication articles.

**Concentration is not amount.** A substance present at a lower concentration
can deliver a larger dose, because the dose is concentration times volume. The
article's three hypothetical samples make that arithmetic explicit: the most
concentrated one is not the largest exposure.

**A relative reduction is not an absolute one.** Halving a risk means very
different things at different starting risks. The two hypothetical populations
share a relative reduction of 50% and differ tenfold in how many events that
actually prevents.

Both examples are stipulated illustrations. No measurement, product, or trial
is being described.
"""

from dataclasses import dataclass

from blog_reproducibility.common.validation import non_negative, positive

__all__ = [
    "EXPOSURES",
    "RISKS",
    "ExposureRow",
    "RiskCommunicationSummary",
    "RiskRow",
    "example_payload",
    "exposure_rows",
    "risk_rows",
]


@dataclass(frozen=True, slots=True)
class ExposureRow:
    """One sample: how concentrated it is, how much was taken, and the dose."""

    name: str
    concentration_mg_per_ml: float
    volume_ml: float
    amount_mg: float


@dataclass(frozen=True, slots=True)
class RiskRow:
    """One population: events before and after, in both absolute and relative terms."""

    name: str
    before_per_1000: float
    after_per_1000: float
    difference_per_1000: float
    relative_reduction: float


@dataclass(frozen=True, slots=True)
class RiskCommunicationSummary:
    """Every number the two articles report."""

    exposures: tuple[ExposureRow, ...]
    risks: tuple[RiskRow, ...]


# Name, concentration in mg/mL, volume in mL.
EXPOSURES: tuple[tuple[str, float, float], ...] = (
    ("A", 10.0, 2.0),
    ("B", 1.0, 30.0),
    ("C", 1.0, 20.0),
)
# Name, five-year events per 1,000 people before and after.
RISKS: tuple[tuple[str, float, float], ...] = (
    ("Population A", 20.0, 10.0),
    ("Population B", 2.0, 1.0),
)


def exposure_rows() -> tuple[ExposureRow, ...]:
    """Return concentration, volume, and the dose each combination delivers."""
    return tuple(
        ExposureRow(
            name=name,
            concentration_mg_per_ml=non_negative(concentration, name="concentration"),
            volume_ml=non_negative(volume, name="volume"),
            amount_mg=concentration * volume,
        )
        for name, concentration, volume in EXPOSURES
    )


def risk_rows() -> tuple[RiskRow, ...]:
    """Return the absolute and relative views of the same reduction."""
    return tuple(
        RiskRow(
            name=name,
            before_per_1000=positive(before, name="before"),
            after_per_1000=non_negative(after, name="after"),
            difference_per_1000=before - after,
            relative_reduction=1 - after / before,
        )
        for name, before, after in RISKS
    )


def example_payload() -> RiskCommunicationSummary:
    """Return the numbers the two articles report."""
    return RiskCommunicationSummary(exposures=exposure_rows(), risks=risk_rows())
