"""Dose arithmetic for the essay on aspartame, fruit, and quantity.

The essay's argument is that "its breakdown products occur in fruit" is true and
incomplete, and that the missing part is dose. Everything below is arithmetic on
published figures: methanol from each source on one scale, cans needed to reach
the acceptable daily intake, and a one-compartment prediction of peak blood
methanol checked against concentrations measured in volunteers.

The one-compartment prediction is deliberately crude. It assumes the whole dose
is absorbed at once and none is cleared, so it is an upper bound rather than a
pharmacokinetic model, and the point of comparing it with the measured peaks is
to show that the bound is already far below any concentration that matters.

Sources are named beside each constant.
"""

from dataclasses import dataclass
from typing import Final

from blog_reproducibility.common.validation import count, non_negative, positive

__all__ = [
    "ADI",
    "APPLES_1KG",
    "BLOOD_THRESHOLDS",
    "CAN_LITRES",
    "DETECTION_LIMIT",
    "DRINK",
    "ENDOGENOUS",
    "HIGH_EXPOSURE",
    "JUICE",
    "LETHAL_PER_KG",
    "METHANOL_SHARE",
    "NUTRINET_HAZARD_INTERVAL",
    "NUTRINET_HAZARD_RATIO",
    "NUTRINET_HIGHER_CONSUMERS_MG_PER_DAY",
    "PHENYLALANINE_SHARE",
    "STEGINK",
    "VOLUME_OF_DISTRIBUTION",
    "AspartameSummary",
    "MeasuredPeak",
    "cans_to_reach",
    "example_payload",
    "methanol_from",
    "predicted_peak",
]

# Share of aspartame that becomes methanol on hydrolysis (WHO EHC 196; EFSA 2013).
METHANOL_SHARE: Final[float] = 0.10
# Share that becomes phenylalanine (EFSA 2013).
PHENYLALANINE_SHARE: Final[float] = 0.56
# Acceptable daily intake, mg of aspartame per kg of body weight per day.
ADI: Final[dict[str, int]] = {"EFSA and JECFA": 40, "FDA": 50}
CAN_LITRES: Final[float] = 0.33
# mg of aspartame per litre of soft drink: the EU maximum permitted level, and the
# means of two Portuguese surveys (Lino et al. 2008; Basilio et al. 2020, both
# Food Addit Contam Part A).
DRINK: Final[dict[str, float]] = {
    "maximum permitted": 600.0,
    "measured mean, 2008": 89.0,
    "measured mean, 2020": 161.5,
}
# mg of methanol per litre of fruit juice, WHO EHC 196: mean and range.
JUICE: Final[dict[str, float]] = {"mean": 140.0, "low": 12.0, "high": 640.0}
# mg of methanol from the pectin in 1 kg of apples (Lindinger et al. 1997).
APPLES_1KG: Final[tuple[int, int]] = (400, 1400)
# mg of methanol the body makes per day (Lindinger et al. 1997).
ENDOGENOUS: Final[tuple[int, int]] = (300, 600)
# Minimum lethal dose of methanol, mg/kg, untreated (WHO EHC 196).
LETHAL_PER_KG: Final[tuple[int, int]] = (300, 1000)
# Volume of distribution for methanol, litres per kg (EFSA 2013, citing Graw et al. 2000).
VOLUME_OF_DISTRIBUTION: Final[float] = 0.77
# Stegink et al. 1981: mean peak blood methanol, mg/L, with SD, after one dose of
# aspartame in mg/kg. At 34 mg/kg it was below the 4 mg/L detection limit in all
# twelve subjects.
STEGINK: Final[dict[int, tuple[float, float]]] = {
    100: (12.7, 4.8),
    150: (21.4, 3.5),
    200: (25.8, 7.8),
}
DETECTION_LIMIT: Final[float] = 4.0
# mg/L of blood methanol at which effects begin, WHO EHC 196.
BLOOD_THRESHOLDS: Final[dict[str, int]] = {
    "effects on the nervous system": 200,
    "effects on the eye": 500,
}
# Debras et al. 2022: mean aspartame intake of the higher-consumer group, mg/day,
# and the hazard ratio reported for it.
NUTRINET_HIGHER_CONSUMERS_MG_PER_DAY: Final[float] = 47.42
NUTRINET_HAZARD_RATIO: Final[float] = 1.15
NUTRINET_HAZARD_INTERVAL: Final[tuple[float, float]] = (1.03, 1.28)
# EFSA 2013, Table 8: highest 95th-percentile exposure, mg/kg/day, by age group.
HIGH_EXPOSURE: Final[dict[str, float]] = {"toddlers": 36.0, "children": 32.4, "adults": 27.5}


@dataclass(frozen=True, slots=True)
class MeasuredPeak:
    """A dose, what the one-compartment bound predicts, and what was measured."""

    dose_mg_per_kg: int
    predicted_mg_per_litre: float
    observed_mg_per_litre: float
    observed_sd: float


@dataclass(frozen=True, slots=True)
class AspartameSummary:
    """Every number the essay reports."""

    aspartame_in_one_can_mg: dict[str, float]
    methanol_from_one_can_mg: dict[str, float]
    methanol_from_a_glass_of_juice_mg: dict[str, float]
    methanol_at_the_adi_70kg_mg: float
    lethal_dose_70kg_mg: tuple[int, int]
    cans_to_reach_the_adi: dict[str, float]
    measured_peaks: tuple[MeasuredPeak, ...]
    predicted_peak_at_34_mg_per_kg: float
    predicted_peak_at_the_adi: float
    predicted_rise_from_one_can_70kg: float
    nervous_system_threshold_over_one_can: float
    high_exposure_share_of_adi_percent: dict[str, float]
    nutrinet_share_of_adi_percent: float
    phenylalanine_from_one_can_mg: float


def methanol_from(aspartame_mg: float) -> float:
    """Return the methanol released by hydrolysing a given mass of aspartame."""
    return METHANOL_SHARE * non_negative(aspartame_mg, name="aspartame_mg")


def predicted_peak(aspartame_mg_per_kg: float) -> float:
    """Upper bound on peak blood methanol, mg/L, from one dose.

    The whole dose is assumed absorbed at once into the volume of distribution
    with nothing cleared, so the value is a ceiling rather than a prediction of
    what a pharmacokinetic model would give.
    """
    dose = non_negative(aspartame_mg_per_kg, name="aspartame_mg_per_kg")
    return methanol_from(dose) / VOLUME_OF_DISTRIBUTION


def cans_to_reach(adi_mg_per_kg: float, weight_kg: float, mg_per_litre: float) -> float:
    """Return how many cans a person of that weight needs to reach the daily intake."""
    intake = positive(adi_mg_per_kg, name="adi_mg_per_kg")
    weight = positive(weight_kg, name="weight_kg")
    concentration = positive(mg_per_litre, name="mg_per_litre")
    return intake * weight / (concentration * CAN_LITRES)


def example_payload() -> AspartameSummary:
    """Return the numbers the essay reports."""
    per_can = {name: round(level * CAN_LITRES, 1) for name, level in DRINK.items()}
    one_can_70kg = DRINK["maximum permitted"] * CAN_LITRES / 70.0
    adi = ADI["EFSA and JECFA"]
    rise = predicted_peak(one_can_70kg)

    return AspartameSummary(
        aspartame_in_one_can_mg=per_can,
        methanol_from_one_can_mg={
            name: round(methanol_from(mass), 1) for name, mass in per_can.items()
        },
        methanol_from_a_glass_of_juice_mg={
            name: round(0.25 * level, 1) for name, level in JUICE.items()
        },
        methanol_at_the_adi_70kg_mg=round(methanol_from(adi * 70)),
        lethal_dose_70kg_mg=(70 * LETHAL_PER_KG[0], 70 * LETHAL_PER_KG[1]),
        cans_to_reach_the_adi={
            f"{weight} kg, {name}": round(cans_to_reach(adi, weight, level), 1)
            for weight in (70, 20)
            for name, level in DRINK.items()
        },
        measured_peaks=tuple(
            MeasuredPeak(
                dose_mg_per_kg=dose,
                predicted_mg_per_litre=round(predicted_peak(dose), 1),
                observed_mg_per_litre=observed,
                observed_sd=sd,
            )
            for dose, (observed, sd) in STEGINK.items()
        ),
        predicted_peak_at_34_mg_per_kg=round(predicted_peak(34), 1),
        predicted_peak_at_the_adi=round(predicted_peak(adi), 1),
        predicted_rise_from_one_can_70kg=round(rise, 2),
        nervous_system_threshold_over_one_can=round(
            BLOOD_THRESHOLDS["effects on the nervous system"] / rise
        ),
        high_exposure_share_of_adi_percent={
            group: round(100 * exposure / adi) for group, exposure in HIGH_EXPOSURE.items()
        },
        nutrinet_share_of_adi_percent=round(
            100 * NUTRINET_HIGHER_CONSUMERS_MG_PER_DAY / (adi * 70), 1
        ),
        phenylalanine_from_one_can_mg=round(
            PHENYLALANINE_SHARE * DRINK["maximum permitted"] * CAN_LITRES
        ),
    )


def dose_grid(points: int = 500) -> tuple[float, ...]:
    """Return the dose axis used by the kinetics figure, 1 to 250 mg/kg."""
    size = count(points, name="points", minimum=2)
    return tuple(step / 2 for step in range(2, size + 1))
