"""Photon energy against absorbed energy, for the microwave article.

The article separates two quantities that both get called "energy" and answer
different questions. A microwave photon at 2.45 GHz carries about ten
microelectronvolts. Breaking a chemical bond takes a few electronvolts, and
ionising an atom takes more, so no number of microwave photons ionises anything:
each one arrives on its own and is far too small.

What a microwave oven does deliver is a great many of them. At 500 W absorbed
that is about 3 × 10^26 photons a second, and over a minute it is 30 kilojoules
of heat. The food gets hot and stays exactly as radioactive as it was.

The constants are the exact SI definitions, so nothing here depends on a
measurement. There is no exposure model and no absorption model: the absorbed
power is a stated assumption, not a prediction.
"""

from dataclasses import dataclass
from typing import Final

from blog_reproducibility.common.validation import positive

__all__ = [
    "ELEMENTARY_CHARGE",
    "GREEN_WAVELENGTH_M",
    "MICROWAVE_FREQUENCY_HZ",
    "PLANCK",
    "SPEED_OF_LIGHT",
    "MicrowaveSummary",
    "absorbed_energy_joules",
    "example_payload",
    "photon_energy_ev",
    "photon_energy_ev_from_wavelength",
    "photons_per_second",
]

# All three are exact by the SI definitions of the second, the metre, and the ampere.
PLANCK: Final[float] = 6.62607015e-34
ELEMENTARY_CHARGE: Final[float] = 1.602176634e-19
SPEED_OF_LIGHT: Final[float] = 299_792_458.0

# The frequency domestic ovens use.
MICROWAVE_FREQUENCY_HZ: Final[float] = 2.45e9
# Green light, for a photon everyone has met.
GREEN_WAVELENGTH_M: Final[float] = 550e-9


@dataclass(frozen=True, slots=True)
class MicrowaveSummary:
    """Every number the article reports."""

    microwave_photon_ev: float
    green_photon_ev: float
    green_over_microwave: float
    photons_per_second_at_500w: float
    absorbed_kilojoules: dict[str, float]


def photon_energy_ev(frequency_hz: float) -> float:
    """Return the energy of one photon at that frequency, in electronvolts."""
    frequency = positive(frequency_hz, name="frequency_hz")
    return PLANCK * frequency / ELEMENTARY_CHARGE


def photon_energy_ev_from_wavelength(wavelength_m: float) -> float:
    """Return the energy of one photon at that wavelength, in electronvolts."""
    wavelength = positive(wavelength_m, name="wavelength_m")
    return photon_energy_ev(SPEED_OF_LIGHT / wavelength)


def photons_per_second(power_watts: float, frequency_hz: float = MICROWAVE_FREQUENCY_HZ) -> float:
    """Return how many photons a second that absorbed power corresponds to."""
    power = positive(power_watts, name="power_watts")
    frequency = positive(frequency_hz, name="frequency_hz")
    return power / (PLANCK * frequency)


def absorbed_energy_joules(power_watts: float, seconds: float) -> float:
    """Return the energy absorbed over a period at a constant power."""
    power = positive(power_watts, name="power_watts")
    duration = positive(seconds, name="seconds")
    return power * duration


def example_payload() -> MicrowaveSummary:
    """Return the numbers the article reports."""
    microwave = photon_energy_ev(MICROWAVE_FREQUENCY_HZ)
    green = photon_energy_ev_from_wavelength(GREEN_WAVELENGTH_M)

    return MicrowaveSummary(
        microwave_photon_ev=microwave,
        green_photon_ev=green,
        green_over_microwave=green / microwave,
        photons_per_second_at_500w=photons_per_second(500.0),
        absorbed_kilojoules={
            f"{int(power)} W": absorbed_energy_joules(power, 60.0) / 1000.0
            for power in (500.0, 1000.0)
        },
    )
