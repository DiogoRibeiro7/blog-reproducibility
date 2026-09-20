"""Solar geometry and heat storage, for the article on why summer follows the tilt.

Two pieces of physics. The first is the geometry: at a given latitude and solar
declination, how high the Sun gets at noon, how long the day is, and how much
energy arrives over the whole day relative to the solar constant. Seasons follow
from that geometry, not from the Earth-Sun distance, which is why the two
hemispheres have opposite seasons at the same instant.

The second is why the warmest weeks come after the solstice rather than on it. A
reservoir with heat capacity, driven periodically, responds with reduced
amplitude and a lag, both set by its time constant.

The model is an ideal sphere and a point Sun. Refraction, the finite angular
size of the Sun, atmospheric absorption, and the eccentricity of the orbit are
all left out, because the article's argument does not depend on any of them.
"""

from dataclasses import dataclass
from math import acos, atan, cos, pi, radians, sin, sqrt, tan

from blog_reproducibility.common.validation import positive, real

__all__ = [
    "OBLIQUITY",
    "SeasonsSummary",
    "SolarDay",
    "ThermalResponse",
    "example_payload",
    "solar_geometry",
    "thermal_response",
]

# Earth's axial tilt in degrees, which is the declination at each solstice.
OBLIQUITY = 23.44


@dataclass(frozen=True, slots=True)
class SolarDay:
    """What the geometry gives for one latitude on one day."""

    latitude: float
    declination: float
    noon_altitude: float
    daylight_hours: float
    daily_mean_over_solar_constant: float

    @property
    def overhead_equivalent_hours(self) -> float:
        """Daily energy expressed as hours of directly overhead sunlight."""
        return 24 * self.daily_mean_over_solar_constant


@dataclass(frozen=True, slots=True)
class ThermalResponse:
    """How a periodically driven reservoir answers its forcing."""

    time_constant_days: float
    amplitude_fraction: float
    lag_days: float


def solar_geometry(latitude: float, declination: float) -> SolarDay:
    """Return noon altitude, daylight hours, and daily mean insolation.

    The daily mean is divided by the solar constant, so it is the fraction of a
    day's worth of directly overhead sunlight that actually arrives. It is
    clamped at zero for polar night, where the Sun never rises.
    """
    place = real(latitude, name="latitude")
    tilt = real(declination, name="declination")
    if not -90.0 <= place <= 90.0:
        raise ValueError("latitude must lie between -90 and 90 degrees")
    if not -90.0 <= tilt <= 90.0:
        raise ValueError("declination must lie between -90 and 90 degrees")

    phi, delta = radians(place), radians(tilt)
    # Clamping covers polar day and polar night, where the sunset angle has no
    # solution and the day is either fully lit or fully dark.
    sunset = acos(max(-1.0, min(1.0, -tan(phi) * tan(delta))))
    daily_mean = (sunset * sin(phi) * sin(delta) + cos(phi) * cos(delta) * sin(sunset)) / pi

    return SolarDay(
        latitude=place,
        declination=tilt,
        noon_altitude=90 - abs(place - tilt),
        daylight_hours=24 * sunset / pi,
        daily_mean_over_solar_constant=max(0.0, daily_mean),
    )


def thermal_response(time_constant_days: float, period_days: float = 365.0) -> ThermalResponse:
    """Amplitude fraction and peak lag for a linear periodically forced reservoir.

    For ``C dT/dt = F(t) - T/R`` driven at angular frequency ``omega``, the
    response amplitude is ``1 / sqrt(1 + (omega*tau)^2)`` and the peak arrives
    ``atan(omega*tau) / omega`` later. A larger reservoir answers later and more
    weakly, which is why the sea lags the land and both lag the solstice.
    """
    tau = positive(time_constant_days, name="time_constant_days")
    period = positive(period_days, name="period_days")

    omega = 2 * pi / period
    return ThermalResponse(
        time_constant_days=tau,
        amplitude_fraction=1 / sqrt(1 + (omega * tau) ** 2),
        lag_days=atan(omega * tau) / omega,
    )


@dataclass(frozen=True, slots=True)
class SeasonsSummary:
    """Every number the article reports."""

    at_45_north: tuple[SolarDay, ...]
    by_latitude: dict[str, tuple[SolarDay, SolarDay]]
    thermal: tuple[ThermalResponse, ...]


def example_payload() -> SeasonsSummary:
    """Return the numbers the article reports."""
    return SeasonsSummary(
        at_45_north=tuple(
            solar_geometry(45, declination) for declination in (-OBLIQUITY, 0.0, OBLIQUITY)
        ),
        by_latitude={
            str(latitude): (
                solar_geometry(latitude, -OBLIQUITY),
                solar_geometry(latitude, OBLIQUITY),
            )
            for latitude in (0, 45, 70)
        },
        thermal=tuple(thermal_response(tau) for tau in (10.0, 30.0, 90.0)),
    )
