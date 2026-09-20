"""Check the solar geometry and the heat-storage response.

The daily mean insolation is checked against direct numerical integration of the
changing solar angle through the day, which does not use the closed form the
model evaluates. The thermal response is checked against numerical integration
of the energy-balance equation it solves.
"""

from math import cos, pi, radians, sin

import pytest

from blog_reproducibility.physics.seasons import (
    OBLIQUITY,
    example_payload,
    solar_geometry,
    thermal_response,
)

SUMMARY = example_payload()


def test_daily_mean_against_direct_integration_of_the_solar_angle() -> None:
    """Integrating max(0, sin of the solar altitude) over the day gives the same mean."""
    for latitude, declination in ((45, -OBLIQUITY), (45, 0.0), (45, OBLIQUITY), (-20, 10.0)):
        phi, delta = radians(latitude), radians(declination)
        steps = 200_000
        total = 0.0
        for index in range(steps):
            hour_angle = -pi + (index + 0.5) * (2 * pi / steps)
            altitude_sine = sin(phi) * sin(delta) + cos(phi) * cos(delta) * cos(hour_angle)
            total += max(0.0, altitude_sine)
        integrated = total / steps

        computed = solar_geometry(latitude, declination).daily_mean_over_solar_constant
        assert computed == pytest.approx(integrated, abs=1e-5)


def test_hemispheres_are_mirror_images() -> None:
    """Flipping the latitude and the declination together reproduces the day exactly."""
    for latitude, declination in ((45, OBLIQUITY), (12, -5.0), (70, OBLIQUITY)):
        north = solar_geometry(latitude, declination)
        south = solar_geometry(-latitude, -declination)

        assert south.daylight_hours == pytest.approx(north.daylight_hours)
        assert south.daily_mean_over_solar_constant == pytest.approx(
            north.daily_mean_over_solar_constant
        )
        assert south.noon_altitude == pytest.approx(north.noon_altitude)


def test_the_equinox_gives_twelve_hours_everywhere() -> None:
    """At zero declination every latitude has the same day length."""
    for latitude in (-70, -45, 0, 30, 45, 70):
        assert solar_geometry(latitude, 0.0).daylight_hours == pytest.approx(12.0)


def test_polar_day_and_polar_night() -> None:
    """Above the Arctic Circle the Sun either never sets or never rises."""
    polar_night = solar_geometry(70, -OBLIQUITY)
    polar_day = solar_geometry(70, OBLIQUITY)

    assert polar_night.daylight_hours == 0.0
    assert polar_night.daily_mean_over_solar_constant == 0.0
    assert polar_day.daylight_hours == 24.0
    assert polar_day.daily_mean_over_solar_constant > 0

    # Midsummer at the pole beats the equator on daily total, despite a low Sun.
    equator = solar_geometry(0, OBLIQUITY)
    assert polar_day.overhead_equivalent_hours > equator.overhead_equivalent_hours


def test_the_tilt_and_not_the_distance_drives_the_seasons() -> None:
    """The two hemispheres are opposite at the same instant, at one distance."""
    december_north = solar_geometry(45, -OBLIQUITY)
    december_south = solar_geometry(-45, -OBLIQUITY)

    assert december_north.daylight_hours < 12 < december_south.daylight_hours
    assert (
        december_north.daily_mean_over_solar_constant
        < december_south.daily_mean_over_solar_constant
    )


def test_thermal_response_against_numerical_integration() -> None:
    """Integrating the energy-balance equation reproduces the amplitude and lag.

    The reservoir obeys ``tau dT/dt + T = cos(omega t)``. Stepping that forward
    and reading off the steady-state peak does not use the closed form.
    """
    period = 365.0
    omega = 2 * pi / period
    steps_per_period = 100_000
    step = period / steps_per_period

    for tau in (10.0, 30.0, 90.0):
        # Run several periods first so the transient decays.
        temperature = 0.0
        warmup = 5 * steps_per_period
        for index in range(warmup):
            temperature += step * (cos(omega * index * step) - temperature) / tau

        peak_value, peak_time = -1e9, 0.0
        for index in range(warmup, warmup + steps_per_period):
            time = index * step
            temperature += step * (cos(omega * time) - temperature) / tau
            if temperature > peak_value:
                peak_value, peak_time = temperature, time

        expected = thermal_response(tau, period)
        assert peak_value == pytest.approx(expected.amplitude_fraction, rel=0.01)
        assert peak_time % period == pytest.approx(expected.lag_days, abs=0.5)


def test_thermal_response_limits() -> None:
    """A tiny reservoir follows the forcing; a large one lags a quarter period."""
    quick = thermal_response(1e-9)
    assert quick.amplitude_fraction == pytest.approx(1.0)
    assert quick.lag_days == pytest.approx(0.0, abs=1e-6)

    slow = thermal_response(1e9)
    assert slow.amplitude_fraction < 1e-6
    assert slow.lag_days == pytest.approx(365 / 4, rel=1e-6)

    # A bigger reservoir always responds later and more weakly.
    values = [thermal_response(tau) for tau in (5.0, 20.0, 80.0, 200.0)]
    assert [row.amplitude_fraction for row in values] == sorted(
        [row.amplitude_fraction for row in values], reverse=True
    )
    assert [row.lag_days for row in values] == sorted([row.lag_days for row in values])


def test_published_numbers() -> None:
    """Every value the article prints should come back unchanged."""
    winter, equinox, summer = SUMMARY.at_45_north

    assert winter.noon_altitude == pytest.approx(21.56)
    assert winter.daylight_hours == pytest.approx(8.574108, abs=5e-7)
    assert winter.daily_mean_over_solar_constant == pytest.approx(0.085598, abs=5e-7)

    assert equinox.noon_altitude == pytest.approx(45.0)
    assert equinox.daylight_hours == pytest.approx(12.0)
    assert equinox.daily_mean_over_solar_constant == pytest.approx(0.225079, abs=5e-7)

    assert summer.noon_altitude == pytest.approx(68.44)
    assert summer.daylight_hours == pytest.approx(15.425892, abs=5e-7)
    assert summer.daily_mean_over_solar_constant == pytest.approx(0.366877, abs=5e-7)

    ten, thirty, ninety = SUMMARY.thermal
    assert ten.amplitude_fraction == pytest.approx(0.985505, abs=5e-7)
    assert ten.lag_days == pytest.approx(9.902944, abs=5e-7)
    assert thirty.amplitude_fraction == pytest.approx(0.888513, abs=5e-7)
    assert thirty.lag_days == pytest.approx(27.692362, abs=5e-7)
    assert ninety.amplitude_fraction == pytest.approx(0.542305, abs=5e-7)
    assert ninety.lag_days == pytest.approx(57.953185, abs=5e-7)

    equator_winter, equator_summer = SUMMARY.by_latitude["0"]
    assert equator_winter.overhead_equivalent_hours == pytest.approx(7.009009, abs=5e-7)
    assert equator_summer.overhead_equivalent_hours == pytest.approx(7.009009, abs=5e-7)


def test_invalid_inputs() -> None:
    """Latitudes and declinations off the sphere, and non-positive constants, are rejected."""
    with pytest.raises(ValueError):
        solar_geometry(95, 0.0)
    with pytest.raises(ValueError):
        solar_geometry(45, 95.0)
    with pytest.raises(ValueError):
        solar_geometry(float("nan"), 0.0)
    with pytest.raises(ValueError):
        thermal_response(0.0)
    with pytest.raises(ValueError):
        thermal_response(30.0, period_days=0.0)
