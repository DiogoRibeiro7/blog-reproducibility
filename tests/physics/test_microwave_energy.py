"""Check the photon-energy arithmetic against the constants it comes from."""

import pytest

from blog_reproducibility.physics.microwave_energy import (
    ELEMENTARY_CHARGE,
    GREEN_WAVELENGTH_M,
    MICROWAVE_FREQUENCY_HZ,
    PLANCK,
    SPEED_OF_LIGHT,
    absorbed_energy_joules,
    example_payload,
    photon_energy_ev,
    photon_energy_ev_from_wavelength,
    photons_per_second,
)

SUMMARY = example_payload()


def test_photon_energy_is_planck_times_frequency() -> None:
    """Recomputing from the constants gives the same electronvolts."""
    expected = PLANCK * MICROWAVE_FREQUENCY_HZ / ELEMENTARY_CHARGE
    assert photon_energy_ev(MICROWAVE_FREQUENCY_HZ) == pytest.approx(expected)
    assert photon_energy_ev(2 * MICROWAVE_FREQUENCY_HZ) == pytest.approx(2 * expected)


def test_wavelength_and_frequency_agree() -> None:
    """The two routes to a photon's energy are the same calculation."""
    frequency = SPEED_OF_LIGHT / GREEN_WAVELENGTH_M
    assert photon_energy_ev_from_wavelength(GREEN_WAVELENGTH_M) == pytest.approx(
        photon_energy_ev(frequency)
    )


def test_a_microwave_photon_is_far_below_any_chemical_bond() -> None:
    """Ten microelectronvolts against the few electronvolts a bond needs."""
    assert SUMMARY.microwave_photon_ev < 1e-4
    assert SUMMARY.green_photon_ev > 2.0
    assert SUMMARY.green_over_microwave > 100_000
    # Even the whole visible photon is well below an ionisation energy.
    assert SUMMARY.green_photon_ev < 13.6


def test_the_oven_delivers_very_many_very_small_photons() -> None:
    """Energy arrives in quantity, not in quality."""
    assert SUMMARY.photons_per_second_at_500w > 1e26
    # Photons a second times the energy of each is the absorbed power.
    joules_each = PLANCK * MICROWAVE_FREQUENCY_HZ
    assert SUMMARY.photons_per_second_at_500w * joules_each == pytest.approx(500.0)
    assert photons_per_second(1000.0) == pytest.approx(2 * photons_per_second(500.0))


def test_absorbed_energy_is_power_times_time() -> None:
    """A minute at 500 W is 30 kJ; twice the power is twice the energy."""
    assert absorbed_energy_joules(500.0, 60.0) == pytest.approx(30_000.0)
    assert SUMMARY.absorbed_kilojoules == {"500 W": 30.0, "1000 W": 60.0}


def test_published_values() -> None:
    """The numbers the article prints."""
    assert SUMMARY.microwave_photon_ev == pytest.approx(1.0132385857e-05, rel=1e-9)
    assert SUMMARY.green_photon_ev == pytest.approx(2.254258153, rel=1e-9)
    assert SUMMARY.green_over_microwave == pytest.approx(222_480.488, rel=1e-8)
    assert SUMMARY.photons_per_second_at_500w == pytest.approx(3.07997996e26, rel=1e-8)


def test_invalid_inputs() -> None:
    """Non-positive frequencies, wavelengths, powers, and durations are rejected."""
    with pytest.raises(ValueError):
        photon_energy_ev(0.0)
    with pytest.raises(ValueError):
        photon_energy_ev_from_wavelength(-1.0)
    with pytest.raises(ValueError):
        photons_per_second(0.0)
    with pytest.raises(ValueError):
        absorbed_energy_joules(500.0, 0.0)
