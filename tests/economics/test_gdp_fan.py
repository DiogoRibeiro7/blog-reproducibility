"""Check the fan chart simulation against its deterministic limit and its claim.

Without shocks every path is the trend path, which gives an exact reference. The
figure's title claims persistence widens the fan faster than i.i.d. shocks, which
is checked from the same draws.
"""

from dataclasses import replace

import numpy as np
import pytest

from blog_reproducibility.economics.gdp_fan import (
    DEFAULT_SETTINGS,
    FanSettings,
    band_widths,
    example_payload,
    fan_percentiles,
    simulate_gdp_paths,
)

SUMMARY = example_payload()


def test_without_shocks_every_path_is_the_trend() -> None:
    """Zero shock scale gives 100 * 1.02 ** t on every path, with a zero-width fan."""
    paths = simulate_gdp_paths(replace(DEFAULT_SETTINGS, shock_scale=0.0, simulations=20))
    trend = 100 * 1.02 ** np.arange(DEFAULT_SETTINGS.years)

    np.testing.assert_allclose(paths, np.broadcast_to(trend, paths.shape))
    assert band_widths(paths) == pytest.approx((0.0,) * DEFAULT_SETTINGS.years)


def test_paths_start_at_the_index_base() -> None:
    """Every simulated path starts at 100 and has one value per year."""
    paths = simulate_gdp_paths(replace(DEFAULT_SETTINGS, simulations=50))

    assert paths.shape == (50, 11)
    assert np.all(paths[:, 0] == 100.0)


def test_percentiles_are_ordered_and_bands_widen() -> None:
    """Each year's percentiles are ordered and the 90% band widens every year."""
    bands = fan_percentiles(simulate_gdp_paths())

    assert np.all(np.diff(bands, axis=0) >= 0)
    assert np.all(np.diff(SUMMARY.band_90_width) > 0)


def test_persistence_widens_the_fan_faster() -> None:
    """The title's claim: persistent shocks widen the band more than i.i.d. shocks."""
    persistent = SUMMARY.band_90_width
    independent = SUMMARY.independent_band_90_width

    # The first year's shock is identical; persistence only matters afterwards.
    assert persistent[1] == pytest.approx(independent[1])
    assert all(left > right for left, right in zip(persistent[2:], independent[2:], strict=True))
    assert persistent[-1] > 2.5 * independent[-1]


def test_published_percentiles_are_reproduced() -> None:
    """The seeded simulation reproduces the percentiles printed by the script."""
    final = SUMMARY.final_year_percentiles

    assert list(final) == ["p5", "p25", "p50", "p75", "p95"]
    assert final["p50"] == pytest.approx(121.75941619528945, rel=1e-9)
    assert final["p95"] - final["p5"] == pytest.approx(SUMMARY.band_90_width[-1])


def test_simulation_is_deterministic() -> None:
    """The same seed gives the same paths; another seed does not."""
    small = replace(DEFAULT_SETTINGS, simulations=100)

    np.testing.assert_array_equal(simulate_gdp_paths(small), simulate_gdp_paths(small))
    assert not np.array_equal(simulate_gdp_paths(small), simulate_gdp_paths(replace(small, seed=1)))


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"simulations": 0}, ValueError),
        ({"years": 0}, ValueError),
        ({"start_level": 0.0}, ValueError),
        ({"persistence": 1.0}, ValueError),
        ({"shock_scale": -0.01}, ValueError),
        ({"shock_degrees_of_freedom": 0.0}, ValueError),
        ({"seed": -1}, ValueError),
        ({"simulations": 1.5}, TypeError),
    ],
)
def test_invalid_settings_are_rejected(overrides: dict[str, float], error: type[Exception]) -> None:
    """Settings outside the simulation's domain are refused."""
    with pytest.raises(error):
        FanSettings(**overrides)  # type: ignore[arg-type]


def test_invalid_paths_are_rejected() -> None:
    """Percentiles need a non-empty matrix of paths."""
    with pytest.raises(ValueError):
        fan_percentiles(np.zeros(5))
    with pytest.raises(ValueError):
        fan_percentiles(np.zeros((0, 5)))
