"""Monte Carlo GDP fan chart for the article on Monte Carlo macroeconomic modelling.

GDP starts at 100 and grows each year at a trend rate plus a deviation that
follows an AR(1) process driven by heavy-tailed Student-t shocks:

    e_t = rho * e_(t-1) + scale * t_df,
    Y_t = Y_(t-1) * (1 + trend + e_t).

The fan summarises many simulated paths by their percentiles each year. The
figure's claim is that persistence makes the fan widen faster than i.i.d.
shocks would, because a shock keeps pushing growth in the same direction for
several years; the payload reports both widths from the same draws.
"""

from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from numpy.typing import NDArray

from blog_reproducibility.common.validation import count, positive, real

__all__ = [
    "DEFAULT_SETTINGS",
    "PERCENTILES",
    "FanSettings",
    "FanSummary",
    "band_widths",
    "example_payload",
    "fan_percentiles",
    "simulate_gdp_paths",
]

# Outer band, inner band, median.
PERCENTILES: Final[tuple[float, ...]] = (5, 25, 50, 75, 95)


@dataclass(frozen=True, slots=True)
class FanSettings:
    """Simulation settings for the published fan chart."""

    simulations: int = 10_000
    years: int = 11
    start_level: float = 100.0
    trend_growth: float = 0.02
    persistence: float = 0.7
    shock_scale: float = 0.011
    shock_degrees_of_freedom: float = 5.0
    seed: int = 20260816

    def __post_init__(self) -> None:
        count(self.simulations, name="simulations", minimum=1)
        count(self.years, name="years", minimum=1)
        positive(self.start_level, name="start_level")
        real(self.trend_growth, name="trend_growth")
        if not -1 < real(self.persistence, name="persistence") < 1:
            raise ValueError("persistence must lie in (-1, 1)")
        if real(self.shock_scale, name="shock_scale") < 0:
            raise ValueError("shock_scale must be zero or greater")
        positive(self.shock_degrees_of_freedom, name="shock_degrees_of_freedom")
        count(self.seed, name="seed")


DEFAULT_SETTINGS: Final[FanSettings] = FanSettings()


@dataclass(frozen=True, slots=True)
class FanSummary:
    """Final-year percentiles and the 90% band width with and without persistence."""

    settings: FanSettings
    final_year_percentiles: dict[str, float]
    band_90_width: tuple[float, ...]
    independent_band_90_width: tuple[float, ...]


def simulate_gdp_paths(settings: FanSettings = DEFAULT_SETTINGS) -> NDArray[np.float64]:
    """Return an array of shape ``(simulations, years)`` of simulated GDP levels."""
    rng = np.random.default_rng(settings.seed)
    paths = np.full((settings.simulations, settings.years), settings.start_level)
    deviation = np.zeros(settings.simulations)
    for year in range(1, settings.years):
        shocks = rng.standard_t(df=settings.shock_degrees_of_freedom, size=settings.simulations)
        deviation = settings.persistence * deviation + settings.shock_scale * shocks
        paths[:, year] = paths[:, year - 1] * (1 + settings.trend_growth + deviation)
    return paths


def fan_percentiles(paths: NDArray[np.float64]) -> NDArray[np.float64]:
    """Percentiles of each year's GDP, one row per entry of ``PERCENTILES``."""
    if paths.ndim != 2 or paths.shape[0] == 0:
        raise ValueError("paths must be a non-empty two-dimensional array")
    return np.percentile(paths, PERCENTILES, axis=0)


def band_widths(paths: NDArray[np.float64]) -> tuple[float, ...]:
    """Width of the 90% band (5th to 95th percentile) in each year."""
    bands = fan_percentiles(paths)
    return tuple(float(width) for width in bands[-1] - bands[0])


def example_payload(settings: FanSettings = DEFAULT_SETTINGS) -> FanSummary:
    """Return the percentiles and band widths behind the article's figure."""
    paths = simulate_gdp_paths(settings)
    final = fan_percentiles(paths)[:, -1]
    independent = simulate_gdp_paths(replace(settings, persistence=0.0))
    return FanSummary(
        settings=settings,
        final_year_percentiles={
            f"p{percentile:g}": float(value)
            for percentile, value in zip(PERCENTILES, final, strict=True)
        },
        band_90_width=band_widths(paths),
        independent_band_90_width=band_widths(independent),
    )
