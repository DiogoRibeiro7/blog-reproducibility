"""Savitzky-Golay smoothing against a moving average, for the article on the filter.

A Savitzky-Golay filter fits a low-order polynomial by least squares in each
window and keeps its centre value. A cubic fit reproduces any cubic exactly, so
the filter follows a peak's curvature and keeps its height; a moving average is
the order-zero case and flattens every peak narrower than its window.

The signal is two Gaussian peaks, heights 1 and 0.8, sampled at 400 points on
[0, 6] with Gaussian noise of standard deviation 0.06.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter

from blog_reproducibility.common.validation import count, non_negative

__all__ = [
    "PEAKS",
    "POLYNOMIAL_ORDER",
    "SEED",
    "WINDOW",
    "PeakHeights",
    "SmoothedSignal",
    "clean_signal",
    "example_payload",
    "moving_average",
    "smooth_signal",
]

SEED: Final[int] = 20260816
POINTS: Final[int] = 400
NOISE_SD: Final[float] = 0.06
WINDOW: Final[int] = 31
POLYNOMIAL_ORDER: Final[int] = 3
# Centre, height, and the denominator of each Gaussian exponent.
PEAKS: Final[tuple[tuple[float, float, float], ...]] = ((1.6, 1.0, 0.05), (3.6, 0.8, 0.03))


@dataclass(frozen=True, slots=True)
class SmoothedSignal:
    """The clean signal, its noisy sample, and both smooths on a shared grid."""

    time: NDArray[np.float64]
    clean: NDArray[np.float64]
    noisy: NDArray[np.float64]
    moving_average: NDArray[np.float64]
    savitzky_golay: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PeakHeights:
    """Height of one peak in the clean signal and under each smooth."""

    centre: float
    clean: float
    moving_average: float
    savitzky_golay: float


def clean_signal(time: ArrayLike) -> NDArray[np.float64]:
    """The noise-free signal: two Gaussian peaks."""
    points = np.asarray(time, dtype=np.float64)
    total = np.zeros_like(points)
    for centre, height, width in PEAKS:
        total += height * np.exp(-((points - centre) ** 2) / width)
    return total


def moving_average(values: ArrayLike, window: int = WINDOW) -> NDArray[np.float64]:
    """Centred moving average; the edges average over zeros, as ``np.convolve`` does."""
    size = count(window, name="window", minimum=1)
    return np.convolve(np.asarray(values, dtype=np.float64), np.ones(size) / size, mode="same")


def smooth_signal(*, seed: int = SEED, noise_sd: float = NOISE_SD) -> SmoothedSignal:
    """Sample the noisy signal and smooth it both ways."""
    rng = np.random.default_rng(count(seed, name="seed"))
    time = np.linspace(0, 6, POINTS)
    clean = clean_signal(time)
    noisy = clean + rng.normal(0, non_negative(noise_sd, name="noise_sd"), time.size)
    return SmoothedSignal(
        time=time,
        clean=clean,
        noisy=noisy,
        moving_average=moving_average(noisy),
        savitzky_golay=np.asarray(savgol_filter(noisy, WINDOW, POLYNOMIAL_ORDER)),
    )


def example_payload() -> tuple[PeakHeights, ...]:
    """Height of each peak before and after smoothing."""
    signal = smooth_signal()
    heights = []
    for centre, _, _ in PEAKS:
        index = int(np.argmin(np.abs(signal.time - centre)))
        heights.append(
            PeakHeights(
                centre=centre,
                clean=float(signal.clean[index]),
                moving_average=float(signal.moving_average[index]),
                savitzky_golay=float(signal.savitzky_golay[index]),
            )
        )
    return tuple(heights)
