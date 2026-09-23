"""Check the Savitzky-Golay property the figure depends on, then the figure's claim.

A cubic Savitzky-Golay filter reproduces any cubic exactly away from the edges,
and a moving average does not, which is why one keeps peak height and the other
does not.
"""

import numpy as np
import pytest
from scipy.signal import savgol_filter

from blog_reproducibility.data_science.savitzky_golay import (
    POLYNOMIAL_ORDER,
    WINDOW,
    clean_signal,
    example_payload,
    moving_average,
    smooth_signal,
)

SUMMARY = example_payload()


def test_cubic_filter_reproduces_cubics() -> None:
    """Any cubic passes through the filter unchanged; the moving average bends it."""
    x = np.linspace(-1, 1, 101)
    cubic = 2 * x**3 - x**2 + 0.5 * x + 3
    interior = slice(WINDOW, -WINDOW)

    filtered = savgol_filter(cubic, WINDOW, POLYNOMIAL_ORDER)
    np.testing.assert_allclose(filtered[interior], cubic[interior], atol=1e-9)
    assert np.max(np.abs(moving_average(cubic)[interior] - cubic[interior])) > 0.01


def test_moving_average_preserves_lines() -> None:
    """A symmetric window averages a straight line back to itself."""
    line = np.linspace(0, 5, 200)
    interior = slice(WINDOW, -WINDOW)

    np.testing.assert_allclose(moving_average(line)[interior], line[interior], atol=1e-12)


def test_savitzky_golay_keeps_peak_height() -> None:
    """The figure's claim: the moving average flattens both peaks, Savitzky-Golay does not."""
    for peak in SUMMARY:
        assert peak.savitzky_golay == pytest.approx(peak.clean, rel=0.08)
        assert peak.moving_average < 0.8 * peak.clean
        assert abs(peak.savitzky_golay - peak.clean) < abs(peak.moving_average - peak.clean)


def test_savitzky_golay_tracks_the_clean_signal_better() -> None:
    """Over the whole record, the filter stays closer to the truth than the average."""
    signal = smooth_signal()
    interior = slice(WINDOW, -WINDOW)

    def error(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean((values[interior] - signal.clean[interior]) ** 2)))

    assert error(signal.savitzky_golay) < error(signal.moving_average)
    assert error(signal.savitzky_golay) < error(signal.noisy)


def test_clean_signal_has_the_published_peaks() -> None:
    """Heights 1 and 0.8 at 1.6 and 3.6."""
    assert clean_signal([1.6, 3.6]) == pytest.approx([1.0, 0.8], abs=1e-3)


def test_without_noise_both_smooths_are_deterministic() -> None:
    """Zero noise leaves the clean signal as the input."""
    signal = smooth_signal(noise_sd=0.0)

    np.testing.assert_array_equal(signal.noisy, signal.clean)
    with pytest.raises(ValueError):
        moving_average([1.0], window=0)
