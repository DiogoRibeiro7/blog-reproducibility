"""Figure renderer for the article on the Savitzky-Golay filter."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.data_science.savitzky_golay import (
    POLYNOMIAL_ORDER,
    WINDOW,
    smooth_signal,
)


def render_smoothing_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot the raw signal against its moving-average and Savitzky-Golay smooths."""
    use_house_style()
    signal = smooth_signal()

    figure, axis = plt.subplots()
    axis.plot(signal.time, signal.noisy, color=INK_MUTED, lw=1.0, alpha=0.7, label="Raw signal")
    axis.plot(
        signal.time, signal.moving_average, color=PALETTE[1], label=f"Moving average ({WINDOW})"
    )
    axis.plot(
        signal.time,
        signal.savitzky_golay,
        color=PALETTE[0],
        label=f"Savitzky-Golay ({WINDOW}, order {POLYNOMIAL_ORDER})",
    )
    axis.set(
        title="Savitzky-Golay preserves peak height; a moving average does not",
        xlabel="time",
        ylabel="amplitude",
    )
    axis.legend()

    return save_figure(figure, slug="savitzky_golay", output_dir=output_dir)
