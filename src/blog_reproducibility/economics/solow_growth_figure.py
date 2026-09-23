"""Figure renderer for the article on the Solow growth model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.economics.solow_growth import (
    break_even_investment,
    output_per_worker,
    saving,
    steady_state_capital,
)


def render_steady_state_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot saving against break-even investment and mark the steady state."""
    use_house_style()
    capital = np.linspace(0.01, 14, 600)
    invested = saving(capital)
    required = break_even_investment(capital)
    kstar = steady_state_capital()

    figure, axis = plt.subplots()
    axis.plot(
        capital,
        output_per_worker(capital),
        color=INK_MUTED,
        lw=1.6,
        label="Output per worker $f(k)$",
    )
    axis.plot(capital, invested, color=PALETTE[0], label="Saving $s\\,f(k)$")
    axis.plot(capital, required, color=PALETTE[1], label="Break-even $(n+g+\\delta)k$")
    axis.fill_between(
        capital, invested, required, where=invested > required, color=PALETTE[0], alpha=0.10
    )
    axis.axvline(kstar, color=INK_PRIMARY, lw=1.0)
    axis.annotate(
        f"steady state $k^*$ = {kstar:.1f}",
        xy=(kstar, float(saving(kstar))),
        xytext=(kstar + 1.1, 0.55),
        fontsize=9.5,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "lw": 1, "color": INK_MUTED},
    )
    axis.set(
        title="Concave returns give the Solow model one stable steady state",
        xlabel="capital per effective worker $k$",
        ylabel="output / investment per worker",
        xlim=(0, 14),
        ylim=(0, 2.6),
    )
    axis.legend()

    return save_figure(figure, slug="solow_steady_state", output_dir=output_dir)
