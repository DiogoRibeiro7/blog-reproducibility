"""Figure renderer for the article on Markov chain Monte Carlo."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blog_reproducibility.common.plotting import (
    INK_PRIMARY,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.mathematics.metropolis import (
    BURN_IN,
    TARGET_MEAN,
    TARGET_SD,
    run_chains,
)


def render_trace_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot early traces of four chains and the pooled posterior against the target."""
    use_house_style()
    chains, _ = run_chains()
    figure, (traces, posterior) = plt.subplots(
        1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    for index, chain in enumerate(chains):
        traces.plot(
            chain[:800], color=PALETTE[index], lw=1.0, alpha=0.9, label=f"Chain {index + 1}"
        )
    traces.set_title("Traces (first 800 draws)", fontsize=11)
    traces.set(xlabel="iteration", ylabel="value")
    traces.legend(ncol=4, fontsize=8.5)

    pooled = np.concatenate([chain[BURN_IN:] for chain in chains])
    posterior.hist(pooled, bins=60, density=True, color=PALETTE[0], alpha=0.85)
    grid = np.linspace(pooled.min(), pooled.max(), 300)
    density = np.exp(-0.5 * ((grid - TARGET_MEAN) / TARGET_SD) ** 2) / (
        TARGET_SD * np.sqrt(2 * np.pi)
    )
    posterior.plot(grid, density, color=INK_PRIMARY, lw=1.6)
    posterior.set_title("Pooled posterior vs analytic density", fontsize=11)
    posterior.set_xlabel("value")
    figure.suptitle(
        "Well-mixed chains converge on the same posterior",
        x=0.012,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )

    return save_figure(figure, slug="mcmc_trace", output_dir=output_dir)
